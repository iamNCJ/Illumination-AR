import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils
from utils import eval_utils
import pickle

# Classification
class FeatExtractor(nn.Module):
    def __init__(self, batchNorm, c_in):
        super(FeatExtractor, self).__init__()
        self.conv1 = model_utils.conv(batchNorm, c_in, 64,    k=3, stride=2, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64,   128,   k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 128,  128,   k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 128,  128,   k=3, stride=2, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 128,  128,   k=3, stride=1, pad=1)
        self.conv6 = model_utils.conv(batchNorm, 128,  256,   k=3, stride=2, pad=1)
        self.conv7 = model_utils.conv(batchNorm, 256,  256,   k=3, stride=1, pad=1)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        return out

class Classifier(nn.Module):
    def __init__(self, batchNorm):
        super(Classifier, self).__init__()
        self.conv1 = model_utils.conv(batchNorm, 512,  256, k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 256,  256, k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 256,  256, k=3, stride=2, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 256,  256, k=3, stride=2, pad=1)
        
        self.dir_x_est = nn.Sequential(
                    model_utils.conv(batchNorm, 256, 64,  k=1, stride=1, pad=0),
                    model_utils.outputConv(64, 36, k=1, stride=1, pad=0))

        self.dir_y_est = nn.Sequential(
                    model_utils.conv(batchNorm, 256, 64,  k=1, stride=1, pad=0),
                    model_utils.outputConv(64, 36, k=1, stride=1, pad=0))

        self.int_est = nn.Sequential(
                    model_utils.conv(batchNorm, 256, 64,  k=1, stride=1, pad=0),
                    model_utils.outputConv(64, 20, k=1, stride=1, pad=0))

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return {
            'dir_x': self.dir_x_est(out),
            'dir_y': self.dir_y_est(out),
            'ints': self.int_est(out),
        }

class LCNet(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(LCNet, self).__init__()
        self.featExtractor = FeatExtractor(batchNorm, c_in)
        self.classifier = Classifier(batchNorm)
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.get_global_feature = False
        self.global_feature_dir = "./data/models/gl_feature.pkl"
        self.use_global_feature = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def prepareInputs(self, x):
        n, c, h, w = x[0].shape
        t_h, t_w = 128, 128
        imgs = torch.nn.functional.interpolate(x[0], size=(t_h, t_w), mode='bilinear')

        inputs = list(torch.split(imgs, 3, 1))
        mask = x[1]
        mask = torch.nn.functional.interpolate(mask, size=(t_h, t_w), mode='bilinear')
        for i in range(len(inputs)):
            inputs[i] = torch.cat([inputs[i], mask], 1)
        return inputs

    def convertMidDirs(self, pred):
        _, x_idx = pred['dirs_x'].data.max(1)
        _, y_idx = pred['dirs_y'].data.max(1)
        dirs = eval_utils.SphericalClassToDirs(x_idx, y_idx, 36)
        return dirs

    def convertMidDirsCont(self, pred):
        """Continuous Version"""
        cls_num = 36
        device = pred['dirs_x'].device
        x_idx = torch.sum(torch.softmax(pred['dirs_x'], dim=-1) * torch.linspace(0, cls_num - 1, cls_num, device=device), dim=-1)
        y_idx = torch.sum(torch.softmax(pred['dirs_y'], dim=-1) * torch.linspace(0, cls_num - 1, cls_num, device=device), dim=-1)
        dirs = eval_utils.SphericalClassToDirs(x_idx, y_idx, cls_num)
        return dirs

    def convertMidIntens(self, pred, img_num):
        _, idx = pred['ints'].data.max(1)
        ints = eval_utils.ClassToLightInts(idx, 20)
        ints = ints.view(-1, 1).repeat(1, 3)
        ints = torch.cat(torch.split(ints, ints.shape[0] // img_num, 0), 1)
        return ints

    def forward(self, x):
        inputs = self.prepareInputs(x)
        inputs = torch.stack(inputs, dim=0)  # [img_num, 1, 4, 128, 128]
        feats = []
        for i in range(len(inputs)):
            out_feat = self.featExtractor(inputs[i])
            feats.append(out_feat)
        feat_fused, _ = torch.stack(feats, 1).max(1)
        if self.get_global_feature:
            with open(self.global_feature_dir,'wb') as f:
                pickle.dump(feat_fused, f)
        if self.use_global_feature:
            with open(self.global_feature_dir,'rb') as f:
                feat_fused = pickle.load(f)

        l_dirs_x, l_dirs_y, l_ints = [], [], []
        for i in range(len(inputs)):
            net_input = torch.cat([feats[i], feat_fused], 1)
            outputs = self.classifier(net_input)
            l_dirs_x.append(outputs['dir_x'])
            l_dirs_y.append(outputs['dir_y'])
            l_ints.append(outputs['ints'])

        pred = {}
        pred['dirs_x'] = torch.cat(l_dirs_x, 0).squeeze()
        pred['dirs_y'] = torch.cat(l_dirs_y, 0).squeeze()
        if pred['dirs_x'].dim() == 1:
            pred['dirs_x'] = pred['dirs_x'].view(1, -1)
        if pred['dirs_y'].dim() == 1:
            pred['dirs_y'] = pred['dirs_y'].view(1, -1)
        pred['dirs']   = self.convertMidDirsCont(pred)
        pred['ints'] = torch.cat(l_ints, 0).squeeze()
        if pred['ints'].dim() == 1:
            pred['ints'] = pred['ints'].view(1, -1)
        pred['intens'] = self.convertMidIntens(pred, len(inputs))
        return pred
