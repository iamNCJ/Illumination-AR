import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import model_utils
import eval_utils

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
        return self.dir_x_est(out), self.dir_y_est(out), self.int_est(out),

class LCNet(nn.Module):
    def __init__(self, batchNorm=False, c_in=3):
        super(LCNet, self).__init__()
        self.featExtractor = FeatExtractor(batchNorm, c_in)
        self.classifier = Classifier(batchNorm)
        self.c_in      = c_in

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def convertMidDirsCont(self, dirs_x, dirs_y):
        """Continuous Version"""
        device = dirs_x.device
        x_idx = torch.sum(torch.softmax(dirs_x, dim=-1) * torch.linspace(0, 35, 36, device=device), dim=-1)
        y_idx = torch.sum(torch.softmax(dirs_y, dim=-1) * torch.linspace(0, 35, 36, device=device), dim=-1)
        dirs = eval_utils.SphericalClassToDirs(x_idx, y_idx, 36)
        return dirs

    def convertMidIntens(self, ints, img_num: int):
        _, idx = ints.data.max(1)
        ints = eval_utils.ClassToLightInts(idx, 20)
        ints = ints.view(-1, 1).repeat(1, 3)
        ints = torch.cat(torch.split(ints, ints.shape[0] // img_num, 0), 1)
        return ints

    def forward(self, inputs):  # [img_num, 3, 128, 128]
        inputs = inputs[:, None]  # [img_num, 1, 3, 128, 128]
        mask = torch.ones_like(inputs[:, :, :1], device=inputs.device)
        inputs = torch.cat([inputs, mask], dim=2)  # [img_num, 1, 4, 128, 128]
        feats = []
        for i in range(len(inputs)):
            out_feat = self.featExtractor(inputs[i])
            feats.append(out_feat)
        feat_fused, _ = torch.stack(feats, 1).max(1)

        l_dirs_x, l_dirs_y, l_ints = [], [], []
        for i in range(len(inputs)):
            net_input = torch.cat([feats[i], feat_fused], 1)
            dir_x, dir_y, ints = self.classifier(net_input)
            l_dirs_x.append(dir_x)
            l_dirs_y.append(dir_y)
            l_ints.append(ints)

        dirs_x = torch.cat(l_dirs_x, 0).squeeze()
        dirs_y = torch.cat(l_dirs_y, 0).squeeze()
        # if pred['dirs_x'].dim() == 1:
        #     pred['dirs_x'] = pred['dirs_x'].view(1, -1)
        # if pred['dirs_y'].dim() == 1:
        #     pred['dirs_y'] = pred['dirs_y'].view(1, -1)
        dirs = self.convertMidDirsCont(dirs_x, dirs_y)
        ints = torch.cat(l_ints, 0).squeeze()
        if ints.dim() == 1:
            ints = ints.view(1, -1)
        ints = self.convertMidIntens(ints, len(inputs))
        return dirs, ints, feat_fused
