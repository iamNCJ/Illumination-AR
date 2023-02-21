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

    def forward(self, inputs, feat_fused):  # [1, 3, 128, 128]
        mask = torch.ones_like(inputs[:, :1], device=inputs.device)
        inputs = torch.cat([inputs, mask], dim=1)  # [1, 4, 128, 128]
        out_feat = self.featExtractor(inputs)
        feat_fused, _ = torch.stack([out_feat, feat_fused], 1).max(1)

        net_input = torch.cat([out_feat, feat_fused], 1)
        dir_x, dir_y, ints = self.classifier(net_input)
        dirs = self.convertMidDirsCont(dir_x[:, :, 0, 0], dir_y[:, :, 0, 0])
        ints = self.convertMidIntens(ints[:, :, 0, 0], len(inputs))
        return dirs, ints, feat_fused
