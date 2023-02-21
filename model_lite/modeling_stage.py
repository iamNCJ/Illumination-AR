import torch

from LCNet import LCNet
from LCNet_Online import LCNet as LCNet_Online
# from NENet import NENet
from torch.utils.mobile_optimizer import optimize_for_mobile

model = LCNet(c_in=4)
model_query = LCNet_Online(c_in=4)
sd = torch.load("./LCNet_CVPR2019.pth.tar")
model.load_state_dict(sd['state_dict'])
model.cpu()
model_query.load_state_dict(sd['state_dict'])
model_query.cpu()

# normal_net = NENet(c_in=6)
# sd = torch.load("./NENet_CVPR2019.pth.tar")
# normal_net.load_state_dict(sd['state_dict'])
# normal_net.cpu()

# scripted_model = torch.jit.script(model)
# optimized_model = optimize_for_mobile(scripted_model)  # some op not supported on mobile gpu
# optimized_model._save_for_lite_interpreter("modeling_stage.ptl")

# scripted_model = torch.jit.script(model_query)
# optimized_model = optimize_for_mobile(scripted_model)  # some op not supported on mobile gpu
# optimized_model._save_for_lite_interpreter("online_stage.ptl")

import imageio
from einops import rearrange
import os
import numpy as np
from tqdm import tqdm
from natsort import natsorted

# imgs = imageio.v3.imread('/minio/illumination-ar/real-capture-230208/pig.mov') / 255.
imgs = []
for file in natsorted(os.listdir('/home/ncj/workspace/codespace/Illumination-AR/model/data/ToyPSDataset/Pig')):
    print(file)
    if '.bmp' in file:
        imgs.append(imageio.v3.imread('/home/ncj/workspace/codespace/Illumination-AR/model/data/ToyPSDataset/Pig/' + file) / 255.)
imgs = np.stack(imgs)
imgs = rearrange(imgs, 'N H W C -> N C H W')
print(imgs.shape)
imgs = torch.from_numpy(imgs).float()
# mask = torch.ones_like(imgs[:1, :1])
t_h, t_w = 128, 128
imgs = torch.nn.functional.interpolate(imgs, size=(t_h, t_w), mode='bilinear')
# inputs = prepareInputs(imgs, mask)
# print(inputs.shape)
with torch.no_grad():
    dirs, ints, feat_fused = model(imgs)
    print(dirs.shape)
    print(ints.shape)
    print(feat_fused.shape)
    print(feat_fused.max())
    print(feat_fused.min())
    feat_fused = torch.ones([1, 256, 8, 8]) * -torch.inf
    for img in imgs:
        dirs, ints, feat_fused = model_query(img[None], feat_fused)
        print(dirs, ints)
