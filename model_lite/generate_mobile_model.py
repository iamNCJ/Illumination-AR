import torch

from LCNet_Online import LCNet as LCNet_Online
from torch.utils.mobile_optimizer import optimize_for_mobile

model_query = LCNet_Online(c_in=4)
sd = torch.load("./LCNet_CVPR2019.pth.tar")
model_query.load_state_dict(sd['state_dict'])
model_query.cpu()

scripted_model = torch.jit.script(model_query)
optimized_model = optimize_for_mobile(scripted_model)  # some op not supported on mobile gpu
optimized_model._save_for_lite_interpreter("online_stage.ptl")
