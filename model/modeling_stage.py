import torch, sys
sys.path.append('.')

from datasets import custom_data_loader
from options  import run_model_opts
from models   import custom_model
from utils    import logger, recorders

import test_stage2 as test_utils

args = run_model_opts.RunModelOpts().parse()
args.stage2    = True
args.test_resc = False
args.retrain = "data/models/LCNet_CVPR2019.pth.tar"
args.retrain_s2 = "data/models/NENet_CVPR2019.pth.tar"
args.benchmark = "UPS_Custom_Dataset"
args.bm_dir = "data/ToyPSDataset/"
log  = logger.Logger(args)

def main(args):
    test_loader = custom_data_loader.benchmarkLoader(args)
    model = custom_model.buildModel(args)
    model.get_global_feature = True
    model_s2 = custom_model.buildModelStage2(args)
    models = [model, model_s2]

    recorder = recorders.Records(args.log_dir)
    test_utils.test(args, 'test', test_loader, models, log, 1, recorder)
    log.plotCurves(recorder, 'test')

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)
