import os
import torch
from models import model_utils
from utils import eval_utils, time_utils 
import numpy as np
import pickle

def get_itervals(args, split):
    if split not in ['train', 'val', 'test']:
        split = 'test'
    args_var = vars(args)
    disp_intv = args_var[split+'_disp']
    save_intv = args_var[split+'_save']
    stop_iters = args_var['max_'+split+'_iter']
    return disp_intv, save_intv, stop_iters

def test(args, split, loader, model, log, epoch, recorder):
    model.eval()
    log.printWrite('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    disp_intv, save_intv, stop_iters = get_itervals(args, split)
    res = []
    with torch.no_grad():
        for i, sample in enumerate(loader):
            data = model_utils.parseData(args, sample, timer, split)
            input = model_utils.getInput(args, data)
            if model.get_global_feature or model.use_global_feature:
                model.global_feature_dir="./data/models/gl_feature_"+str(i)+".pkl"
            pred = model(input); timer.updateTime('Forward')
            with open('./data/models/stage1_result/'+str(i)+'.txt','w') as f:
                print("dirs:",file=f)
                print(pred['dirs'],file=f)
                print("intens",file=f)
                print(pred['intens'], file=f)
                print("dirs_x",file=f)
                for j in range(pred["dirs_x"].shape[0]):
                    print(pred["dirs_x"][j],",",file=f)
                print("dirs_y",file=f)
                for j in range(pred["dirs_y"].shape[0]):
                    print(pred["dirs_y"][j],",",file=f)
            with open('./data/models/stage1_result/'+str(i)+'.pkl', 'wb') as f:
                pickle.dump(pred, f)

            recoder, iter_res, error = prepareRes(args, data, pred, recorder, log, split)

            res.append(iter_res)
            iters = i + 1
            if iters % disp_intv == 0:
                opt = {'split':split, 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                        'timer':timer, 'recorder': recorder}
                log.printItersSummary(opt)

            if iters % save_intv == 0:
                results, nrow = prepareSave(args, data, pred)
                log.saveImgResults(results, split, epoch, iters, nrow=nrow, error=error)
                log.plotCurves(recorder, split, epoch=epoch, intv=disp_intv)

            if stop_iters > 0 and iters >= stop_iters: break
    res = np.vstack([np.array(res), np.array(res).mean(0)])
    save_name = '%s_res.txt' % (args.suffix)
    np.savetxt(os.path.join(args.log_dir, split, save_name), res, fmt='%.2f')
    opt = {'split': split, 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)

def prepareRes(args, data, pred, recorder, log, split):
    data_batch = args.val_batch if split == 'val' else args.test_batch
    iter_res = []
    error = ''
    if args.s1_est_d:
        l_acc, data['dir_err'] = eval_utils.calDirsAcc(data['dirs'].data, pred['dirs'].data, data_batch)
        recorder.updateIter(split, l_acc.keys(), l_acc.values())
        iter_res.append(l_acc['l_err_mean'])
        error += 'D_%.3f-' % (l_acc['l_err_mean']) 
    if args.s1_est_i:
        int_acc, data['int_err'] = eval_utils.calIntsAcc(data['ints'].data, pred['intens'].data, data_batch)
        recorder.updateIter(split, int_acc.keys(), int_acc.values())
        iter_res.append(int_acc['ints_ratio'])
        error += 'I_%.3f-' % (int_acc['ints_ratio'])

    if args.s1_est_n:
        acc, error_map = eval_utils.calNormalAcc(data['n'].data, pred['n'].data, data['m'].data)
        recorder.updateIter(split, acc.keys(), acc.values())
        iter_res.append(acc['n_err_mean'])
        error += 'N_%.3f-' % (acc['n_err_mean'])
        data['error_map'] = error_map['angular_map']

    return recorder, iter_res, error


def prepareSave(args, data, pred):
    results = [data['img'].data, data['m'].data, (data['n'].data+1)/2]
    if args.s1_est_n:
        pred_n = (pred['n'].data + 1) / 2
        masked_pred = pred_n * data['m'].data.expand_as(pred['n'].data)
        res_n = [pred_n, masked_pred, data['error_map']]
        results += res_n

    nrow = data['img'].shape[0]
    return results, nrow
