
import torch
import torch.nn as nn
from data_scripts.csv_dataset_common import PatchDataset_WSI
from torch.utils.data import DataLoader
from torchmetrics import AUROC
import utilities.utils as utils
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import os

def train_one_epoch(model, criterion,
    data_loader, optimizer: torch.optim.Optimizer,
    device, epoch, max_norm= 0, args = None):

    model.train()

    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    for iter, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        targets = targets.to(device, non_blocking=True)
        if type(samples) == torch.Tensor:
            samples = samples.to(device, non_blocking=True)
        elif type(samples) == list:
            samples = [i.to(device, non_blocking=True) for i in samples]
        
        with torch.cuda.amp.autocast():

            logits = model(samples)

            targets = torch.tensor(targets, dtype=logits.dtype)
            loss = criterion(logits, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        loss.backward()

        if max_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        optimizer.zero_grad()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def fwd_patches(data, model, device, return_path=False):

    dset = PatchDataset_WSI(wsi_info=data, return_path=return_path)

    dloader = DataLoader(dset, batch_size=64, num_workers=10, shuffle=False, drop_last=False)

    outs = []

    for _, ims in enumerate(dloader):

        ims = ims.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            out = model.forward_fn(ims)

        outs.append(out)

    outputs = torch.cat(outs, dim=1)

    # torch.save(outputs.cpu(), f'visualizations/outputs_{idx}.pt')

    with torch.cuda.amp.autocast():
        logits = model.mil_model(outputs)

    return logits

import pickle

# def fwd_patches(data, model, device, return_path=True):

#     dset = PatchDataset_WSI(wsi_info=data, return_path=return_path)

#     dloader = DataLoader(dset, batch_size=64, num_workers=10, shuffle=False, drop_last=False)

#     outs = []
#     coords_all = []

#     wsi_name = data[0][0][0].split('/')[-3]

#     wsi_dir = f'visualizations/MIDL_sup/{wsi_name}'

#     os.makedirs(wsi_dir, exist_ok=True)

#     for _, dta in enumerate(dloader):

#         if return_path:
#             ims, coords = dta
#         else:
#             ims = dta

#         ims = ims.to(device, non_blocking=True)
#         with torch.cuda.amp.autocast():
#             out = model.forward_fn(ims)

#         outs.append(out)
#         coord_list = list(coords[0][0])
#         coord_list = [s.split('/')[-1].split('.')[0] for s in coord_list]
#         coords_all.extend(coord_list)


#     outputs = torch.cat(outs, dim=1)

#     coord_save_path = f'{wsi_dir}/coords.pkl'
#     with open(coord_save_path, "wb") as f:
#         pickle.dump(coords_all, f)

#     with torch.cuda.amp.autocast():

#         agg = model.mil_model.agg_and_classify

#         tiles_embs = agg.tiles_embs(outputs)
#         scaled_embs = agg.attention_layer(tiles_embs, dim=[1,2]).squeeze(1,2)

#         logits = agg.mlp(scaled_embs)

#         if type(logits)!=tuple:
#             logits = logits.squeeze(1)

#     torch.save(outputs.cpu(), f'{wsi_dir}/enc_outputs.pt')
#     torch.save(tiles_embs.cpu(), f'{wsi_dir}/tiles_embs.pt')
#     torch.save(F.sigmoid(logits).cpu(), f'{wsi_dir}/logits.pt')

#     return logits


def fwd_features(data, model, device):

    if type(data) == torch.Tensor:
        data = data.to(device, non_blocking=True)
    elif type(data) == list:
        data = [i.to(device, non_blocking=True) for i in data]

    with torch.cuda.amp.autocast():
        logits = model(data)

    return logits


@torch.no_grad()
def evaluate(data_loader, model, device, forward_function_name, args=None):


    if 'patches' in forward_function_name:
        forward_function = fwd_patches
    else:
        forward_function = fwd_features
    
    targets, preds = [], []
    
    criterion = torch.nn.BCEWithLogitsLoss()
    auroc = AUROC(task='binary')

    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    loss_value=0.0

    for idx, (data, target) in enumerate(metric_logger.log_every(data_loader, 20, header)):

        # # assume a batch size of 1 otherwise it will bug out!
        # wsi_name = data[0][0][0].split('/')[-3]
        # wsi_dir = f'visualizations/MIDL/{wsi_name}'
        # os.makedirs(wsi_dir, exist_ok=True)

        target = target.to(device, non_blocking=True)

        logits = forward_function(data, model, device)

        target = torch.tensor(target, dtype=logits.dtype)

        loss = criterion(logits, target)
        pred = F.sigmoid(logits)

        loss_value+=loss.item()
    
        # to mAP calculation
        preds.append(pred.cpu().detach())
        targets.append(target.cpu().detach())

    loss_value /= len(data_loader)

    tt = torch.cat(targets)
    pp = torch.cat(preds)
    
    AUC = auroc(pp.float(), tt)
    
    ll = pp > 0.5
    ACC = 100 * (ll == tt).float().mean()

    torch.save(pp, f=f'{args.output_dir}/logits.pt')
    torch.save(tt, f=f'{args.output_dir}/targets.pt')

    print('wrong samples: ', data_loader.dataset.df.iloc[np.where(ll != tt)[0], :])
    
    return AUC, ACC, loss_value




@torch.no_grad()
def evaluate_agg(data_loader, model, device, args=None):

    targets, preds = [], []

    # targets = torch.load('analysis/cam_target.pt')
    
    criterion = torch.nn.BCEWithLogitsLoss()
    auroc = AUROC(task='binary')

    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    loss_value=0.0

    for idx, (data, target) in enumerate(metric_logger.log_every(data_loader, 20, header)):

        # # assume a batch size of 1 otherwise it will bug out!
        # wsi_name = data[0][0][0].split('/')[-3]
        # wsi_dir = f'visualizations/MIDL/{wsi_name}'
        # os.makedirs(wsi_dir, exist_ok=True)

        tiles_embs = data.to(device, non_blocking=True)[0]
        target = target.to(device, non_blocking=True)

        agg = model.mil_model.agg_and_classify

        with torch.cuda.amp.autocast():
            scaled_embs = agg.attention_layer(tiles_embs, dim=[1,2]).squeeze(1,2)
            logits = agg.mlp(scaled_embs)

            if type(logits)!=tuple:
                logits=  logits.squeeze(1)
            else:
                logits=   logits
            

        target = torch.tensor(target, dtype=logits.dtype)

        loss = criterion(logits, target)
        pred = F.sigmoid(logits)

        loss_value+=loss.item()
    
        # to mAP calculation
        preds.append(pred.cpu().detach())
        targets.append(target.cpu().detach())

    loss_value /= len(data_loader)

    tt = torch.cat(targets)
    pp = torch.cat(preds)
    
    AUC = auroc(pp.float(), tt)
    
    ll = pp > 0.5
    ACC = 100 * (ll == tt).float().mean()


    best_t, best_acc = find_best_threshold_acc(pp, tt, 100)

    if args.eval_only:

        torch.save(pp, f=f'{args.output_dir}/logits.pt')
        torch.save(tt, f=f'{args.output_dir}/targets.pt')

    print('wrong samples: ', data_loader.dataset.df.iloc[np.where(ll != tt)[0], :])
    
    return AUC, ACC, loss_value



def find_best_threshold_acc(scores: torch.Tensor,
                            labels: torch.Tensor,
                            num_thresholds: int = 11):
    """
    scores : Tensor [N] in [0, 1]
    labels : Tensor [N] in {0, 1}
    """
    assert scores.ndim == 1 and labels.ndim == 1
    assert scores.shape[0] == labels.shape[0]

    thresholds = torch.linspace(0, 1, num_thresholds, device=scores.device)

    best_t = 0.5
    best_acc = -1.0

    for t in thresholds:
        preds = (scores >= t).int()
        acc = (preds == labels).float().mean()

        if acc > best_acc:
            best_acc = acc
            best_t = t.item()

    return best_t, best_acc.item()
