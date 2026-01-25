
import torch
import torch.nn as nn
from data_scripts.csv_dataset import (PatchDataset_WSI, Camelyon_feature_subtype,
                                    Camelyon_patch_subtype
                                      )
from torch.utils.data import DataLoader
from torchmetrics import AUROC
import utilities.utils as utils
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

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

            bag_prediction, max_prediction = model(samples)

            bag_loss = criterion(bag_prediction, targets)
            max_loss = criterion(max_prediction, targets)

            weigth_param = model.mil_model.agg_and_classify.single_weight_parameter

            loss = weigth_param * bag_loss + (1 - weigth_param) * max_loss

        with torch.no_grad():
            bag_prediction = (
                    (1 - weigth_param) * torch.sigmoid(max_prediction) +
                    weigth_param * torch.sigmoid(bag_prediction)
            ).squeeze().cpu().numpy()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        optimizer.zero_grad()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def fwd_patches(data, model, device, args=None):

    dset = PatchDataset_WSI(wsi_info=data)
    dloader = DataLoader(dset, batch_size=64, num_workers=args.num_workers, shuffle=False, drop_last=False)

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


def fwd_features(data, model, device, args=None):

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

    nb_neg = sum(data_loader.dataset.df['label']==0)
    nb_pos = sum(data_loader.dataset.df['label']==1)

    pos_weight = torch.Tensor([nb_neg / nb_pos]).float().cuda()  
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    auroc = AUROC(task='binary')

    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    loss_value=0.0

    for _, (data, target) in enumerate(metric_logger.log_every(data_loader, 40, header)):

        target = target.to(device, non_blocking=True)

        bag_prediction, max_prediction = forward_function(data, model, device)

        bag_loss = criterion(bag_prediction, target)
        max_loss = criterion(max_prediction, target)

        loss = 0.5 * bag_loss + (1 - 0.5) * max_loss

        pred = ((1 - 0.5) * torch.sigmoid(max_prediction) +
                    0.5 * torch.sigmoid(bag_prediction)
            ).detach().cpu()

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

    if args.eval_only:

        torch.save(pp, f=f'{args.output_dir}/logits.pt')
        torch.save(tt, f=f'{args.output_dir}/targets.pt')
    
    return AUC, ACC, loss_value
    


