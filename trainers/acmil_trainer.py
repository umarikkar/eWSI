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

def train_one_epoch(model, criterion,
    data_loader, optimizer: torch.optim.Optimizer,
    device, epoch, max_norm= 0, args = None):

    model.train()

    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    n_token=5
    for iter, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        targets = targets.to(device, non_blocking=True)
        if type(samples) == torch.Tensor:
            samples = samples.to(device, non_blocking=True)
        elif type(samples) == list:
            samples = [i.to(device, non_blocking=True) for i in samples]
        
        with torch.cuda.amp.autocast():

            sub_preds, slide_preds, attn = model(samples)

            targets = torch.tensor(targets, dtype=sub_preds.dtype)

            loss0 = criterion(sub_preds, targets.unsqueeze(-1).repeat_interleave(n_token,dim=-1))
            loss1 = criterion(slide_preds.squeeze(-1), targets)

            diff_loss = torch.tensor(0).to(device, dtype=torch.float)
            attn = torch.softmax(attn, dim=-1)

            for i in range(n_token):
                for j in range(i + 1, n_token):
                    diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                                n_token * (n_token - 1) / 2)

            loss = diff_loss + loss0 + loss1

        loss_value = loss.item()

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        optimizer.zero_grad()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def fwd_patches(data, model, device, args):

    dset = PatchDataset_WSI(wsi_info=data)
    dloader = DataLoader(dset, batch_size=128, num_workers=args.num_workers, shuffle=False, drop_last=False)

    outs = []

    for _, ims in enumerate(dloader):

        ims = ims.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            out = model.forward_fn(ims)

        outs.append(out)

    outputs = torch.cat(outs, dim=1)

    with torch.cuda.amp.autocast():
        _, logits, _ = model.mil_model(outputs)

    return logits.squeeze(-1)


def fwd_features(data, model, device, args):

    if type(data) == torch.Tensor:
        data = data.to(device, non_blocking=True)
    elif type(data) == list:
        data = [i.to(device, non_blocking=True) for i in data]

    with torch.cuda.amp.autocast():
        _, logits, _ = model(data)

    return logits.squeeze(-1)


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

    for _, (data, target) in enumerate(metric_logger.log_every(data_loader, 20, header)):

        target = target.to(device, non_blocking=True)

        logits = forward_function(data, model, device, args)

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

    # if args.eval_only:

    torch.save(pp, f=f'{args.output_dir}/logits.pt')
    torch.save(tt, f=f'{args.output_dir}/targets.pt')


    print('wrong samples: ', data_loader.dataset.df.iloc[np.where(ll != tt)[0], :])
    
    return AUC, ACC, loss_value