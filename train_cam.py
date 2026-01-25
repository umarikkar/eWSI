
"""
TRAINING CODE FOR eWSI. Started from the DeiT ViT fine-tuning code. 
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import datetime
import json
import math
import os
import sys
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch.utils.data import DataLoader
from torchmetrics import AUROC

import utilities.utils as utils

from data_scripts.csv_dataset_common import FeatureSubtypeBase, PatchSubtypeBase
from models.get_models import get_encoder_list
from models.wsi_models import Combined_WSI_model
from trainers import default_trainer, acmil_trainer



def get_args_parser():

    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)

    # Training and hardware parameters
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--avg_weight_num', default=5, type=int)
    parser.add_argument('--epochs', default=101, type=int)
    parser.add_argument('--eval_freq', default=100, type=int)
    parser.add_argument('--eval_only', action='store_true') 
    parser.set_defaults(eval_only=False)
    parser.add_argument('--resume', action='store_true') 
    parser.set_defaults(resume=True)
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)

    # Model parameters
    parser.add_argument("--model_config", type=str, default='vit_small_patch16_224.ibot_inet.lora_q_f1_f2.maxpool3_tanh', help='for lora, choose q, k, v, o, f1, f2 seperated by _')
    # parser.add_argument("--model_config", type=str, default='vit_base_patch16_224.owkin_pancancer.lora_q_f1_f2.maxpool3_tanh', help='for lora, choose q, k, v, o, f1, f2 seperated by _')
    # parser.add_argument("--model_config", type=str, default='vit_base_patch16_224.owkin_pancancer.features.maxpool3_tanh', help='for lora, choose q, k, v, o, f1, f2 seperated by _')
    # parser.add_argument("--model_config", type=str, default='vit_small_patch16_224.sup_inet.full.maxpool3_relu', help='for lora, choose q, k, v, o, f1, f2 seperated by _')
    parser.add_argument('--img_size', default=224, type=int, help="size of the input image.")
    parser.add_argument("--lora_rank", type=int, default=4, help="the dataset")
    parser.add_argument("--patch_drop_rate", type=float, default=0.5)

    # Data parameters
    parser.add_argument("--data_root", type=str, default='/work/um00109/Hist/Datasets/Camelyon16/', help="the dataset")
    parser.add_argument("--dataset_name", type=str, default='camelyon16', help="the dataset")
    parser.add_argument("--fold_num", type=int, default=0, help="the model used")
    parser.add_argument('--augment', default=1, type=int)
    parser.add_argument('--skip_val', default=1, type=int)
    parser.add_argument("--mag_subimg_config", type=str, default='20x_64', help="magnification configs")
    parser.add_argument('--load_features', action='store_true') 
    # parser.set_defaults(load_features=True)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=0.5, metavar='NORM',help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-7, metavar='LR',help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',help='LR decay rate (default: 0.1)')

    parser.add_argument('--aggregator_cpt_path', 
                        type=str, 
                        default='results/camelyon16_full/vit_small_patch16_224.ibot_inet/lora_q_f1_f2/maxpool3_relu/20x_64/bs_8_LR_1.0e-03_WD_5.0e-02_avg_5_seed_0_drop_0.5/aggregator.pth',
                        help='aggreagator checkpoint path')
    
    parser.add_argument('--load_aggregator', default=0, type=int)
    parser.add_argument('--train_bias', default=0, type=int)

    return parser



def main(args):

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    # Dataset and DataLoaders ------------------------------------------------------------
    
    # encoder_name = args.encoder_name + '_seed_%d'%args.seed
    encoder_name = args.encoder_name

    if args.lora_config == 'features':
        dataset_train = FeatureSubtypeBase(encoder_name=encoder_name,        
                                load_features=args.load_features, 
                                magnification=args.magnification, 
                                num_subImgs=args.num_subImgs,

                                dataset_name=args.dataset_name, 
                                data_root=args.data_root,
                                istrain='train',
                                fold_num=args.fold_num
                                )
        dataset_val = FeatureSubtypeBase(encoder_name=encoder_name,        
                                load_features=args.load_features, 
                                magnification=args.magnification, 
                                num_subImgs=args.num_subImgs_eval,

                                dataset_name=args.dataset_name, 
                                data_root=args.data_root,
                                istrain='test',
                                fold_num=args.fold_num
                                )
        forward_function = 'features'
    else:
        dataset_train = PatchSubtypeBase(image_size=args.img_size,        
                        augment=args.augment, 
                        magnification=args.magnification, 
                        return_img=True,
                        num_subImgs=args.num_subImgs,

                        dataset_name=args.dataset_name, 
                        data_root=args.data_root,
                        istrain='train',
                        fold_num=args.fold_num
                        )
        dataset_val = PatchSubtypeBase(image_size=args.img_size,             
                        augment=False, 
                        magnification=args.magnification, 
                        return_img=False,
                        num_subImgs=args.num_subImgs_eval,

                        dataset_name=args.dataset_name, 
                        data_root=args.data_root,
                        istrain='test',
                        fold_num=args.fold_num
                        )

        forward_function = 'patches'

    data_loader_train = DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size,num_workers=args.num_workers, pin_memory=True,drop_last=False)
    data_loader_val = DataLoader(dataset_val, shuffle=False, batch_size=1,num_workers=args.num_workers, pin_memory=True,drop_last=False)
    
    # Define WSI model -----------------------------------------------------------

    print(f"Creating model: {args.model_name}")
    output_dir = Path(args.output_dir)

    encoders = get_encoder_list(args)
    model = Combined_WSI_model(encoders=encoders, out_features=1, mil_config=args.mil_config, lora_config=args.lora_config, encoder_name=args.encoder_name)
    
    if args.load_aggregator:
        agg_cpt = torch.load(args.aggregator_cpt_path, map_location='cpu')
        msg2 = model.mil_model.load_state_dict(agg_cpt, strict=True)
        print('aggregator loading: ', msg2)

        if args.train_bias:
            print('training the final classifier layer only.')
            for n, p in model.mil_model.agg_and_classify.named_parameters():
                if 'mlp' not in n:
                    p.requires_grad=False
    else:
        print('aggregator trained from scratch')

    model.to(device)

    print('trainable params in total... :')
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, p.shape)

    # Define LR, optimizer, scheduler and criterion --------------------------------
    nb_neg = sum(dataset_train.df['label']==0)
    nb_pos = sum(dataset_train.df['label']==1)
    args.pos_weight = torch.Tensor([nb_neg / nb_pos]).float().cuda()  

    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=args.pos_weight)
        
    print(f"Loss, optimizer and schedulers ready.")

    # Resume training or evaluate only ---------------------------------------------
    to_restore = {"epoch": 0}
    resume_dir = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume and os.path.exists(resume_dir):

        print('RESUMING FROM CHECKPOINT!')
        avg_weights = torch.load(resume_dir)['model_avg']
        original_weights = torch.load(resume_dir)['model']
        set_trainable_flattened_weights(model, original_weights)

        utils.restart_from_checkpoint(resume_dir, run_variables=to_restore, optimizer=optimizer, lr_scheduler=lr_scheduler)

    args.start_epoch = to_restore["epoch"]

    # set train and eval functions -----------------------------------------------
    if 'acmil' not in args.mil_config.lower():
        train_one_epoch = default_trainer.train_one_epoch
        evaluate = default_trainer.evaluate
    else:
        train_one_epoch = acmil_trainer.train_one_epoch
        evaluate = acmil_trainer.evaluate

    if args.eval_only:
        set_trainable_flattened_weights(model, avg_weights)
        AUC, ACC, loss_value = evaluate(data_loader_val, model, device, forward_function, args)
        print(f"Accuracy of the network: AUC= {AUC:.2f} ACC= {ACC:.2f}")
        return
    
    # TRAINING STARTS HERE!!!! -----------------------------------------------------

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    # To store the past {args.avg_weight_num} epochs' weights of only params with requires_grad=True
    weight_history = []

    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_one_epoch(
                            model, criterion, data_loader_train,
                            optimizer, device, epoch, args.clip_grad, args = args)
        
        current_weights = get_trainable_flattened_weights(model)
        weight_history.append(current_weights)

        if len(weight_history) > args.avg_weight_num:
            weight_history.pop(0)

        original_weights = get_trainable_flattened_weights(model)
        avg_weights = torch.mean(torch.stack(weight_history), dim=0)

        lr_scheduler.step(epoch)

        if args.output_dir:
            if lr_scheduler is not None:
                lrsd = lr_scheduler.state_dict()
            else:
                lrsd = None
            utils.save_on_master({'model': original_weights,'model_avg':avg_weights,'optimizer': optimizer.state_dict(),
                'lr_scheduler': lrsd,'epoch': epoch,'args': args,}, os.path.join(args.output_dir, 'checkpoint.pth'))

        if (epoch !=0 and epoch%args.eval_freq == 0) or epoch==args.epochs-1:

            # evaluate with ensemble of weights
            set_trainable_flattened_weights(model, avg_weights)
            AUC, ACC, loss_value = evaluate(data_loader_val, model, device, forward_function, args)
            print(f"Accuracy of the network on the {len(dataset_val)} test images with weighted CPT: AUC= {AUC:.2f} ACC= {ACC:.2f}")

            # set back to original weights
            set_trainable_flattened_weights(model, original_weights)

            if max_accuracy < AUC:
                max_accuracy = AUC
    
            print(f'Max accuracy: {max_accuracy:.2f}%')
            
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'test_AUC': AUC.item(), 'test_ACC': ACC.item(), 
                        # 'test_AUC_current': AUC_current.item(), 'test_ACC_current': ACC_current.item(), 
                        'test_loss': loss_value,
                        'epoch': epoch, 'n_parameters': n_parameters}
            
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch, 'n_parameters': n_parameters}
 
 
        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


# Function to get flattened weights for params that require grad
def get_trainable_flattened_weights(model):
    return torch.cat([param.flatten() for param in model.parameters() if param.requires_grad])

# Function to set model's state_dict from a flat tensor (for params that require grad)
def set_trainable_flattened_weights(model, flat_weights):
    idx = 0
    for param in model.parameters():
        if param.requires_grad:
            numel = param.numel()
            param.data.copy_(flat_weights[idx:idx+numel].view_as(param))
            idx += numel


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    [args.model_name, args.pretrain_cpt, args.lora_config, args.mil_config] = args.model_config.lower().split('.')
    training_config = 'bs_%d_LR_%.1e_WD_%.1e_avg_%d_seed_%d'%(args.batch_size, args.lr, args.weight_decay, args.avg_weight_num, args.seed)

    
    base_bs = 8
    lr_new = args.lr * args.batch_size / base_bs
    args.lr = lr_new


    if args.lora_config != 'features':
        training_config += f'_drop_{args.patch_drop_rate}'
        if args.augment==0:
            args.lora_config += '_noAUG'


    if "lora" in args.lora_config:
        dddd = f"{args.lora_config}_{args.lora_rank}"
    else:
        dddd = f"{args.lora_config}"

    args.encoder_name = ('.').join([args.model_name, args.pretrain_cpt])


    if args.train_bias:
        nk = '_bias'
        args.load_aggregator=1
    elif args.load_aggregator:
        nk = '_load_agg'
    else:
        nk = '_full'

    args.output_dir = os.path.join('results_MIDL_full', args.dataset_name + nk, '%s.%s'%(args.model_name, args.pretrain_cpt), dddd, args.mil_config, args.mag_subimg_config,
                                   training_config)
    
    print('\n LOGGING DIRECTORY >>>>> : %s\n'%args.output_dir)

    # sys.exit()
    if not os.path.exists(args.output_dir) and utils.is_main_process():
        os.makedirs(args.output_dir)

    args = utils.get_lora_subimg_settings(args)

    if args.eval_only:
        print('\n\n only evaluating!!! \n\n')
    
    main(args)

