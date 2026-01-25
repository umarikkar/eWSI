# import timm.models.vision_transformer as vits
import models.vision_transformer_lora as vits
import models.vision_transformer_dora as vits_dora
import timm
import torch
import torch.nn as nn
import numpy as np

def remove_qkv(cpt_dict, lora_cfg):
    if 'adalora' not in lora_cfg and 'dora' not in lora_cfg:
        return cpt_dict

    new_cpt_dict = {}
    for key, value in cpt_dict.items():
        if 'qkv' not in key:
            new_cpt_dict[key] = value
        else:
            split_values = torch.split(value, len(value) // 3, dim=0)
            new_cpt_dict.update({key.replace('qkv.', 'qkv.%d.'%d): part for d, part in enumerate(split_values)})
            
    return new_cpt_dict

def get_encoder(args):

    if args.lora_config=='features':
        return None
    
    cpt_dict=None

    if 'dora' in args.lora_config:
        model = vits_dora.__dict__[args.model_name](pretrained=False, num_classes=-1, img_size=args.img_size, patch_drop_rate=args.patch_drop_rate, lora_cfg=args.lora_cfg, lora_config=args.lora_config)
    else:
        model = vits.__dict__[args.model_name](pretrained=False, num_classes=-1, img_size=args.img_size, patch_drop_rate=args.patch_drop_rate, lora_cfg=args.lora_cfg, lora_config=args.lora_config, lora_rank=args.lora_rank)

    if args.pretrain_cpt.lower() == 'owkin_pancancer':
        backbone = timm.create_model('hf-hub:1aurent/vit_base_patch16_224.owkin_pancancer', pretrained=True, num_classes=-1, img_size=args.img_size)
        cpt_dict=backbone.state_dict()
        # msg = model.load_state_dict(backbone.state_dict(), strict=False)

    elif args.pretrain_cpt.lower() == 'ibot_inet':
        cpt_path = 'pretrained_cpts/vit_small_patch16_224.ibot_inet.pth'
        cpt_dict = torch.load(cpt_path, map_location='cpu')['teacher']
        cpt_dict = {key.replace('module.', '').replace('backbone.', ''): value for key, value in cpt_dict.items()}
        msg = model.load_state_dict(cpt_dict, strict=False)

        # cpt_path = '/vol/research/fmodel_medical/people/umar/histopathology/preprocessing_v2/pretrained_cpts/vit_base_patch16_224.ibot_inet.pth'
        # cpt_dict = torch.load(cpt_path, map_location='cpu')
        # cpt_dict = cpt_dict['state_dict']

        # print(msg)

    elif args.pretrain_cpt.lower() == 'ibot_pan4m':
        cpt_path = 'pretrained_cpts/vit_small_patch16_224.ibot_pan4m.pth'
        cpt_dict = torch.load(cpt_path, map_location='cpu')['teacher']
        cpt_dict = {key.replace('module.', '').replace('backbone.', ''): value for key, value in cpt_dict.items()}
        # msg = model.load_state_dict(cpt_dict, strict=False)

    elif 'ibotv2' in args.pretrain_cpt.lower():
        cpt_path = 'pretrained_cpts/vit_small_patch16_224.%s.pth'%args.pretrain_cpt
        cpt_dict = torch.load(cpt_path, map_location='cpu')['teacher']
        cpt_dict = {key.replace('module.', '').replace('backbone.', ''): value for key, value in cpt_dict.items()}

    elif args.pretrain_cpt.lower() == 'sup_inet':
        # backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=-1, img_size=args.img_size)
        backbone = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=-1, img_size=args.img_size)
        cpt_dict=backbone.state_dict()
        # msg = model.load_state_dict(backbone.state_dict(), strict=False)

    else:
        msg = 'no pretrained cpt!'

    if cpt_dict is not None:
        cpt_dict = remove_qkv(cpt_dict, args.lora_config)
        msg = model.load_state_dict(cpt_dict, strict=False)
        
    print(args.pretrain_cpt, msg)

    return model



def get_encoder_list(args):

    encoders = [get_encoder(args) for _ in range(len(args.magnification))]

    return encoders
