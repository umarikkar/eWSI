
import torch
import numpy as np
import os
from tqdm import tqdm
import pickle
import struct
from torchvision import transforms

# Save an integer to a binary file
def save_integer(num, filename):
    with open(filename, 'wb') as file:
        file.write(struct.pack('i', num))

# Load an integer from a binary file
def load_integer(filename):
    with open(filename, 'rb') as file:
        return struct.unpack('i', file.read(4))[0]

def get_transform(train_mode=True, tf='none'):

    normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) ])
        
    flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)])

    color_jitter = transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.6, 
                                        saturation=0.2, hue=0.1)],
                p=0.8)


    if train_mode==True:
        transform = transforms.Compose([
            flip,
            color_jitter,
            normalize,
        ])
    else:
        transform = normalize

    return transform
    
def sample_images(num_imgs, num_samples=-1, sample_type='random', attn=None, metadata=None):
    
    all_images = list(range(num_imgs))

    if num_samples == -1:
        return torch.tensor(all_images).int()

    if sample_type=='random':
        indices = torch.randint(0, len(all_images), (num_samples,)).int()

    return indices
        

def align_by_intersects(features, intersect_vals):
    
    if not type(intersect_vals)==torch.Tensor:
        intersect_vals = torch.Tensor(intersect_vals)
    features = torch.stack([feat[intersect_vals[:, i]] \
                            for i, feat in enumerate(features)], dim=1)

    return features


def get_intersect_vals(coords):

    if len(coords)>1:
        cond=False
        for cx, cy in zip(coords[:-1], coords[1:]):
            if not np.all(cx==cy):
                cond=True
                break
        if cond:
            intersect_vals = paired_list_features(coords)
        else:
            intersect_vals = [[i for _ in range(len(coords))] for i in range(len(coords[0]))] 
    else:
        intersect_vals = [[i] for i in range(len(coords[0]))]

    return intersect_vals


def paired_list_features(coords):
    
    p1 = coords[0]

    # there is more than one set of magnficiation
    pr = []
    for r_id, coords_aux in enumerate(coords[1:]):

        paired_list = []

        for coord_idx, coord in enumerate(p1):
            cond = np.all(coord == coords_aux, axis=1)
            if np.any(cond):
                paired_list.append((coord_idx, int(np.where(cond)[0])))
            else:
                paired_list.append(0)
    
        pr.append(paired_list)

    intersect_vals = [
    [items[0][0],] + [item[1] for item in items]
    for items in zip(*pr)
    if all(item != 0 and item[0] == items[0][0] for item in items)
    ]

    return intersect_vals


def stretch_list(original_list, target_length):
    # Create an array of the indices of the original list
    original_indices = np.linspace(0, len(original_list) - 1, num=target_length)
    
    # Interpolate the values at the new indices
    stretched_list = np.interp(original_indices, np.arange(len(original_list)), original_list).astype(int)
    
    return stretched_list.tolist()



def get_aligned_features(wsi, data_path, mags, feat_dir='features_aligned/vit_small_patch16_224.ibot_pan4m', return_clusters=False):

    metadata_name = 'coords.pkl'

    wsi_data_path=None
    for datapath in data_path:
        dpath = os.path.join(datapath, feat_dir, '%dx'%mags[0], wsi, metadata_name)
        if os.path.exists(dpath):
            wsi_data_path = datapath

    if wsi_data_path==None:
        print(f'{dpath} unavailable')


    continue_func=True
    for mag in mags:
        if not os.path.exists(os.path.join(wsi_data_path, feat_dir, f'{mag}x', wsi, metadata_name)):
            continue_func=False
            print(f'mag: {mag} not available for WSI: {wsi}')
        
    if not continue_func:
        return None

    coords, feats, patches = [], [], []

    for mag in mags:
        feat=os.path.join(wsi_data_path, feat_dir, f'{mag}x', wsi, 'all_features.pt')
        meta = os.path.join(wsi_data_path, feat_dir, f'{mag}x', wsi, metadata_name)
        with open(meta, 'rb') as f:
            patch_list = pickle.load(f)
        coord = np.array([[int(x) for x in nm.split('_')[:-1]] for nm in patch_list])

        patches.append(patch_list)
        feats.append(feat)
        coords.append(coord)

    intersect_vals = get_intersect_vals(coords)

    return feats, intersect_vals, patches




def get_aligned_patches(wsi, data_path, mags, feat_dir='subimages_aligned'):

    wsi_data_path = None
    for datapath in data_path:
        dpath = os.path.join(datapath, feat_dir, 'metadata', f'{wsi}_{mags[0]}x.pkl')
        if os.path.exists(dpath):
            wsi_data_path = datapath
        
    if wsi_data_path==None:
        print(f'{dpath} unavailable')
        

    continue_func = all(
    os.path.exists(os.path.join(wsi_data_path, feat_dir, 'metadata', f'{wsi}_{mag}x.pkl'))
    for mag in mags
        )
        
    if not continue_func:
        print(f'mags {mags} not available for WSI: {wsi}')
        return None

    coords, feats, res, patches = [], [], [], []

    file_dir=os.path.join(wsi_data_path, feat_dir, 'patches_v5', wsi)

    for mag in mags:
        meta = os.path.join(wsi_data_path, feat_dir, 'metadata', f'{wsi}_{mag}x.pkl')
        with open(meta, 'rb') as f:
            patch_list = pickle.load(f)
        coord = np.array([[int(x) for x in nm.split('_')[:-1]] for nm in patch_list])

        patches.append(patch_list)
        coords.append(coord)

    intersect_vals = get_intersect_vals(coords)

    return file_dir, intersect_vals, patches