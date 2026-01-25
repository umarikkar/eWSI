import os
import pickle
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from functools import partial
from tqdm import tqdm

from data_scripts.get_splits import get_splits_subtype
from data_scripts.helpers import *

"""
Base datasets ====================================================
"""

class PatchDataset_WSI(Dataset):
    
    def __init__(self, wsi_info, transform=None, return_path=False):
        
        super().__init__()
        
        self.all_images = wsi_info

        self.transform = get_transform(False) if transform is None else transform

        self.return_path = return_path

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        
        img_path = self.all_images[idx]

        img = torch.stack([self.transform(Image.open(im[0]).convert('RGB').resize((224, 224))) \
                           for im in img_path])
        
        if self.return_path:
            return img, img_path
        
        return img


class SmallFeatureDataset(Dataset):
    
    def __init__(self, data_list, label_list, sample_rate=2048):
        
        super().__init__()
        
        self.data_list = data_list
        self.label_list = label_list
    
    def __getitem__(self, idx):
        
        img_path = self.all_images[idx]

        img = torch.stack([self.transform(Image.open(im[0]).convert('RGB').resize((224, 224))) \
                           for im in img_path])
        
        return img

import ast

class PatchSubtypeBase(Dataset):

    def __init__(self, image_size='vit_base_patch16_224.owkin_pancancer', 
                 augment=False, magnification=20, return_img=True, num_subImgs=32, transform=None, 
                 **kwargs): 
        
        self.mag = magnification if isinstance(magnification, list) else [magnification]
        self.image_size = image_size
        self.return_img = return_img
        self.transform = transform
        self.num_subImgs = num_subImgs

        self.df, self.data_root, self.file_ext = get_df(**kwargs)

        if isinstance(self.data_root, list):
            self.feat_dir = [os.path.join(f, 'patches', f'skip_1_{self.mag[0]}x') for f in self.data_root]
        else:
            self.feat_dir = os.path.join(self.data_root, 'patches', f'skip_1_{self.mag[0]}x')

        # self.feat_dir = os.path.join(self.data_root, 'patches_v23')
        self.feat_dir = os.path.join(self.data_root, 'patches_MIDL_full')


        print(f'Feature directory: {self.feat_dir}')

        def get_feats(x, feat_dir, meta_df):

            x = x.replace('.tif', '')

            idx = meta_df.index[meta_df["wsi"] == x]

            coords = ast.literal_eval(list(meta_df["filenames"][idx])[0])

            # coord_path = os.path.join(feat_dir, x, 'coords.pkl')[0]

            # with open(coord_path, 'rb') as f:
            #     coords = pickle.load(f)

            feat_path = [os.path.join(feat_dir, x, 'patches', i) for i in coords]

            return feat_path, coords

        tqdm.pandas()

        self.meta_df = pd.read_csv('camelyon_metadata.csv')

        print("Creating set of aligned patches and filenames...")
        self.df[['filename', 'coordinates']] = self.df['wsi'].progress_apply(lambda x: get_feats(x, self.feat_dir, self.meta_df)).apply(pd.Series)
        print("Finished creating set of aligned features...")

        if augment:
            self.transform = get_transform(True, tf=transform)
        else:
            self.transform = get_transform(False)
        

        print('%s data size=% d'%(kwargs['istrain'], self.df.shape[0]))
        print('number of sub images: ', self.num_subImgs)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img_paths_all = self.df['filename'][idx]
        label = int(self.df['label'][idx])

        idxs_to_sample = sample_images(len(img_paths_all), num_samples=self.num_subImgs)

        img_paths = [[img_paths_all[i] for i in idxs_to_sample]]
        # n_images * n_mags * c * h * w
        imgs = list(map(list, zip(*img_paths)))

        if self.return_img:
            imgs = torch.stack([torch.stack([self.transform(Image.open(im).convert('RGB').resize((self.image_size, 
                                                        self.image_size))) for im in im_mag]) for im_mag in imgs])

        return imgs, label
    


class FeatureSubtypeBase(Dataset):

    def __init__(self, encoder_name='vit_base_patch16_224.owkin_pancancer', 
                 load_features=False, magnification=20, num_subImgs=8192, **kwargs): 
        
        self.load_features = load_features
        self.mag = magnification if isinstance(magnification, list) else [magnification]
        self.num_subImgs=num_subImgs

        self.encoder_name = encoder_name.split('/')[-1]

        self.df, self.data_root, self.file_ext = get_df(**kwargs)
        # if isinstance(self.data_root, list):
        #     self.feat_dir = [os.path.join(f, 'features') for f in self.data_root]
        # else:
        #     self.feat_dir = os.path.join(self.data_root, 'features')

        self.feat_dir = os.path.join(self.data_root, 'features_MIDL')

        print(f'Feature directory: {self.feat_dir}')

        tqdm.pandas()
        print("Creating aligned patches and filenames...")
        self.df[['filename', 'coordinates']] = self.df['wsi'].progress_apply(
            lambda x: self.get_feats(x, self.feat_dir)).apply(pd.Series)
        print("Finished creating aligned features.")

        if self.load_features:
            print("Loading all features into memory...")
            self.all_feats = [
                [torch.load(f, map_location='cpu').half() for f in filename]
                for filename in tqdm(self.df['filename'])
            ]
            print("Features loaded into memory.")

    def get_feats(self, x, feat_dir):
        x = x.replace(self.file_ext, '')
        feats_, coords_ = [], []
        for mag in self.mag:
            feat_path = os.path.join(feat_dir, self.encoder_name, x, 'all_features.pt')
            coord_path = os.path.join(feat_dir, self.encoder_name, x, 'coords.pkl')
            with open(coord_path, 'rb') as f:
                coords = pickle.load(f)
            feats_.append(feat_path)
            coords_.append(coords)
        return feats_, coords_

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.load_features:
            imgs = self.all_feats[idx]
        else:
            filepath = self.df['filename'][idx]
            imgs = [torch.load(f).half() for f in filepath]

        label = int(self.df['label'][idx])
        idxs_to_sample = sample_images(len(imgs[0]), num_samples=self.num_subImgs)
        imgs = torch.stack([img[idxs_to_sample] for img in imgs], dim=1)

        return imgs, label

"""
DataFrame helpers ====================================================================
"""
def get_tcga_df(dataset_name=None, data_root=None, istrain='train', fold_num=0):

    data_root = data_root or '/vol/research/scratch1/NOBACKUP/um00109/tcga'
    df, data_path = get_splits_subtype(data_root, dataset_name, fold_num, istrain)

    return df, data_path, '.svs'

def get_camelyon_df(dataset_name=None, data_root=None, istrain='train', fold_num=0):

    data_root = data_root or '/vol/research/scratch1/NOBACKUP/um00109/Camelyon/Camelyon16_v2'
    istrain = istrain == 'train'
    csv_file = 'data_scripts/csvs/camelyon16/reference_train.csv' if istrain else 'data_scripts/csvs/camelyon16/reference_test.csv'
    df = pd.read_csv(csv_file)
    df = df.rename(columns={'image': 'wsi', 'type': 'label_name'})
    df['label'] = df['label_name'].factorize(sort=True)[0]

    return df, data_root, '.tif'

def get_bracs_df(dataset_name=None, data_root=None, istrain='train', fold_num=0):

    data_root = data_root or '/vol/research/scratch1/NOBACKUP/um00109/BRACS'
    df = pd.read_csv('data_scripts/csvs/BRACS.csv').iloc[:, :5]
    df = df.dropna(subset=['WSI Filename'])
    set_map = {'train': 'Training', 'val': 'Validation', 'test': 'Testing'}
    df = df[df['Set'] == set_map.get(istrain, 'Testing')].reset_index(drop=True)

    df = df.rename(columns={'WSI Filename': 'wsi', 'WSI label': 'label_name'})
    df['label_name'] = df['label_name'].replace({
        'N': 'benign', 'PB': 'benign', 'UDH': 'benign',
        'FEA': 'atypical', 'ADH': 'atypical',
        'DCIS': 'malignant', 'IC': 'malignant'})
    df['label'] = df['label_name'].factorize(sort=True)[0].astype(int)

    return df, data_root, '.svs'

def get_df(dataset_name=None, **kwargs):
    if 'camelyon' in dataset_name:
        df_fn = get_camelyon_df
    elif 'tcga' in dataset_name or 'brca' in dataset_name:
        df_fn = get_tcga_df
    elif 'bracs' in dataset_name:
        df_fn = get_bracs_df

    return df_fn(dataset_name, **kwargs)


if __name__ == '__main__':

    
    dataset = FeatureSubtypeBase(encoder_name='vit_small_patch16_224.ibot_inet',        
                                load_features=False, 
                                magnification=20, 
                                num_subImgs=8192,

                                dataset_name='tcga-brca', 
                                data_root=None,
                                istrain='train',
                                fold_num=0
                                )


    print(dataset[0])
