from torch.utils.data import Dataset
import pandas as pd
import os
import glob

import torch

from torchvision import transforms


from PIL import Image


from data_scripts.get_splits import get_splits_subtype_kfold
from data_scripts.helpers import *

    
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

        

"""
Dataset to get a list of images per WSI, given the WSI
"""
class PatchDataset_WSI(Dataset):
    
    def __init__(self, wsi_info, transform=None):
        
        super().__init__()
        
        self.all_images = wsi_info
        self.transform = get_transform(False) if transform is None else transform

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        
        img_path = self.all_images[idx]

        img = torch.stack([self.transform(Image.open(im[0]).convert('RGB').resize((224, 224))) \
                           for im in img_path])
        
        return img
    
    
class PatchDataset_WSI_aligned(Dataset):
    
    def __init__(self, wsi_info, transform=None):
        
        super().__init__()
        
        self.all_images = wsi_info[0]
        self.transform = get_transform(False) if transform is None else transform
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        
        img_path = self.all_images[idx]

        img = torch.stack([self.transform(Image.open(im).convert('RGB').resize((224, 224))) \
                           for im in img_path])
        
        
        return img, 0
    

"""
TCGA ==================================================================================================================
"""

class TCGA_subtype(Dataset):

    def __init__(self, fold_num=0, num_subImgs=8, data_root=None, dataset_name='tcga-brca', magnification=[20],
                 istrain='train'):  
         
        data_root = '/vol/research/scratch1/NOBACKUP/um00109/tcga' if data_root is None else data_root

        self.num_subImgs = num_subImgs
        self.istrain = True if istrain=='train' else False
        self.mag = sorted(magnification)[::-1]
        
        df, self.data_path = get_splits_subtype_kfold(data_root, dataset_name, fold_num, istrain)

        # df = df.iloc[:50]

        self.df = df

        return

class TCGA_feature_subtype(TCGA_subtype):

    def __init__(self, encoder_name='vit_base_patch16_224.owkin_pancancer', 
                 load_features=False, return_clusters=False, mil_config='maxpool', **kwargs): 
        
        super().__init__(**kwargs)

        self.mil_config = mil_config
        self.load_features = load_features
        self.return_cluster = return_clusters
        self.r = np.random.RandomState(1)

        self.encoder_name = encoder_name.split('/')[-1]
        self.feat_dir = os.path.join('features_aligned', encoder_name)

        tqdm.pandas()
        print("Creating set of aligned patches and filenames...")
        self.df[['filename', 'intersect_vals', 'coordinates']] = self.df['wsi'].progress_apply(lambda x: get_aligned_features(x, 
                                                self.data_path, self.mag, self.feat_dir)).apply(pd.Series)
        print("Finished creating set of aligned features...")

        if self.load_features:
            print("Loading all features to memory...here we align the features as well.")
            self.all_feats = []
            for _, filename in tqdm(enumerate(self.df['filename'])):
                features = [torch.load(f, map_location='cpu').half() for f in filename]
                self.all_feats.append(features)
            print("Done loading all features to memory...")

        print('%s data size=% d'%(kwargs['istrain'], self.df.shape[0]))
        print('number of sub images: ', self.num_subImgs)

    def create_bags(self, idx, feats, intersect_idxs, bag_size=24):

        int_indexes = torch.Tensor(intersect_idxs).int()

        mean_bag_sz = int(bag_size/8)

        bag_length = max(int(self.r.normal(mean_bag_sz, 1, 1)), 2)

        cluster_labels = torch.Tensor(self.df['cluster_labels'][idx]).int()
        cluster_labels = cluster_labels[int_indexes[:,0]]
        cluster_cnt = [int(sum(cluster_labels==i)) for i in range(8)]

        cluster_pickup = [min(bag_length, cnt) for cnt in cluster_cnt]
        cluster_idxs = [np.where(cluster_labels==i)[0] for i in range(8)]

        idxs=[]
        for cl_idx, pickup in zip(cluster_idxs, cluster_pickup):
            if pickup>0:
                id = self.r.choice(cl_idx, pickup).tolist()
                idxs = idxs + id

        idxs = torch.Tensor(idxs).int()

        len_init = len(idxs)
        image_idxs = torch.randperm(len_init)

        cb = torch.cat([idxs]*(bag_size//len_init) + [idxs[image_idxs[:bag_size%len_init]]])

        feats = torch.stack([feat[int_indexes[cb, i]] for i, feat in enumerate(feats)])

        feats = feats.permute(1,0,2)

        return feats, 0

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        if self.load_features:
            imgs = self.all_feats[idx]
        else:
            filepath = self.df['filename'][idx]
            imgs = [torch.load(f).half() for f in filepath]

        label = self.df['label'][idx].astype(float)
        intersect_idxs = self.df['intersect_vals'][idx]

        if self.mil_config == 'zoommil':
            # zoomMIL
            subimgs= [1024, 512, 256]
            for im_idx, img in enumerate(imgs):
                idxs = torch.tensor(sample_images(list(range(len(img))), num_samples=subimgs[im_idx]))
                imgs[im_idx] = img[idxs]
        elif self.return_cluster:
            feats_all = []
            for ki in range(64):
                bag, skip = self.create_bags(idx, imgs, intersect_idxs)
                if skip:
                    return [], 1
                
                feats_all.append(bag)

            imgs = torch.stack(feats_all)
        else:
            idxs_to_sample = torch.tensor(sample_images(intersect_idxs, num_samples=self.num_subImgs))
            imgs = torch.stack([img[idxs_to_sample[:,idxs]] for idxs, img in enumerate(imgs)], dim=1)

        
        return imgs, label
    
  

    

class TCGA_patch_subtype(TCGA_subtype):

    def __init__(self, image_size=224, return_img=True, transform=None, augment=True, return_dupl_mat=False, **kwargs):   

        super().__init__(**kwargs)  
        
        self.image_size = image_size
        self.return_img = return_img
        self.return_dupl_mat = return_dupl_mat
        self.transform = transform
        
        # df = df.head()

        tqdm.pandas()
        print("Creating set of aligned patches and filenames...")
        self.df[['filename', 'intersect_vals', 'coordinates']] = self.df['wsi'].progress_apply(lambda x: get_aligned_patches(x, 
                                                self.data_path, self.mag, 'subimages_aligned')).apply(pd.Series)
        
        print("Finished creating set of aligned features...")

        if augment:
            self.transform = get_transform(self.istrain, tf=transform)
        else:
            self.transform = get_transform(False)
        
        print('%s data size=% d'%(kwargs['istrain'], self.df.shape[0]))
        print('data features path: %s'%self.data_path)

        

    def __len__(self):
        return self.df.shape[0]

        
    def __getitem__(self, idx):

        filepath = self.df['filename'][idx]

        intersect_idxs = self.df['intersect_vals'][idx]
        label = self.df['label'][idx].astype(float)

        idxs_to_sample = torch.tensor(sample_images(intersect_idxs, num_samples=self.num_subImgs)).T
        
        patch_list_all = self.df['coordinates'][idx]

        img_paths = [[os.path.join(filepath, im_name) for im_name in [patches[i] for i in idxs]] for idxs, patches \
                in zip(idxs_to_sample, patch_list_all)]            
        
        # n_images * n_mags * c * h * w
        imgs = list(map(list, zip(*img_paths)))

        if self.return_img:   
            imgs = torch.stack([torch.stack([self.transform(Image.open(im).convert('RGB').resize((self.image_size, 
                                                        self.image_size))) for im in im_mag]) for im_mag in imgs])

  
        return imgs, label
        

"""
CAMELYON ==================================================================================================================
"""

class Camelyon_subtype(Dataset):

    def __init__(self, num_subImgs=8, data_root=None, istrain='train', fold_num=0):  
         
        data_root = '/vol/research/scratch1/NOBACKUP/um00109/Camelyon/Camelyon16' if data_root is None else data_root

        self.num_subImgs = num_subImgs
        self.istrain = True if istrain=='train' else False

        self.data_root = data_root

        
        if istrain=='train':
            df = pd.read_csv(f'data_scripts/csvs/camelyon16/train_10fold_{fold_num}.csv')
        elif istrain=='val':
            df = pd.read_csv(f'data_scripts/csvs/camelyon16/val_10fold_{fold_num}.csv')
        else:
            df = pd.read_csv('data_scripts/csvs/camelyon16/reference_test.csv')

        df = df.rename(columns={'image':'wsi'})

        df = df.rename(columns={'type':'label_name'})
        df['label'] = df['label_name'].factorize(sort=True)[0]
        
        # df, self.data_path = get_splits_subtype(data_root, dataset_name, fold_num, istrain)

        # df = df.iloc[:50]

        self.df = df

        return


class Camelyon_feature_subtype(Camelyon_subtype):

    def __init__(self, encoder_name='vit_base_patch16_224.owkin_pancancer', 
                 load_features=False, mil_config='maxpool', skip_val=1, mag=20, **kwargs): 
        
        super().__init__(**kwargs)

        self.mil_config = mil_config
        self.load_features = load_features
        self.mag=mag
        self.skip_val=skip_val

        self.encoder_name = encoder_name.split('/')[-1]
        self.feat_dir = os.path.join(self.data_root, 'features')
        
        # f'skip_{skip_val}_{mag[0]}x', encoder_name)

        print(f'feat_dir: {self.feat_dir}')

        tqdm.pandas()
        print("Creating set of aligned patches and filenames...")
        self.df[['filename', 'coordinates']] = self.df['wsi'].progress_apply(lambda x: self.get_feats(x, self.feat_dir)).apply(pd.Series)
        print("Finished creating set of aligned features...")

        if self.load_features:
            print("Loading all features to memory...here we align the features as well.")
            self.all_feats = []
            for _, filename in tqdm(enumerate(self.df['filename'])):
                features = [torch.load(f, map_location='cpu').half() for f in filename]
                self.all_feats.append(features)
            print("Done loading all features to memory...")

        print('%s data size=% d'%(kwargs['istrain'], self.df.shape[0]))
        print('number of sub images: ', self.num_subImgs)

    def get_feats(self, x, feat_dir):

        x = x.replace('.tif', '')
        feats_ = []
        coords_ = []
        for mag in self.mag:
            feat_path = os.path.join(feat_dir, f'skip_{self.skip_val}_{mag}x', self.encoder_name, x, 'all_features.pt')
            coord_path = os.path.join(feat_dir, f'skip_{self.skip_val}_{mag}x', self.encoder_name, x, 'coords.pkl')
            with open(coord_path, 'rb') as f:
                coords = pickle.load(f)
            feats_.append(feat_path)
            coords_.append(coords)


        return feats_, coords_


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        if self.load_features:
            imgs = self.all_feats[idx]
        else:
            filepath = self.df['filename'][idx]
            imgs = [torch.load(f).half() for f in filepath]

        label = self.df['label'][idx].astype(float)
        # intersect_idxs = self.df['intersect_vals'][idx]

        if 'zoommil' in self.mil_config:
            # zoomMIL
            if self.istrain:
                subimgs = [self.num_subImgs, self.num_subImgs//4, self.num_subImgs//8]
            else:
                subimgs = [16384, 4096, 1024]
            for im_idx, img in enumerate(imgs):
                idxs = torch.tensor(sample_images(list(range(len(img))), num_samples=subimgs[im_idx]))
                imgs[im_idx] = img[idxs]

        else:
            idxs_to_sample = torch.tensor(sample_images(list(range(len(imgs[0]))), num_samples=self.num_subImgs))
            imgs = torch.stack([img[idxs_to_sample] for _, img in enumerate(imgs)], dim=1)

        
        return imgs, label
    
  

class Camelyon_patch_subtype(Camelyon_subtype):

    def __init__(self, image_size=224, 
                 augment=False, skip_val=1, mag=[20], return_img=True, transform=None, **kwargs): 
        
        super().__init__(**kwargs)

        self.image_size = image_size
        self.return_img = return_img
        self.transform = transform

        self.feat_dir = os.path.join(self.data_root, 'patches', f'skip_{skip_val}_{mag[0]}x')

        print(f'feat_dir: {self.feat_dir}')

        def get_feats(x, feat_dir):

            x = x.replace('.tif', '')
            coord_path = os.path.join(feat_dir, x, 'coords.pkl')

            with open(coord_path, 'rb') as f:
                coords = pickle.load(f)

            feat_path = [os.path.join(feat_dir, x, i) for i in coords]

            return feat_path, coords

        tqdm.pandas()
        print("Creating set of aligned patches and filenames...")
        self.df[['filename', 'coordinates']] = self.df['wsi'].progress_apply(lambda x: get_feats(x, self.feat_dir)).apply(pd.Series)
        print("Finished creating set of aligned features...")

        if augment:
            self.transform = get_transform(self.istrain, tf=transform)
        else:
            self.transform = get_transform(False)
        

        print('%s data size=% d'%(kwargs['istrain'], self.df.shape[0]))
        print('number of sub images: ', self.num_subImgs)


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        img_paths_all = self.df['filename'][idx]
        label = self.df['label'][idx].astype(float)

        idxs_to_sample = torch.tensor(sample_images(list(range(len(img_paths_all))),\
                                                     num_samples=self.num_subImgs))

        img_paths = [[img_paths_all[i] for i in idxs_to_sample]]

        # n_images * n_mags * c * h * w
        imgs = list(map(list, zip(*img_paths)))

        if self.return_img:
            imgs = torch.stack([torch.stack([self.transform(Image.open(im).convert('RGB').resize((self.image_size, 
                                                        self.image_size))) for im in im_mag]) for im_mag in imgs])


        return imgs, label
    
  
"""
BRACS ==================================================================================================================
"""

class BRACS_subtype(Dataset):

    def __init__(self, num_subImgs=8, data_root=None, istrain='train', fold_num=0, dataset_name='bracs'):  
         
        data_root = '/vol/research/scratch1/NOBACKUP/um00109/Camelyon/Camelyon16' if data_root is None else data_root

        df = pd.read_csv('data_scripts/csvs/BRACS.csv')
        df = df.iloc[:,:5]
        df = df.dropna(subset=['WSI Filename'])

        self.num_subImgs = num_subImgs
        self.istrain = True if istrain=='train' else False

        self.data_root = data_root

        if istrain=='train':
            df = df[df['Set'] == 'Training'].reset_index(drop=True)
        elif istrain=='val':
            df = df[df['Set'] == 'Validation'].reset_index(drop=True)
        else:
            df = df[df['Set'] == 'Testing'].reset_index(drop=True)

        df = df.rename(columns={'WSI Filename':'wsi'})
        df = df.rename(columns={'WSI label':'label_name'})

        df['label_name'][df['label_name']=='N']='benign'
        df['label_name'][df['label_name']=='PB']='benign'
        df['label_name'][df['label_name']=='UDH']='benign'

        df['label_name'][df['label_name']=='FEA']='atypical'
        df['label_name'][df['label_name']=='ADH']='atypical'
        
        df['label_name'][df['label_name']=='DCIS']='malignant'
        df['label_name'][df['label_name']=='IC']='malignant'       

        
        df['label'] = df['label_name'].factorize(sort=True)[0]
        df['label'] = df['label'].astype('int')
        
        self.df = df

        return
    

class BRACS_feature_subtype(BRACS_subtype):

    def __init__(self, encoder_name='vit_base_patch16_224.owkin_pancancer', 
                 load_features=False, mil_config='maxpool', skip_val=1, magnification=20, **kwargs): 
        
        super().__init__(**kwargs)

        self.mil_config = mil_config
        self.load_features = load_features
        self.mag=magnification
        self.skip_val=skip_val

        self.encoder_name = encoder_name.split('/')[-1]
        self.feat_dir = os.path.join(self.data_root, 'features')
        
        print(f'feat_dir: {self.feat_dir}')

        tqdm.pandas()
        print("Creating set of aligned patches and filenames...")
        self.df[['filename', 'coordinates']] = self.df['wsi'].progress_apply(lambda x: self.get_feats(x, self.feat_dir)).apply(pd.Series)
        print("Finished creating set of aligned features...")

        if self.load_features:
            print("Loading all features to memory...here we align the features as well.")
            self.all_feats = []
            for _, filename in tqdm(enumerate(self.df['filename'])):
                features = [torch.load(f, map_location='cpu').half() for f in filename]
                self.all_feats.append(features)
            print("Done loading all features to memory...")

        print('%s data size=% d'%(kwargs['istrain'], self.df.shape[0]))
        print('number of sub images: ', self.num_subImgs)

    def get_feats(self, x, feat_dir):

        x = x.replace('.svs', '')
        feats_ = []
        coords_ = []
        for mag in self.mag:
            feat_path = os.path.join(feat_dir, f'skip_{self.skip_val}_{mag}x', self.encoder_name, x, 'all_features.pt')
            coord_path = os.path.join(feat_dir, f'skip_{self.skip_val}_{mag}x', self.encoder_name, x, 'coords.pkl')
            with open(coord_path, 'rb') as f:
                coords = pickle.load(f)
            feats_.append(feat_path)
            coords_.append(coords)

        return feats_, coords_


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        if self.load_features:
            imgs = self.all_feats[idx]
        else:
            filepath = self.df['filename'][idx]
            imgs = [torch.load(f).half() for f in filepath]

        label = self.df['label'][idx].astype(float)

        idxs_to_sample = torch.tensor(sample_images(list(range(len(imgs[0]))), num_samples=self.num_subImgs))
        imgs = torch.stack([img[idxs_to_sample] for _, img in enumerate(imgs)], dim=1)

        
        return imgs, label.astype(int)
    

class BRACS_patch_subtype(BRACS_subtype):

    def __init__(self, image_size=224, 
                 augment=False, skip_val=1, magnification=[20], return_img=True, transform=None, **kwargs): 
        
        super().__init__(**kwargs)

        self.image_size = image_size
        self.return_img = return_img
        self.transform = transform

        self.feat_dir = os.path.join(self.data_root, 'patches', f'skip_{skip_val}_{magnification[0]}x')

        print(f'feat_dir: {self.feat_dir}')

        def get_feats(x, feat_dir):

            # x = x.replace('.tif', '')
            # coord_path = os.path.join(feat_dir, x, 'coords.pkl')

            # with open(coord_path, 'rb') as f:
            #     coords = pickle.load(f)

            coords = os.listdir(os.path.join(feat_dir, x))

            feat_path = [os.path.join(feat_dir, x, i) for i in coords]

            return feat_path, coords

        tqdm.pandas()
        print("Creating set of aligned patches and filenames...")
        self.df[['filename', 'coordinates']] = self.df['wsi'].progress_apply(lambda x: get_feats(x, self.feat_dir)).apply(pd.Series)
        print("Finished creating set of aligned features...")

        if augment:
            self.transform = get_transform(self.istrain, tf=transform)
        else:
            self.transform = get_transform(False)
        

        print('%s data size=% d'%(kwargs['istrain'], self.df.shape[0]))
        print('number of sub images: ', self.num_subImgs)


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        img_paths_all = self.df['filename'][idx]
        label = self.df['label'][idx].astype(int)

        idxs_to_sample = torch.tensor(sample_images(list(range(len(img_paths_all))),\
                                                     num_samples=self.num_subImgs))

        img_paths = [[img_paths_all[i] for i in idxs_to_sample]]

        # n_images * n_mags * c * h * w
        imgs = list(map(list, zip(*img_paths)))

        if self.return_img:
            imgs = torch.stack([torch.stack([self.transform(Image.open(im).convert('RGB').resize((self.image_size, 
                                                        self.image_size))) for im in im_mag]) for im_mag in imgs])


        return imgs, label