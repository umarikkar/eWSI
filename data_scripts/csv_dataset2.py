from torch.utils.data import Dataset
import pandas as pd
import os

import torch

from torchvision import transforms


from PIL import Image


from data_scripts.get_splits import get_splits_subtype
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
    


class TCGA_subtype(Dataset):

    def __init__(self, fold_num=0, num_subImgs=8, data_root=None, dataset_name='tcga-brca', magnification=[20],
                 istrain='train'):  
         
        data_root = '/vol/research/scratch1/NOBACKUP/um00109/tcga' if data_root is None else data_root

        self.num_subImgs = num_subImgs
        self.istrain = True if istrain=='train' else False
        self.mag = sorted(magnification)[::-1]
        
        df, self.data_path = get_splits_subtype(data_root, dataset_name, fold_num, istrain)

        # df = df.iloc[:50]

        self.df = df

        return
    




class TCGA_feature_subtype(TCGA_subtype):

    def __init__(self, encoder_name='vit_base_patch16_224.owkin_pancancer', 
                 load_features=False, return_clusters=False, align=True, **kwargs): 
        
        super().__init__(**kwargs)

        self.align = align
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

        if not self.return_cluster and self.align:
            idxs_to_sample = torch.tensor(sample_images(intersect_idxs, num_samples=self.num_subImgs))
            imgs = torch.stack([img[idxs_to_sample[:,idxs]] for idxs, img in enumerate(imgs)], dim=1)
        elif self.return_cluster:
            feats_all = []
            for ki in range(64):
                bag, skip = self.create_bags(idx, imgs, intersect_idxs)
                if skip:
                    return [], 1
                
                feats_all.append(bag)

            imgs = torch.stack(feats_all)
        else:
            # zoomMIL
            subimgs= [1024, 512, 256]
            for im_idx, img in enumerate(imgs):
                idxs = torch.tensor(sample_images(list(range(len(img))), num_samples=subimgs[im_idx]))
                imgs[im_idx] = img[idxs]
        
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
        


class ChunkedImageSampleDataset(TCGA_subtype):

    def __init__(self, image_size=224, return_img=False, return_chunks=True, transform=None, augment=True, return_dupl_mat=False, **kwargs):   

        super().__init__(**kwargs)  
        
        self.image_size = image_size
        self.return_img = return_img
        self.return_dupl_mat = return_dupl_mat
        self.transform = transform
        self.chunk_size=128
        self.return_chunks=return_chunks

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
        

    def custom_collate_fn(batch):
        """
        Custom collate function to deal with lazy image loading.
        Each batch will be a generator that loads images in chunks of 128.
        """
        # Since we are using batch_size=1, we'll get a list of one item
        image_gen, label = batch[0]
        
        # Return the sample (which is a generator) directly
        return image_gen, torch.tensor([label]).half()


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        Return the entire sample, but chunked into groups of `chunk_size`.
        """

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
        elif self.return_chunks:        
            def image_chunk_generator():
                for i in range(0, len(imgs), self.chunk_size):
                    chunk_paths = imgs[i:i+self.chunk_size]
                    images = torch.stack([torch.stack([self.transform(Image.open(im).convert('RGB').resize((self.image_size, 
                                                self.image_size))) for im in im_mag]) for im_mag in chunk_paths])
                    yield images

        return image_chunk_generator(), label