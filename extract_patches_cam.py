import os
import math
import numpy as np
from tqdm import tqdm
from PIL import Image
import openslide
import cv2
import matplotlib.pyplot as plt
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


def get_best_level_for_target_downsample(slide, target_downsample: float) -> int:
    downsamples = np.array(slide.level_downsamples, dtype=np.float32)
    level = int(np.argmin(np.abs(downsamples - target_downsample)))
    return level


def get_objective_power(slide) -> float:
    props = slide.properties
    # Common keys
    for k in ["openslide.objective-power", "aperio.AppMag"]:
        if k in props:
            try:
                return float(props[k])
            except Exception:
                pass
    if "openslide.mpp-x" in props:
        mpp = float(props["openslide.mpp-x"])
        if mpp <= 0.3:
            return 40.0
        if mpp <= 0.6:
            return 20.0
        
    return 40.0


def make_tissue_mask(slide, mask_level: int = None, sat_thresh=13, val_thresh=255):

    if mask_level is None:
        mask_level = len(slide.level_dimensions) - 1  
    w, h = slide.level_dimensions[mask_level]
    img = slide.read_region((0, 0), mask_level, (w, h)).convert("RGB")
    img_np = np.array(img)

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    tissue = (s > sat_thresh) & (v < val_thresh)

    tissue_u8 = tissue.astype(np.uint8) * 255

    tissue_u8 = cv2.morphologyEx(tissue_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    tissue_u8 = cv2.morphologyEx(tissue_u8, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    tissue = tissue_u8 > 0


    return img_np, tissue, mask_level





def extract_patches_20x(
    wsi_path: str,
    out_dir: str,
    patch_size: int = 256,
    stride: int = 256,  
    target_mag: float = 20.0,
    tissue_frac_thresh: float = 0.5,  
):
    os.makedirs(out_dir, exist_ok=True)
    slide = openslide.OpenSlide(wsi_path)

    base_mag = get_objective_power(slide)
    target_downsample = base_mag / target_mag

    level = get_best_level_for_target_downsample(slide, target_downsample)
    ds = slide.level_downsamples[level]
    level_w, level_h = slide.level_dimensions[level]

    # tissue masks
    img_np, tissue_mask, mask_level = make_tissue_mask(slide)
    mask_w, mask_h = tissue_mask.shape[1], tissue_mask.shape[0]

    ds_mask = slide.level_downsamples[mask_level]
    scale = ds / ds_mask  

    n_x = math.floor((level_w - patch_size) / stride) + 1
    n_y = math.floor((level_h - patch_size) / stride) + 1

    patch_dir = os.path.join(out_dir, "patches")
    os.makedirs(patch_dir, exist_ok=True)

    patch_id = 0
    for iy in tqdm(range(n_y), desc=os.path.basename(wsi_path)):
        for ix in range(n_x):
            x_lv = ix * stride
            y_lv = iy * stride

            # Check tissue coverage
            x_m = int(x_lv * scale)
            y_m = int(y_lv * scale)
            ps_m = max(1, int(patch_size * scale))

            x_m2 = min(mask_w, x_m + ps_m)
            y_m2 = min(mask_h, y_m + ps_m)

            if x_m >= mask_w or y_m >= mask_h or x_m2 <= x_m or y_m2 <= y_m:
                continue

            frac = tissue_mask[y_m:y_m2, x_m:x_m2].mean()
            if frac < tissue_frac_thresh:
                # if no tissue in this area
                continue

            # level-0 coords for read_region
            x0 = int(x_lv * ds)
            y0 = int(y_lv * ds)

            patch = slide.read_region((x0, y0), level, (patch_size, patch_size)).convert("RGB")
            patch.save(os.path.join(patch_dir, f"patch_{patch_id:08d}_x{x0}_y{y0}_lv{level}.jpg"))

            patch_id += 1

    wsi_name = os.path.basename(wsi_path).replace(".tif", "")

    slide_dir = os.path.join(out_dir, "slides")
    os.makedirs(slide_dir, exist_ok=True)
    plt.imsave(os.path.join(slide_dir, f"slide_{wsi_name}.png"), img_np)

    mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    plt.imsave(os.path.join(mask_dir, f"mask_{wsi_name}.png"), tissue_mask)

    print(f'saved patches for {wsi_name}')

    slide.close()

    return patch_id, level, ds, base_mag



def get_args():
    parser = argparse.ArgumentParser(
        description="Extract 20x patches from Camelyon16 WSIs"
    )
    parser.add_argument("--data_dir",type=str,default="/vol/research/datasets/pathology/Camelyon/Camelyon16/",)
    parser.add_argument("--out_root",type=str,default="./patches/",)
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    data_dir = args.data_dir
    out_root = args.out_root

    wsi_list = glob.glob(os.path.join(data_dir, "*/*", "*.tif"))
    wsi_list.sort()

    for wsi in tqdm(wsi_list):
        wsi_name = os.path.basename(wsi).replace(".tif", "")
        out_dir = os.path.join(out_root, wsi_name)

        n, level, ds, base_mag = extract_patches_20x(wsi, out_dir)





