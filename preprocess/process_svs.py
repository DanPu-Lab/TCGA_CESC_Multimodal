import os
import numpy as np
import cv2
import openslide
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import config

PATCH_SIZE = config.PATCH_SIZE
MIN_PATCHES = config.MIN_PATCHES
CHUNK_SIZE = 1000 # 内存安全块大小

def create_tissue_mask(thumbnail):
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)
    _, sat_mask = cv2.threshold(hsv[:,:,1], 20, 255, cv2.THRESH_BINARY)
    _, val_mask = cv2.threshold(hsv[:,:,2], 40, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(sat_mask, val_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) > 0

def dynamic_stride(tissue_density):
    if tissue_density > 0.7: return int(PATCH_SIZE * 0.4)
    elif tissue_density > 0.3: return int(PATCH_SIZE * 0.6)
    else: return int(PATCH_SIZE * 0.8)

def process_single_svs(svs_path):
    """独立的单文件处理函数，方便多进程调用"""
    sample_id = os.path.basename(svs_path).split('.')[0]
    out_dir = os.path.join(config.DATA_DIR, "svs")
    final_npz = os.path.join(out_dir, f"{sample_id}_patches.npz")
    
    if os.path.exists(final_npz):
        return True, sample_id
        
    try:
        slide = openslide.OpenSlide(svs_path)
        thumb = np.array(slide.get_thumbnail((512,512)))
        mask = create_tissue_mask(thumb)
        
        all_patches = []
        for level in [1, 0, 2]:
            if level >= slide.level_count: continue
            
            w, h = slide.level_dimensions[level]
            ds = slide.level_downsamples[level]
            mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            stride = dynamic_stride(np.sum(mask_resized) / max(1, mask_resized.size))
            
            for x in range(0, w - PATCH_SIZE + 1, stride):
                for y in range(0, h - PATCH_SIZE + 1, stride):
                    cx, cy = x + PATCH_SIZE//2, y + PATCH_SIZE//2
                    if cx < w and cy < h and mask_resized[cy, cx]:
                        patch = np.array(slide.read_region((int(x*ds), int(y*ds)), level, (PATCH_SIZE, PATCH_SIZE)))[:,:,:3]
                        patch = patch.astype(np.float32) / 255.0
                        patch = (patch - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
                        all_patches.append(patch)
                        
                        if len(all_patches) >= MIN_PATCHES * 2:
                            break
                if len(all_patches) >= MIN_PATCHES * 2:
                    break
            if len(all_patches) >= MIN_PATCHES:
                break
                
        if len(all_patches) < MIN_PATCHES:
            return False, f"{sample_id}: Only {len(all_patches)} patches extracted."
            
        np.savez_compressed(final_npz, patches=np.array(all_patches, dtype=np.float32))
        return True, f"{sample_id} success"
        
    except Exception as e:
        return False, f"{sample_id} failed: {str(e)}"

def main():
    svs_dir = os.path.join(config.DATA_DIR, "svs")
    os.makedirs(svs_dir, exist_ok=True)
    
    # 获取原始数据的路径
    raw_svs_files = []
    for root, _, files in os.walk(config.RAW_DATA_DIR):
        for f in files:
            if f.endswith('.svs'):
                raw_svs_files.append(os.path.join(root, f))
                
    print(f"Found {len(raw_svs_files)} SVS files. Starting multi-processing...")
    
    num_workers = max(1, cpu_count() - 2)
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_svs, raw_svs_files), total=len(raw_svs_files), desc="Processing WSI"))
        
    success = sum(1 for r, _ in results if r)
    print(f"Finished. Success: {success}/{len(raw_svs_files)}")

if __name__ == "__main__":
    main()