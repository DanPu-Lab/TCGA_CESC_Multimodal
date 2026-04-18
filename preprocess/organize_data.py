import os
import pandas as pd
from tqdm import tqdm
import sys

# 确保能导入外层的 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def extract_label(folder_name):
    """根据文件夹名称后缀判断标签，C1为0，C2为1"""
    return 0 if folder_name.endswith("C1") else 1

def main():
    print(f"Scanning raw data directory: {config.RAW_DATA_DIR}")
    labels_dir = os.path.join(config.DATA_DIR, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    labels = []
    # 假设你的原始数据存放在 config.RAW_DATA_DIR 下
    if not os.path.exists(config.RAW_DATA_DIR):
        raise FileNotFoundError(f"Raw data directory not found: {config.RAW_DATA_DIR}")

    for folder in tqdm(os.listdir(config.RAW_DATA_DIR), desc="Organizing labels"):
        if not folder.startswith("TCGA"):
            continue
            
        sample_id = "-".join(folder.split("-")[:3])
        label = extract_label(folder)
        labels.append({"sample_id": sample_id, "label": label})
        
    if not labels:
        print("Warning: No TCGA folders found in raw data directory.")
        return

    df = pd.DataFrame(labels)
    output_path = os.path.join(labels_dir, "all_labels.csv")
    df.to_csv(output_path, index=False)
    print(f"Successfully generated labels for {len(df)} samples at {output_path}")

if __name__ == "__main__":
    main()