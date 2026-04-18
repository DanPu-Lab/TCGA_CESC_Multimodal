import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import config

def main():
    train_ids = pd.read_csv(os.path.join(config.DATA_DIR, "splits/train_ids.csv"))['sample_id'].tolist()
    labels = pd.read_csv(os.path.join(config.DATA_DIR, "labels/all_labels.csv"))
    train_labels = labels[labels['sample_id'].isin(train_ids)]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out_dir = os.path.join(config.RESULT_DIR, "cross_validation_80percent/splits")
    os.makedirs(out_dir, exist_ok=True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_ids, train_labels['label'])):
        train_ids_fold = train_labels.iloc[train_idx]['sample_id']
        val_ids_fold = train_labels.iloc[val_idx]['sample_id']
        train_ids_fold.to_csv(os.path.join(out_dir, f"fold_{fold}_train_ids.csv"), index=False)
        val_ids_fold.to_csv(os.path.join(out_dir, f"fold_{fold}_val_ids.csv"), index=False)
    print("CV splits generated.")

if __name__ == "__main__":
    main()