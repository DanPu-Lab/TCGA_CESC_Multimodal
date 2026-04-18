import os
import pandas as pd
from sklearn.model_selection import train_test_split
import config

def main():
    labels = pd.read_csv(os.path.join(config.DATA_DIR, "labels/all_labels.csv"))
    train, test = train_test_split(labels['sample_id'], test_size=0.2, stratify=labels['label'], random_state=42)
    os.makedirs(os.path.join(config.DATA_DIR, "splits"), exist_ok=True)
    train.to_csv(os.path.join(config.DATA_DIR, "splits/train_ids.csv"), index=False)
    test.to_csv(os.path.join(config.DATA_DIR, "splits/test_ids.csv"), index=False)

if __name__ == "__main__":
    main()