import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
from sklearn.preprocessing import StandardScaler
import config

class MultimodalDataset(Dataset):
    def __init__(self, sample_ids, mode='image', is_train=True, scaler=None):
        self.sample_ids = sample_ids
        self.mode = mode
        self.is_train = is_train
        labels = pd.read_csv(os.path.join(config.DATA_DIR, "labels/all_labels.csv"))
        self.labels = {row['sample_id']: row['label'] for _, row in labels.iterrows()}

        if mode in ['gene', 'early_fusion', 'late_fusion']:
            adata = sc.read_h5ad(os.path.join(config.DATA_DIR, "merged_genes.h5ad"))
            gene_df = adata.to_df()
            if is_train:
                self.scaler = StandardScaler()
                self.gene_data = self.scaler.fit_transform(gene_df.values)
            else:
                self.scaler = scaler
                self.gene_data = self.scaler.transform(gene_df.values)
            self.gene_df = pd.DataFrame(self.gene_data, index=gene_df.index, columns=gene_df.columns)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        label = self.labels[sid]

        if self.mode == 'image':
            return self._get_image(sid, label)
        elif self.mode == 'gene':
            return self._get_gene(sid, label)
        elif self.mode in ['early_fusion', 'late_fusion']:
            img, _ = self._get_image(sid, label)
            gene, _ = self._get_gene(sid, label)
            return img, gene, label

    def _get_image(self, sid, label):
        data = np.load(os.path.join(config.DATA_DIR, f"svs/{sid}_patches.npz"))
        patches = data['patches']
        if len(patches) > config.NUM_PATCHES_SAMPLE:
            idx = np.random.choice(len(patches), config.NUM_PATCHES_SAMPLE, replace=False)
        else:
            idx = np.random.choice(len(patches), config.NUM_PATCHES_SAMPLE, replace=True)
        patches = patches[idx]
        if self.is_train:
            if np.random.rand() > 0.5:
                patches = np.flip(patches, axis=2).copy()
            if np.random.rand() > 0.5:
                patches = np.flip(patches, axis=1).copy()
            if np.random.rand() > 0.5:
                patches = np.rot90(patches, k=1, axes=(1,2)).copy()
        patches = np.transpose(patches, (0,3,1,2))
        return torch.FloatTensor(patches), torch.tensor(label, dtype=torch.long)

    def _get_gene(self, sid, label):
        feat = self.gene_df.loc[sid].values.astype(np.float32)
        return torch.FloatTensor(feat), torch.tensor(label, dtype=torch.long)

def get_dataloaders():
    train_ids = pd.read_csv(os.path.join(config.DATA_DIR, "splits/train_ids.csv"))['sample_id'].tolist()
    test_ids = pd.read_csv(os.path.join(config.DATA_DIR, "splits/test_ids.csv"))['sample_id'].tolist()

    datasets = {}
    for mode in ['image', 'gene', 'early_fusion', 'late_fusion']:
        train_ds = MultimodalDataset(train_ids, mode=mode, is_train=True)
        if mode in ['gene', 'early_fusion', 'late_fusion']:
            scaler = train_ds.scaler
        else:
            scaler = None
        test_ds = MultimodalDataset(test_ids, mode=mode, is_train=False, scaler=scaler)
        datasets[mode] = {
            'train': DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
            'test': DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        }
    return datasets