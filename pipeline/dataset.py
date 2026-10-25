"""One patient per item; deterministic evaluation bags shared across models."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config
from pipeline.data import (FrozenScaler, read_expression, read_labels, read_ids,
                           check_partitions, fixed_indices)

MODES = ("image", "gene", "early_fusion", "late_fusion")


class PatientDataset(Dataset):
    def __init__(self, sample_ids, mode="image", is_train=True, scaler=None,
                 data_dir=None, expression=None, cohort="tcga", bag_size=None, frozen_bags=None):
        if mode not in MODES:
            raise ValueError(mode)
        self.data_dir = Path(data_dir or config.DATA_DIR)
        self.sample_ids = list(sample_ids)
        self.mode, self.is_train, self.cohort = mode, is_train, cohort
        self.bag_size = bag_size or config.NUM_PATCHES_SAMPLE
        self.frozen_bags = frozen_bags
        labels = read_labels(self.data_dir)
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("A dataset must contain unique patients")
        self.labels = labels.loc[self.sample_ids].to_dict()
        self.scaler = scaler
        if mode != "image":
            frame = expression if expression is not None else read_expression(self.data_dir)
            if is_train and scaler is None:
                self.scaler = FrozenScaler.fit(frame, self.sample_ids)
            if self.scaler is None:
                raise ValueError("Evaluation requires the saved TCGA training scaler")
            if is_train and set(self.scaler.fit_ids) != set(self.sample_ids):
                raise ValueError("Scaler fit IDs differ from this training partition")
            self.gene_df = self.scaler.transform(frame.loc[self.sample_ids])

    def __len__(self):
        return len(self.sample_ids)

    def indices(self, sid, count):
        if self.is_train:
            return np.random.choice(count, self.bag_size, replace=count < self.bag_size)
        idx = (np.asarray(self.frozen_bags[sid], dtype=int) if self.frozen_bags is not None
               else fixed_indices(sid, count, self.bag_size, config.RANDOM_SEED, self.cohort))
        if (idx.shape != (self.bag_size,) or len(set(idx)) != self.bag_size
                or (idx < 0).any() or (idx >= count).any()):
            raise ValueError(f"Invalid fixed evaluation bag: {sid}")
        return idx

    def freeze_bags(self):
        if self.is_train:
            raise ValueError("Training bags are stochastic; only evaluation bags may be frozen")
        bags = {}
        for sid in self.sample_ids:
            root = self.data_dir / "svs"
            path = root / f"{sid}_patches.h5"
            if path.exists():
                import h5py
                with h5py.File(path, "r") as handle:
                    count = len(handle["patches"])
            else:
                with np.load(root / f"{sid}_patches.npz", allow_pickle=False) as handle:
                    count = len(handle["patches"])
            bags[sid] = self.indices(sid, count)
        return bags

    def image(self, sid):
        root = self.data_dir / "svs"
        h5_path, npz_path = root / f"{sid}_patches.h5", root / f"{sid}_patches.npz"
        if h5_path.exists():
            import h5py
            with h5py.File(h5_path, "r") as handle:
                if handle.attrs.get("pixel_encoding", "") != "rgb_uint8_gamma_0.9":
                    raise ValueError(f"Unknown pixel encoding: {h5_path}")
                n = len(handle["patches"])
                idx = self.indices(sid, n)
                # Preserve sampled order; h5py fancy indexing requires sorted indices.
                patches = np.stack([handle["patches"][int(i)] for i in idx]).astype(np.float32) / 255
            patches = (patches - config.IMAGENET_MEAN) / config.IMAGENET_STD
        else:
            # Legacy NPZ files produced by preprocess/process_svs.py were already
            # ImageNet-normalised. They must not be normalised a second time.
            with np.load(npz_path, allow_pickle=False) as handle:
                data = handle["patches"]
                n = len(data)
                idx = self.indices(sid, n)
                patches = data[idx].astype(np.float32)
            if patches.shape[-1] != 3 or not np.isfinite(patches).all():
                raise ValueError(f"Expected finite NHWC normalised patches: {npz_path}")
        if self.is_train:
            if np.random.rand() > .5:
                patches = np.flip(patches, axis=1)
            if np.random.rand() > .5:
                patches = np.flip(patches, axis=2)
            if np.random.rand() > .5:
                patches = np.rot90(patches, axes=(1, 2))
        return torch.from_numpy(patches.transpose(0, 3, 1, 2).copy()).float()

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        label = torch.tensor(self.labels[sid], dtype=torch.long)
        if self.mode == "image":
            return self.image(sid), label
        gene = torch.tensor(self.gene_df.loc[sid].to_numpy(), dtype=torch.float32)
        if self.mode == "gene":
            return gene, label
        return self.image(sid), gene, label


def loader(dataset, training=False):
    if training and len(dataset) < 2:
        raise ValueError("BatchNorm training needs at least two patients")
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=training,
                      num_workers=config.NUM_WORKERS,
                      drop_last=training and len(dataset) % config.BATCH_SIZE == 1,
                      generator=torch.Generator().manual_seed(config.RANDOM_SEED))


def make_loaders(train_ids, val_ids, mode, data_dir=None):
    labels = read_labels(data_dir or config.DATA_DIR)
    check_partitions(train_ids, val_ids, labels.index)
    train = PatientDataset(train_ids, mode, True, data_dir=data_dir)
    val = PatientDataset(val_ids, mode, False, scaler=train.scaler, data_dir=data_dir)
    return loader(train, True), loader(val), train.scaler
