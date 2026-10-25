"""Data alignment and training-only standardisation.

Expression inputs are unstandardised log1p(TPM). A scaler is always fitted on
explicit training IDs, never on the complete loaded matrix or evaluation data.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def read_ids(path):
    frame = pd.read_csv(path, dtype={"sample_id": str})
    if "sample_id" not in frame:
        raise ValueError(f"Missing sample_id column: {path}")
    ids = frame.sample_id.tolist()
    if not ids or len(set(ids)) != len(ids) or frame.sample_id.isna().any():
        raise ValueError(f"Empty or duplicated patient IDs: {path}")
    return ids


def read_labels(data_dir):
    frame = pd.read_csv(Path(data_dir) / "labels/all_labels.csv", dtype={"sample_id": str})
    if not {"sample_id", "label"}.issubset(frame):
        raise ValueError("Labels require sample_id and label columns")
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("Patient labels must be unique and nonmissing")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Only reference labels C1=0 and C2=1 are supported")
    return frame.set_index("sample_id").label.astype(int)


def read_expression(data_dir):
    import anndata as ad
    frame = ad.read_h5ad(Path(data_dir) / "merged_genes.h5ad").to_df()
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Expression patient IDs and gene names must be unique")
    if not np.isfinite(frame.to_numpy()).all() or (frame.to_numpy() < 0).any():
        raise ValueError("Require nonnegative unstandardised log1p(TPM), not pre-standardised expression")
    return frame


def check_partitions(train_ids, evaluation_ids, allowed_ids=None):
    for ids in (train_ids, evaluation_ids):
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicated patient IDs within a partition")
    if set(train_ids) & set(evaluation_ids):
        raise ValueError("Patient overlap between training and evaluation")
    if allowed_ids is not None and not (set(train_ids) | set(evaluation_ids)) <= set(allowed_ids):
        raise ValueError("Partition includes unknown patients")


def order_digest(genes):
    return hashlib.sha256("\n".join(map(str, genes)).encode()).hexdigest()


@dataclass
class FrozenScaler:
    genes: list
    fit_ids: list
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame, train_ids):
        if not train_ids or len(set(train_ids)) != len(train_ids):
            raise ValueError("Explicit unique training IDs are required")
        selected = frame.loc[list(train_ids)]
        if not np.isfinite(selected.to_numpy()).all():
            raise ValueError("Nonfinite training expression")
        fitted = StandardScaler().fit(selected.to_numpy(dtype=np.float64))
        return cls(frame.columns.tolist(), list(train_ids), fitted.mean_, fitted.scale_)

    def transform(self, frame):
        if not frame.columns.is_unique:
            raise ValueError("Duplicated input genes")
        missing = set(self.genes) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing {len(missing)} required genes; no imputation is performed")
        values = frame.loc[:, self.genes].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite expression values")
        return pd.DataFrame((values - self.mean) / self.scale,
                            index=frame.index, columns=self.genes)

    def save(self, path):
        np.savez_compressed(path, genes=np.asarray(self.genes, dtype=str),
                            fit_ids=np.asarray(self.fit_ids, dtype=str),
                            mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            result = cls(data["genes"].astype(str).tolist(), data["fit_ids"].astype(str).tolist(),
                         data["mean"].copy(), data["scale"].copy())
        n = len(result.genes)
        if (len(set(result.genes)) != n or not result.fit_ids
                or len(set(result.fit_ids)) != len(result.fit_ids)
                or result.mean.shape != (n,) or result.scale.shape != (n,)
                or not np.isfinite(result.mean).all() or not np.isfinite(result.scale).all()
                or not (result.scale > 0).all()):
            raise ValueError("Invalid frozen scaler artifact")
        return result


def fixed_indices(case_id, n_patches, bag_size=50, seed=42, cohort="tcga"):
    if n_patches < bag_size:
        raise ValueError(f"{case_id}: fewer than {bag_size} patches; review WSI QC")
    token = f"{cohort}-fixed-bag-v1|{seed}|{case_id}".encode()
    case_seed = int.from_bytes(hashlib.sha256(token).digest()[:4], "little")
    return np.random.RandomState(case_seed).choice(n_patches, bag_size, replace=False)


def manifest_metadata(train_ids, eval_ids, genes):
    return {"training_ids": list(train_ids), "evaluation_ids": list(eval_ids),
            "gene_order_sha256": order_digest(genes), "positive_class": "C2",
            "scaler_fit_partition": "training_only"}
