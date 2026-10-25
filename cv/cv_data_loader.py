"""Fold-local scalers; the outer held-out cohort is never used here."""
from pathlib import Path
import config
from pipeline.dataset import PatientDataset, make_loaders
from pipeline.data import read_ids, check_partitions

CVDataset = PatientDataset

def get_cv_dataloaders(fold):
    root = Path(config.RESULT_DIR) / "cross_validation_80percent/splits"
    tr = read_ids(root / f"fold_{fold}_train_ids.csv")
    val = read_ids(root / f"fold_{fold}_val_ids.csv")
    outer = read_ids(Path(config.DATA_DIR) / "splits/train_ids.csv")
    check_partitions(tr, val, outer)
    if set(tr) | set(val) != set(outer):
        raise ValueError("CV fold does not cover exactly the outer training partition")
    result = {}
    for mode in ("image", "gene", "early_fusion", "late_fusion"):
        train, validation, _ = make_loaders(tr, val, mode)
        result[mode] = {"train": train, "val": validation}
    return result
