"""Compatibility API backed by explicit training-only preprocessing."""
from pathlib import Path
import config
from pipeline.dataset import PatientDataset, make_loaders
from pipeline.data import read_ids

MultimodalDataset = PatientDataset

def get_dataloaders():
    root = Path(config.DATA_DIR) / "splits"
    train_ids, test_ids = read_ids(root / "train_ids.csv"), read_ids(root / "test_ids.csv")
    result = {}
    for mode in ("image", "gene", "early_fusion", "late_fusion"):
        train, test, _ = make_loaders(train_ids, test_ids, mode)
        result[mode] = {"train": train, "test": test}
    return result
