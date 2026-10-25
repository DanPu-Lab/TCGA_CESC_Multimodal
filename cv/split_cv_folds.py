from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import config
from pipeline.data import read_ids, read_labels, check_partitions

def main():
    outer = read_ids(Path(config.DATA_DIR) / "splits/train_ids.csv")
    test = read_ids(Path(config.DATA_DIR) / "splits/test_ids.csv")
    check_partitions(outer, test)
    labels = read_labels(config.DATA_DIR).loc[outer]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    out = Path(config.RESULT_DIR) / "cross_validation_80percent/splits"
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("fold_*_ids.csv")):
        raise FileExistsError("Existing CV patient splits will not be overwritten")
    # IDs and labels have exactly the same order; no isin/filter reordering.
    for fold, (tr, val) in enumerate(folds.split(outer, labels.to_numpy())):
        for name, indices in [("train", tr), ("val", val)]:
            pd.DataFrame({"sample_id": [outer[i] for i in indices]}).to_csv(
                out / f"fold_{fold}_{name}_ids.csv", index=False)

if __name__ == "__main__":
    main()
