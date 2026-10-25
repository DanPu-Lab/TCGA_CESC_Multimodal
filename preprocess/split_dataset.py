from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.model_selection import train_test_split
import config
from pipeline.data import read_labels, check_partitions

def main():
    labels = read_labels(config.DATA_DIR).sort_index()
    train, test = train_test_split(labels.index.to_numpy(), test_size=.2,
                                  stratify=labels.to_numpy(), random_state=config.RANDOM_SEED)
    check_partitions(train, test, labels.index)
    out = Path(config.DATA_DIR) / "splits"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train_ids.csv").exists() or (out / "test_ids.csv").exists():
        raise FileExistsError("Existing patient split will not be overwritten")
    for name, ids in [("train", train), ("test", test)]:
        pd.DataFrame({"sample_id": ids}).to_csv(out / f"{name}_ids.csv", index=False)

if __name__ == "__main__":
    main()
