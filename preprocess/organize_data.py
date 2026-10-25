"""Validate a curated patient manifest; never infer labels from missing suffixes."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import pandas as pd
import config


def extract_label(folder_name):
    if folder_name.endswith("-C1"):
        return 0
    if folder_name.endswith("-C2"):
        return 1
    raise ValueError(f"No explicit C1/C2 annotation: {folder_name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="CSV: sample_id,label,rna_path,wsi_path; one selected primary diagnostic WSI per patient")
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest, dtype={"sample_id": str})
    if not {"sample_id", "label", "rna_path", "wsi_path"} <= set(frame):
        raise ValueError("Required columns: sample_id,label,rna_path,wsi_path")
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("Manifest must identify unique patients, not slides")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Reference labels must be C1=0, C2=1")
    for column in ["rna_path", "wsi_path"]:
        if not all(Path(path).is_file() for path in frame[column]):
            raise ValueError(f"Missing files in {column}")
    out = Path(config.DATA_DIR) / "labels/all_labels.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    frame[["sample_id", "label"]].to_csv(out, index=False)


if __name__ == "__main__":
    main()
