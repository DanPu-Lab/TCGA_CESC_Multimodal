"""Audit externally inferred reference labels; no performance-based selection."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import pandas as pd


def select_cohort(manifest, calls, require_agreement=True):
    required = {"sample_id", "rna_path", "wsi_path"}
    if not required <= set(manifest) or manifest.sample_id.duplicated().any():
        raise ValueError("Paired, cervical-cancer-only manifest needs unique sample_id,rna_path,wsi_path")
    if "SampleIDs" in calls and "sample_id" not in calls:
        calls = calls.rename(columns={"SampleIDs": "sample_id"})
    if not {"sample_id", "BestCall", *map(str, range(1, 7))} <= set(calls):
        raise ValueError("Require official ImmuneSubtypeClassifier BestCall and scores 1..6")
    if calls.sample_id.duplicated().any() or set(calls.sample_id) != set(manifest.sample_id):
        raise ValueError("Classifier output must correspond exactly to the paired input patients")
    joined = manifest.merge(calls, on="sample_id", validate="one_to_one")
    scores = joined[list(map(str, range(1, 7)))].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores.sum(1) <= 0).any():
        raise ValueError("Invalid classifier scores")
    scores /= scores.sum(1, keepdims=True)
    best = joined.BestCall.astype(str).str.replace(r"^C", "", regex=True).astype(int)
    if not best.isin(range(1, 7)).all():
        raise ValueError("BestCall must be an official C1-C6 assignment")
    top = scores.argmax(1) + 1
    joined["non_C1_C2"] = ~best.isin([1, 2])
    joined["BestCall_argmax_discordant"] = best.to_numpy() != top
    joined["retained"] = ~joined.non_C1_C2 & (~joined.BestCall_argmax_discordant if require_agreement else True)
    joined["label"] = best - 1
    # No score-based cut-off. Keep all rejection reasons, including their overlap.
    return joined.loc[joined.retained].copy(), joined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--classifier-output", required=True, help="Official callSubtypes() Pred table exported as CSV")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bestcall-only", action="store_true", help="Explicit sensitivity analysis including discordant C1/C2 BestCall cases")
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    kept, audit = select_cohort(pd.read_csv(args.manifest), pd.read_csv(args.classifier_output),
                                not args.bestcall_only)
    if args.expected_count is not None and len(kept) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} retained cases, obtained {len(kept)}; do not alter labels to force counts")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "labels").mkdir()
    audit.to_csv(args.output / "eligibility_audit.csv", index=False)
    kept[["sample_id", "label"]].to_csv(args.output / "labels/all_labels.csv", index=False)
    kept[["sample_id", "rna_path", "wsi_path"]].to_csv(args.output / "paired_manifest.csv", index=False)
    print("Retained", len(kept), "patients; C1/C2 counts", kept.label.value_counts().to_dict())


if __name__ == "__main__":
    main()
