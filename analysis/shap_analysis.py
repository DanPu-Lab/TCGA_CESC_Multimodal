"""Kernel SHAP for CIS-GnE using frozen scaling and training-only background."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import pandas as pd
import torch
import shap
import config
from pipeline.artifacts import load_checkpoint, sha256_file
from pipeline.data import read_expression, read_ids, read_labels, check_partitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(Path(config.RESULT_DIR) / "gene_model/best_model.pth"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nsamples", type=int, default=30)
    parser.add_argument("--background", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--l1-reg", default="num_features(10)", help="Explicit version-sensitive feature-selection setting")
    args = parser.parse_args()
    model, scaler, meta = load_checkpoint(args.checkpoint, "gene", config.DEVICE)
    train_ids = meta["training_ids"]
    test_ids = read_ids(Path(config.DATA_DIR) / "splits/test_ids.csv")
    check_partitions(train_ids, test_ids)
    if meta.get("stage") != "final" or set(test_ids) != set(meta["evaluation_ids"]):
        raise ValueError("SHAP requires the final model and its original held-out IDs")
    if args.background > len(train_ids) or args.background < 1 or args.nsamples < 1:
        raise ValueError("Invalid background or SHAP evaluation budget")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    frame = scaler.transform(read_expression(config.DATA_DIR))
    selected = np.random.RandomState(args.seed).choice(train_ids, args.background, replace=False)
    background = frame.loc[selected].to_numpy()
    test = frame.loc[test_ids].to_numpy()

    def predict(values):
        with torch.no_grad():
            x = torch.as_tensor(values, dtype=torch.float32, device=config.DEVICE)
            return torch.softmax(model(x), dim=1).cpu().numpy()

    explainer = shap.KernelExplainer(predict, background)
    values = explainer.shap_values(test, nsamples=args.nsamples, l1_reg=args.l1_reg)
    c2 = values[1] if isinstance(values, list) else np.asarray(values)[..., 1]
    if c2.shape != test.shape or not np.isfinite(c2).all():
        raise ValueError("Unexpected C2 SHAP output shape or nonfinite attributions")
    ranking = pd.DataFrame({"gene": scaler.genes, "mean_abs_shap": np.abs(c2).mean(axis=0)})
    ranking = ranking.sort_values("mean_abs_shap", ascending=False, kind="stable")
    args.output.mkdir(parents=True, exist_ok=False)
    ranking.to_csv(args.output / "feature_importance_ranked.csv", index=False)
    ranking.loc[ranking.mean_abs_shap > 0].to_csv(args.output / "feature_importance_nonzero.csv", index=False)
    pd.DataFrame(c2, index=test_ids, columns=scaler.genes).rename_axis("sample_id").to_csv(
        args.output / "shap_values_matrix.csv")
    labels = read_labels(config.DATA_DIR).loc[test_ids]
    pd.DataFrame({"gene": scaler.genes,
                  **{f"C{label+1}_mean_shap": c2[labels.to_numpy() == label].mean(0) for label in [0, 1]}}
                 ).to_csv(args.output / "subtype_specific_shap.csv", index=False)
    (args.output / "shap_metadata.json").write_text(json.dumps(
        {"background_ids": list(selected), "test_ids": test_ids, "seed": args.seed,
         "nsamples": args.nsamples, "l1_reg": args.l1_reg, "shap_version": shap.__version__,
         "checkpoint_sha256": sha256_file(args.checkpoint),
         "scope": "unimodal CIS-GnE attribution, not causal effects or fusion-model explanation"}, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
