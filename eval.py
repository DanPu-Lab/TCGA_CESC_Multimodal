"""Evaluate frozen models; no fitting, model selection or threshold optimisation."""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import config
from pipeline.artifacts import load_checkpoint, sha256_file
from pipeline.data import read_ids, read_labels, check_partitions
from pipeline.dataset import PatientDataset, loader, MODES
from pipeline.engine import predict, seed_all
from pipeline.metrics import classification_metrics, metric_intervals


def evaluate_cohort(data_dir, checkpoint_root, output, cohort="tcga", n_resamples=1000):
    seed_all(config.RANDOM_SEED)
    data_dir, checkpoint_root, output = map(Path, (data_dir, checkpoint_root, output))
    labels = read_labels(data_dir)
    ids = (read_ids(data_dir / "splits/test_ids.csv") if cohort == "tcga"
           else sorted(labels.index.tolist()))
    if output.exists():
        raise FileExistsError("Evaluation output exists; previous results will not be overwritten")
    loaded = {mode: load_checkpoint(checkpoint_root / f"{mode}_model/best_model.pth",
                                    mode, config.DEVICE) for mode in MODES}
    training_sets = [set(meta["training_ids"]) for _, _, meta in loaded.values()]
    if any(x != training_sets[0] for x in training_sets):
        raise ValueError("Models were not fitted on the same TCGA training patients")
    for _, _, meta in loaded.values():
        check_partitions(meta["training_ids"], ids)
        if meta.get("stage") != "final":
            raise ValueError("Cohort evaluation requires final models, not CV-fold checkpoints")
        if cohort == "tcga" and set(meta["evaluation_ids"]) != set(ids):
            raise ValueError("Held-out IDs differ from checkpoint provenance")
    bags = PatientDataset(ids, "image", False, data_dir=data_dir, cohort=cohort).freeze_bags()
    output.mkdir(parents=True)
    np.savez_compressed(output / "frozen_patch_indices.npz", sample_ids=np.asarray(ids, dtype=str),
                        patch_indices=np.stack([bags[sid] for sid in ids]))
    metrics, provenance = [], {}
    for mode in MODES:
        model, scaler, meta = loaded[mode]
        ds = PatientDataset(ids, mode, False, scaler=scaler, data_dir=data_dir, cohort=cohort,
                            frozen_bags=bags)
        pred = predict(model, loader(ds), mode, config.DEVICE)
        pred.to_csv(output / f"{mode}_predictions.csv", index=False)
        row = {"model": mode, **classification_metrics(pred.true, pred.pred_prob_C2)}
        row.update(metric_intervals(pred.true, pred.pred_prob_C2, n_resamples=n_resamples))
        metrics.append(row)
        provenance[mode] = {"checkpoint_sha256": sha256_file(
            checkpoint_root / f"{mode}_model/best_model.pth"),
            "scaler_sha256": meta.get("scaler_sha256"), "prediction_rule": "argmax_C1_C2"}
    pd.DataFrame(metrics).to_csv(output / "model_comparison.csv", index=False)
    (output / "evaluation_metadata.json").write_text(json.dumps(
        {"cohort": cohort, "patients": ids, "seed": config.RANDOM_SEED,
         "n_resamples": n_resamples, "frozen_models": provenance}, indent=2), encoding="utf8")
    return pd.DataFrame(metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--checkpoint-root", default=config.RESULT_DIR)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cohort", choices=["tcga", "cgci"], default="tcga")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    print(evaluate_cohort(args.data_dir, args.checkpoint_root, args.output,
                          args.cohort, args.bootstrap).to_string(index=False))


if __name__ == "__main__":
    main()
