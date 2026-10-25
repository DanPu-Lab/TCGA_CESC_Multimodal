"""Package original weights with original split/scaler artifacts, without fitting."""
from pathlib import Path
import argparse
import torch
from pipeline.data import FrozenScaler, read_ids, check_partitions, manifest_metadata
from pipeline.engine import build_model
from pipeline.artifacts import save_checkpoint, sha256_file


def remap_state(state, mode, layout="public"):
    result = {}
    prefixes = ({"gene_net.": "net.", "classifier.": "net.16."} if mode == "gene" else
                {"image_net.": "image_encoder.", "image_attention.": "image_attn.",
                 "gene_net.": "gene_encoder.", "fusion_classifier.": "classifier."}
                if mode == "early_fusion" else
                {"image_net.": "image_encoder.", "image_attention.": "image_attn.",
                 "gene_net.": "gene_encoder."} if mode == "late_fusion" else {})
    for old, value in state.items():
        key = old.removeprefix("module.")
        if layout == "server-v1":
            for source, target in prefixes.items():
                if key.startswith(source):
                    key = target + key[len(source):]
                    break
        if key in result or not isinstance(value, torch.Tensor):
            raise ValueError("Invalid or colliding parameter names")
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model", choices=["image", "gene", "early_fusion", "late_fusion"], required=True)
    parser.add_argument("--layout", choices=["public", "server-v1"], default="public")
    parser.add_argument("--train-ids", type=Path, required=True, help="Original final-training IDs, not a regenerated split")
    parser.add_argument("--test-ids", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, help="Original training-only FrozenScaler NPZ; never reconstructed from all 257 cases")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train, test = read_ids(args.train_ids), read_ids(args.test_ids)
    check_partitions(train, test)
    scaler = FrozenScaler.load(args.scaler) if args.scaler else None
    if args.model != "image" and scaler is None:
        parser.error("Original training scaler is required for this model")
    if scaler is not None and set(scaler.fit_ids) != set(train):
        parser.error("Scaler fit IDs do not match original final-training IDs")
    if args.output.exists() or args.output.with_suffix(".scaler.npz").exists():
        raise FileExistsError("Destination checkpoint/scaler exists")
    raw = torch.load(args.weights, map_location="cpu", weights_only=True)
    state = raw.get("model_state_dict", raw.get("state_dict", raw))
    dim = len(scaler.genes) if scaler else 0
    model = build_model(args.model, dim, pretrained=False)
    model.load_state_dict(remap_state(state, args.model, args.layout), strict=True)
    meta = manifest_metadata(train, test, scaler.genes if scaler else [])
    meta.update(stage="final", input_dim=dim, epoch_selection="original_artifact_import",
                original_weights_sha256=sha256_file(args.weights), original_layout=args.layout,
                original_train_ids_sha256=sha256_file(args.train_ids),
                original_test_ids_sha256=sha256_file(args.test_ids))
    save_checkpoint(args.output, model, args.model, scaler, meta)
    print("Strictly loaded original tensors; no training, recalibration or scaler fitting performed.")


if __name__ == "__main__":
    main()
