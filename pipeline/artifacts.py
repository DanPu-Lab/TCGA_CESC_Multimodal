"""Checkpoints bind a model to its exact training scaler and patient split."""
import hashlib
import os
import tempfile
from pathlib import Path
import torch
from pipeline.data import FrozenScaler, order_digest


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_checkpoint(path, model, mode, scaler, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.module if hasattr(model, "module") else model
    meta = dict(metadata)
    if scaler is not None:
        scaler_path = path.with_suffix(".scaler.npz")
        scaler.save(scaler_path)
        meta.update(scaler_file=scaler_path.name, scaler_sha256=sha256_file(scaler_path),
                    gene_order_sha256=order_digest(scaler.genes))
    fd, temporary = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save({"format_version": 2, "model_type": mode,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "metadata": meta}, temporary)
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def load_checkpoint(path, expected_mode=None, device="cpu"):
    from pipeline.engine import build_model
    path = Path(path)
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or bundle.get("format_version") != 2:
        raise ValueError("Legacy weights lack split/scaler provenance. Use the explicit "
                         "import_legacy_checkpoint.py tool with original training artifacts; "
                         "do not infer a scaler from the complete TCGA matrix.")
    mode, meta = bundle["model_type"], bundle["metadata"]
    if expected_mode is not None and mode != expected_mode:
        raise ValueError("Checkpoint/model type mismatch")
    if meta.get("scaler_fit_partition") != "training_only":
        raise ValueError("Checkpoint has no training-only preprocessing provenance")
    train_ids, test_ids = meta["training_ids"], meta["evaluation_ids"]
    if set(train_ids) & set(test_ids):
        raise ValueError("Checkpoint training and evaluation patient IDs overlap")
    scaler = None
    if mode != "image":
        filename = meta["scaler_file"]
        if Path(filename).name != filename:
            raise ValueError("Scaler must be stored alongside the checkpoint")
        scaler_path = path.parent / filename
        if sha256_file(scaler_path) != meta["scaler_sha256"]:
            raise ValueError("Scaler file checksum mismatch")
        scaler = FrozenScaler.load(scaler_path)
        if set(scaler.fit_ids) != set(train_ids) or order_digest(scaler.genes) != meta["gene_order_sha256"]:
            raise ValueError("Scaler does not match the checkpoint training IDs/gene order")
    model = build_model(mode, meta["input_dim"], pretrained=False)
    state = {k.removeprefix("module."): v for k, v in bundle["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, scaler, meta
