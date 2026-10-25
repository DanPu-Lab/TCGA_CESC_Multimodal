"""Patch-specific Grad-CAM for correctly classified held-out CIS-AtN cases."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import config
from pipeline.artifacts import load_checkpoint
from pipeline.data import read_ids, read_labels
from pipeline.dataset import PatientDataset, loader
from pipeline.engine import predict


def make_cams(model, images, target_class):
    captured = []
    def capture(module, inputs, output):
        output.retain_grad()
        captured.append(output)
    handle = model.resnet[-2].register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        output = model(images.requires_grad_(True))
        output[0, target_class].backward()
        features = captured[0]
        # Keep the convolutional CHANNEL dimension until weighted summation.
        weights = features.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * features).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        values = cam[:, 0].detach().cpu().numpy()
        low, high = np.percentile(values, [5, 95], axis=(1, 2), keepdims=True)
        return np.clip((values-low) / np.maximum(high-low, 1e-8), 0, 1)
    finally:
        handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(Path(config.RESULT_DIR) / "image_model/best_model.pth"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--per-class", type=int, default=4)
    args = parser.parse_args()
    model, _, meta = load_checkpoint(args.checkpoint, "image", config.DEVICE)
    ids = read_ids(Path(config.DATA_DIR) / "splits/test_ids.csv")
    if meta.get("stage") != "final" or set(ids) != set(meta["evaluation_ids"]):
        raise ValueError("Use the final model and original held-out patients")
    ds = PatientDataset(ids, "image", False)
    pred = predict(model, loader(ds), "image", config.DEVICE)
    correct = pred.loc[(pred.pred_prob_C2 > .5).astype(int) == pred.true]
    rng = np.random.default_rng(args.seed)
    selected = []
    for label in [0, 1]:
        eligible = correct.loc[correct.true == label, "sample_id"].to_numpy()
        if len(eligible) < args.per_class:
            raise ValueError("Insufficient correctly classified cases for the requested stratified display")
        selected.extend(rng.choice(eligible, args.per_class, replace=False).tolist())
    args.output.mkdir(parents=True, exist_ok=False)
    for sid in selected:
        images, label = ds[ids.index(sid)]
        cams = make_cams(model, images.unsqueeze(0).to(config.DEVICE), int(label))
        np.savez_compressed(args.output / f"{sid}_cams.npz", cams=cams)
        # Display the first frozen patch; save every patch CAM for inspection.
        raw = images[0].permute(1, 2, 0).numpy() * config.IMAGENET_STD + config.IMAGENET_MEAN
        raw = np.clip(raw, 0, 1)
        overlay = .6*raw + .4*plt.get_cmap("viridis")(cams[0])[..., :3]
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(raw); axes[1].imshow(overlay)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.output / f"{sid}_gradcam.png", dpi=300)
        plt.close(fig)
    (args.output / "selection.json").write_text(json.dumps(
        {"case_ids": selected, "seed": args.seed, "correct_predictions_only": True,
         "colormap": "viridis", "opacity": .4, "percentile_normalisation": [5, 95],
         "scope": "unimodal image attribution, not cell-type or causal validation"}, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
