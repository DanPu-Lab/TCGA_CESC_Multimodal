"""Frozen external evaluation with synthetic inputs and lightweight test models."""
import json
import numpy as np
import pandas as pd
import torch
import anndata as ad
from pipeline.data import FrozenScaler
import eval as evaluation


def test_external_evaluation_never_refits_and_uses_identical_saved_bags(tmp_path, monkeypatch):
    import config
    root, weights = tmp_path / "external", tmp_path / "weights"
    (root / "labels").mkdir(parents=True)
    (root / "svs").mkdir()
    ids = ["e1", "e2", "e3", "e4"]
    pd.DataFrame({"sample_id": ids, "label": [0, 0, 1, 1]}).to_csv(root / "labels/all_labels.csv", index=False)
    frame = pd.DataFrame(np.full((4, 2), 1e8), index=ids, columns=["a", "b"])
    ad.AnnData(frame).write_h5ad(root / "merged_genes.h5ad")
    for sid in ids:
        np.savez(root / f"svs/{sid}_patches.npz", patches=np.arange(100*2*2*3, dtype=np.float32).reshape(100, 2, 2, 3))
    scaler = FrozenScaler.fit(pd.DataFrame([[1., 2.], [3., 4.]], index=["t1", "t2"], columns=["a", "b"]), ["t1", "t2"])
    before = scaler.mean.copy()
    seen = {}
    class TinyModel(torch.nn.Module):
        def __init__(self, mode):
            super().__init__()
            self.mode = mode
        def forward(self, *inputs):
            if self.mode != "gene":
                seen.setdefault(self.mode, []).append(inputs[0].detach().clone())
            logits = torch.tensor([[0., 1.]]).repeat(len(inputs[0]), 1)
            return (torch.softmax(logits, 1),)*3 if self.mode == "late_fusion" else logits
    def frozen_load(path, mode, device):
        return TinyModel(mode), None if mode == "image" else scaler, {
            "training_ids": ["t1", "t2"], "evaluation_ids": ["held1", "held2"], "stage": "final"}
    for mode in evaluation.MODES:
        folder = weights / f"{mode}_model"
        folder.mkdir(parents=True)
        (folder / "best_model.pth").write_bytes(b"synthetic-test-placeholder-not-a-checkpoint")
    monkeypatch.setattr(evaluation, "load_checkpoint", frozen_load)
    monkeypatch.setattr(config, "DEVICE", "cpu")
    monkeypatch.setattr(config, "BATCH_SIZE", 2)
    monkeypatch.setattr(config, "NUM_WORKERS", 0)
    def forbidden(*a, **kw):
        raise AssertionError("External evaluation attempted scaler fitting")
    monkeypatch.setattr(FrozenScaler, "fit", forbidden)
    out = tmp_path / "evaluation"
    result = evaluation.evaluate_cohort(root, weights, out, "cgci", n_resamples=100)
    assert len(result) == 4 and (result.n == 4).all()
    np.testing.assert_array_equal(scaler.mean, before)
    for a, b, c in zip(seen["image"], seen["early_fusion"], seen["late_fusion"]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(a, c, rtol=0, atol=0)
    with np.load(out / "frozen_patch_indices.npz") as bags:
        assert bags["patch_indices"].shape == (4, 50)
        assert all(len(set(x)) == 50 for x in bags["patch_indices"])
    assert json.loads((out / "evaluation_metadata.json").read_text())["patients"] == ids
