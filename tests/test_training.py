import json
import numpy as np
import pandas as pd
import pytest
import torch
import anndata as ad
from pipeline import engine
from pipeline.data import FrozenScaler
from pipeline.dataset import PatientDataset


@pytest.fixture
def data(tmp_path, monkeypatch):
    import config
    root = tmp_path / "data"
    (root / "labels").mkdir(parents=True)
    ids = [f"p{i}" for i in range(10)]
    frame = pd.DataFrame(np.arange(40).reshape(10, 4)/10, index=ids, columns=list("abcd"))
    frame.loc[ids[-2:]] = 1e8  # Test outliers must not influence training scaling.
    ad.AnnData(frame).write_h5ad(root / "merged_genes.h5ad")
    pd.DataFrame({"sample_id": ids, "label": [0, 1]*5}).to_csv(root / "labels/all_labels.csv", index=False)
    monkeypatch.setattr(config, "DATA_DIR", str(root))
    monkeypatch.setattr(config, "BATCH_SIZE", 4)
    monkeypatch.setattr(config, "NUM_WORKERS", 0)
    monkeypatch.setattr(config, "DEVICE", "cpu")
    monkeypatch.setattr(config, "PATIENCE", 2)
    monkeypatch.setattr(engine, "build_model", lambda mode, input_dim, pretrained: torch.nn.Linear(input_dim, 2))
    return root, ids, frame


def test_final_training_does_not_evaluate_or_select_on_test(data, tmp_path, monkeypatch):
    root, ids, frame = data
    def forbidden(*args, **kwargs):
        raise AssertionError("Final training accessed an evaluation loader")
    monkeypatch.setattr(engine, "predict", forbidden)
    out = tmp_path / "final"
    engine.fit("gene", ids[:8], ids[8:], out, epochs=2, pretrained=False, final=True)
    assert (out / "best_model.pth").exists()
    scaler = FrozenScaler.load(out / "best_model.scaler.npz")
    np.testing.assert_allclose(scaler.mean, frame.iloc[:8].mean(0))
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["selected_epoch"] == 2
    assert meta["epoch_selection"] == "predeclared_fixed_budget"


def test_cv_reports_saved_best_epoch_not_last_epoch(data, tmp_path, monkeypatch):
    root, ids, frame = data
    calls = []
    def varying_prediction(model, batches, mode, device):
        calls.append(True)
        return pd.DataFrame({"sample_id": ids[8:], "true": [0, 1],
                             "pred_prob_C2": [.1, .9] if len(calls) == 1 else [.9, .1]})
    monkeypatch.setattr(engine, "predict", varying_prediction)
    result = engine.fit("gene", ids[:8], ids[8:], tmp_path / "cv", epochs=3, pretrained=False)
    assert len(calls) == 3
    assert result["auroc"] == 1 and result["selected_epoch"] == 1
    saved = pd.read_csv(tmp_path / "cv/val_results.csv").iloc[0]
    assert saved.auroc == 1 and saved.selected_epoch == 1


def test_external_dataset_cannot_fit_scaler(data):
    root, ids, frame = data
    with pytest.raises(ValueError, match="saved TCGA training scaler"):
        PatientDataset(ids[8:], "gene", False, data_dir=root, cohort="cgci")


def test_evaluation_image_bag_is_unchanged_across_model_modes(data):
    root, ids, frame = data
    (root / "svs").mkdir()
    patches = np.arange(100*4*4*3, dtype=np.float32).reshape(100, 4, 4, 3) / 100
    np.savez(root / f"svs/{ids[8]}_patches.npz", patches=patches)
    scaler = FrozenScaler.fit(frame, ids[:8])
    a = PatientDataset([ids[8]], "image", False, data_dir=root, cohort="cgci")[0][0]
    b = PatientDataset([ids[8]], "early_fusion", False, scaler, data_dir=root, cohort="cgci")[0][0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
