import numpy as np
import pytest
import torch
from pipeline.engine import build_model, forward
from pipeline.data import FrozenScaler, manifest_metadata
from pipeline.artifacts import save_checkpoint, load_checkpoint


@pytest.fixture(autouse=True)
def fast_cpu():
    torch.set_num_threads(2)


@pytest.mark.parametrize("mode", ["image", "gene", "early_fusion", "late_fusion"])
def test_forward_and_valid_probabilities_without_downloading_weights(mode):
    model = build_model(mode, 8, pretrained=False).eval()
    image, gene, label = torch.randn(2, 2, 3, 32, 32), torch.randn(2, 8), torch.tensor([0, 1])
    batch = ((image, label) if mode == "image" else (gene, label) if mode == "gene"
             else (image, gene, label))
    with torch.no_grad():
        loss_input, probs, y = forward(model, batch, mode, "cpu")
    assert probs.shape == (2, 2)
    torch.testing.assert_close(probs.sum(1), torch.ones(2))
    if mode == "late_fusion":
        direct = model(image, gene)[0]
        torch.testing.assert_close(probs, direct)
        assert not torch.allclose(probs, torch.softmax(direct, dim=1))
        torch.testing.assert_close(loss_input, direct.clamp_min(1e-10).log())


def test_checkpoint_binds_original_scaler_and_rejects_bare_or_tampered_artifacts(tmp_path):
    import pandas as pd
    frame = pd.DataFrame([[1., 2., 5.], [3., 4., 5.]], index=["a", "b"], columns=list("xyz"))
    scaler = FrozenScaler.fit(frame, ["a", "b"])
    model = build_model("gene", 3, pretrained=False).eval()
    meta = manifest_metadata(["a", "b"], ["heldout"], scaler.genes)
    meta.update(input_dim=3, stage="final")
    path = tmp_path / "model.pth"
    save_checkpoint(path, model, "gene", scaler, meta)
    restored, saved, metadata = load_checkpoint(path, "gene")
    torch.testing.assert_close(model(torch.zeros(2, 3)), restored(torch.zeros(2, 3)))
    assert saved.fit_ids == ["a", "b"]
    with path.with_suffix(".scaler.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(path, "gene")
    torch.save(model.state_dict(), tmp_path / "legacy.pth")
    with pytest.raises(ValueError, match="Legacy"):
        load_checkpoint(tmp_path / "legacy.pth")


def test_grad_cam_produces_one_map_per_patch():
    from analysis.gradcam_analysis import make_cams
    model = build_model("image", 0, pretrained=False).eval()
    maps = make_cams(model, torch.randn(1, 2, 3, 64, 64), 1)
    assert maps.shape == (2, 64, 64)
    assert np.isfinite(maps).all()
    assert maps.min() >= 0 and maps.max() <= 1
