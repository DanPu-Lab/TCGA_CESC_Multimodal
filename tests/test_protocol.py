"""Synthetic tests of scientific protocol invariants; no patient data required."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pipeline.data import FrozenScaler, check_partitions, fixed_indices
from pipeline.metrics import (classification_metrics, paired_auc_bootstrap,
                              align_prediction_tables)
from external.prepare_cohort import select_cohort
from analysis.treatment_analysis import normalise_response, response_table, odds_ratio
from preprocess.process_tsv import read_tpm


def test_extreme_validation_values_cannot_change_training_scaler(tmp_path):
    frame = pd.DataFrame([[1., 3., 5.], [3., 7., 5.], [1e9, -1e8, 1e7]],
                         index=["train1", "train2", "heldout"], columns=["a", "b", "constant"])
    scaler = FrozenScaler.fit(frame, ["train1", "train2"])
    np.testing.assert_allclose(scaler.mean, [2, 5, 5])
    np.testing.assert_allclose(scaler.scale, [1, 2, 1])
    changed = frame.copy()
    changed.loc["heldout"] = [-1e15, 1e15, 1e15]
    other = FrozenScaler.fit(changed, ["train1", "train2"])
    np.testing.assert_array_equal(scaler.mean, other.mean)
    np.testing.assert_array_equal(scaler.scale, other.scale)
    scaler.save(tmp_path / "scaler.npz")
    loaded = FrozenScaler.load(tmp_path / "scaler.npz")
    np.testing.assert_array_equal(loaded.transform(frame).values, scaler.transform(frame).values)
    with pytest.raises(ValueError, match="Missing"):
        loaded.transform(frame[["a", "b"]])
    pd.testing.assert_frame_equal(loaded.transform(frame), loaded.transform(frame[["b", "a", "constant"]]))


def test_patient_overlap_and_duplicates_fail():
    with pytest.raises(ValueError, match="overlap"):
        check_partitions(["a", "b"], ["b", "c"])
    with pytest.raises(ValueError, match="Duplicated"):
        check_partitions(["a", "a"], ["b"])


def test_frozen_bags_are_shared_unique_and_patient_specific():
    first = fixed_indices("patient-a", 100, cohort="cgci")
    np.testing.assert_array_equal(first, fixed_indices("patient-a", 100, cohort="cgci"))
    assert len(set(first)) == 50
    assert not np.array_equal(first, fixed_indices("patient-b", 100, cohort="cgci"))
    assert len(set(fixed_indices("patient-a", 50))) == 50
    with pytest.raises(ValueError):
        fixed_indices("insufficient", 49)


def test_external_metrics_match_confusion_counts():
    # Synthetic predictions with the published EFF confusion counts. These test
    # metric arithmetic, not the reported AUROC or any patient-level probability.
    labels = np.array([0]*35 + [1]*84)
    scores = np.array([.2]*27 + [.8]*8 + [.2]*6 + [.8]*78)
    result = classification_metrics(labels, scores)
    assert result["balanced_accuracy"] == pytest.approx(.85)
    assert result["f1_C2"] == pytest.approx(156/170)
    assert result["accuracy"] == pytest.approx(105/119)
    assert result["C1_correct"] == 27 and result["C2_correct"] == 78


def test_argmax_tie_is_C1():
    row = classification_metrics([0, 1], [.5, .9])
    assert row["accuracy"] == 1


def test_paired_bootstrap_identical_models_and_sign_symmetry():
    y = np.array([0]*10 + [1]*10)
    a = np.linspace(.05, .95, 20)
    b = np.roll(a, 4)
    same = paired_auc_bootstrap(y, a, a, 100, 5)
    assert same["difference"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    assert same["p_two_sided"] == 1
    ab, ba = paired_auc_bootstrap(y, a, b, 100, 5), paired_auc_bootstrap(y, b, a, 100, 5)
    assert ab["difference"] == pytest.approx(-ba["difference"])
    assert ab["ci_low"] == pytest.approx(-ba["ci_high"])
    assert ab["p_two_sided"] == ba["p_two_sided"]
    assert 0 < ab["p_two_sided"] <= 1


def test_paired_predictions_align_by_id_and_reject_different_patients():
    a = pd.DataFrame({"sample_id": ["a", "b"], "true": [0, 1], "pred_prob_C2": [.1, .9]})
    y, x, z = align_prediction_tables(a, a.iloc[::-1])
    np.testing.assert_array_equal(x, z)
    with pytest.raises(ValueError, match="same patients"):
        align_prediction_tables(a, a.assign(sample_id=["c", "b"]))


def test_adjuvant_is_not_stable_disease_and_patient_rows_are_unique():
    assert normalise_response("adjuvant") is None
    assert normalise_response("recurrence") is None
    assert normalise_response(" Stable Disease ") == "SD"
    labels = pd.Series([0, 0, 1, 1], index=["a", "b", "c", "d"], name="label")
    rows = pd.DataFrame({"sample_id": list("abcd"), "response": ["CR", "adjuvant", "CR", "SD"]})
    table, audit = response_table(rows, labels)
    assert table.values.sum() == 3
    assert audit.standardised_response.isna().sum() == 1
    assert odds_ratio(table, ["SD"])["zero_cell_correction"]
    with pytest.raises(ValueError, match="per patient"):
        response_table(pd.concat([rows, rows.iloc[:1]]), labels)


def test_external_eligibility_tracks_overlapping_reasons_without_score_cutoff():
    manifest = pd.DataFrame({"sample_id": ["a", "b", "c", "d"], "rna_path": ["r"]*4, "wsi_path": ["s"]*4})
    calls = pd.DataFrame({"sample_id": list("abcd"), "BestCall": [1, 2, 3, 1],
                          "1": [.4, .1, .5, .1], "2": [.2, .4, .1, .6],
                          "3": [.1, .1, .2, .1], "4": [.1]*4, "5": [.1]*4, "6": [.1, .2, 0., 0.]})
    kept, audit = select_cohort(manifest, calls)
    assert kept.sample_id.tolist() == ["a", "b"]
    assert audit.non_C1_C2.sum() == 1 and audit.BestCall_argmax_discordant.sum() == 2
    sensitive, _ = select_cohort(manifest, calls, require_agreement=False)
    assert sensitive.sample_id.tolist() == ["a", "b", "d"]


def test_duplicate_tpm_is_aggregated_before_log1p(tmp_path):
    path = tmp_path / "sample.tsv"
    pd.DataFrame({"gene_name": ["a", "a", "b", "ignored"],
                  "gene_type": ["protein_coding"]*3 + ["lncRNA"],
                  "tpm_unstranded": [1., 2., 4., 100.]}).to_csv(path, sep="\t", index=False)
    values = read_tpm(path)
    assert values.to_dict() == {"a": 3., "b": 4.}
