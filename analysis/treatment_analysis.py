"""Clinical context from reference subtypes, never model-predicted subtypes."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, MonteCarloMethod
from pipeline.data import read_labels
import config

RESPONSE_MAP = {"cr": "CR", "complete response": "CR", "complete remission/response": "CR",
                "pr": "PR", "partial response": "PR", "partial remission/response": "PR",
                "sd": "SD", "stable disease": "SD", "pd": "PD",
                "progressive disease": "PD", "progression": "PD"}


def normalise_response(value):
    # Adjuvant describes treatment intent, NOT a response. Recurrence alone is
    # also not a response measurement. Unknown entries remain excluded/audited.
    return RESPONSE_MAP.get(str(value).strip().lower())


def response_table(records, reference_labels):
    if not {"sample_id", "response"} <= set(records) or records.sample_id.duplicated().any():
        raise ValueError("Require one explicitly curated response per patient; do not count timeline rows as patients")
    joined = records.merge(reference_labels.rename("label"), left_on="sample_id", right_index=True,
                            validate="one_to_one", how="inner")
    joined["standardised_response"] = joined.response.map(normalise_response)
    included = joined.dropna(subset=["standardised_response"])
    table = pd.crosstab(included.label, included.standardised_response).reindex(index=[0, 1], fill_value=0)
    table = table.reindex(columns=[c for c in ["CR", "PR", "SD", "PD"] if c in table], fill_value=0)
    if (table.sum(axis=1) == 0).any() or table.shape[1] < 2:
        raise ValueError("Insufficient response categories or reference subtypes")
    return table, joined


def odds_ratio(table, category):
    selected = table[list(category)].sum(axis=1).to_numpy(dtype=float)
    total = table.sum(axis=1).to_numpy(dtype=float)
    # C2 versus C1 odds of the specified category against all other responses.
    a, b, c, d = selected[1], total[1]-selected[1], selected[0], total[0]-selected[0]
    correction = any(v == 0 for v in [a, b, c, d])
    values = np.asarray([a, b, c, d]) + (.5 if correction else 0)
    a, b, c, d = values
    ratio = a*d/(b*c)
    se = np.sqrt(np.sum(1/values))
    return {"OR_C2_vs_C1": float(ratio), "ci_low": float(np.exp(np.log(ratio)-1.96*se)),
            "ci_high": float(np.exp(np.log(ratio)+1.96*se)), "zero_cell_correction": correction}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", help="CSV: sample_id,response; endpoint selection must be curated before analysis")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table, audit = response_table(pd.read_csv(args.responses), read_labels(args.data_dir))
    chi, p, dof, expected = chi2_contingency(table.to_numpy(), correction=False)
    stats = {"chi_square": float(chi), "p_pearson": float(p), "dof": int(dof),
             "expected_cells_below_5": int((expected < 5).sum()), "n_patients": int(table.to_numpy().sum()),
             "label_source": "pre_existing_reference_subtypes; not model predictions"}
    if (expected < 5).any():
        exact = fisher_exact(table.to_numpy(), method=MonteCarloMethod(
            n_resamples=100000, rng=np.random.default_rng(123)))
        stats["p_fisher_freeman_halton_monte_carlo_sensitivity"] = float(exact.pvalue)
    args.output.mkdir(parents=True, exist_ok=False)
    audit.to_csv(args.output / "response_mapping_audit.csv", index=False)
    table.to_csv(args.output / "treatment_response_counts.csv")
    comparisons = [{"response": col, **odds_ratio(table, [col])} for col in table]
    if {"CR", "PR"} <= set(table):
        comparisons.append({"response": "ORR", **odds_ratio(table, ["CR", "PR"])})
        successes = table[["CR", "PR"]].sum(axis=1).to_numpy()
        orr_table = np.column_stack([successes, table.sum(axis=1).to_numpy()-successes])
        if (orr_table.sum(axis=0) > 0).all():
            stats["ORR_p_pearson"] = float(chi2_contingency(orr_table, correction=False).pvalue)
            stats["ORR_p_fisher_two_sided"] = float(fisher_exact(orr_table).pvalue)
        else:
            stats["ORR_test"] = "not estimable: all patients share response status"
    pd.DataFrame(comparisons).to_csv(args.output / "odds_ratios.csv", index=False)
    (args.output / "statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf8")
    print(stats)


if __name__ == "__main__":
    main()
