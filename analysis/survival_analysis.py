"""Overall-survival associations of reference C1/C2 annotations, not model predictions."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import config
from pipeline.data import read_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("survival", help="CSV: sample_id,OS.time (days),OS (0/1); one row per patient")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_csv(args.survival)
    if not {"sample_id", "OS.time", "OS"} <= set(table) or table.sample_id.duplicated().any():
        raise ValueError("Require unique patient sample_id,OS.time,OS")
    table = table.merge(read_labels(args.data_dir).rename("label"), left_on="sample_id",
                        right_index=True, validate="one_to_one")
    for column in ["OS.time", "OS"]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    valid = np.isfinite(table["OS.time"]) & (table["OS.time"] > 0) & table.OS.isin([0, 1])
    included, excluded = table.loc[valid].copy(), table.loc[~valid].copy()
    groups = [included.loc[included.label == label] for label in [0, 1]]
    if any(g.empty for g in groups):
        raise ValueError("Both reference subtypes are required")
    logrank = logrank_test(groups[0]["OS.time"], groups[1]["OS.time"], groups[0].OS, groups[1].OS)
    cox = CoxPHFitter().fit(included[["OS.time", "OS", "label"]], duration_col="OS.time", event_col="OS")
    lower, upper = np.exp(cox.confidence_intervals_.loc["label"])
    result = {"n": len(included), "C1": len(groups[0]), "C2": len(groups[1]),
              "logrank_p_two_sided": float(logrank.p_value),
              "HR_C2_vs_C1": float(cox.hazard_ratios_["label"]),
              "HR_ci_low": float(lower), "HR_ci_high": float(upper),
              "label_source": "pre_existing_reference_subtypes", "time_unit": "days"}
    args.output.mkdir(parents=True, exist_ok=False)
    for label, group in enumerate(groups):
        km = KaplanMeierFitter().fit(group["OS.time"]/365.25, group.OS, label=f"C{label+1}")
        for year in [1, 3, 5]:
            result[f"C{label+1}_{year}yr_probability"] = float(km.predict(year))
        km.survival_function_.to_csv(args.output / f"C{label+1}_kaplan_meier.csv")
        km.confidence_interval_.to_csv(args.output / f"C{label+1}_survival_ci.csv")
        km.event_table.to_csv(args.output / f"C{label+1}_event_table.csv")
    included.to_csv(args.output / "included_survival_patients.csv", index=False)
    excluded.to_csv(args.output / "excluded_survival_patients.csv", index=False)
    (args.output / "survival_statistics.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(result)


if __name__ == "__main__":
    main()
