"""Generate Figure 2f source numbers directly from paired patient predictions."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import pandas as pd
from pipeline.metrics import align_prediction_tables, paired_auc_bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_directory", type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    rows = []
    for fusion in ("late_fusion", "early_fusion"):
        for baseline in ("image", "gene"):
            a, b = [pd.read_csv(args.prediction_directory / f"{m}_predictions.csv")
                    for m in (fusion, baseline)]
            y, first, second = align_prediction_tables(a, b)
            rows.append({"model": fusion, "baseline": baseline,
                         **paired_auc_bootstrap(y, first, second, args.bootstrap, args.seed)})
    table = pd.DataFrame(rows)
    table["multiplicity"] = "four predeclared comparisons; unadjusted P values"
    table.to_csv(args.prediction_directory / "paired_auroc_comparisons.csv", index=False)
    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, row in table.iterrows():
            ax.bar(i, row.difference, color="#748A82")
            ax.vlines(i, row.ci_low, row.ci_high, color="black")
            ax.hlines([row.ci_low, row.ci_high], i-.06, i+.06, color="black")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(range(len(table)), [f"{r.model} vs {r.baseline}" for r in table.itertuples()],
                      rotation=30, ha="right")
        ax.set_ylabel("AUROC difference (95% CI)")
        fig.tight_layout()
        fig.savefig(args.prediction_directory / "paired_auroc_comparisons.pdf")
        plt.close(fig)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
