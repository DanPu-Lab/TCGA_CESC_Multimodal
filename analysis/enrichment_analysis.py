"""Exploratory ORA; BH within each list/database, no pooled global FDR claim."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import pandas as pd
import gseapy as gp
import config

DATABASES = {"KEGG": "KEGG_2021_Human.gmt", "Reactome": "Reactome_2022.gmt",
             "GO_BP": "GO_Biological_Process_2023.gmt"}


def gene_lists(frame):
    if not {"gene", "mean_abs_shap"} <= set(frame) or frame.gene.duplicated().any():
        raise ValueError("Require unique gene/mean_abs_shap rows")
    if frame.empty or frame.mean_abs_shap.isna().any() or (frame.mean_abs_shap < 0).any():
        raise ValueError("Invalid SHAP input table")
    ranked = frame.sort_values("mean_abs_shap", ascending=False, kind="stable")
    return {"top_200": ranked.head(200).gene.tolist(), "top_150": ranked.head(150).gene.tolist(),
            "shap_gt_mean": ranked.loc[ranked.mean_abs_shap > ranked.mean_abs_shap.mean(), "gene"].tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="The exact input importance table used to define all three lists")
    parser.add_argument("--databases", type=Path, default=Path(config.BASE_DIR) / "databases")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lists = gene_lists(pd.read_csv(args.input))
    args.output.mkdir(parents=True, exist_ok=False)
    all_rows = []
    for name, genes in lists.items():
        if len(genes) < 20:
            continue
        for db, filename in DATABASES.items():
            gmt = args.databases / filename
            if not gmt.is_file():
                raise FileNotFoundError(gmt)
            # Local GMT, default annotated-gene background as specified in Methods.
            result = gp.enrichr(gene_list=genes, gene_sets=str(gmt),
                                organism="Human", outdir=None, cutoff=1.)
            full = result.results.copy()
            full["Gene_Set"], full["Database"], full["Input_Genes"] = name, db, len(genes)
            full.to_csv(args.output / f"{name}_{db}_all_terms.csv", index=False)
            all_rows.append(full)
    if not all_rows:
        raise ValueError("No gene list passed the minimum size")
    complete = pd.concat(all_rows, ignore_index=True)
    complete.to_csv(args.output / "all_analyses.csv", index=False)
    pooled = complete.loc[complete["Adjusted P-value"] < .10].copy()
    pooled = pooled.sort_values("Adjusted P-value").drop_duplicates("Term")
    pooled["Global_FDR_control"] = False
    pooled.to_csv(args.output / "pooled_exploratory_terms.csv", index=False)
    (args.output / "gene_lists.json").write_text(json.dumps(lists, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
