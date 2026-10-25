"""Build unstandardised log1p(TPM) inputs; preserve an explicit gene order."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import config
from pipeline.data import order_digest


def read_tpm(path):
    frame = pd.read_csv(path, sep="\t", comment="#")
    if not {"gene_name", "gene_type", "tpm_unstranded"} <= set(frame):
        raise ValueError(f"Missing GDC RNA columns: {path}")
    frame = frame.loc[frame.gene_type == "protein_coding"].copy()
    frame["tpm_unstranded"] = pd.to_numeric(frame.tpm_unstranded, errors="raise")
    if (frame.gene_name.isna().any() or not np.isfinite(frame.tpm_unstranded).all()
            or (frame.tpm_unstranded < 0).any()):
        raise ValueError(f"Invalid protein-coding expression: {path}")
    return frame.groupby("gene_name", sort=False).tpm_unstranded.sum()


def prepare(manifest, output, ordered_genes=None, expected_genes=None):
    manifest = pd.read_csv(manifest)
    if not {"sample_id", "rna_path"} <= set(manifest) or manifest.sample_id.duplicated().any():
        raise ValueError("Manifest requires one sample_id/rna_path row per patient")
    if manifest.sample_id.isna().any() or len(manifest) == 0:
        raise ValueError("Missing patient IDs or empty manifest")
    series = {str(row.sample_id): read_tpm(row.rna_path) for row in manifest.itertuples()}
    if ordered_genes is None:
        first = next(iter(series.values()))
        common = set.intersection(*(set(x.index) for x in series.values()))
        genes = [x for x in first.index if x in common]
    else:
        genes = list(ordered_genes)
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("Gene order must be nonempty and unique")
    if expected_genes is not None and len(genes) != expected_genes:
        raise ValueError(f"Expected {expected_genes} genes, obtained {len(genes)}; review annotation mapping")
    for sid, values in series.items():
        if not set(genes) <= set(values.index):
            raise ValueError(f"{sid}: missing required genes; no imputation is permitted")
    frame = np.log1p(pd.DataFrame({sid: s.loc[genes] for sid, s in series.items()}).T)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix = ad.AnnData(frame)
    matrix.uns["preprocessing"] = "protein_coding; duplicate TPM sum; log1p; not standardised"
    matrix.uns["gene_order_sha256"] = order_digest(genes)
    matrix.write_h5ad(output)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV: sample_id,rna_path")
    parser.add_argument("--output", default=str(Path(config.DATA_DIR) / "merged_genes.h5ad"))
    parser.add_argument("--scaler", help="Original final-TCGA scaler supplies the exact CGCI gene order")
    parser.add_argument("--expected-genes", type=int, default=19914)
    args = parser.parse_args()
    genes = None
    if args.scaler:
        from pipeline.data import FrozenScaler
        genes = FrozenScaler.load(args.scaler).genes
    frame = prepare(args.manifest, args.output, genes, args.expected_genes)
    print(f"Saved {len(frame)} patients and {len(frame.columns)} unstandardised log1p(TPM) genes")


if __name__ == "__main__":
    main()
