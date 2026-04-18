import os
import pandas as pd
import numpy as np
import scanpy as sc
from tqdm import tqdm
import config

def process_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t', comment='#')
    # Standardize column names if necessary
    if 'gene_type' not in df.columns:
        df.columns = ['gene_id','gene_name','gene_type','unstranded','stranded_first','stranded_second',
                      'tpm_unstranded','fpkm_unstranded','fpkm_uq_unstranded']
    protein_coding = df[df['gene_type'] == 'protein_coding'].copy()
    if protein_coding.empty:
        return None
    protein_coding = protein_coding[~protein_coding['gene_name'].duplicated()]
    expr = protein_coding.set_index('gene_name')[['tpm_unstranded']]
    sample_id = os.path.basename(tsv_path).split('.')[0]
    expr.columns = [sample_id]
    expr = np.log1p(expr).T
    return expr

def main():
    tsv_dir = os.path.join(config.DATA_DIR, "tsv")
    tsv_files = [f for f in os.listdir(tsv_dir) if f.endswith('.tsv')]
    # First pass: collect all protein-coding genes and their occurrence
    gene_counter = {}
    for f in tqdm(tsv_files, desc="Scanning genes"):
        df = pd.read_csv(os.path.join(tsv_dir, f), sep='\t', comment='#')
        if 'gene_type' not in df.columns:
            df.columns = ['gene_id','gene_name','gene_type','unstranded','stranded_first','stranded_second',
                          'tpm_unstranded','fpkm_unstranded','fpkm_uq_unstranded']
        prot = df[df['gene_type'] == 'protein_coding']
        for gene in prot['gene_name'].unique():
            gene_counter[gene] = gene_counter.get(gene, 0) + 1
    common_genes = [g for g, cnt in gene_counter.items() if cnt == len(tsv_files)]
    print(f"Common protein-coding genes across all samples: {len(common_genes)}")

    # Second pass: extract expression
    all_expr = []
    for f in tqdm(tsv_files, desc="Extracting expression"):
        expr = process_tsv(os.path.join(tsv_dir, f))
        if expr is not None:
            # Keep only common genes
            expr = expr.reindex(columns=common_genes)
            all_expr.append(expr)
    merged = pd.concat(all_expr)
    adata = sc.AnnData(merged)
    # Add labels
    labels = pd.read_csv(os.path.join(config.DATA_DIR, "labels/all_labels.csv"), index_col='sample_id')
    adata.obs['c1c2'] = labels.loc[adata.obs_names, 'label'].map({0: 'C1', 1: 'C2'})
    adata.write_h5ad(os.path.join(config.DATA_DIR, "merged_genes.h5ad"))
    print(f"Merged {len(all_expr)} samples, {merged.shape[1]} genes.")

if __name__ == "__main__":
    main()