import os
import pandas as pd
import gseapy as gp
import config

def main():
    shap_dir = os.path.join(config.RESULT_DIR, "analysis/shap")
    enrich_dir = os.path.join(config.RESULT_DIR, "analysis/enrichment")
    os.makedirs(enrich_dir, exist_ok=True)

    shap_file = os.path.join(shap_dir, "feature_importance_ranked.csv")
    if not os.path.exists(shap_file):
        print("SHAP feature importance file not found. Run SHAP analysis first.")
        return
        
    feature_imp = pd.read_csv(shap_file)
    top_genes = feature_imp.head(200)['gene'].tolist()

    # 多数据库字典配置
    databases = {
        'KEGG': 'KEGG_2021_Human.gmt',
        'Reactome': 'Reactome_2022.gmt', 
        'GO_BP': 'GO_Biological_Process_2023.gmt',
        'Hallmark': 'MSigDB_Hallmark_2020.gmt',
        'Immune': 'Immunologic_Signatures.gmt'
    }
    
    all_results = []
    
    for db_name, file_name in databases.items():
        gmt_path = os.path.join(config.BASE_DIR, "databases", file_name)
        if not os.path.exists(gmt_path):
            print(f"Missing {db_name} database at {gmt_path}. Skipping...")
            continue
            
        print(f"Running enrichment for {db_name}...")
        enr = gp.enrichr(gene_list=top_genes, gene_sets=gmt_path, organism='Human', cutoff=0.05)
        
        if enr.results is not None and not enr.results.empty:
            # 过滤显著通路 (FDR < 0.05)
            sig_res = enr.results[enr.results['Adjusted P-value'] < 0.05].copy()
            if not sig_res.empty:
                sig_res['Database'] = db_name
                all_results.append(sig_res)

    if all_results:
        final_df = pd.concat(all_results, ignore_col=True)
        final_df = final_df.sort_values('Adjusted P-value')
        output_path = os.path.join(enrich_dir, "all_significant_pathways.csv")
        final_df.to_csv(output_path, index=False)
        print(f"Enrichment completed. Found {len(final_df)} significant pathways across databases.")
        print(f"Saved to {output_path}")
    else:
        print("No significant pathways found across any available databases.")

if __name__ == "__main__":
    main()