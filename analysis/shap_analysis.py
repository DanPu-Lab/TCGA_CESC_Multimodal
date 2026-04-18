import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import StandardScaler
import shap
import config
from models.gene_model import GeneModel

def main():
    os.makedirs(os.path.join(config.RESULT_DIR, "analysis/shap"), exist_ok=True)

    # Load data
    adata = sc.read_h5ad(os.path.join(config.DATA_DIR, "merged_genes.h5ad"))
    gene_df = adata.to_df()
    scaler = StandardScaler()
    gene_scaled = scaler.fit_transform(gene_df.values)
    gene_scaled_df = pd.DataFrame(gene_scaled, index=gene_df.index, columns=gene_df.columns)

    test_ids = pd.read_csv(os.path.join(config.DATA_DIR, "splits/test_ids.csv"))['sample_id'].tolist()
    X_test = gene_scaled_df.loc[test_ids].values
    labels_df = pd.read_csv(os.path.join(config.DATA_DIR, "labels/all_labels.csv"))
    y_test = [labels_df[labels_df['sample_id']==sid]['label'].values[0] for sid in test_ids]

    # Load model
    model = GeneModel(input_dim=X_test.shape[1]).to(config.DEVICE)
    state = torch.load(os.path.join(config.RESULT_DIR, "gene_model/best_model.pth"), map_location='cpu')
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # SHAP explainer
    background = X_test[np.random.choice(len(X_test), 20, replace=False)]
    explainer = shap.KernelExplainer(lambda x: torch.softmax(model(torch.FloatTensor(x).to(config.DEVICE)), dim=1).cpu().numpy(), background)
    shap_values = explainer.shap_values(X_test, nsamples=30)

    # SHAP for class 1 (C2)
    shap_c2 = shap_values[1] if isinstance(shap_values, list) else shap_values[:,:,1]
    mean_abs_shap = np.abs(shap_c2).mean(axis=0)
    feature_importance = pd.DataFrame({'gene': gene_df.columns, 'mean_abs_shap': mean_abs_shap}).sort_values('mean_abs_shap', ascending=False)
    feature_importance.to_csv(os.path.join(config.RESULT_DIR, "analysis/shap/feature_importance_ranked.csv"), index=False)

    # Save full SHAP matrix
    pd.DataFrame(shap_c2, index=test_ids, columns=gene_df.columns).to_csv(os.path.join(config.RESULT_DIR, "analysis/shap/shap_values_matrix.csv"))
    # Sample labels
    pd.DataFrame({'sample_id': test_ids, 'label': y_test}).to_csv(os.path.join(config.RESULT_DIR, "analysis/shap/sample_labels.csv"), index=False)
    # Subtype specific
    c1_idx = [i for i, l in enumerate(y_test) if l==0]
    c2_idx = [i for i, l in enumerate(y_test) if l==1]
    subtype_shap = pd.DataFrame({
        'gene': gene_df.columns,
        'C1_mean_shap': shap_c2[c1_idx].mean(axis=0) if c1_idx else 0,
        'C2_mean_shap': shap_c2[c2_idx].mean(axis=0) if c2_idx else 0
    })
    subtype_shap.to_csv(os.path.join(config.RESULT_DIR, "analysis/shap/subtype_specific_shap.csv"), index=False)

    print("SHAP analysis completed. Files saved.")

if __name__ == "__main__":
    main()