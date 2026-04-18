import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score, roc_curve
import config
from models.image_model import ImageModel
from models.gene_model import GeneModel
from models.early_fusion_model import EarlyFusionModel
from models.late_fusion_model import LateFusionModel
from data.dataset import get_dataloaders

def evaluate(model, loader, model_type):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            if model_type == 'image':
                x, y = batch
                x = x.to(config.DEVICE)
                out = model(x)
            elif model_type == 'gene':
                x, y = batch
                x = x.to(config.DEVICE)
                out = model(x)
            elif model_type == 'early_fusion':
                img, gene, y = batch
                img, gene = img.to(config.DEVICE), gene.to(config.DEVICE)
                out = model(img, gene)
            elif model_type == 'late_fusion':
                img, gene, y = batch
                img, gene = img.to(config.DEVICE), gene.to(config.DEVICE)
                out, _, _ = model(img, gene)
            else:
                raise ValueError
            probs_batch = torch.softmax(out, dim=1).cpu().numpy()
            preds_batch = probs_batch.argmax(1)
            preds.extend(preds_batch)
            labels.extend(y.numpy())
            probs.extend(probs_batch)
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    auc = roc_auc_score(labels, [p[1] for p in probs])
    f1 = f1_score(labels, preds)
    return acc, bal_acc, auc, f1, probs, labels

def main():
    os.makedirs(os.path.join(config.RESULT_DIR, "evaluation"), exist_ok=True)
    dataloaders = get_dataloaders()
    test_loaders = {m: dataloaders[m]['test'] for m in ['image', 'gene', 'early_fusion', 'late_fusion']}

    # Get gene input dimension
    input_dim = next(iter(test_loaders['gene']))[0].shape[1]

    models = {
        'image': ImageModel().to(config.DEVICE),
        'gene': GeneModel(input_dim).to(config.DEVICE),
        'early_fusion': EarlyFusionModel(input_dim).to(config.DEVICE),
        'late_fusion': LateFusionModel(input_dim).to(config.DEVICE)
    }
    for name, model in models.items():
        path = os.path.join(config.RESULT_DIR, f"{name}_model/best_model.pth")
        if os.path.exists(path):
            state = torch.load(path, map_location='cpu')
            # Remove 'module.' prefix if present
            state = {k.replace('module.', ''): v for k, v in state.items()}
            model.load_state_dict(state)
            print(f"Loaded {name} model")

    results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        acc, bal_acc, auc, f1, probs, labels = evaluate(model, test_loaders[name], name)
        results[name] = {'accuracy': acc, 'balanced_accuracy': bal_acc, 'auc': auc, 'f1_score': f1}
        # Save metrics
        pd.DataFrame([results[name]]).to_csv(os.path.join(config.RESULT_DIR, f"evaluation/{name}_metrics.csv"), index=False)
        # Save predictions
        pred_df = pd.DataFrame({'true': labels, 'pred_prob_C2': [p[1] for p in probs]})
        pred_df.to_csv(os.path.join(config.RESULT_DIR, f"evaluation/{name}_predictions.csv"), index=False)
        # ROC data
        fpr, tpr, _ = roc_curve(labels, [p[1] for p in probs])
        pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(os.path.join(config.RESULT_DIR, f"evaluation/{name}_roc.csv"), index=False)

    # Comparison table
    comp = pd.DataFrame(results).T
    comp.to_csv(os.path.join(config.RESULT_DIR, "evaluation/model_comparison.csv"))
    print("\n=== Model Comparison ===")
    print(comp)

if __name__ == "__main__":
    main()