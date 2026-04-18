import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score
from tqdm import tqdm
import config
from models.gene_model import GeneModel
from cv.cv_data_loader import get_cv_dataloaders

def train_fold(fold):
    result_dir = os.path.join(config.RESULT_DIR, f"cross_validation_80percent/gene_model/fold_{fold}")
    os.makedirs(result_dir, exist_ok=True)
    dataloaders = get_cv_dataloaders(fold)
    train_loader = dataloaders['gene']['train']
    val_loader = dataloaders['gene']['val']
    input_dim = next(iter(train_loader))[0].shape[1]

    labels = pd.read_csv(os.path.join(config.DATA_DIR, "labels/all_labels.csv"))
    train_ids = pd.read_csv(os.path.join(config.RESULT_DIR, f"cross_validation_80percent/splits/fold_{fold}_train_ids.csv"))['sample_id'].tolist()
    train_labels = [labels[labels['sample_id']==sid]['label'].values[0] for sid in train_ids]
    class_counts = np.bincount(train_labels)
    class_weights = len(train_labels) / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(config.DEVICE)

    model = GeneModel(input_dim).to(config.DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    best_auc = 0
    patience_counter = 0
    for epoch in range(30):
        model.train()
        train_loss, train_preds, train_labels_epoch = 0.0, [], []
        for x, y in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}"):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_preds.extend(out.argmax(1).cpu().numpy())
            train_labels_epoch.extend(y.cpu().numpy())
        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_labels_epoch, train_preds)

        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                out = model(x)
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())
                val_probs.extend(torch.softmax(out, dim=1).cpu().numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        val_bal_acc = balanced_accuracy_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, [p[1] for p in val_probs])
        val_f1 = f1_score(val_labels, val_preds)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} AUC: {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(result_dir, "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                break

    pd.DataFrame({'accuracy': [val_acc], 'balanced_accuracy': [val_bal_acc], 'auc': [val_auc], 'f1_score': [val_f1]}).to_csv(
        os.path.join(result_dir, "val_results.csv"), index=False)
    return {'accuracy': val_acc, 'balanced_accuracy': val_bal_acc, 'auc': val_auc, 'f1_score': val_f1}

def main():
    all_results = []
    for fold in range(5):
        print(f"\n===== Fold {fold} =====")
        res = train_fold(fold)
        all_results.append(res)
    summary = pd.DataFrame(all_results)
    summary.to_csv(os.path.join(config.RESULT_DIR, "cross_validation_80percent/gene_model/cv_summary.csv"), index=False)
    print("\nCV Summary:\n", summary.mean())

if __name__ == "__main__":
    main()