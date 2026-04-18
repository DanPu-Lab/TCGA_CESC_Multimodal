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
from models.early_fusion_model import EarlyFusionModel
from data.dataset import get_dataloaders

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, preds, labels = 0.0, [], []
    for img, gene, y in tqdm(loader, desc="Train"):
        img, gene, y = img.to(config.DEVICE), gene.to(config.DEVICE), y.to(config.DEVICE)
        optimizer.zero_grad()
        out = model(img, gene)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * img.size(0)
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    return running_loss / len(loader.dataset), accuracy_score(labels, preds)

def validate(model, loader, criterion):
    model.eval()
    running_loss, preds, labels, probs = 0.0, [], [], []
    with torch.no_grad():
        for img, gene, y in loader:
            img, gene, y = img.to(config.DEVICE), gene.to(config.DEVICE), y.to(config.DEVICE)
            out = model(img, gene)
            loss = criterion(out, y)
            running_loss += loss.item() * img.size(0)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(torch.softmax(out, dim=1).cpu().numpy())
    loss = running_loss / len(loader.dataset)
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    auc = roc_auc_score(labels, [p[1] for p in probs])
    f1 = f1_score(labels, preds)
    return loss, acc, bal_acc, auc, f1, probs, labels

def main():
    os.makedirs(os.path.join(config.RESULT_DIR, "early_fusion_model"), exist_ok=True)
    dataloaders = get_dataloaders()
    train_loader = dataloaders['early_fusion']['train']
    test_loader = dataloaders['early_fusion']['test']

    input_dim = next(iter(train_loader))[1].shape[1]
    labels = pd.read_csv(os.path.join(config.DATA_DIR, "labels/all_labels.csv"))
    class_counts = labels['label'].value_counts().sort_index().values
    class_weights = len(labels) / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(config.DEVICE)

    model = EarlyFusionModel(input_dim).to(config.DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)

    best_auc = 0
    patience_counter = 0
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_bal_acc, val_auc, val_f1, _, _ = validate(model, test_loader, criterion)
        scheduler.step()
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} BalAcc: {val_bal_acc:.4f} AUC: {val_auc:.4f} F1: {val_f1:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(config.RESULT_DIR, "early_fusion_model/best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print("Early stopping")
                break

if __name__ == "__main__":
    main()