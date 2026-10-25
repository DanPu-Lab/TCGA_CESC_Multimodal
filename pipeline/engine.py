"""Cross-validation uses training-fold scalers; final fitting never reads test data."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import config
from pipeline.data import read_ids, read_labels, read_expression, check_partitions, manifest_metadata
from pipeline.dataset import PatientDataset, loader, make_loaders
from pipeline.metrics import classification_metrics
from pipeline.artifacts import save_checkpoint


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(mode, input_dim, pretrained=True):
    if mode == "image":
        from models.image_models import ImageModel
        return ImageModel(pretrained=pretrained)
    if mode == "gene":
        from models.gene_models import GeneModel
        return GeneModel(input_dim)
    if mode == "early_fusion":
        from models.early_fusion_models import EarlyFusionModel
        return EarlyFusionModel(input_dim, pretrained=pretrained)
    if mode == "late_fusion":
        from models.late_fusion_models import LateFusionModel
        return LateFusionModel(input_dim, pretrained=pretrained)
    raise ValueError(mode)


def forward(model, batch, mode, device):
    inputs, labels = batch[:-1], batch[-1].to(device)
    outputs = model(*(value.to(device) for value in inputs))
    if mode == "late_fusion":
        probabilities = outputs[0]
        if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("Invalid late-fusion probabilities")
        if not torch.allclose(probabilities.sum(1), torch.ones(len(labels), device=device), atol=1e-5):
            raise ValueError("Late-fusion probabilities do not sum to one")
        loss_input = probabilities.clamp_min(1e-10).log()
    else:
        loss_input = outputs
        probabilities = torch.softmax(outputs, dim=1)
    return loss_input, probabilities, labels


def train_epoch(model, batches, criterion, optimizer, mode, device):
    model.train()
    loss_sum, count = 0., 0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        logits, _, labels = forward(model, batch, mode, device)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        loss_sum += float(loss.detach()) * len(labels)
        count += len(labels)
    return loss_sum / count


def predict(model, batches, mode, device):
    model.eval()
    ys, probabilities = [], []
    with torch.no_grad():
        for batch in batches:
            _, p, y = forward(model, batch, mode, device)
            ys.extend(y.cpu().tolist())
            probabilities.extend(p[:, 1].cpu().tolist())
    return pd.DataFrame({"sample_id": batches.dataset.sample_ids, "true": ys,
                         "pred_prob_C2": probabilities})


def class_weights(labels):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    if (counts == 0).any():
        raise ValueError("Training partition must include both C1 and C2")
    return torch.tensor(len(labels) / (2 * counts), dtype=torch.float32)


def fit(mode, train_ids, evaluation_ids, output, epochs, pretrained=True, final=False):
    seed_all(config.RANDOM_SEED)
    labels = read_labels(config.DATA_DIR)
    check_partitions(train_ids, evaluation_ids, labels.index)
    if not final:
        tr, val, scaler = make_loaders(train_ids, evaluation_ids, mode)
    else:
        # Do not construct an evaluation dataset/loader or use evaluation values
        # for scaler fitting, stopping or selection during final fitting.
        ds = PatientDataset(train_ids, mode, True)
        tr, val, scaler = loader(ds, True), None, ds.scaler
    input_dim = len(scaler.genes) if scaler is not None else 0
    model = build_model(mode, input_dim, pretrained).to(config.DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    weight = class_weights(labels.loc[train_ids]).to(config.DEVICE)
    criterion = (nn.NLLLoss(weight=weight) if mode == "late_fusion"
                 else nn.CrossEntropyLoss(weight=weight))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    meta = manifest_metadata(train_ids, evaluation_ids, scaler.genes if scaler else [])
    meta.update(input_dim=input_dim, stage="final" if final else "cv",
                seed=config.RANDOM_SEED, max_epochs=epochs,
                epoch_selection="predeclared_fixed_budget" if final else "training_fold_validation_AUROC",
                pretrained=pretrained)
    best_auc, stale, history, best_row = -np.inf, 0, [], None
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, tr, criterion, optimizer, mode, config.DEVICE)
        scheduler.step()
        row = {"epoch": epoch, "training_loss": loss}
        if final:
            if epoch == epochs:
                meta["selected_epoch"] = epoch
                save_checkpoint(output / "best_model.pth", model, mode, scaler, meta)
        else:
            pred = predict(model, val, mode, config.DEVICE)
            metrics = classification_metrics(pred.true, pred.pred_prob_C2)
            row.update(metrics)
            if metrics["auroc"] > best_auc:
                best_auc, stale, best_row = metrics["auroc"], 0, dict(metrics)
                meta["selected_epoch"] = epoch
                best_row["selected_epoch"] = epoch
                save_checkpoint(output / "best_model.pth", model, mode, scaler, meta)
                pred.to_csv(output / "best_validation_predictions.csv", index=False)
            else:
                stale += 1
        history.append(row)
        print(json.dumps(row), flush=True)
        if not final and stale >= config.PATIENCE:
            break
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    (output / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
    # Metrics correspond to the saved best epoch, not to the last epoch seen.
    if best_row is not None:
        pd.DataFrame([best_row]).to_csv(output / "val_results.csv", index=False)
    return best_row


def training_main(mode, stage="final"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, help="Final budget fixed using training-only CV; required for final fitting")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--no-pretrained", action="store_true", help="For smoke tests, not the stated ImageNet protocol")
    args = parser.parse_args()
    if stage == "final" and (args.epochs is None or args.epochs < 1):
        parser.error("--epochs must be fixed from training-only CV before opening the held-out test set")
    epochs = args.epochs or config.CV_EPOCHS[mode]
    if epochs < 1:
        parser.error("--epochs must be positive")
    train_ids = read_ids(Path(config.DATA_DIR) / "splits/train_ids.csv")
    test_ids = read_ids(Path(config.DATA_DIR) / "splits/test_ids.csv")
    check_partitions(train_ids, test_ids)
    if stage == "final":
        fit(mode, train_ids, test_ids, Path(config.RESULT_DIR) / f"{mode}_model",
            epochs, not args.no_pretrained, final=True)
        return
    folds, records = ([args.fold] if args.fold is not None else range(5)), []
    for fold in folds:
        root = Path(config.RESULT_DIR) / "cross_validation_80percent"
        tr = read_ids(root / f"splits/fold_{fold}_train_ids.csv")
        val = read_ids(root / f"splits/fold_{fold}_val_ids.csv")
        check_partitions(tr, val, train_ids)
        if set(tr) | set(val) != set(train_ids):
            raise ValueError("Each CV fold must partition exactly the outer training patients")
        row = fit(mode, tr, val, root / f"{mode}_model/fold_{fold}", epochs,
                  not args.no_pretrained)
        records.append({"fold": fold, **row})
    if args.fold is None:
        pd.DataFrame(records).to_csv(root / f"{mode}_model/cv_summary.csv", index=False)
