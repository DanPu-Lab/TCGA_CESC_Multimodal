"""Portable configuration; paths may be overridden without editing source."""
import os
from pathlib import Path
import numpy as np
import torch

BASE_DIR = str(Path(__file__).resolve().parent)
DATA_DIR = os.environ.get("CESC_DATA_DIR", str(Path(BASE_DIR) / "data/processed"))
RAW_DATA_DIR = os.environ.get("CESC_RAW_DATA_DIR", str(Path(BASE_DIR) / "data/raw"))
RESULT_DIR = os.environ.get("CESC_RESULT_DIR", str(Path(BASE_DIR) / "result"))
PATCH_SIZE, MIN_PATCHES, NUM_PATCHES_SAMPLE = 224, 100, 50
BATCH_SIZE = int(os.environ.get("CESC_BATCH_SIZE", "16"))
NUM_WORKERS = int(os.environ.get("CESC_NUM_WORKERS", "0"))
NUM_EPOCHS, PATIENCE = 50, 10
CV_EPOCHS = {"image": 50, "gene": 50, "early_fusion": 30, "late_fusion": 30}
LEARNING_RATE, WEIGHT_DECAY = 1e-4, 1e-5
RANDOM_SEED = 42
DEVICE = os.environ.get("CESC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = np.asarray([.485, .456, .406], dtype=np.float32)
IMAGENET_STD = np.asarray([.229, .224, .225], dtype=np.float32)
