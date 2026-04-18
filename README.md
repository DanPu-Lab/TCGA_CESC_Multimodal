# TCGA Cervical Cancer (CESC) Multi‑modal Immune Subtype Prediction 

This repository contains the complete code for predicting **C1/C2 immune subtypes** in cervical cancer using histopathology whole‑slide images (SVS) and RNA‑Seq transcriptomic data (TSV) from **TCGA‑CESC**.

We compare:
- Single‑modal models (image‑only, gene‑only)
- Multi‑modal models (early fusion, late fusion)

Additionally, we provide:
- 5‑fold cross‑validation
- SHAP analysis for gene importance
- Grad‑CAM for image region localization
- Pathway enrichment analysis
- Survival analysis (Kaplan‑Meier, Cox regression)
- Treatment response sensitivity analysis


Environment Setup
Create a virtual environment (conda or venv) and install dependencies:
pip install -r requirements.txt
Note: This code requires a GPU for reasonable training speed. CPU is possible but very slow.

---

##  Data Preparation

1. **Download TCGA‑CESC data**  
   - Obtain permission from [TCGA](https://portal.gdc.cancer.gov/) .  
   - Download **SVS** files (Biospecimen, image) and **TSV** files (Transcriptome Profiling, RNA‑Seq).  
   - Place them in the following structure inside `data/raw/`:

2. **Run preprocessing**  
   All scripts will read `RAW_DATA_DIR` and write processed files to `DATA_DIR` (see `config.py`).

```bash
python preprocess/organize_data.py
python preprocess/process_svs.py   # Extract image patches (WSI → patches.npz)
python preprocess/process_tsv.py  # Merge gene expression matrices → merged_genes.h5ad
python preprocess/split_dataset.py

After successful preprocessing, you will have:
data/processed/svs/*_patches.npz (image patches)
data/processed/tsv/ (individual TSV files)
data/processed/merged_genes.h5ad (gene expression matrix)
data/processed/labels/all_labels.csv (sample labels)
data/processed/splits/train_ids.csv and test_ids.csv


Training
Train each model separately. The best checkpoint (by validation AUC) will be saved to result/{model_name}_model/best_model.pth.
# Single‑modal
python train/train_image.py
python train/train_gene.py
# Multi‑modal
python train/train_early_fusion.py
python train/train_late_fusion.py
Training logs and validation curves are saved in the respective model folders.


Cross‑Validation (5‑fold)
Generate the fold splits first:
python cv/split_cv_folds.py

Then run each model’s cross‑validation:
python cv/train_cv_image.py
python cv/train_cv_gene.py
python cv/train_cv_early_fusion.py
python cv/train_cv_late_fusion.py

Results (per‑fold metrics and summary) are stored in result/cross_validation_80percent/.


Evaluation
After training, evaluate all four models on the held‑out test set:
python eval.py
Outputs:
result/evaluation/{model}_metrics.csv – accuracy, balanced accuracy, AUC, F1
result/evaluation/{model}_predictions.csv – predicted probabilities
result/evaluation/{model}_roc.csv – ROC curve data
result/evaluation/model_comparison.csv – side‑by‑side comparison


Interpretability & Downstream Analyses
All analysis scripts read the trained models and output results into result/analysis/.

SHAP (Gene importance)
python analysis/shap_analysis.py

Grad‑CAM (Image region localization)
python analysis/gradcam_analysis.py

Pathway Enrichment
python analysis/enrichment_analysis.py

Survival Analysis
python analysis/survival_analysis.py

Treatment Response
python analysis/treatment_analysis.py


Notes
The code automatically uses all available GPUs (via DataParallel). To use only specific GPUs, set CUDA_VISIBLE_DEVICES.
If you encounter out‑of‑memory errors, reduce BATCH_SIZE or NUM_PATCHES_SAMPLE.
