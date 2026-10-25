# Cervical cancer immune subtype prediction with histopathology and transcriptomics

Public implementation accompanying **Interpretable multimodal deep learning integrates histopathology and transcriptomics for immune subtype prediction in cervical cancer**.

The four architectures are CIS-AtN (histopathology), CIS-GnE (transcriptomics), CIS-EFF (early feature fusion), and CIS-LDF (late decision fusion). The task is classification of the established C1/C2 immune states. CGCI reference labels are **transcriptome-derived classifier assignments**, not an independent clinical or histological ground truth. Clinical analyses use pre-existing reference labels, not model predictions.

## Release scope and numerical reproduction

This maintenance release replaces the obsolete public training/data-loading scripts with an implementation of the manuscript's patient-level validation protocol. It enforces training-partition-only preprocessing and checkpoint selection, aligned fold IDs/labels, best-epoch reporting, correct late-fusion probability handling, and explicit clinical response semantics. This is a source-code update; historical reported results were **not rerun or independently verified in this maintenance session**.

Exact numerical reproduction requires the **original** curated patient manifests, original train/test and CV split files, ordered gene list, frozen model weights, and matching original training-only scaler parameters. These artifacts are not bundled here. A seed is not a replacement for the original split manifests. Do not fabricate a scaler, reconstruct it from all 257 TCGA patients, or regenerate splits and describe them as the original experimental artifacts. No manuscript AUROC, P value, patient prediction or clinical outcome is hard-coded into the analysis code.

The repository does not automatically download patient data. Obtain TCGA-CESC and CGCI-HTMCP-CC files from the GDC under their applicable access terms. Use the GENCODE version and patient eligibility criteria stated in the manuscript; the scripts do not silently translate annotation releases.

## Installation

Use Python 3.11 or 3.12. Reported experiments used Python 3.11; see the manuscript for experiment-specific software versions. Install the appropriate PyTorch 2.5.1 / TorchVision 0.20.1 build for your CUDA environment, then:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

OpenSlide also requires its native library. On Ubuntu install `libopenslide0`; on Windows use the OpenSlide-supported binary installation. Tests use synthetic slides and do not require SVS files or pretrained weight downloads. Test predictions are explicitly synthetic and are not research results.

## Paths and patient manifests

Default paths are `data/raw`, `data/processed`, and `result`. Override `CESC_RAW_DATA_DIR`, `CESC_DATA_DIR`, `CESC_RESULT_DIR`, `CESC_DEVICE`, `CESC_BATCH_SIZE` and `CESC_NUM_WORKERS` as needed.

Prepare a CSV with columns `sample_id,label,rna_path,wsi_path`: one unique patient per row, C1=0 and C2=1, and explicit existing file paths. Curate primary-tumour diagnostic slides and RNA aliquots at patient level before processing. Multiple slides/aliquots must not become independent patients. File paths are never inferred from arbitrary GDC UUID filenames. For multiple eligible slides, document the original slide-selection/aggregation rule rather than silently duplicating the patient.

```text
data/processed/
  labels/all_labels.csv                 # sample_id,label
  merged_genes.h5ad                      # unstandardised log1p(TPM), patients x ordered genes
  svs/<sample_id>_patches.h5              # gamma-corrected RGB patches and coordinates
  splits/train_ids.csv                   # original outer training patients
  splits/test_ids.csv                    # original held-out patients
```

```bash
python preprocess/organize_data.py --manifest tcga_manifest.csv
python preprocess/process_tsv.py --manifest tcga_manifest.csv
python preprocess/process_svs.py --manifest tcga_manifest.csv --cohort tcga
```

RNA: restrict to protein-coding genes, aggregate duplicate TPM entries by gene symbol, then apply log1p. No expression-abundance filter is used. Preserve the exact original 19,914-gene order for reproduction. A saved scaler is fitted only on the explicit training IDs of each CV fold or the final outer training partition. Missing external genes are an error, never silently imputed. Legacy normalised NPZ patches are supported; new HDF5 caches are ImageNet-normalised once at loading.

WSI: HSV thresholds S>20 and V>40, 5x5 morphological opening/closing, 224-pixel patches, adaptive stride, tissue-coverage filtering, gamma 0.9, and ImageNet normalisation. For TCGA, 20x/10x/5x target scales are derived from objective power and recorded native pyramid levels; native reads are resized when necessary. Unknown objective power must be supplied from verified metadata. The extraction metadata records achieved sampling, coordinates and mpp; an index such as level 1 is never assumed to mean 20x across scanners. CGCI uses levels 1,0,2 with its specified mean-RGB and tissue filters.

## Patient-level splitting and CV

For a new run only (do not overwrite the original reproduction manifests):

```bash
python preprocess/split_dataset.py
python cv/split_cv_folds.py
python cv/train_cv_image.py
python cv/train_cv_gene.py
python cv/train_cv_early_fusion.py
python cv/train_cv_late_fusion.py
```

The outer split is stratified 80/20, seed 42. All five CV folds are confined to the outer 80%. IDs and reference labels are aligned by identifier before splitting. Each fold estimates and saves its own scaler. Adam uses learning rate 1e-4, weight decay 1e-5, cosine scheduling, inverse-frequency **training-partition-only** class weights, gradient clipping 1.0 and validation-AUROC early stopping (patience 10). Default CV budgets are 50 epochs for unimodal models and 30 for fusion models. Only training images are augmented. CV metrics are exported from the saved best validation epoch, not the last epoch.

## Final fitting and held-out evaluation

After locking model settings using only training CV, declare an integer epoch budget for each final model and fit the complete outer training partition. Replace `EPOCHS_FROM_CV` below with that locked integer; it is intentionally not an invented historical value.

```bash
python train/train_image.py --epochs EPOCHS_FROM_CV
python train/train_gene.py --epochs EPOCHS_FROM_CV
python train/train_early_fusion.py --epochs EPOCHS_FROM_CV
python train/train_late_fusion.py --epochs EPOCHS_FROM_CV
python eval.py --output result/heldout_evaluation
python analysis/compare_auroc.py result/heldout_evaluation --plot
```

Final fitting constructs no held-out DataLoader and performs no test-based early stopping or checkpoint selection. Each checkpoint binds weights to its training IDs, held-out IDs, ordered features and checksum-protected training scaler. Existing run directories are not overwritten. Evaluation never falls back to an uninitialised/random model when a weight file is absent. Late-fusion outputs are already probabilities and are not passed through softmax again.

For existing weights, use `python import_legacy_checkpoint.py --help`. This explicitly requires original split/scaler artifacts, loads tensors strictly and preserves them without training. `--layout server-v1` provides checked parameter-name remapping for the earlier server modules; dimensions and every key must match. No scaler is fitted by this import tool. A bare weight file alone cannot establish its training/preprocessing provenance.

## External CGCI evaluation

Infer reference labels with the official CRI-iAtlas `ImmuneSubtypeClassifier` version 0.2.6 and `immune_optimized_99_pairs.rds`. Export the official prediction table with patient IDs, `BestCall` and the six subtype scores. Use a cervical-cancer-only paired input manifest. The selection audit records non-C1/C2 and BestCall/score-argmax discordance separately, including overlapping reasons; it applies no score cut-off. `--bestcall-only` is an explicit optional sensitivity analysis and is never substituted for the main cohort silently.

```bash
python external/prepare_cohort.py --manifest cgci_paired_manifest.csv --classifier-output cgci_classifier_output.csv --output cgci_processed --expected-count 119
python preprocess/process_tsv.py --manifest cgci_processed/paired_manifest.csv --scaler result/gene_model/best_model.scaler.npz --output cgci_processed/merged_genes.h5ad
python preprocess/process_svs.py --manifest cgci_processed/paired_manifest.csv --cohort cgci --output-dir cgci_processed/svs
python eval.py --cohort cgci --data-dir cgci_processed --checkpoint-root result --output result/cgci_evaluation
```

External RNA is ordered and transformed using the saved TCGA training scaler, without any external fit. All four fixed models use the same patient-level 50-patch bag; indices are derived from SHA256 of cohort, seed 42 and patient ID, without replacement. No external retraining, batch correction, threshold optimisation or test-time augmentation is performed. Classification uses argmax, with C2 positive. Exported patient-level predictions and metrics include bootstrap uncertainty. Actual cohort counts are checked, not forced by editing labels.

## Attribution, enrichment and clinical context

```bash
python analysis/shap_analysis.py --output result/analysis/shap
python analysis/gradcam_analysis.py --output result/analysis/gradcam
python analysis/enrichment_analysis.py result/analysis/shap/feature_importance_nonzero.csv --output result/analysis/enrichment
python analysis/treatment_analysis.py curated_patient_responses.csv --output result/analysis/treatment
python analysis/survival_analysis.py curated_patient_survival.csv --output result/analysis/survival
```

SHAP explains CIS-GnE, not the fusion model: 20 training-background patients, seed 123 and the stated 30-evaluation budget. `--nsamples`, `--background` and explicit `--l1-reg` are recorded; increase budgets and vary backgrounds for stability analyses. The budget and sparse-regression settings can influence rankings. Save the full attribution matrix and exact input ranking table. The enrichment mean threshold is calculated from the supplied table, so a nonzero-only table and an all-gene table define different analyses and must not be interchanged silently.

Enrichment uses three lists (top 200, top 150, above the supplied-table mean), minimum size 20, and local KEGG 2021 / Reactome 2022 / GO BP 2023 GMT files with their annotated-gene backgrounds. BH adjustment is within each list/database analysis. All term results are saved; the minimum-adjusted-P representation of duplicate pooled terms is exploratory and does not confer global FDR control. Attribution and enrichment do not establish mechanisms.

Grad-CAM uses the terminal ResNet convolutional block, preserves the channel dimension when computing gradient weights, normalises maps to the 5th/95th percentiles, and uses 40% opacity with the jet rainbow colour map. Representative patients are randomly selected within correctly classified held-out subtypes; selection IDs and all patch maps are saved. Heatmaps do not identify immune cell types or establish multimodal spatial correspondence. Physical scale bars require original mpp metadata and are not guessed.

Clinical response inputs require one explicitly curated endpoint per patient (`sample_id,response`). `adjuvant` is treatment intent, not stable disease; recurrence alone is not a response category. Unrecognised values are excluded with an audit trail. Multiple timeline rows are rejected rather than counted as separate patients. Pearson chi-square is reported with sparse-cell diagnostics and an exact/Monte-Carlo sensitivity test. Odds ratios are C2 versus C1, with a 0.5 correction for zero cells. Survival uses `sample_id,OS.time,OS`, days, Kaplan-Meier, two-sided log-rank and univariable Cox. These analyses use reference subtype labels and do not validate model-predicted prognosis.

Paired AUROC inference resamples patients jointly across models within reference class, with 1,000 replicates by default. It reports percentile 95% CIs and a centred-bootstrap, two-sided approximate P value with add-one correction. Replicates are not independent patient observations. The four comparisons are explicitly unadjusted; do not present them as globally multiplicity-corrected. No P-value or significance star is filled from the manuscript.

## References and data access

- GDC: https://portal.gdc.cancer.gov/
- TCGA immune annotations: https://xenabrowser.net/
- Official reference classifier: https://github.com/CRI-iAtlas/ImmuneSubtypeClassifier
- Clinical source: https://www.cbioportal.org/

Keep access-controlled data, identifiers not cleared for public sharing, raw slides, expression matrices, weights and credentials outside Git. Share original reproducibility artifacts only under their applicable data-use permissions.
