# Public implementation update

## 2026-10-25

- Replace old public training/data-loading paths with patient-level training-only scaling, explicit frozen scaler artifacts and held-out-independent final fitting.
- Correct CV label/ID ordering and export metrics for the saved best epoch.
- Remove duplicate softmax from late-fusion evaluation and fail on absent/incompatible checkpoints.
- Add frozen external-cohort evaluation, classifier-eligibility audits and gene-order checks.
- Implement documented RNA/WSI preprocessing, training-background SHAP, channel-correct Grad-CAM and transparent per-analysis enrichment.
- Correct clinical response semantics, require patient-level endpoints and audit exclusions.
- Add paired patient-bootstrap analysis and synthetic regression tests.
- Preserve the original public architecture parameter names; provide explicit import/remapping for earlier server checkpoints.

This maintenance release changes source code, not manuscript result values. Historical numerical results require the original frozen experimental artifacts and have not been rerun as part of this update.
