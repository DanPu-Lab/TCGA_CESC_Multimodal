import os
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import config

def main():
    save_dir = os.path.join(config.RESULT_DIR, "analysis/survival")
    os.makedirs(save_dir, exist_ok=True)

    surv = pd.read_csv(os.path.join(config.DATA_DIR, "TCGA-CESC.survival.tsv"), sep='\t')
    labels = pd.read_csv(os.path.join(config.DATA_DIR, "labels/all_labels.csv"))
    # Extract patient ID
    surv['sample_id'] = surv['sample'].str.extract(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})')
    merged = surv.merge(labels, on='sample_id').dropna(subset=['OS.time', 'OS'])
    merged = merged[merged['OS.time'] > 0]
    merged['subtype'] = merged['label'].map({0: 'C1', 1: 'C2'})

    # Log-rank test
    c1 = merged[merged['subtype']=='C1']
    c2 = merged[merged['subtype']=='C2']
    if len(c1) > 0 and len(c2) > 0:
        result = logrank_test(c1['OS.time'], c2['OS.time'], c1['OS'], c2['OS'])
        p_value = result.p_value
    else:
        p_value = np.nan

    # Cox regression
    from lifelines import CoxPHFitter
    cox_data = merged[['OS.time', 'OS', 'label']].copy()
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col='OS.time', event_col='OS')
    hr = cph.hazard_ratios_['label']
    hr_ci_lower, hr_ci_upper = np.exp(cph.confidence_intervals_.loc['label'])

    # Survival rates
    kmf = KaplanMeierFitter()
    survival_rates = {}
    for sub, data in [('C1', c1), ('C2', c2)]:
        if len(data) > 0:
            kmf.fit(data['OS.time']/365.25, data['OS'])
            survival_rates[sub] = {'1yr': kmf.predict(1), '3yr': kmf.predict(3), '5yr': kmf.predict(5)}
        else:
            survival_rates[sub] = {'1yr': np.nan, '3yr': np.nan, '5yr': np.nan}

    # Summary
    summary = pd.DataFrame([{
        'logrank_p': p_value,
        'hazard_ratio': hr,
        'hr_ci_lower': hr_ci_lower,
        'hr_ci_upper': hr_ci_upper,
        'c1_1yr': survival_rates['C1']['1yr'],
        'c1_3yr': survival_rates['C1']['3yr'],
        'c1_5yr': survival_rates['C1']['5yr'],
        'c2_1yr': survival_rates['C2']['1yr'],
        'c2_3yr': survival_rates['C2']['3yr'],
        'c2_5yr': survival_rates['C2']['5yr'],
    }])
    summary.to_csv(os.path.join(save_dir, "statistical_summary.csv"), index=False)

    # Save merged data
    merged.to_csv(os.path.join(save_dir, "merged_survival_data.csv"), index=False)

    # Generate report
    with open(os.path.join(save_dir, "final_analysis_report.txt"), "w") as f:
        f.write(f"Log-rank p-value: {p_value:.4f}\n")
        f.write(f"Hazard Ratio (C2 vs C1): {hr:.3f} (95% CI {hr_ci_lower:.3f}-{hr_ci_upper:.3f})\n")
        f.write(f"C1 5-year survival: {survival_rates['C1']['5yr']:.1%}\n")
        f.write(f"C2 5-year survival: {survival_rates['C2']['5yr']:.1%}\n")
    print("Survival analysis completed.")

if __name__ == "__main__":
    main()