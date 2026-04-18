import os
import pandas as pd
from scipy.stats import chi2_contingency
import config

def main():
    save_dir = os.path.join(config.RESULT_DIR, "analysis/treatment")
    os.makedirs(save_dir, exist_ok=True)

    treatment = pd.read_csv(os.path.join(config.DATA_DIR, "data_timeline_treatment.txt"), sep='\t')
    labels = pd.read_csv(os.path.join(config.DATA_DIR, "labels/all_labels.csv"))
    labels['PATIENT_ID'] = labels['sample_id'].str.extract(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})')
    merged = treatment.merge(labels, on='PATIENT_ID', how='inner')
    merged['subtype'] = merged['label'].map({0: 'C1', 1: 'C2'})

    # Filter for response data
    resp = merged[merged['MEASURE_OF_RESPONSE'].notna()]
    resp = resp[resp['MEASURE_OF_RESPONSE'] != '']
    mapping = {
        'complete response': 'CR', 'partial response': 'PR', 'stable disease': 'SD',
        'progression': 'PD', 'recurrence': 'PD', 'adjuvant': 'SD'
    }
    resp['response'] = resp['MEASURE_OF_RESPONSE'].str.lower().map(mapping)
    resp = resp.dropna(subset=['response'])

    # Contingency table
    table = pd.crosstab(resp['subtype'], resp['response'])
    if table.shape[1] >= 2 and table.shape[0] >= 2:
        chi2, p, dof, exp = chi2_contingency(table)
    else:
        p = np.nan

    # Summary
    summary = table.copy()
    summary.to_csv(os.path.join(save_dir, "treatment_response_distribution.csv"))
    pd.DataFrame({'chi2_p_value': [p]}).to_csv(os.path.join(save_dir, "statistical_test_results.csv"), index=False)

    # ORR/DCR
    orr_dcr = []
    for subtype in ['C1', 'C2']:
        if subtype in table.index:
            row = table.loc[subtype]
            total = row.sum()
            cr_pr = row.get('CR', 0) + row.get('PR', 0)
            sd = row.get('SD', 0)
            orr_dcr.append({'Subtype': subtype, 'ORR': cr_pr/total, 'DCR': (cr_pr+sd)/total})
    pd.DataFrame(orr_dcr).to_csv(os.path.join(save_dir, "treatment_response_summary.csv"), index=False)

    print("Treatment analysis completed.")

if __name__ == "__main__":
    main()