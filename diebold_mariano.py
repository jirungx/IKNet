import os
import pandas as pd
import numpy as np
from scipy import stats

def dm_test(e1, e2, h=1, crit='MSE'):
    T = len(e1)
    if crit == 'MSE':
        d = (e1 ** 2) - (e2 ** 2)
    elif crit == 'MAD':
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("지원되지 않는 손실 기준입니다.")
    
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    DM_stat = d_mean / np.sqrt(d_var / T)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(DM_stat)))
    return DM_stat, p_value

years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
baseline_models = [
    "IKNetKO", "IKNetTO"
]
target_model = "IKNet"
folder = "results/diebold_mariano"

results = []

for model in baseline_models:
    for year in years:
        target_file = os.path.join(folder, f"{target_model}_{year}.csv")
        model_file = os.path.join(folder, f"{model}_{year}.csv")

        if not os.path.exists(target_file) or not os.path.exists(model_file):
            print(f"파일 없음: {year}, {model}")
            continue

        df_target = pd.read_csv(target_file)[['date', 'y_true', 'y_pred']].rename(columns={'y_pred': 'y_pred_target'})
        df_model = pd.read_csv(model_file)[['date', 'y_pred']].rename(columns={'y_pred': 'y_pred_model'})

        df_merged = pd.merge(df_target, df_model, on='date', how='inner')

        if len(df_merged) < 30:
            print(f"샘플 부족: {model} - {year} ({len(df_merged)}개)")
            continue

        e1 = df_merged['y_true'] - df_merged['y_pred_target']
        e2 = df_merged['y_true'] - df_merged['y_pred_model']

        DM_stat, p_value = dm_test(e1.values, e2.values)

        results.append({
            'Year': year,
            'Compared Model': model,
            'DM Statistic': round(DM_stat, 3),
            'p-value_raw': p_value,
            'p-value': "< 0.001" if p_value < 0.001 else round(p_value, 5),
            'Significant (p<0.05)': 'Yes' if p_value < 0.05 else 'No',
            'Samples Compared': len(e1)
        })

# 결과 정리 및 저장
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by=['Compared Model', 'Year'])
df_results = df_results.drop(columns=['p-value_raw'])
df_results.to_csv("dm_test_results.csv", index=False)

print("분석 완료")
