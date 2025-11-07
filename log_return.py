import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from glob import glob
from collections import defaultdict

# 경로 설정
folder_path = "results/final_preds_with_direction"
output_folder = "final_cumulative_return"
os.makedirs(output_folder, exist_ok=True)

# 모델 순서 및 색상 설정 (예측 그래프와 동일)
model_order = [
    "Long-only",
    "Ridge",
    "LSTM",
    "Transformer",
    "TCN",
    "FinBERT-Embedding-LSTM",
    "FinBERT-Sentiment-LSTM",
    "IKGNet"
]

model_color_map = {
    "Long-only": "black",
    "Ridge": "gray",
    "LSTM": "blue",
    "Transformer": "magenta",
    "TCN": "cyan",
    "FinBERT-Embedding-LSTM": "purple",
    "FinBERT-Sentiment-LSTM": "green",
    "IKGNet": "red"
}

# 폰트 설정
plt.rcParams["font.family"] = "Times New Roman"

# 연도별 모델별 누적 수익률 저장
year_model_returns = defaultdict(dict)
commission = 0.003
results = []

csv_files = glob(os.path.join(folder_path, "*.csv"))

# 수익률 계산
for file_path in csv_files:
    df = pd.read_csv(file_path)
    file_name = os.path.basename(file_path).replace(".csv", "")
    try:
        model_name, year = file_name.rsplit("_", 1)
        model_name = model_name.strip()
        year = int(year)
    except ValueError:
        continue

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    if df.empty:
        continue

    df['P_t'] = df['y_true'].shift(1)
    df['P_t+1'] = df['y_true']
    df['log_return'] = np.log(df['P_t+1'] / df['P_t'])
    df = df.dropna()

    strategy_returns = []
    flag = 0
    buy_count = 0
    sell_count = 0

    for i in range(len(df)):
        row = df.iloc[i]
        pred_dir = row['predicted_direction']
        log_ret = row['log_return']
        if flag == 0 and pred_dir == 1:
            strategy_ret = -commission
            flag = 1
            buy_count += 1
        elif flag == 1 and pred_dir == 1:
            strategy_ret = log_ret
        elif flag == 1 and pred_dir == 0:
            strategy_ret = -commission
            flag = 0
            sell_count += 1
        else:
            strategy_ret = 0
        strategy_returns.append(strategy_ret)

    df['strategy_return'] = strategy_returns
    df['cumulative_return'] = (np.exp(np.cumsum(df['strategy_return'])) - 1) * 100
    year_model_returns[year][model_name] = (df['date'], df['cumulative_return'])
    results.append((model_name, year, buy_count, sell_count, len(df)))

# Long-only 전략 추가 
for year in range(2018, 2025):
    for file_path in csv_files:
        if f"_{year}" in file_path and "LSTM" in file_path:
            df_long = pd.read_csv(file_path)
            df_long['date'] = pd.to_datetime(df_long['date'])
            df_long = df_long.sort_values('date').reset_index(drop=True)
            df_long['P_t'] = df_long['y_true'].shift(1)
            df_long['P_t+1'] = df_long['y_true']
            df_long['log_return_longonly'] = np.log(df_long['P_t+1'] / df_long['P_t'])
            df_long['longonly_cum_return'] = (np.exp(np.cumsum(df_long['log_return_longonly'])) - 1) * 100
            year_model_returns[year]['Long-only'] = (df_long['date'], df_long['longonly_cum_return'])
            break

# 시각화
for year, model_data in sorted(year_model_returns.items()):
    plt.figure(figsize=(12, 8))

    for model_name in model_order:
        if model_name not in model_data:
            continue
        dates, cumulative_return = model_data[model_name]

        # 범례 라벨 설정
        if model_name == "IKGNet":
            legend_label = "IK-MWFNet (Ours)"
        else:
            legend_label = model_name

        # 선 굵기 설정
        if model_name == "IKGNet":
            line_width = 1.7
        elif model_name == "Long-only":
            line_width = 1.5
        else:
            line_width = 1.2

        if model_name == "Long-only":
            plt.plot(
                dates, cumulative_return,
                label=legend_label,
                linestyle='--',
                color=model_color_map[model_name],
                linewidth=line_width
            )
        else:
            plt.plot(
                dates, cumulative_return,
                label=legend_label,
                color=model_color_map[model_name],
                linewidth=line_width
            )

    plt.xlabel("Date", fontsize=16)
    plt.ylabel("Cumulative Return (%)", fontsize=16)
    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # 한 달 간격
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    plt.legend(
        loc='best',
        fontsize=16,
        frameon=False,
        ncol=2
    )

    save_path = os.path.join(output_folder, f"Cumulative_Return_{year}.png")
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()

