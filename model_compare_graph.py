import pandas as pd
import matplotlib.pyplot as plt
import os

# 모델 리스트와 연도 리스트
model_names = [
    "Ridge", "LSTM", "Transformer", "TCN",
    "FinBERT-Attention-LSTM", "FinBERT-Sentiment-LSTM", "IKNet"
]
years = list(range(2022, 2023))  # 2018 ~ 2024

# 모델별 색상 매핑 (정확한 이름으로 반영)
model_color_map = {
    "Ridge": "gray",
    "LSTM": "blue",
    "Transformer": "magenta",
    "TCN": "cyan",
    "FinBERT-Attention-LSTM": "purple",
    "FinBERT-Sentiment-LSTM": "green",
    "IKNet": "red"
}

# 폰트 설정
plt.rcParams['font.family'] = 'Times New Roman'

# 시각화 시작
for year in years:
    plt.figure(figsize=(12, 9))
    has_data = False
    ground_truth_plotted = False

    for model_name in model_names:
        file_name = f"results/preds/{model_name}_{year}.csv"
        if not os.path.exists(file_name):
            print(f"[Warning] File not found: {file_name}")
            continue

        df = pd.read_csv(file_name)
        df["date"] = pd.to_datetime(df["date"])

        # Ground Truth는 한 번만 그림
        if not ground_truth_plotted:
            plt.plot(
                df["date"], df["y_true"],
                label="Ground Truth",
                color="black", linestyle="--", linewidth=2
            )
            ground_truth_plotted = True

        # 모델 예측선 그리기
        plt.plot(
            df["date"], df["y_pred"],
            label=model_name,
            linewidth=1.5,
            color=model_color_map.get(model_name, "gray")
        )

        has_data = True

    if has_data:
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Close Price", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.legend(
            loc="best",
            # bbox_to_anchor=(0.5, -0.2),
            ncol=2,
            fontsize=14,
            frameon=False,
            # columnspacing=1.5
        )

        plt.grid(True)
        plt.tight_layout()

        # 저장
        save_path = f"model_compare_graph/compare_graph_{year}.png"
        os.makedirs("model_compare_graph", exist_ok=True)
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.savefig("model_compare_graph/figure_2022.png", bbox_inches='tight', dpi=400)
        plt.close()
    else:
        print(f"[Skipped] {year}: No prediction files available for any model.")
