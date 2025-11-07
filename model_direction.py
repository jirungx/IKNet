import os
import pandas as pd
from glob import glob

# 경로 설정
input_folder = "results/preds"           # 원본 CSV들이 있는 폴더
output_folder = "results/classfication_preds"  # 수정된 파일을 저장할 폴더
os.makedirs(output_folder, exist_ok=True)

# 모든 CSV 파일 처리
csv_files = glob(os.path.join(input_folder, "*.csv"))

for file_path in csv_files:
    df = pd.read_csv(file_path)
    
    # 파일 이름 가져오기
    file_name = os.path.basename(file_path)
    
    # 오늘 종가 (shift -1 해야 내일 종가와 비교 가능)
    df['today_price'] = df['y_true'].shift(1)  # 오늘 실제 종가
    df['predicted_tomorrow_price'] = df['y_pred']  # 모델이 예측한 내일 종가
    df['real_tomorrow_price'] = df['y_true']   # 실제 내일 종가

    # 모델이 상승을 예측했는지 (모델이 예측한 내일 가격 > 오늘 가격)
    df['predicted_direction'] = (df['predicted_tomorrow_price'] > df['today_price']).astype(int)

    # 실제로 상승했는지 (실제 내일 가격 > 오늘 가격)
    df['real_direction'] = (df['real_tomorrow_price'] > df['today_price']).astype(int)
    
    # 중간 계산 열 제거
    df = df.drop(columns=['today_price', 'predicted_tomorrow_price', 'real_tomorrow_price'])

    # 저장
    output_path = os.path.join(output_folder, file_name)
    df.to_csv(output_path, index=False)

print("모든 파일 방향 정보 수정 및 저장 완료")
