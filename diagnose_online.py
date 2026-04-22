import torch
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 1. 경로 및 모델 설정 (inference.py와 동일)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_PATH = os.path.join(BASE_DIR, 'kaggle/working/final_delivery/code/ecg_training')
WEIGHTS_PATH = os.path.join(BASE_DIR, 'kaggle/working/final_delivery/model_weights/best_model.pt')

if CODE_PATH not in sys.path:
    sys.path.append(CODE_PATH)

from ecg_training.models import PTBXLClassifier

def load_model():
    model_config = {
        "input_leads": 12,
        "num_classes": 5,
        "embedding_dim": 256,
        "blocks": [2, 2, 2, 2],
        "base_channels": 32,
        "dropout": 0.2
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PTBXLClassifier(**model_config)
    
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

# 2. 전처리 함수 (Z-score Normalization)
def preprocess_signal(signal):
    # signal shape: (length, leads) -> (1000, 12)
    # Z-score normalization per lead
    means = signal.mean(axis=0, keepdims=True)
    stds = signal.std(axis=0, keepdims=True) + 1e-7
    normalized = (signal - means) / stds
    # 모델 입력 형태: (1, 12, 1000)
    return torch.tensor(normalized.T, dtype=torch.float32).unsqueeze(0)

import urllib.request

# 3. 데이터 다운로드 및 진단 실행
def run_diagnosis():
    print("\n🌐 PhysioNet PTB-XL 데이터 다운로드 중...")
    record_id = '00001'
    base_url = 'https://physionet.org/files/ptb-xl/1.0.3/records100/00000/'
    
    try:
        # 헤더와 데이터 파일 수동 다운로드
        for ext in ['.hea', '.dat']:
            file_name = f'{record_id}_lr{ext}'
            if not os.path.exists(file_name):
                print(f"  - {file_name} 받는 중...")
                urllib.request.urlretrieve(base_url + file_name, file_name)
        
        # 로컬에서 읽기
        record = wfdb.rdrecord(f'{record_id}_lr')
        print(f"✅ 데이터 로드 성공: {record_id}_lr")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    signal = record.p_signal


    
    # 모델 로드
    model, device = load_model()
    
    # 전처리
    input_tensor = preprocess_signal(signal).to(device)
    
    # 추론
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.sigmoid(output).cpu().numpy()[0]
    
    # 결과 출력
    classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    print(f"\n🔍 [진단 결과 - Record {record_id}]")
    for cls, prob in zip(classes, probabilities):
        indicator = "🔴" if prob > 0.5 else "⚪"
        print(f" {indicator} {cls}: {prob*100:5.2f}%")

    # 4. 시각화 (첫 3개 리드만 예시로 출력)
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(signal[:, i])
        plt.title(f'Lead {record.sig_name[i]}')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('ecg_plot.png')
    print(f"\n📈 ECG 파형이 'ecg_plot.png'로 저장되었습니다.")

if __name__ == "__main__":
    # 인자 없이 실행 (내부에서 자동으로 레코드 선택)
    run_diagnosis()
