import torch
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 1. 경로 및 모델 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_PATH = os.path.join(BASE_DIR, 'kaggle/working/final_delivery/code/ecg_training')
WEIGHTS_PATH = os.path.join(BASE_DIR, 'kaggle/working/final_delivery/model_weights/best_model.pt')
ARRHYTHMIA_WEIGHTS_PATH = os.path.join(BASE_DIR, 'runs/arrhythmia_specialist/arrhythmia_best.pt')

if CODE_PATH not in sys.path:
    sys.path.append(CODE_PATH)

from ecg_training.models import PTBXLClassifier, ArrhythmiaSpecialist

def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 공통 설정
    model_config = {
        "input_leads": 12,
        "num_classes": 5,
        "embedding_dim": 256,
        "blocks": [2, 2, 2, 2],
        "base_channels": 32,
        "dropout": 0.2
    }
    
    # 메인 모델 (PTB-XL 5종)
    main_model = PTBXLClassifier(**model_config)
    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
        main_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        print("✅ 메인 진단 모델 로드 완료")
    
    main_model.to(device).eval()

    # 부정맥 정밀 모델 (6종)
    # 메인 모델의 백본을 공유하여 전문가 모델 생성
    specialist = ArrhythmiaSpecialist(backbone=main_model.backbone, num_arrhythmia_classes=6)
    if os.path.exists(ARRHYTHMIA_WEIGHTS_PATH):
        checkpoint = torch.load(ARRHYTHMIA_WEIGHTS_PATH, map_location=device)
        specialist.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        print("✅ 부정맥 정밀 분석 모델 로드 완료")
    else:
        print("💡 부정맥 정밀 모델 가중치가 없어 초기 상태로 작동합니다. (학습 필요)")
    
    specialist.to(device).eval()
    return main_model, specialist, device

# ... (preprocess_signal 함수는 동일)

# 3. 데이터 다운로드 및 진단 실행
def run_diagnosis():
    # ... (데이터 다운로드 로직 동일)
    
    # 모델 로드
    main_model, specialist, device = load_models()
    
    # 전처리
    input_tensor = preprocess_signal(signal).to(device)
    
    # 추론
    with torch.no_grad():
        main_out = main_model(input_tensor)
        main_probs = torch.sigmoid(main_out).cpu().numpy()[0]
        
        spec_out = specialist(input_tensor)
        spec_probs = torch.sigmoid(spec_out).cpu().numpy()[0]
    
    # 결과 출력
    main_classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    spec_classes = ["AFIB", "AFLT", "SVPB", "PVC", "SVTA", "VTA"]
    
    print(f"\n🔍 [1. 일반 진단 결과 - Record {record_id}]")
    for cls, prob in zip(main_classes, main_probs):
        indicator = "🔴" if prob > 0.5 else "⚪"
        print(f" {indicator} {cls:5}: {prob*100:6.2f}%")

    print(f"\n🩺 [2. 부정맥 정밀 분석 결과]")
    for cls, prob in zip(spec_classes, spec_probs):
        indicator = "🔶" if prob > 0.3 else "⚪" # 부정맥은 더 민감하게(0.3) 표시
        print(f" {indicator} {cls:5}: {prob*100:6.2f}%")

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
