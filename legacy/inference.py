import torch
import sys
import os
import numpy as np

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_PATH = os.path.join(BASE_DIR, 'kaggle/working/final_delivery/code/ecg_training')
WEIGHTS_PATH = os.path.join(BASE_DIR, 'kaggle/working/final_delivery/model_weights/best_model.pt')

# 파이썬 모듈 검색 경로에 추가
if CODE_PATH not in sys.path:
    sys.path.append(CODE_PATH)

try:
    from ecg_training.models import PTBXLClassifier
    print("✅ 모델 정의(Class)를 성공적으로 불러왔습니다.")
except ImportError as e:
    print(f"❌ 모델 정의를 불러오는데 실패했습니다: {e}")
    sys.exit(1)

# 2. 모델 설정 및 초기화
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

# 3. 가중치 로드
try:
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    # 체크포인트 파일이 딕셔너리 형태이므로 'model_state_dict' 키를 사용합니다.
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"✅ 모델 가중치 로드 완료! (장치: {device})")
except Exception as e:
    print(f"❌ 가중치 로드 중 오류 발생: {e}")
    sys.exit(1)

# 4. 가상 데이터로 추론 테스트 (10초, 100Hz = 1000 샘플)
print("\n--- 추론 테스트 시작 ---")
# 입력 형태: (Batch, Leads, Length) -> (1, 12, 1000)
dummy_input = torch.randn(1, 12, 1000).to(device)

with torch.no_grad():
    output = model(dummy_input)
    # Sigmoid를 통과시켜 확률값으로 변환
    probabilities = torch.sigmoid(output).cpu().numpy()[0]

classes = ["NORM", "MI", "STTC", "CD", "HYP"]
print("가상 데이터 진단 결과 (확률):")
for cls, prob in zip(classes, probabilities):
    print(f" - {cls}: {prob:.4f}")

print("\n🎉 모든 준비가 끝났습니다! 이제 실제 ECG 데이터를 넣을 수 있습니다.")
