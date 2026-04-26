import torch
import torch.nn.functional as F
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import urllib.request
from datetime import datetime

# Grad-CAM 기능을 위한 전역 변수
activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

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
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        main_model.load_state_dict(state_dict)
        print("✅ 메인 진단 모델 로드 완료")
    else:
        print("⚠️ 메인 모델 가중치 파일을 찾을 수 없습니다.")
    
    main_model.to(device).eval()

    # 부정맥 정밀 모델 (6종)
    specialist = ArrhythmiaSpecialist(backbone=main_model.backbone, num_arrhythmia_classes=6)
    if os.path.exists(ARRHYTHMIA_WEIGHTS_PATH):
        checkpoint = torch.load(ARRHYTHMIA_WEIGHTS_PATH, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        specialist.load_state_dict(state_dict)
        print("✅ 부정맥 정밀 분석 모델 로드 완료")
    else:
        print("💡 부정맥 정밀 모델 가중치가 없어 초기 상태로 작동합니다. (학습 필요)")
    
    specialist.to(device).eval()
    return main_model, specialist, device

# 2. 전처리 함수 (Z-score Normalization)
def preprocess_signal(signal):
    # signal shape: (length, leads) -> (1000, 12)
    # Z-score normalization per lead
    means = signal.mean(axis=0, keepdims=True)
    stds = signal.std(axis=0, keepdims=True) + 1e-7
    normalized = (signal - means) / stds
    # 모델 입력 형태: (1, 12, 1000)
    return torch.tensor(normalized.T, dtype=torch.float32).unsqueeze(0)

# 3. 데이터 다운로드 및 진단 실행
def run_diagnosis():
    print("\n🌐 PhysioNet PTB-XL 데이터 다운로드 중...")
    record_id = '00002'
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
    main_model, specialist, device = load_models()
    
    # Grad-CAM 타겟 설정 (백본의 마지막 블록)
    target_layer = specialist.backbone.lead_encoder.layers[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)
    
    # 전처리
    input_tensor = preprocess_signal(signal).to(device)
    input_tensor.requires_grad = True
    
    # 추론 및 Grad-CAM 계산
    main_out = main_model(input_tensor)
    main_probs = torch.sigmoid(main_out).detach().cpu().numpy()[0]
    
    spec_out = specialist(input_tensor)
    spec_probs = torch.sigmoid(spec_out)
    
    # 가장 확률이 높은 부정맥 클래스에 대해 Grad-CAM 수행
    target_class_idx = spec_probs.argmax().item()
    spec_probs[0, target_class_idx].backward()
    
    # Grad-CAM 맵 생성
    with torch.no_grad():
        pooled_gradients = torch.mean(gradients, dim=[0, 2])
        # activations: (batch*leads, channels, seq_len) -> (12, 256, 63)
        # pooled_gradients: (256,)
        
        # 가중치 적용
        cam_activations = activations.clone()
        for i in range(cam_activations.shape[1]):
            cam_activations[:, i, :] *= pooled_gradients[i]
        
        # 채널별 평균 및 ReLU
        heatmap = torch.mean(cam_activations, dim=1).squeeze() # (leads, seq_len)
        heatmap = F.relu(heatmap)
        
        # 전체 리드에 대해 합산하여 공통 관심도 계산
        combined_heatmap = torch.mean(heatmap, dim=0) # (seq_len,)
        combined_heatmap /= (torch.max(combined_heatmap) + 1e-7)
        combined_heatmap = combined_heatmap.cpu().numpy()
        
        # 히트맵을 원본 신호 길이에 맞게 보간
        heatmap_interp = np.interp(
            np.linspace(0, len(combined_heatmap), len(signal)),
            np.arange(len(combined_heatmap)),
            combined_heatmap
        )

    spec_probs_np = spec_probs.detach().cpu().numpy()[0]
    
    # 결과 해석 및 리포트 생성
    main_classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    spec_classes = ["AFIB", "AFLT", "SVPB", "PVC", "SVTA", "VTA"]
    target_class_name = spec_classes[target_class_idx]
    
    # 임상 소견 생성 로직
    findings = []
    if main_probs[0] > 0.5:
        findings.append("정상 동리듬(Normal Sinus Rhythm) 가능성 높음")
    
    for i, (cls, prob) in enumerate(zip(main_classes[1:], main_probs[1:])):
        if prob > 0.5:
            desc = {"MI": "심근경색(Myocardial Infarction)", "STTC": "ST/T파 변화", 
                    "CD": "전도 장애(Conduction Disturbance)", "HYP": "비대(Hypertrophy)"}
            findings.append(f"{desc.get(cls, cls)} 의심")

    for cls, prob in zip(spec_classes, spec_probs_np):
        if prob > 0.3: # 부정맥은 민감하게 포착
            desc = {"AFIB": "심방세동(Atrial Fibrillation)", "AFLT": "심방조동(Atrial Flutter)",
                    "SVPB": "상심실성 조기수축", "PVC": "심실성 조기수축",
                    "SVTA": "상심실성 빈맥", "VTA": "심실성 빈맥"}
            findings.append(f"부정맥 확인: {desc.get(cls, cls)} 소견")

    # 리포트 텍스트 구성
    report_lines = []
    report_lines.append("="*50)
    report_lines.append("        AI ECG ANALYSIS CLINICAL REPORT")
    report_lines.append("="*50)
    report_lines.append(f" [데이터 정보] Record ID: {record_id}_lr")
    report_lines.append(f" [분석 시간] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (KST)")
    report_lines.append("-"*50)
    
    report_lines.append(" [1. 종합 판독 소견 (Clinical Impression)]")
    if findings:
        for i, find in enumerate(findings):
            report_lines.append(f"  {i+1}. {find}")
    else:
        report_lines.append("  - 특이 소견 없음")
    report_lines.append("-"*50)

    report_lines.append(" [2. 상세 진단 데이터]")
    report_lines.append("  <일반 진단 (PTB-XL 5개 대분류)>")
    for cls, prob in zip(main_classes, main_probs):
        indicator = "🔴" if prob > 0.5 else "⚪"
        report_lines.append(f"   {indicator} {cls:5}: {prob*100:6.2f}%")

    report_lines.append("\n  <부정맥 정밀 분석 (Arrhythmia Specialist)>")
    for cls, prob in zip(spec_classes, spec_probs_np):
        indicator = "🔶" if prob > 0.3 else "⚪"
        report_lines.append(f"   {indicator} {cls:5}: {prob*100:6.2f}%")
    report_lines.append("-"*50)

    report_lines.append(f" [3. XAI 분석 (Grad-CAM)]")
    report_lines.append(f"  - 타겟 클래스: {target_class_name}")
    report_lines.append(f"  - 이미지('ecg_plot.png')의 붉은 하이라이트 구간이 {target_class_name} 판독의 주요 근거임.")
    report_lines.append("-"*50)

    report_lines.append(" [4. 의학적 주의사항]")
    report_lines.append("  ※ 본 리포트는 AI 모델의 분석 결과이며 전문의의 최종 판독을")
    report_lines.append("     대체할 수 없습니다. 임상적 결정 전 반드시 전문가와 상의하십시오.")
    report_lines.append("="*50)

    # 리포트 출력 및 저장
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    with open("diagnosis_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n📄 진단 리포트가 'diagnosis_report.txt'로 저장되었습니다.")

    # 4. 시각화 (Grad-CAM 포함)
    plt.figure(figsize=(12, 8))
    for i in range(3):
        ax = plt.subplot(3, 1, i+1)
        ax.plot(signal[:, i], color='black', linewidth=0.8, label=f'Lead {record.sig_name[i]}')
        
        # Grad-CAM 히트맵 오버레이
        for j in range(len(heatmap_interp)-1):
            if heatmap_interp[j] > 0.2: # 중요도가 낮은 구간은 표시 안 함
                ax.axvspan(j, j+1, color='red', alpha=heatmap_interp[j] * 0.3)
        
        ax.set_title(f'Lead {record.sig_name[i]} (Target: {target_class_name})')
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend(loc='upper right')
            
    plt.tight_layout()
    plt.savefig('ecg_plot.png')
    print(f"📈 Grad-CAM 분석 결과가 포함된 ECG 파형이 'ecg_plot.png'로 저장되었습니다.")

if __name__ == "__main__":
    run_diagnosis()
