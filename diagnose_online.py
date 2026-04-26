import torch
import torch.nn.functional as F
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import urllib.request
from datetime import datetime
from scipy.signal import find_peaks

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

# 3. 신호 분석 함수 (2단계: Signal-to-Feature)
def analyze_rr_intervals(signal, fs=100):
    """
    R-peak를 탐지하고 RR 간격의 통계를 계산합니다.
    주로 Lead II (보통 index 1) 또는 전반적인 리드를 사용합니다.
    """
    # 12리드 중 R-peak가 잘 보이는 리드 선택 (여기선 1번 리드 예시)
    lead_signal = signal[:, 1]
    
    # R-peak 탐지 (임계값 및 거리 설정)
    # 100Hz 기준, 심박수 200bpm(30 samples) 이상은 드물다고 가정
    peaks, _ = find_peaks(lead_signal, distance=30, prominence=0.5)
    
    if len(peaks) < 2:
        return None

    # RR 간격 계산 (ms 단위)
    rr_intervals = np.diff(peaks) * (1000 / fs)
    avg_hr = 60000 / np.mean(rr_intervals)
    
    # RR 간격 변동성 (Coefficient of Variation)
    rr_std = np.std(rr_intervals)
    rr_mean = np.mean(rr_intervals)
    rr_cv = (rr_std / rr_mean) * 100 # % 단위
    
    return {
        "peaks": peaks,
        "avg_hr": avg_hr,
        "rr_cv": rr_cv,
        "intervals": rr_intervals
    }

# 4. 의학적 지식 베이스 기반 추론 엔진 (3단계: Knowledge Base & Template)
def get_clinical_reasoning(main_probs, spec_probs, rr_metrics):
    """
    AI 확률값과 정량적 지표를 결합하여 의학적 근거가 포함된 설명문을 생성합니다.
    """
    reasons = []
    
    # 심방세동(AFIB) 추론 로직
    if spec_probs[0] > 0.3:
        if rr_metrics and rr_metrics['rr_cv'] > 15:
            reasons.append("심방세동(AFIB): AI가 높은 확률로 탐지했으며, 실제 측정된 RR 간격 변동률이 {:.2f}%로 매우 불규칙하여 임상적으로 일치함.".format(rr_metrics['rr_cv']))
        else:
            reasons.append("심방세동(AFIB) 의심: AI는 탐지했으나, 측정된 RR 간격은 규칙적임. P파 소실 여부에 대한 추가 확인이 필요함.")

    # 서맥/빈맥 추론 로직
    if rr_metrics:
        if rr_metrics['avg_hr'] < 60:
            reasons.append("서맥(Bradycardia): 평균 심박수가 {:.1f} BPM으로 정상 범위(60-100)보다 낮음.".format(rr_metrics['avg_hr']))
        elif rr_metrics['avg_hr'] > 100:
            reasons.append("빈맥(Tachycardia): 평균 심박수가 {:.1f} BPM으로 정상 범위보다 높음.".format(rr_metrics['avg_hr']))

    # 심근경색(MI) 추론 로직
    if main_probs[1] > 0.5:
        reasons.append("심근경색(MI) 의심: AI가 ST-T 파형의 비정상적 변화를 감지함. 효소 검사 및 임상 증상 확인 권장.")

    return reasons

# 5. LLM 연계 리포트 생성 엔진 (4단계: LLM-Linked Report)
def generate_llm_report(record_id, findings, clinical_reasons, rr_metrics):
    """
    추출된 모든 정보를 종합하여 LLM(예: GPT-4)이 작성한 듯한 자연스러운 서술형 리포트를 생성합니다.
    (현재는 시뮬레이션된 템플릿 사용)
    """
    summary = f"본 환자(Record: {record_id})의 ECG 분석 결과, "
    
    if len(findings) == 1 and "정상 동리듬(Normal Sinus Rhythm) 가능성 높음" in findings[0]:
        summary += "전반적으로 특이 소견이 없는 정상적인 심박동을 보이고 있습니다. "
    else:
        summary += f"총 {len(findings)}가지의 주요 소견이 관찰됩니다. "
        
    if rr_metrics:
        summary += f"평균 심박수는 {rr_metrics['avg_hr']:.1f} BPM이며, "
        if rr_metrics['rr_cv'] > 15:
            summary += "RR 간격이 매우 불규칙하여 심방세동과 같은 부정맥 가능성을 시사합니다. "
        else:
            summary += "심박 리듬은 비교적 규칙적입니다. "

    summary += "\n\n[전문 판독 제언]\n"
    if clinical_reasons:
        for reason in clinical_reasons:
            summary += f"- {reason}\n"
    else:
        summary += "- AI 탐지 결과와 정량적 지표 사이에 특이한 불일치가 없으며 안정적인 상태임.\n"
        
    summary += "\n[임상적 권고]\n상기 소견을 종합할 때, 환자의 실제 임상 증상(두근거림, 가슴 통증 등)과 대조하여 분석하시기 바라며, 필요시 24시간 홀터 모니터링 검사를 고려할 것을 권장합니다."
    
    return summary

# 6. 데이터 다운로드 및 진단 실행
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

    # 2단계: 신호 정량 분석 추가
    rr_metrics = analyze_rr_intervals(signal, fs=100)

    spec_probs_np = spec_probs.detach().cpu().numpy()[0]
    
    # 3단계: 지식 베이스 기반 추론 엔진 실행
    clinical_reasons = get_clinical_reasoning(main_probs, spec_probs_np, rr_metrics)

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

    # 4단계: LLM 연계 서술형 리포트 생성
    llm_narrative = generate_llm_report(record_id, findings, clinical_reasons, rr_metrics)

    # 리포트 텍스트 구성
    report_lines = []
    report_lines.append("="*50)
    report_lines.append("        AI ECG ANALYSIS CLINICAL REPORT")
    report_lines.append("="*50)
    report_lines.append(f" [데이터 정보] Record ID: {record_id}_lr")
    report_lines.append(f" [분석 시간] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (KST)")
    report_lines.append("-" * 50)

    report_lines.append(" [1. AI 전문의 분석 요약 (Expert Interpretation)]")
    report_lines.append(llm_narrative)
    report_lines.append("-" * 50)

    report_lines.append(" [2. 신호 정량 분석 (Signal Metrics)]")
    if rr_metrics:
        report_lines.append(f"  - 평균 심박수: {rr_metrics['avg_hr']:.1f} BPM")
        report_lines.append(f"  - RR 간격 변동률(CV): {rr_metrics['rr_cv']:.2f}%")
        status = "불규칙(Irregular)" if rr_metrics['rr_cv'] > 15 else "규칙(Regular)"
        report_lines.append(f"  - 리듬 상태: {status}")
    else:
        report_lines.append("  - R-peak 탐지 실패 (분석 불가)")
    report_lines.append("-" * 50)

    report_lines.append(" [3. 종합 판독 소견 (Clinical Impression)]")
    if findings:
        for i, find in enumerate(findings):
            report_lines.append(f"  {i+1}. {find}")
    else:
        report_lines.append("  - 특이 소견 없음")
    report_lines.append("-" * 50)

    report_lines.append(" [4. 의학적 추론 및 근거 (Clinical Reasoning)]")
    if clinical_reasons:
        for i, reason in enumerate(clinical_reasons):
            report_lines.append(f"  ● {reason}")
    else:
        report_lines.append("  - 추가적인 추론 근거 없음")
    report_lines.append("-" * 50)

    report_lines.append(" [5. 상세 진단 데이터]")
    report_lines.append("  <일반 진단 (PTB-XL 5개 대분류)>")
    for cls, prob in zip(main_classes, main_probs):
        indicator = "🔴" if prob > 0.5 else "⚪"
        report_lines.append(f"   {indicator} {cls:5}: {prob*100:6.2f}%")

    report_lines.append("\n  <부정맥 정밀 분석 (Arrhythmia Specialist)>")
    for cls, prob in zip(spec_classes, spec_probs_np):
        indicator = "🔶" if prob > 0.3 else "⚪"
        report_lines.append(f"   {indicator} {cls:5}: {prob*100:6.2f}%")
    report_lines.append("-" * 50)

    report_lines.append(f" [6. XAI 분석 (Grad-CAM)]")
    report_lines.append(f"  - 타겟 클래스: {target_class_name}")
    report_lines.append(f"  - 이미지('ecg_plot.png')의 붉은 하이라이트 구간이 {target_class_name} 판독의 주요 근거임.")
    report_lines.append("-" * 50)

    report_lines.append(" [7. 의학적 주의사항]")


    report_lines.append("  ※ 본 리포트는 AI 모델의 분석 결과이며 전문의의 최종 판독을")
    report_lines.append("     대체할 수 없습니다. 임상적 결정 전 반드시 전문가와 상의하십시오.")
    report_lines.append("="*50)

    # 리포트 출력 및 저장
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    with open("diagnosis_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n📄 진단 리포트가 'diagnosis_report.txt'로 저장되었습니다.")

    # 4. 시각화 (Grad-CAM 및 R-peak 표시)
    plt.figure(figsize=(12, 8))
    for i in range(3):
        ax = plt.subplot(3, 1, i+1)
        ax.plot(signal[:, i], color='black', linewidth=0.8, label=f'Lead {record.sig_name[i]}')
        
        # R-peak 표시 (주로 리드 II인 1번 인덱스에 표시)
        if rr_metrics and i == 1:
            ax.plot(rr_metrics['peaks'], signal[rr_metrics['peaks'], i], "ro", markersize=4, label='R-peak')
        
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
