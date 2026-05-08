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
THRESHOLDS_PATH = os.path.join(BASE_DIR, 'kaggle/working/final_delivery/model_weights/thresholds.json')
ARRHYTHMIA_WEIGHTS_PATH = os.path.join(BASE_DIR, 'runs/arrhythmia_specialist/arrhythmia_best.pt')

if CODE_PATH not in sys.path:
    sys.path.append(CODE_PATH)

from ecg_training.models import PTBXLClassifier, ArrhythmiaSpecialist

def load_thresholds():
    """학습 과정에서 생성된 thresholds.json을 로드합니다."""
    default_thresholds = {"NORM": 0.5, "MI": 0.5, "STTC": 0.5, "CD": 0.5, "HYP": 0.5}
    if os.path.exists(THRESHOLDS_PATH):
        try:
            import json
            with open(THRESHOLDS_PATH, 'r') as f:
                data = json.load(f)
                print(f"✅ 학습된 임계값 로드 완료: {THRESHOLDS_PATH}")
                return data.get("thresholds", default_thresholds)
        except Exception as e:
            print(f"⚠️ 임계값 로드 실패 ({e}), 기본값(0.5)을 사용합니다.")
    else:
        print("💡 thresholds.json이 없어 기본 임계값(0.5)을 사용합니다.")
    return default_thresholds

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
def get_clinical_reasoning(main_probs, spec_probs, rr_metrics, thresholds):
    """
    AI 확률값과 정량적 지표를 결합하여 구조화된 의학적 근거를 생성합니다.
    """
    structured_findings = {
        "Emergency": [],
        "High-Risk": [],
        "Non-Urgent": [],
        "Evidence": {
            "HR": rr_metrics['avg_hr'] if rr_metrics else "N/A",
            "RR_CV": rr_metrics['rr_cv'] if rr_metrics else "N/A",
            "Confidence_Scores": {
                "MI": float(main_probs[1]),
                "AFIB": float(spec_probs[0])
            }
        },
        "Metadata": {
            "Threshold_Source": "PTB-XL/Arrhythmia Specialist Baseline",
            "Threshold_Values": thresholds
        }
    }
    
    # 1. Emergency Findings
    # (VTA 등은 PTB-XL 데이터셋 부족으로 Specialist에서 제외됨)
    
    if rr_metrics and rr_metrics['avg_hr'] < 40:
        structured_findings["Emergency"].append({
            "Finding": "심한 서맥(Severe Bradycardia)",
            "Reason": f"평균 심박수가 {rr_metrics['avg_hr']:.1f} BPM으로 매우 낮아 순환 부전 위험이 있음.",
            "Confidence": 1.0
        })

    # 2. High-Risk Findings
    if main_probs[1] > 0.5: # MI
        structured_findings["High-Risk"].append({
            "Finding": "심근경색(MI) 가능성",
            "Reason": "ST-T 파형의 허혈성 변화가 감지됨. 급성 관상동맥 증후군 배제 필요.",
            "Confidence": float(main_probs[1])
        })
    
    if spec_probs[0] > 0.3: # AFIB
        if rr_metrics and rr_metrics['rr_cv'] > 15:
            structured_findings["High-Risk"].append({
                "Finding": "심방세동(AFIB)",
                "Reason": f"AI 탐지 결과와 RR 간격의 불규칙성(CV: {rr_metrics['rr_cv']:.2f}%)이 일치함.",
                "Confidence": float(spec_probs[0])
            })
        else:
            structured_findings["High-Risk"].append({
                "Finding": "심방세동(AFIB) 의심",
                "Reason": "AI는 AFIB를 탐지했으나 RR 간격은 비교적 규칙적임. P파 소실 여부 확인 필요.",
                "Confidence": float(spec_probs[0])
            })

    # 3. Non-Urgent Findings
    if rr_metrics:
        if 40 <= rr_metrics['avg_hr'] < 60:
            structured_findings["Non-Urgent"].append({
                "Finding": "경증 서맥(Mild Bradycardia)",
                "Reason": "정상 범위보다 다소 낮은 심박수이나 즉각적인 처치는 불필요할 수 있음.",
                "Confidence": 0.8
            })
        elif rr_metrics['avg_hr'] > 100:
            structured_findings["Non-Urgent"].append({
                "Finding": "빈맥(Tachycardia)",
                "Reason": "심박수가 다소 높음. 생리적 요인 또는 긴장 확인 요망.",
                "Confidence": 0.8
            })

    return structured_findings

# 5. LLM 연계 리포트 생성 엔진 (4단계: LLM-Linked Report)
def generate_llm_report(record_id, structured_findings):
    """
    구조화된 진단 결과를 바탕으로 EMR 형식의 요약문을 생성합니다.
    LLM은 새로운 진단을 내리지 않고, 전달된 JSON의 논리적 순서만 정리합니다.
    """
    # 1. 시스템 인스트럭션 (제한된 어시스턴트 역할)
    system_instruction = """
    당신은 심장 전문의를 보조하여 구조화된 분석 결과를 EMR(Electronic Medical Record) 문장으로 정리하는 어시스턴트입니다.
    
    [핵심 지침]
    1. 제공된 JSON 데이터('structured_findings')에 포함된 소견만 서술하십시오.
    2. 절대 새로운 진단을 추측하거나 hallucination을 생성하지 마십시오.
    3. 결과는 전문적이고 건조한 의학적 톤(EMR style)으로 작성하십시오.
    4. Emergency -> High-Risk -> Non-Urgent 순서로 중요도를 반영하여 요약하십시오.
    5. 'Evidence' 섹션의 수치(HR, RR_CV)를 소견의 근거로 적절히 인용하십시오.
    """

    # 2. 데이터 컨텍스트 준비
    import json
    data_context = {
        "record_id": record_id,
        "structured_findings": structured_findings
    }

    # 3. LLM API 호출 (생략 가능, 여기서는 프롬프트 구성만 보여줌)
    # 실제 구현 시 Ollama 또는 Gemini API 사용
    
    # 시뮬레이션된 안전한 리포트 (Fallback & Target Style)
    summary = f"""
[EMR SUMMARY - RECORD {record_id}]

1. 주요 소견 및 응급도 (Clinical Impression):
"""
    if structured_findings["Emergency"]:
        summary += "  - [EMERGENCY]: " + ", ".join([f["Finding"] for f in structured_findings["Emergency"]]) + "\n"
    if structured_findings["High-Risk"]:
        summary += "  - [HIGH-RISK]: " + ", ".join([f["Finding"] for f in structured_findings["High-Risk"]]) + "\n"
    if not (structured_findings["Emergency"] or structured_findings["High-Risk"]):
        summary += "  - 특이 소견 없음\n"

    summary += f"""
2. 임상적 근거 (Evidence Based):
  - 심박수(HR): {structured_findings['Evidence']['HR']} BPM
  - 리듬 규칙성(RR_CV): {structured_findings['Evidence']['RR_CV']}%
"""
    
    # 상세 서술 (LLM이 할 일의 예시)
    for cat in ["Emergency", "High-Risk", "Non-Urgent"]:
        for f in structured_findings[cat]:
            summary += f"  ● {f['Finding']}: {f['Reason']} (신뢰도: {f['Confidence']:.2f})\n"

    summary += """
3. 시스템 제약 사항 및 안내 (Disclaimers):
  - 본 분석은 AI 모델(PTB-XL Backbone + Arrhythmia Specialist)에 의한 자동 판독 결과입니다.
  - 임계값(Threshold)은 연구용 Baseline을 기준으로 설정되었으며 임상적 절대 기준이 아닙니다.
  - 데이터 누락, 신호 잡음, 또는 부정확한 디지털 복원으로 인한 판독 오류 가능성이 존재합니다.
  - 최종 진단 및 치료 결정은 반드시 담당 전문의의 육안 판독과 임상 증상을 종합하여 내려져야 합니다.
"""
    return summary.strip()

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
    
    # 임계값 로드 및 적용
    tuned_thresholds_dict = load_thresholds()
    main_classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    # 리스트 형태로 변환 (순서 유지)
    tuned_thresholds = [tuned_thresholds_dict.get(cls, 0.5) for cls in main_classes]

    # 3단계: 지식 베이스 기반 추론 엔진 실행
    structured_findings = get_clinical_reasoning(main_probs, spec_probs_np, rr_metrics, tuned_thresholds_dict)

    # 결과 해석 및 리포트 생성
    spec_classes = ["AFIB", "AFLT", "PVC"]
    target_class_name = spec_classes[target_class_idx]
    
    # 임상 소견 생성 로직 (Tuned Threshold 기반)
    findings_high = []    # 임계값 이상 (유의 소견)
    findings_medium = []  # 임계값의 70% 이상 (의심 소견)
    findings_low = []     # 그 외 (관찰 필요)

    # 1. 일반 진단 (PTB-XL) 분류
    main_desc = {"MI": "심근경색(Myocardial Infarction)", "STTC": "ST/T파 변화", 
                 "CD": "전도 장애(Conduction Disturbance)", "HYP": "비대(Hypertrophy)"}

    for i, (cls, prob) in enumerate(zip(main_classes[1:], main_probs[1:])):
        desc = main_desc.get(cls, cls)
        t = tuned_thresholds[i+1] # NORM 제외 index
        
        if prob >= t:
            findings_high.append(f"{desc}: AI 탐지 확률({prob:.2f})이 학습된 임계값({t:.2f})을 상과하여 유의함.")
        elif prob >= t * 0.7:
            findings_medium.append(f"{desc}: 임계값 부근({prob:.2f}/{t:.2f})의 변화가 관찰되어 정밀 확인 요망.")
        elif prob > 0.2:
            findings_low.append(f"{desc}: 미세한 변화({prob:.2f})가 있으나 비특이적일 수 있음.")

    # 2. 부정맥 (Specialist) 분류
    spec_desc = {"AFIB": "심방세동(Atrial Fibrillation)", "AFLT": "심방조동(Atrial Flutter)",
                 "PVC": "심실성 조기수축"}

    for cls, prob in zip(spec_classes, spec_probs_np):
        desc = spec_desc.get(cls, cls)
        if prob > 0.7:
            findings_high.append(f"부정맥: {desc} 가능성이 매우 높으며 즉각적인 임상 대응 고려.")
        elif prob > 0.4:
            findings_medium.append(f"부정맥: {desc} 의심 소견. 리듬 모니터링 필요.")
        elif prob > 0.2:
            findings_low.append(f"부정맥: {desc} 가능성을 완전히 배제할 수 없음 (감별 진단).")

    # 리포트 텍스트 구성

    # 4단계: LLM 연계 서술형 리포트 생성
    llm_narrative = generate_llm_report(record_id, structured_findings)

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
    
    report_lines.append("  <주요 소견 (High Confidence)>")
    if findings_high:
        for f in findings_high: report_lines.append(f"   ● {f}")
    else: report_lines.append("   - 해당 없음")
    
    report_lines.append("\n  <참고 소견 (Medium Confidence)>")
    if findings_medium:
        for f in findings_medium: report_lines.append(f"   ○ {f}")
    else: report_lines.append("   - 해당 없음")
    
    report_lines.append("\n  <미세 소견 및 감별 진단 (Low Confidence)>")
    if findings_low:
        for f in findings_low: report_lines.append(f"   △ {f}")
    else: report_lines.append("   - 해당 없음")
    report_lines.append("-" * 50)

    report_lines.append(" [4. 의학적 추론 및 근거 (Clinical Reasoning)]")
    # structured_findings에서 가져오기
    all_reasons = structured_findings["Emergency"] + structured_findings["High-Risk"] + structured_findings["Non-Urgent"]
    if all_reasons:
        for f in all_reasons:
            report_lines.append(f"  ● {f['Finding']}: {f['Reason']}")
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
AM 분석 결과가 포함된 ECG 파형이 'ecg_plot.png'로 저장되었습니다.")

if __name__ == "__main__":
    run_diagnosis()
