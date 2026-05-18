import torch
import torch.nn.functional as F
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import urllib.request
from datetime import datetime
from scipy.signal import find_peaks, butter, filtfilt
import google.generativeai as genai
from dotenv import load_dotenv
import neurokit2 as nk
import onnxruntime as ort

# Add current dir and core to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from core.feature_engine import ClinicalFeatureEngine

# .env 파일 로드
load_dotenv(override=True)

# 1. 경로 및 모델 설정
HYBRID_V2_ONNX_PATH = os.path.join(BASE_DIR, 'hybrid_v2_model.onnx')
U_NET_ONNX_PATH = os.path.join(BASE_DIR, 'hybrid_ekg_model.onnx') # For segmentation

def load_models():
    """하이브리드 V2 모델을 로드합니다."""
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    
    models = {}
    if os.path.exists(HYBRID_V2_ONNX_PATH):
        models['v2_engine'] = ort.InferenceSession(HYBRID_V2_ONNX_PATH, sess_options, providers=['CPUExecutionProvider'])
        print(f"✅ 하이브리드 V2 ONNX 진단 엔진 로드 완료")
    
    # 신규 6클래스 Torch 모델 체크
    v2_weights = "runs/hybrid_v2/specialist_v2.pt"
    if os.path.exists(v2_weights):
        try:
            from ecg_training.models import PTBXLClassifier, ArrhythmiaSpecialistV2
            base_v2 = PTBXLClassifier(input_leads=12, num_classes=5, embedding_dim=256)
            v2_model = ArrhythmiaSpecialistV2(backbone=base_v2.backbone, feature_dim=5, num_classes=6)
            v2_model.load_state_dict(torch.load(v2_weights, map_location='cpu'))
            v2_model.eval()
            models['v2_torch'] = v2_model
            print(f"✅ 하이브리드 V2 6클래스 Torch 모델 로드 완료")
        except Exception as e:
            print(f"⚠️ Hybrid V2 Torch Load Error: {e}")

    if os.path.exists(U_NET_ONNX_PATH):
        models['unet_engine'] = ort.InferenceSession(U_NET_ONNX_PATH, sess_options, providers=['CPUExecutionProvider'])
        print(f"✅ 심박 정밀 탐지(U-Net) 엔진 로드 완료")
        
    return models, None, torch.device('cpu')

def preprocess_signal(signal):
    # Z-score normalization per lead
    means = signal.mean(axis=0, keepdims=True)
    stds = signal.std(axis=0, keepdims=True) + 1e-7
    normalized = (signal - means) / stds
    return torch.tensor(normalized.T, dtype=torch.float32).unsqueeze(0)

def analyze_rr_intervals(signal, fs=100, models=None):
    """
    [V2 Hybrid] NeuroKit2 + UNet(AI) + Clinical Features 결합 분석
    """
    if models is None or 'v2_engine' not in models:
        return None

    # 1. 임상 지표 추출 (Clinical Feature Engine)
    engine = ClinicalFeatureEngine(sampling_rate=fs)
    raw_features = engine.extract_all(signal[:, 1]) # Lead II
    
    # 5가지 핵심 지표 추출
    feat_keys = ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "QRS_Duration"]
    features_vec = np.array([[raw_features.get(k, 0.0) for k in feat_keys]], dtype=np.float32)
    features_vec = np.nan_to_num(features_vec, nan=0.0)

    # 2. V2 하이브리드 추론 (진단)
    input_signal = preprocess_signal(signal).numpy().astype(np.float32)
    
    if 'v2_torch' in models:
        # 최신 6클래스 Torch 모델 사용
        with torch.no_grad():
            sig_t = torch.FloatTensor(input_signal)
            feat_t = torch.FloatTensor(features_vec)
            logits_t = models['v2_torch'](sig_t, feat_t)
            main_probs = torch.softmax(logits_t, dim=1).numpy()[0]
    elif 'v2_engine' in models:
        v2_outputs = models['v2_engine'].run(None, {'signal': input_signal, 'features': features_vec})
        logits = v2_outputs[0][0]
        # Softmax 적용 (CrossEntropyLoss 학습 모델)
        exp_logits = np.exp(logits - np.max(logits)) # Numerical stability
        main_probs = exp_logits / exp_logits.sum()
    else:
        main_probs = np.zeros(6)

    # 3. U-Net 기반 심박 보정 (동시 실행)
    unet_mask = np.zeros(1000)
    if 'unet_engine' in models:
        unet_out = models['unet_engine'].run(None, {'input': input_signal})
        unet_mask = unet_out[1][0, 0, :]

    # 4. 최종 피크 보정 (NeuroKit2 앙상블)
    try:
        cleaned = nk.ecg_clean(signal[:, 1], sampling_rate=fs, method="neurokit")
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")
        nk_peaks = info["ECG_R_Peaks"]
    except:
        nk_peaks = []

    final_peaks = []
    for p in nk_peaks:
        if p < len(unet_mask) and unet_mask[p] > 0.2:
            final_peaks.append(p)
            
    ai_peaks, _ = find_peaks(unet_mask, height=0.6, distance=int(0.3 * fs))
    for ap in ai_peaks:
        if not any(abs(ap - fp) < int(0.1 * fs) for fp in final_peaks):
            final_peaks.append(ap)
    final_peaks = sorted(final_peaks)

    if len(final_peaks) < 2:
        return {"avg_hr": 0, "rr_cv": 0, "main_probs": main_probs, "features": raw_features}

    rr_intervals = np.diff(final_peaks) * (1000 / fs)
    avg_hr = 60000 / np.mean(rr_intervals)
    rr_cv = (np.std(rr_intervals) / np.mean(rr_intervals)) * 100
    
    return {
        "peaks": final_peaks,
        "avg_hr": avg_hr,
        "rr_cv": rr_cv,
        "intervals": rr_intervals,
        "main_probs": main_probs,
        "features": raw_features
    }

def get_clinical_reasoning(main_probs, spec_probs, rr_metrics, thresholds):
    """
    [V2 Hybrid Reasoning] AI 확률 + HRV 지표 + 파형 분석 결합
    """
    structured_findings = {"Emergency": [], "High-Risk": [], "Non-Urgent": [], "Evidence": {}}
    feat = rr_metrics.get('features', {})
    
    # Evidence 주입
    structured_findings["Evidence"] = {
        "HR": f"{rr_metrics['avg_hr']:.1f} BPM",
        "HRV_SDNN": f"{feat.get('HRV_SDNN', 0):.2f} ms",
        "QRS_Width": f"{feat.get('QRS_Duration', 0):.1f} samples"
    }

    # 1. 고위험군 및 주요 소견 진단
    # [0:NORM, 1:MI, 2:STTC, 3:CD, 4:HYP]
    
    if main_probs[1] > 0.4: # MI
        structured_findings["High-Risk"].append({"Finding": "심근경색(MI) 의심", "Reason": "파형 분석 결과 ST-T 구간의 허혈성 변화가 지배적임."})
    
    if main_probs[2] > 0.4: # STTC
        structured_findings["Non-Urgent"].append({"Finding": "ST-T 구간 변화(STTC) 의심", "Reason": "비특이적 ST-T 구간 변화가 관찰되어 추가적인 임상 관찰이 필요함."})

    if main_probs[3] > 0.4: # CD (Conduction Disturbance)
        structured_findings["Non-Urgent"].append({"Finding": "전도 장애(CD) 의심", "Reason": "전도 시스템의 이상(예: 각차단 등)이 의심되는 파형 패턴임."})
    
    if main_probs[4] > 0.4: # HYP (Hypertrophy)
        structured_findings["Non-Urgent"].append({"Finding": "심비대(HYP) 의심", "Reason": "심실 비대를 시사하는 높은 R파 전압 등이 관찰됨."})

    # 심방세동(AFIB) 판독: 현재 AI 모델 클래스에 없으므로 임상 지표(RR-CV, SDNN)에 의존
    # RR interval의 변동계수(CV)가 15% 이상이고 SDNN이 높을 때 의심
    if rr_metrics['rr_cv'] > 15 and feat.get('HRV_SDNN', 0) > 40:
        structured_findings["High-Risk"].append({"Finding": "심방세동(AFIB) 가능성", "Reason": f"심박 변동성(RR-CV: {rr_metrics['rr_cv']:.1f}%)이 매우 높고 리듬이 불규칙하여 임상적 AFIB 패턴을 보임."})

    return structured_findings

def generate_llm_report(record_id, structured_findings):
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = f"심전도 전문 분석 리포트 작성해줘 (V2 Hybrid 엔진 결과): {structured_findings}"
            return model.generate_content(prompt).text
        except: pass
    return "Gemini 연동 실패 (시뮬레이션 모드)"

def load_thresholds():
    return {"NORM": 0.5, "MI": 0.5, "STTC": 0.5, "CD": 0.5, "HYP": 0.5, "AFIB": 0.5}

if __name__ == "__main__":
    pass
d 엔진 결과): {structured_findings}"
            return model.generate_content(prompt).text
        except: pass
    return "Gemini 연동 실패 (시뮬레이션 모드)"

def load_thresholds():
    return {"NORM": 0.5, "MI": 0.5, "STTC": 0.5, "CD": 0.5, "HYP": 0.5}

if __name__ == "__main__":
    pass
