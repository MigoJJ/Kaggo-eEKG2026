import os
import torch
import numpy as np
import sqlite3
from datetime import datetime
from signal_restoration import ECGDigitizerV2, get_digitizer_disclaimer
from diagnose_online import load_models, preprocess_signal, analyze_rr_intervals, get_clinical_reasoning, generate_llm_report

def save_to_db(db_path, data):
    """
    분석 결과를 SQLite 데이터베이스에 저장합니다.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ecg_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            analysis_date TEXT,
            avg_hr REAL,
            rr_cv REAL,
            findings TEXT,
            report_text TEXT
        )
    ''')
    cursor.execute('''
        INSERT INTO ecg_analyses (filename, analysis_date, avg_hr, rr_cv, findings, report_text)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        data['filename'],
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        data['avg_hr'],
        data['rr_cv'],
        ", ".join(data['findings']),
        data['report_text']
    ))
    conn.commit()
    conn.close()
    print(f"📂 Analysis results saved to database: {db_path}")

def run_pdf_clinical_workflow(pdf_path):
    print(f"\n🚀 Starting Clinical Workflow for: {os.path.basename(pdf_path)}")
    
    # 1. PDF에서 12리드 디지털 신호 추출 (V2 Pipeline)
    try:
        digitizer = ECGDigitizerV2()
        multi_lead_signal, q_score = digitizer.process(pdf_path) # (1000, 12), dict
        print(f"✅ Digitization Complete. Overall Confidence: {q_score['overall_confidence']:.2f}")
        
        # 품질 점수 출력
        print(f"   - Lead Labels Detected: {q_score['lead_labels_detected']}/12")
        print(f"   - Grid Calibration: {q_score['grid_calibration']}")
        print(f"   - Skew Detected: {q_score['skew_detected']:.2f}°")
        
        # 품질 미달 시 차단
        if q_score['overall_confidence'] < 0.5:
            print("⚠️ Digitization quality too low for AI inference. Workflow stopped.")
            return
            
    except Exception as e:
        print(f"❌ Digitization Failed: {e}")
        return

    # 2. 모델 로드 및 추론
    models, _, _ = load_models()
    if 'v2_engine' not in models:
        print("❌ Model loading failed.")
        return

    # 3. 정량 지표 분석 및 추론 (analyze_rr_intervals에서 통합 수행)
    # 내부적으로 Softmax 적용 및 하이브리드 지표(HRV, QRS) 결합 추론 수행
    rr_metrics = analyze_rr_intervals(multi_lead_signal, fs=100, models=models)
    
    if rr_metrics is None:
        print("❌ Signal analysis failed.")
        return
        
    main_probs = rr_metrics['main_probs']

    # 4. 임상 소견 및 추론 (구조화된 JSON 생성)
    thresholds = load_thresholds()
    spec_probs_placeholder = np.zeros(6) 
    structured_findings = get_clinical_reasoning(main_probs, spec_probs_placeholder, rr_metrics, thresholds)

    # 5. Gemini AI 판독 리포트 생성
    report_id = os.path.basename(pdf_path)
    llm_draft = generate_llm_report(report_id, structured_findings)

    # 6. 실험적 연구 고지 및 품질 지표 추가
    disclaimer = get_digitizer_disclaimer()
    quality_report = f"\n[DIGITIZATION QUALITY REPORT]\n"
    for k, v in q_score.items():
        quality_report += f" - {k}: {v}\n"
    
    final_report = disclaimer + quality_report + "\n" + llm_draft

    # 7. 결과 출력 및 저장
    print("\n" + "="*50)
    print("      DIGITIZED 12-LEAD ECG CLINICAL REPORT (V2)")
    print("="*50)
    print(final_report)
    print("-" * 50)
    
    # DB 및 파일 저장
    db_data = {
        'filename': report_id,
        'avg_hr': rr_metrics['avg_hr'] if rr_metrics else 0,
        'rr_cv': rr_metrics['rr_cv'] if rr_metrics else 0,
        'findings': [f["Finding"] for f in structured_findings["Emergency"] + structured_findings["High-Risk"]],
        'report_text': final_report
    }
    save_to_db("ecg_history.db", db_data)
    
    output_file = f"{report_id}_analysis.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_report)
    print(f"📄 Report saved to: {output_file}")

if __name__ == "__main__":
    target_pdf = "/home/ittia/git/Kaggo-eEKG2026/data/EKGPDFdata/ekg1.pdf"
    run_pdf_clinical_workflow(target_pdf)
