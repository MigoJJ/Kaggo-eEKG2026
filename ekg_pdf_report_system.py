import os
import torch
import numpy as np
import sqlite3
from datetime import datetime
from ecg_digitizer_proto import process_ecg_document
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
    
    # 1. PDF에서 12리드 디지털 신호 추출
    try:
        multi_lead_signal = process_ecg_document(pdf_path) # (1000, 12)
        print(f"✅ Digitization Complete. Signal shape: {multi_lead_signal.shape}")
    except Exception as e:
        print(f"❌ Digitization Failed: {e}")
        return

    # 2. 모델 로드 및 추론
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main_model, specialist, device = load_models()
    input_tensor = preprocess_signal(multi_lead_signal).to(device)
    
    with torch.no_grad():
        main_out = main_model(input_tensor)
        main_probs = torch.sigmoid(main_out).cpu().numpy()[0]
        spec_out = specialist(input_tensor)
        spec_probs_np = torch.sigmoid(spec_out).cpu().numpy()[0]

    # 3. 정량 지표 분석
    rr_metrics = analyze_rr_intervals(multi_lead_signal, fs=100)

    # 4. 임상 소견 및 추론
    main_classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    spec_classes = ["AFIB", "AFLT", "SVPB", "PVC", "SVTA", "VTA"]
    findings_high = []
    for cls, prob in zip(main_classes[1:], main_probs[1:]):
        if prob > 0.7: findings_high.append(f"{cls} 의심 (강함)")
    for cls, prob in zip(spec_classes, spec_probs_np):
        if prob > 0.7: findings_high.append(f"부정맥: {cls} 가능성 높음")
    clinical_reasons = get_clinical_reasoning(main_probs, spec_probs_np, rr_metrics)

    # 5. 로컬 AI(Gemma) 판독 초안 생성
    report_id = os.path.basename(pdf_path)
    llm_draft = generate_llm_report(
        report_id, findings_high, clinical_reasons, rr_metrics, main_probs, spec_probs_np
    )

    # 6. 결과 출력 및 저장
    print("\n" + "="*50)
    print("      DIGITIZED 12-LEAD ECG CLINICAL REPORT")
    print("="*50)
    print(llm_draft)
    print("-" * 50)
    
    # DB 및 파일 저장
    db_data = {
        'filename': report_id,
        'avg_hr': rr_metrics['avg_hr'] if rr_metrics else 0,
        'rr_cv': rr_metrics['rr_cv'] if rr_metrics else 0,
        'findings': findings_high,
        'report_text': llm_draft
    }
    save_to_db("ecg_history.db", db_data)
    
    output_file = f"{report_id}_analysis.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(llm_draft)
    print(f"📄 Report saved to: {output_file}")

if __name__ == "__main__":
    target_pdf = "/home/ittia/git/Kaggo-eEKG2026/data/EKGPDFdata/ekg1.pdf"
    run_pdf_clinical_workflow(target_pdf)
