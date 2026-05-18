import streamlit as st
import os
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import sqlite3
import tempfile
from signal_restoration import ECGDigitizerV2, get_digitizer_disclaimer
from diagnose_online import (
    load_models, preprocess_signal, analyze_rr_intervals, 
    get_clinical_reasoning, generate_llm_report, load_thresholds
)

# 페이지 설정
st.set_page_config(
    page_title="AI EKG Clinical Dashboard",
    page_icon="❤️",
    layout="wide"
)

# DB 연결 함수
def get_db_connection():
    conn = sqlite3.connect("ecg_history.db")
    return conn

# 분석 결과 저장 함수
def save_to_db(data):
    conn = get_db_connection()
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
        data['date'],
        data['avg_hr'],
        data['rr_cv'],
        data['findings'],
        data['report']
    ))
    conn.commit()
    conn.close()

# 메인 UI
def main():
    st.title("❤️ AI 12-Lead EKG Clinical Dashboard")
    st.markdown("""
    이 대시보드는 PDF 형식의 심전도 리포트를 디지털 신호로 복원하고, AI 모델을 통해 부정맥 및 심혈관 질환을 분석합니다.
    최종 리포트는 **Gemini 2.5 Flash**를 통해 전문 의학 리포트 형식으로 생성됩니다.
    """)

    # 사이드바: 파일 업로드 및 기록
    st.sidebar.header("📥 데이터 입력")
    uploaded_file = st.sidebar.file_uploader("EKG PDF 파일을 업로드하세요", type=["pdf"])
    
    st.sidebar.divider()
    st.sidebar.header("📜 분석 이력")
    try:
        conn = get_db_connection()
        history_df = pd.read_sql_query("SELECT id, filename, analysis_date, avg_hr FROM ecg_analyses ORDER BY id DESC", conn)
        conn.close()
        if not history_df.empty:
            selected_id = st.sidebar.selectbox("과거 분석 결과 선택", history_df['id'].tolist(), 
                                               format_func=lambda x: f"ID {x}: {history_df[history_df['id']==x]['filename'].values[0]}")
            if st.sidebar.button("결과 불러오기"):
                st.session_state.selected_history_id = selected_id
        else:
            st.sidebar.info("저장된 분석 이력이 없습니다.")
    except Exception:
        st.sidebar.info("데이터베이스를 초기화 중입니다...")

    # 메인 분석 섹션
    if uploaded_file is not None:
        if st.sidebar.button("분석 시작"):
            with st.spinner("🚀 심전도 디지털화 및 AI 진단 중..."):
                # 임시 파일 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    # 1. 디지털화
                    digitizer = ECGDigitizerV2()
                    multi_lead_signal, q_score = digitizer.process(tmp_path)
                    
                    # 2. 모델 추론
                    models, _, device = load_models()
                    
                    # 3. 하이브리드 지표 분석 (진단 + 심박수 + 임상지표 동시 처리)
                    analysis_res = analyze_rr_intervals(multi_lead_signal, fs=100, models=models)
                    
                    if analysis_res:
                        main_probs = analysis_res['main_probs']
                        rr_metrics = analysis_res
                    else:
                        st.error("❌ 신호 분석에 실패했습니다.")
                        st.stop()
                    
                    # 4. 의학적 추론
                    thresholds = {"MI": 0.5, "AFIB": 0.3}
                    # spec_probs는 하이브리드 세션에서 확장 가능하나 현재는 placeholder
                    spec_probs_placeholder = np.zeros(6) 
                    structured_findings = get_clinical_reasoning(main_probs, spec_probs_placeholder, rr_metrics, thresholds)
                    
                    # 5. Gemini 리포트 생성
                    llm_report = generate_llm_report(uploaded_file.name, structured_findings)
                    
                    # 6. 결과 정리
                    st.session_state.analysis_result = {
                        "filename": uploaded_file.name,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "signal": multi_lead_signal,
                        "q_score": q_score,
                        "rr_metrics": rr_metrics,
                        "findings_list": [f["Finding"] for f in structured_findings["Emergency"] + structured_findings["High-Risk"]],
                        "report": llm_report,
                        "avg_hr": rr_metrics['avg_hr'] if rr_metrics else 0,
                        "rr_cv": rr_metrics['rr_cv'] if rr_metrics else 0
                    }
                    
                    # DB 저장
                    save_to_db({
                        "filename": uploaded_file.name,
                        "date": st.session_state.analysis_result["date"],
                        "avg_hr": st.session_state.analysis_result["avg_hr"],
                        "rr_cv": st.session_state.analysis_result["rr_cv"],
                        "findings": ", ".join(st.session_state.analysis_result["findings_list"]),
                        "report": llm_report
                    })
                    
                    st.success("✅ 분석 완료!")
                except Exception as e:
                    st.error(f"❌ 분석 중 오류 발생: {e}")
                finally:
                    os.unlink(tmp_path)

    # 결과 디스플레이
    if "analysis_result" in st.session_state:
        res = st.session_state.analysis_result
        
        # 상단 요약 카드
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("평균 심박수 (HR)", f"{res['avg_hr']:.1f} BPM")
        with col2:
            st.metric("리듬 변동성 (CV)", f"{res['rr_cv']:.2f}%")
        with col3:
            status = "위험" if res['findings_list'] else "정상"
            st.metric("상태 요약", status)
        with col4:
            st.metric("복원 신뢰도", f"{res['q_score']['overall_confidence']:.2f}")

        # 탭 구성
        tab1, tab2, tab3 = st.tabs(["📈 심전도 파형", "📝 AI 전문 리포트", "🔍 상세 품질 지표"])
        
        with tab1:
            st.subheader("Restored 12-Lead ECG Signal")
            # Plotly를 이용한 리드별 그래프
            leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            fig = go.Figure()
            for i in range(12):
                # 가독성을 위해 리드별 오프셋 추가
                fig.add_trace(go.Scatter(y=res['signal'][:, i] - (i * 3), name=f"Lead {leads[i]}", mode='lines'))
            
            fig.update_layout(
                height=800,
                title="12-Lead Restored Signal (Stacked View)",
                xaxis_title="Time (Samples @100Hz)",
                yaxis_title="Amplitude (normalized + offset)",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Gemini AI Clinical Interpretation")
            st.info("본 리포트는 AI 모델의 분석 결과이며 의학적 판단의 보조 자료로만 사용하십시오.")
            st.markdown(res['report'])
            
            # 다운로드 버튼
            st.download_button("리포트 다운로드 (.txt)", res['report'], file_name=f"{res['filename']}_report.txt")

        with tab3:
            st.subheader("Digitization Quality Report")
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                st.json(res['q_score'])
            with col_q2:
                st.write("**시스템 고지 사항:**")
                st.write(get_digitizer_disclaimer())

    # 과거 기록 불러오기 로직
    if "selected_history_id" in st.session_state:
        conn = get_db_connection()
        record = pd.read_sql_query(f"SELECT * FROM ecg_analyses WHERE id = {st.session_state.selected_history_id}", conn).iloc[0]
        conn.close()
        
        st.divider()
        st.subheader(f"📜 과거 기록 조회: {record['filename']} ({record['analysis_date']})")
        col_h1, col_h2 = st.columns([1, 2])
        with col_h1:
            st.write(f"**평균 심박수:** {record['avg_hr']:.1f} BPM")
            st.write(f"**리듬 변동성:** {record['rr_cv']:.2f}%")
            st.write(f"**주요 소견:** {record['findings']}")
        with col_h2:
            st.markdown(record['report_text'])
        
        if st.button("과거 기록 닫기"):
            del st.session_state.selected_history_id
            st.rerun()

if __name__ == "__main__":
    main()

    main()
