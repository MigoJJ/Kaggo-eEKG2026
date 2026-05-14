import os
import json
from diagnose_online import generate_llm_report
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 테스트용 구조화된 진단 데이터
structured_findings = {
    "Emergency": [
        {"Finding": "심한 서맥(Severe Bradycardia)", "Reason": "평균 심박수가 38 BPM으로 매우 낮아 즉각적인 조치가 필요함.", "Confidence": 1.0}
    ],
    "High-Risk": [
        {"Finding": "심근경색(MI) 의심", "Reason": "ST-T 파형의 허혈성 변화와 고위험 AI 점수 감지.", "Confidence": 0.89}
    ],
    "Non-Urgent": [
        {"Finding": "경미한 PR 간격 연장", "Reason": "일시적인 현상일 수 있으나 추적 관찰 요망.", "Confidence": 0.7}
    ],
    "Evidence": {
        "HR": 38,
        "RR_CV": 4.5,
        "Confidence_Scores": {"MI": 0.89, "AFIB": 0.05}
    }
}

print("🚀 Gemini API 연동 테스트 시작...")
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    # 마스킹하여 키 확인
    masked_key = api_key[:5] + "*" * (len(api_key) - 10) + api_key[-5:]
    print(f"🔑 API Key 감지됨: {masked_key}")
else:
    print("❌ API Key를 찾을 수 없습니다. .env 파일을 확인해주세요.")

try:
    print("\n--- Gemini 생성 리포트 ---")
    report = generate_llm_report("VERIFICATION_TEST", structured_findings)
    print(report)
    
    if "(Simulation)" in report:
        print("\n⚠️ 결과에 (Simulation)이 포함되어 있습니다. API 호출이 실패했거나 시뮬레이션 모드로 작동 중일 수 있습니다.")
    else:
        print("\n✅ 성공! Gemini가 실시간으로 리포트를 생성했습니다.")
except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
