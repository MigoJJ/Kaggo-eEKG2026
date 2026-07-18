package emr.ekg.pipeline;

/**
 * 진단 파이프라인 단계. 실행 순서를 명시한다:
 * 전처리 → 정상 우선순위 분류 → 응급 → 룰 전도장애 → Beat 부정맥 → ST 허혈 → 합성.
 *
 * NORMAL_TRIAGE는 게이팅하지 않는다 — 모든 ECG는 예외 없이 EMERGENCY 이후 단계를 전부 거친다.
 * P(NORM) 점수는 의사 검토 큐의 우선순위 정렬용 메타데이터일 뿐이다. (20개 요약 피처 기반
 * AND게이트로 자동확정을 시도했을 때 실측 79% 오탐률이 나와 게이팅을 포기했다 — ARCHITECTURE.md P2 참고.)
 */
public enum PipelineStage {
    PREPROCESS,      // 0
    NORMAL_TRIAGE,   // 1 — P(NORM) 우선순위 점수 산출 (게이팅 없음)
    EMERGENCY,       // E — STEMI/ST하강/LongQT (모든 ECG 대상)
    CONDUCTION_RULE, // 2 — RBBB/LBBB/AVB/축
    BEAT_ARRHYTHMIA, // 3 — MIT-BIH 비트 검증
    ST_ISCHEMIA,     // 4 — ST-T 허혈
    SYNTHESIS        // R — 결과 합성 & 리포트
}
