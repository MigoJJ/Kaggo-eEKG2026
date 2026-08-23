package emr.ekg.rules;

import emr.ekg.features.ClinicalConstants;

/**
 * E/2단계 룰엔진의 조정 가능한 임계치. {@link ClinicalConstants}의 기본값(AHA/ACC/ESC 가이드
 * 고정값)과 달리, 여기 담긴 값들은 실측 검증 결과에 따라 재보정이 필요한 파라미터다
 * (예: ischemiaStDepressionMv — Phase 6 실측에서 과발화 확인, Kaggle 학습·대규모 검증 후
 * 데이터 기반 재조정 예정). config/pipeline.yaml에서 읽어 이 레코드를 구성하면 코드 변경 없이
 * 재보정할 수 있다(backbone 목표).
 */
public record RuleThresholds(
        double stemiElevationGeneralMv,
        double stemiElevationV2V3Mv,
        int stemiContiguousLeads,
        double ischemiaDepressionMv,
        int ischemiaContiguousLeads,
        double qtcCriticalMs,
        double vtRateBpm,
        double vtQrsMs,
        double bbbQrsMs,
        double avb1PrMs) {

    public static RuleThresholds defaults() {
        return new RuleThresholds(
                ClinicalConstants.ST_ELEVATION_GENERAL_MV,
                ClinicalConstants.ST_ELEVATION_V2V3_MV,
                2,
                ClinicalConstants.ST_DEPRESSION_MV,
                2,
                ClinicalConstants.QTC_CRITICAL_MS,
                ClinicalConstants.VT_RATE_BPM,
                ClinicalConstants.VT_QRS_MS,
                ClinicalConstants.QRS_COMPLETE_BBB_MS,
                ClinicalConstants.PR_MAX_MS);
    }
}
