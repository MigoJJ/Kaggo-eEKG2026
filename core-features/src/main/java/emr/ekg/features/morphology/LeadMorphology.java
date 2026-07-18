package emr.ekg.features.morphology;

import emr.ekg.features.ClinicalConstants;

/**
 * 공유 QRS/ST/T 시간창(FiducialExtractor 산출)을 특정 리드에 적용해 측정한 형태학적 값.
 * 여러 심박이 검출된 경우 중앙값으로 대표값을 산출한다.
 *
 * @param leadName      리드 명칭
 * @param rAmplitudeMv  QRS 구간 내 최대 양(+) 진폭(PQ접합부 기준, 0 이상)
 * @param sAmplitudeMv  QRS 구간 내 최대 음(-) 진폭의 절대값(0 이상)
 * @param qDurationMs   QRS 시작부 초기 음성 편위(Q파) 폭
 * @param qDepthRatio   Q파 깊이 / R파 높이 비율 (병적 Q 판정용)
 * @param stDeviationMv J-point 이후 40ms 구간의 PQ접합부 대비 평균 편위
 * @param tAmplitudeMv  T파 정점 진폭(부호 포함, PQ접합부 기준)
 */
public record LeadMorphology(
        String leadName,
        double rAmplitudeMv,
        double sAmplitudeMv,
        double qDurationMs,
        double qDepthRatio,
        double stDeviationMv,
        double tAmplitudeMv) {

    public boolean isPathologicQ() {
        return qDurationMs >= ClinicalConstants.Q_PATHOLOGIC_DURATION_MS
                && qDepthRatio >= ClinicalConstants.Q_PATHOLOGIC_R_RATIO;
    }

    /** 심전기축 계산에 사용하는 QRS 순전위 근사치 (R - S). */
    public double netDeflectionMv() {
        return rAmplitudeMv - sAmplitudeMv;
    }
}
