package emr.ekg.features;

/**
 * 20종 핵심 임상 피처 중 하나의 정량 측정 결과.
 * 정상범위(normalLow~normalHigh) 대비 in-range 여부를 함께 보관하여
 * 1단계 정상 사전선별의 룰 교차검증에 사용한다.
 *
 * @param rank       Rank Feature 번호 (1~20)
 * @param name       피처명 (예: "HR", "PR", "QRS_DURATION", "QTc", "ST_DEVIATION")
 * @param value      측정값
 * @param unit       단위 (예: "bpm", "ms", "uV", "deg")
 * @param normalLow  생리적 정상 하한
 * @param normalHigh 생리적 정상 상한
 */
public record ClinicalFeature(
        int rank,
        String name,
        double value,
        String unit,
        double normalLow,
        double normalHigh) {

    public boolean inRange() {
        return value >= normalLow && value <= normalHigh;
    }
}
