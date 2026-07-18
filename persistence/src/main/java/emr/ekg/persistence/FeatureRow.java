package emr.ekg.persistence;

/** ecg_feature 테이블 한 행 (ClinicalFeature의 평면화된 표현). */
public record FeatureRow(
        String recordId,
        int rank,
        String name,
        double value,
        String unit,
        double normalLow,
        double normalHigh) {
}
