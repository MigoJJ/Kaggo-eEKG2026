package emr.ekg.signal;

/**
 * 0단계 전처리 결과. 12리드 신호를 500Hz로 리샘플한 뒤 대역통과·notch·기선보정을 거친 상태.
 * 물리단위(mV)를 유지하므로 룰엔진의 절대 임계치(mm/mV 기반 AHA/ACC/ESC 기준)를 그대로 적용할 수 있다.
 * Z-정규화 등 모델 입력 변환은 별도(추론 직전)로 적용한다.
 *
 * @param samples   [12][N] 리드별 신호 (mV)
 * @param fs        샘플링 주파수(Hz), 표준 500
 * @param leadNames 리드 명칭 (I, II, III, aVR, aVL, aVF, V1..V6)
 * @param sqi       신호품질지수 0.0~1.0 (1.0 = 최상)
 * @param leadOk    리드별 사용가능 여부 (리드 탈락/포화 검출)
 */
public record PreprocessedEcg(
        double[][] samples,
        int fs,
        String[] leadNames,
        double sqi,
        boolean[] leadOk) {

    public static final String[] STANDARD_12_LEADS = {
            "I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6"
    };

    public int leadCount() {
        return samples.length;
    }

    public int sampleCount() {
        return samples.length == 0 ? 0 : samples[0].length;
    }

    /** SQI가 임계 미만이면 판독불가로 반려. */
    public boolean isInterpretable(double sqiThreshold) {
        return sqi >= sqiThreshold;
    }
}
