package emr.ekg.signal;

/**
 * WFDB 리더가 산출하는 원신호. ADC 원시값을 게인·기선으로 물리단위(mV)까지 변환한 상태이며,
 * 리샘플·필터링 등 0단계 전처리는 아직 적용되지 않았다.
 *
 * @param samples   [nSignals][nSamples] 물리단위(mV) 신호
 * @param fs        원본 샘플링 주파수(Hz)
 * @param leadNames 리드 명칭 (WFDB 헤더의 description 필드)
 */
public record RawEcgSignal(double[][] samples, int fs, String[] leadNames) {

    public int leadCount() {
        return samples.length;
    }

    public int sampleCount() {
        return samples.length == 0 ? 0 : samples[0].length;
    }
}
