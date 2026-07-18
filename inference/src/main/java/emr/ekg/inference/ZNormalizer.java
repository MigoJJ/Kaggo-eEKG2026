package emr.ekg.inference;

/**
 * 모델 입력 직전 리드별 Z-정규화(평균0/표준편차1). Java {@code EcgPreprocessor}(0단계)는
 * 룰엔진의 절대 임계치(mV) 적용을 위해 정규화 없이 mV를 유지하므로, 정규화는 추론 직전
 * 여기서 별도로 적용한다. Python 학습 파이프라인의 {@code ecgml.preprocess.znormalize_per_lead}와
 * 동일한 공식(표준편차 하한 1e-9)을 사용해 train/serve 일치를 보장한다.
 */
public final class ZNormalizer {

    private static final double STD_FLOOR = 1e-9;

    private ZNormalizer() {
    }

    public static float[][] normalize(double[][] samples) {
        float[][] out = new float[samples.length][];
        for (int i = 0; i < samples.length; i++) {
            out[i] = normalizeLead(samples[i]);
        }
        return out;
    }

    private static float[] normalizeLead(double[] lead) {
        double mean = 0;
        for (double v : lead) {
            mean += v;
        }
        mean /= lead.length;

        double variance = 0;
        for (double v : lead) {
            variance += (v - mean) * (v - mean);
        }
        variance /= lead.length;
        double std = Math.sqrt(variance);
        if (std < STD_FLOOR) {
            std = 1.0;
        }

        float[] out = new float[lead.length];
        for (int i = 0; i < lead.length; i++) {
            out[i] = (float) ((lead[i] - mean) / std);
        }
        return out;
    }
}
