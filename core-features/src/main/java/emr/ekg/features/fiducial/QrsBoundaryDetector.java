package emr.ekg.features.fiducial;

/**
 * R-peak 주변의 QRS 시작(onset)/끝(offset, J-point) 검출.
 * 단시간 신호 에너지(제곱 미분, 20ms 평활)가 R-peak 지점 에너지 대비 임계 비율 이하로
 * 떨어지는 첫 지점을 경계로 삼는다.
 */
final class QrsBoundaryDetector {

    private static final double ENERGY_THRESHOLD_RATIO = 0.08;
    private static final double MAX_HALF_WIDTH_SEC = 0.12; // 편측 최대 120ms

    private QrsBoundaryDetector() {
    }

    /** @return {onset, offset} 샘플 인덱스 */
    static int[] onsetOffset(double[] signal, int rIndex, int fs) {
        double[] energy = shortTermEnergy(signal, fs);
        int maxHalfWidth = (int) Math.round(MAX_HALF_WIDTH_SEC * fs);
        double threshold = energy[rIndex] * ENERGY_THRESHOLD_RATIO;

        int onset = Math.max(0, rIndex - maxHalfWidth);
        for (int i = rIndex; i > onset; i--) {
            if (energy[i] < threshold) {
                onset = i;
                break;
            }
        }

        int offset = Math.min(signal.length - 1, rIndex + maxHalfWidth);
        for (int i = rIndex; i < offset; i++) {
            if (energy[i] < threshold) {
                offset = i;
                break;
            }
        }

        return new int[] {onset, offset};
    }

    private static double[] shortTermEnergy(double[] signal, int fs) {
        int n = signal.length;
        double[] diff = new double[n];
        for (int i = 1; i < n; i++) {
            diff[i] = signal[i] - signal[i - 1];
        }
        int window = Math.max(1, (int) Math.round(fs * 0.02)); // 20ms
        double[] energy = new double[n];
        double sum = 0;
        for (int i = 0; i < n; i++) {
            double sq = diff[i] * diff[i];
            sum += sq;
            if (i >= window) {
                sum -= diff[i - window] * diff[i - window];
            }
            energy[i] = sum / Math.min(i + 1, window);
        }
        return energy;
    }
}
