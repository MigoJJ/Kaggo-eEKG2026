package emr.ekg.features.fiducial;

/**
 * QRS offset(J-point) 이후 T파 정점/종료를 탐색한다(QT 간격 산출용).
 * 종료점은 임계값 교차법(threshold-crossing)으로 근사한다 — 접선법(tangent method) 대비
 * 단순화된 방식이나, 자동화 QT 측정에서 널리 쓰이는 대안이다.
 */
final class TWaveDetector {

    private static final double SEARCH_MARGIN_SEC = 0.04;
    private static final double MAX_SEARCH_FRACTION_OF_RR = 0.7;
    private static final double END_THRESHOLD_RATIO = 0.2;

    private TWaveDetector() {
    }

    /** @return {peak, offset}; 탐색 불가 시 {-1, -1} */
    static int[] detect(double[] signal, int qrsOffset, int nextBoundaryIndex, int fs) {
        int searchStart = qrsOffset + (int) Math.round(SEARCH_MARGIN_SEC * fs);
        if (searchStart >= signal.length) {
            return new int[] {-1, -1};
        }
        int maxEnd = qrsOffset + (int) Math.round((nextBoundaryIndex - qrsOffset) * MAX_SEARCH_FRACTION_OF_RR);
        int searchEnd = Math.min(signal.length - 1, Math.max(searchStart, maxEnd));

        double baseline = signal[qrsOffset];
        int peak = searchStart;
        double peakAbs = Math.abs(signal[searchStart] - baseline);
        for (int i = searchStart; i <= searchEnd; i++) {
            double a = Math.abs(signal[i] - baseline);
            if (a > peakAbs) {
                peakAbs = a;
                peak = i;
            }
        }

        double sign = Math.signum(signal[peak] - baseline);
        if (sign == 0) {
            sign = 1;
        }
        double threshold = peakAbs * END_THRESHOLD_RATIO;
        int offset = -1;
        for (int i = peak; i <= searchEnd; i++) {
            double v = (signal[i] - baseline) * sign;
            if (v < threshold) {
                offset = i;
                break;
            }
        }
        // 탐색 구간 내에서 기저선 복귀(임계값 교차)를 못 찾으면 T-offset을 확정하지 않는다
        // (searchEnd로 대충 채우면 QT가 비정상적으로 길게 측정되는 오류가 실측에서 확인됨).
        if (offset < 0) {
            return new int[] {peak, -1};
        }

        return new int[] {peak, offset};
    }
}
