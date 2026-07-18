package emr.ekg.features.fiducial;

/**
 * QRS onset 이전 PR 구간에서 P파(시작/정점/끝)를 탐색한다.
 * 국소 등전위선(직전 PQ 접합부) 대비 진폭이 노이즈 플로어 이하면 P파 미검출로 처리한다.
 */
final class PWaveDetector {

    private static final double SEARCH_START_SEC = 0.28;
    private static final double SEARCH_END_MARGIN_SEC = 0.02;
    private static final double NOISE_FLOOR_MV = 0.02;
    private static final double EDGE_RATIO = 0.15;
    private static final int BASELINE_SAMPLES = 5;

    private PWaveDetector() {
    }

    /** @return {onset, offset, peak}; P파 미검출 시 {-1, -1, -1} */
    static int[] detect(double[] signal, int qrsOnset, int previousQrsOffset, int fs) {
        int searchStart = Math.max(previousQrsOffset + 1, qrsOnset - (int) Math.round(SEARCH_START_SEC * fs));
        int searchEnd = qrsOnset - (int) Math.round(SEARCH_END_MARGIN_SEC * fs);
        if (searchEnd <= searchStart) {
            return new int[] {-1, -1, -1};
        }

        double baseline = mean(signal, searchStart, Math.min(searchStart + BASELINE_SAMPLES, searchEnd));

        int peak = searchStart;
        double peakVal = signal[searchStart] - baseline;
        for (int i = searchStart; i < searchEnd; i++) {
            double v = signal[i] - baseline;
            if (v > peakVal) {
                peakVal = v;
                peak = i;
            }
        }
        if (peakVal < NOISE_FLOOR_MV) {
            return new int[] {-1, -1, -1};
        }

        double edgeThreshold = peakVal * EDGE_RATIO;
        int onset = searchStart;
        for (int i = peak; i >= searchStart; i--) {
            if (signal[i] - baseline < edgeThreshold) {
                onset = i;
                break;
            }
        }
        int offset = searchEnd - 1;
        for (int i = peak; i < searchEnd; i++) {
            if (signal[i] - baseline < edgeThreshold) {
                offset = i;
                break;
            }
        }

        return new int[] {onset, offset, peak};
    }

    private static double mean(double[] signal, int from, int to) {
        if (to <= from) {
            return signal[Math.max(0, Math.min(from, signal.length - 1))];
        }
        double sum = 0;
        int count = 0;
        for (int i = from; i < to; i++) {
            sum += signal[i];
            count++;
        }
        return sum / count;
    }
}
