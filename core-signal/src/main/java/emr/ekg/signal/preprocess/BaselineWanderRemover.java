package emr.ekg.signal.preprocess;

/**
 * 2단 중앙값 필터 기반 기선(baseline wander) 제거.
 * 200ms 윈도우로 P/T파를 눌러 QRS 위주 신호를 만든 뒤, 600ms 윈도우로 QRS까지 눌러
 * 순수 기선을 추정하고 원신호에서 뺀다.
 */
final class BaselineWanderRemover {

    private BaselineWanderRemover() {
    }

    static double[] remove(double[] signal, int fs) {
        int w200 = toOdd(Math.max(1, (int) Math.round(fs * 0.2)));
        int w600 = toOdd(Math.max(1, (int) Math.round(fs * 0.6)));
        double[] stage1 = MedianFilter.apply(signal, w200);
        double[] baseline = MedianFilter.apply(stage1, w600);

        double[] out = new double[signal.length];
        for (int i = 0; i < signal.length; i++) {
            out[i] = signal[i] - baseline[i];
        }
        return out;
    }

    private static int toOdd(int v) {
        return (v % 2 == 0) ? v + 1 : v;
    }
}
