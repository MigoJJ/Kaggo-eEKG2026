package emr.ekg.features.fiducial;

import java.util.ArrayList;
import java.util.List;

/**
 * Pan-Tompkins 계열 R-peak 검출기: 5점 미분 → 제곱 → 이동적분 → 임계값 교차 → 원신호 역탐색.
 * 원 논문의 적응형 이중 임계값 대신 스트립 전역 최대값 기반 고정 비율 임계값을 사용한
 * 단순화 버전이다 — 10초 내외의 단일 스트립 판독에는 충분하지만, 진폭이 크게 변하는
 * 장시간 기록에는 적응형 임계값이 더 적합하다.
 */
final class RPeakDetector {

    private RPeakDetector() {
    }

    static int[] detect(double[] signal, int fs) {
        double[] derivative = derivative(signal);
        double[] squared = square(derivative);
        int windowSize = Math.max(1, (int) Math.round(fs * 0.15)); // 150ms
        double[] integrated = movingAverage(squared, windowSize);

        int refractorySamples = (int) Math.round(fs * 0.2); // 200ms 최소 RR
        List<Integer> integratedPeaks = findPeaks(integrated, refractorySamples);

        int searchRadius = (int) Math.round(fs * 0.1); // ±100ms
        List<Integer> rPeaks = new ArrayList<>();
        for (int p : integratedPeaks) {
            int lo = Math.max(0, p - searchRadius);
            int hi = Math.min(signal.length - 1, p + searchRadius);
            int best = lo;
            double bestAbs = Math.abs(signal[lo]);
            for (int i = lo; i <= hi; i++) {
                double a = Math.abs(signal[i]);
                if (a > bestAbs) {
                    bestAbs = a;
                    best = i;
                }
            }
            rPeaks.add(best);
        }

        return rPeaks.stream().mapToInt(Integer::intValue).distinct().sorted().toArray();
    }

    private static double[] derivative(double[] x) {
        int n = x.length;
        double[] y = new double[n];
        for (int i = 2; i < n - 2; i++) {
            y[i] = (-x[i - 2] - 2 * x[i - 1] + 2 * x[i + 1] + x[i + 2]) / 8.0;
        }
        return y;
    }

    private static double[] square(double[] x) {
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] * x[i];
        }
        return y;
    }

    private static double[] movingAverage(double[] x, int window) {
        int n = x.length;
        double[] y = new double[n];
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += x[i];
            if (i >= window) {
                sum -= x[i - window];
            }
            y[i] = sum / Math.min(i + 1, window);
        }
        return y;
    }

    private static List<Integer> findPeaks(double[] x, int refractorySamples) {
        double max = 0;
        for (double v : x) {
            max = Math.max(max, v);
        }
        double threshold = max * 0.25;

        List<Integer> peaks = new ArrayList<>();
        int lastPeak = -refractorySamples;
        for (int i = 1; i < x.length - 1; i++) {
            if (x[i] > threshold && x[i] >= x[i - 1] && x[i] >= x[i + 1] && i - lastPeak >= refractorySamples) {
                peaks.add(i);
                lastPeak = i;
            }
        }
        return peaks;
    }
}
