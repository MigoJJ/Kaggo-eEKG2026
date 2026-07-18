package emr.ekg.signal.preprocess;

import java.util.Arrays;

/**
 * 슬라이딩 윈도우 중앙값 필터. 기선(baseline) 추정에 사용.
 *
 * 매 샘플마다 윈도우를 통째로 복사·재정렬하는 대신, 빠지는/들어오는 원소만 정렬된 버퍼에
 * 이진탐색으로 삽입·삭제해 유지한다(O(n·w·log w) → O(n·w)). 매 시점의 윈도우 원소 집합은
 * 기존 구현과 동일하고 그 집합에서 같은 순위(size/2번째) 값을 그대로 뽑는 것이므로 산술 연산이
 * 없어 결과는 기존 구현과 비트 단위로 동일하다 — 속도만 다르다.
 */
final class MedianFilter {

    private MedianFilter() {
    }

    static double[] apply(double[] signal, int windowSamples) {
        int half = windowSamples / 2;
        int n = signal.length;
        double[] out = new double[n];

        double[] buf = new double[windowSamples];
        int size = 0;
        int prevLo = 0;
        int prevHi = 0;

        for (int i = 0; i < n; i++) {
            int lo = Math.max(0, i - half);
            int hi = Math.min(n, i + half + 1);

            for (int k = prevLo; k < lo; k++) {
                size = removeSorted(buf, size, signal[k]);
            }
            for (int k = prevHi; k < hi; k++) {
                size = insertSorted(buf, size, signal[k]);
            }

            out[i] = buf[size / 2];
            prevLo = lo;
            prevHi = hi;
        }
        return out;
    }

    private static int insertSorted(double[] buf, int size, double value) {
        int idx = Arrays.binarySearch(buf, 0, size, value);
        if (idx < 0) {
            idx = -(idx + 1);
        }
        System.arraycopy(buf, idx, buf, idx + 1, size - idx);
        buf[idx] = value;
        return size + 1;
    }

    private static int removeSorted(double[] buf, int size, double value) {
        int idx = Arrays.binarySearch(buf, 0, size, value);
        System.arraycopy(buf, idx + 1, buf, idx, size - idx - 1);
        return size - 1;
    }
}
