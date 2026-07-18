package emr.ekg.signal.preprocess;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * MedianFilter의 O(n·w) 증분 구현이, 매 샘플마다 윈도우를 통째로 재정렬하는
 * 브루트포스 구현과 비트 단위로 동일한 결과를 내는지 검증한다(성능만 바뀌고 판독 결과는
 * 절대 바뀌면 안 되므로 — 무작위/경계/실제와 유사한 스케일 입력 모두 포함).
 */
class MedianFilterTest {

    @Test
    void matchesBruteForceOnRandomSignal() {
        Random random = new Random(42);
        double[] signal = random.doubles(5000, -1.0, 1.0).toArray();
        assertArrayEquals(bruteForce(signal, 101), MedianFilter.apply(signal, 101));
        assertArrayEquals(bruteForce(signal, 301), MedianFilter.apply(signal, 301));
    }

    @Test
    void matchesBruteForceOnShortSignalShorterThanWindow() {
        Random random = new Random(7);
        double[] signal = random.doubles(37, -1.0, 1.0).toArray();
        assertArrayEquals(bruteForce(signal, 101), MedianFilter.apply(signal, 101));
    }

    @Test
    void matchesBruteForceWithDuplicateValues() {
        double[] signal = new double[500];
        Arrays.fill(signal, 0, 200, 0.0);
        Arrays.fill(signal, 200, 500, 1.0);
        assertArrayEquals(bruteForce(signal, 101), MedianFilter.apply(signal, 101));
    }

    @Test
    void matchesBruteForceWithWindowSizeOne() {
        Random random = new Random(3);
        double[] signal = random.doubles(50, -1.0, 1.0).toArray();
        assertArrayEquals(bruteForce(signal, 1), MedianFilter.apply(signal, 1));
    }

    /** 원래(교체 전) 구현: 매 샘플마다 윈도우를 복사·정렬. */
    private static double[] bruteForce(double[] signal, int windowSamples) {
        int half = windowSamples / 2;
        int n = signal.length;
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            int lo = Math.max(0, i - half);
            int hi = Math.min(n, i + half + 1);
            double[] window = Arrays.copyOfRange(signal, lo, hi);
            Arrays.sort(window);
            out[i] = window[window.length / 2];
        }
        return out;
    }
}
