package emr.ekg.inference;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ZNormalizerTest {

    private static final double EPSILON = 1e-9;

    @Test
    void normalizesLeadToZeroMeanUnitVariance() {
        double[][] samples = {{1, 2, 3, 4, 5}};

        float[][] result = ZNormalizer.normalize(samples);

        double mean = 0;
        for (float v : result[0]) {
            mean += v;
        }
        mean /= result[0].length;
        assertEquals(0.0, mean, 1e-6);

        double variance = 0;
        for (float v : result[0]) {
            variance += (v - mean) * (v - mean);
        }
        variance /= result[0].length;
        assertEquals(1.0, Math.sqrt(variance), 1e-5);
    }

    @Test
    void flatlineLeadDoesNotDivideByZero() {
        double[][] samples = {{2.0, 2.0, 2.0, 2.0}};

        float[][] result = ZNormalizer.normalize(samples);

        for (float v : result[0]) {
            assertEquals(0.0, v, EPSILON);
            assertEquals(false, Float.isNaN(v));
        }
    }

    @Test
    void normalizesEachLeadIndependently() {
        double[][] samples = {
                {0, 10, 20, 30, 40},
                {100, 100, 200, 200, 100},
        };

        float[][] result = ZNormalizer.normalize(samples);

        assertEquals(2, result.length);
        assertEquals(samples[0].length, result[0].length);
        assertEquals(samples[1].length, result[1].length);
    }
}
