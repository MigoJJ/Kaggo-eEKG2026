package emr.ekg.signal.preprocess;

/** 선형보간 리샘플러. 원본 fs를 목표 fs로 표준화한다(MIT-BIH 360Hz → 500Hz 등). */
final class Resampler {

    private Resampler() {
    }

    static double[] resample(double[] signal, int originalFs, int targetFs) {
        if (originalFs == targetFs) {
            return signal.clone();
        }
        int originalN = signal.length;
        double durationSec = (double) originalN / originalFs;
        int targetN = Math.max(1, (int) Math.round(durationSec * targetFs));
        double ratio = (double) (originalN - 1) / Math.max(1, targetN - 1);

        double[] out = new double[targetN];
        for (int i = 0; i < targetN; i++) {
            double srcPos = i * ratio;
            int idx0 = (int) Math.floor(srcPos);
            int idx1 = Math.min(idx0 + 1, originalN - 1);
            double frac = srcPos - idx0;
            out[i] = signal[idx0] * (1 - frac) + signal[idx1] * frac;
        }
        return out;
    }
}
