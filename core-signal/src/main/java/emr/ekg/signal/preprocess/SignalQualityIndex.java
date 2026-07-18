package emr.ekg.signal.preprocess;

/**
 * 리드별 신호품질지수(SQI) 산출. flatline(무변화)·포화(clipping)·과잉 고주파 노이즈를
 * 검출하여 0.0(판독불가)~1.0(최상) 점수를 산출한다. 1단계(정상 사전선별) 진입 전
 * 판독불가 신호를 반려하는 근거로 사용된다.
 */
final class SignalQualityIndex {

    private SignalQualityIndex() {
    }

    static double[] perLeadScore(double[][] leads) {
        double[] scores = new double[leads.length];
        for (int i = 0; i < leads.length; i++) {
            scores[i] = scoreOne(leads[i]);
        }
        return scores;
    }

    private static double scoreOne(double[] signal) {
        double flatlinePenalty = isFlatline(signal) ? 1.0 : 0.0;
        double saturationPenalty = saturationRatio(signal);
        double noisePenalty = highFrequencyNoiseRatio(signal);

        double score = 1.0 - flatlinePenalty - saturationPenalty - noisePenalty;
        return Math.max(0.0, Math.min(1.0, score));
    }

    private static boolean isFlatline(double[] signal) {
        double mean = mean(signal);
        double variance = 0;
        for (double v : signal) {
            variance += (v - mean) * (v - mean);
        }
        variance /= signal.length;
        return variance < 1e-8;
    }

    private static double saturationRatio(double[] signal) {
        double max = Double.NEGATIVE_INFINITY;
        double min = Double.POSITIVE_INFINITY;
        for (double v : signal) {
            max = Math.max(max, v);
            min = Math.min(min, v);
        }
        double range = max - min;
        if (range < 1e-9) {
            return 0.0;
        }
        double railHigh = max - range * 0.01;
        double railLow = min + range * 0.01;
        int railed = 0;
        for (double v : signal) {
            if (v >= railHigh || v <= railLow) {
                railed++;
            }
        }
        double ratio = (double) railed / signal.length;
        return ratio > 0.3 ? ratio : 0.0;
    }

    private static double highFrequencyNoiseRatio(double[] signal) {
        double diffEnergy = 0;
        double signalEnergy = 0;
        for (int i = 1; i < signal.length; i++) {
            double d = signal[i] - signal[i - 1];
            diffEnergy += d * d;
            signalEnergy += signal[i] * signal[i];
        }
        if (signalEnergy < 1e-12) {
            return 0.0;
        }
        double ratio = diffEnergy / signalEnergy;
        return ratio > 2.0 ? Math.min(1.0, (ratio - 2.0) / 5.0) : 0.0;
    }

    private static double mean(double[] signal) {
        double sum = 0;
        for (double v : signal) {
            sum += v;
        }
        return sum / signal.length;
    }
}
