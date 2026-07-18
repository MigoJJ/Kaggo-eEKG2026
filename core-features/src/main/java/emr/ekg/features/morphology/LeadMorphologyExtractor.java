package emr.ekg.features.morphology;

import emr.ekg.features.fiducial.BeatFiducials;
import emr.ekg.signal.PreprocessedEcg;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 기준 리드(Lead II)에서 추출한 공유 QRS/ST/T 시간창을 12리드 각각에 적용해
 * R/S/Q/ST/T 형태학적 값을 측정한다. 여러 심박이 검출된 경우 리드별로 중앙값을 취해
 * 잡음에 강건한 대표값을 산출한다.
 */
public final class LeadMorphologyExtractor {

    // J-point 직후는 QRS 말단(S파 하강)의 잔여 신호가 아직 안정화되지 않아 ST를 인위적으로
    // 낮게 측정하는 계통적 편향을 만든다(실측 확인됨) — 임상에서도 흔히 J+60~J+80 지점을
    // 쓰는 이유와 같다. J-point에서 20ms 띄운 뒤 40ms 구간을 평균해 ST 레벨을 산정한다.
    private static final int ST_OFFSET_MS = 20;
    private static final int ST_WINDOW_MS = 40;
    private static final int BASELINE_LOOKBACK_SAMPLES = 5;

    private LeadMorphologyExtractor() {
    }

    public static Map<String, LeadMorphology> extract(PreprocessedEcg ecg, List<BeatFiducials> beats) {
        Map<String, LeadMorphology> result = new LinkedHashMap<>();
        int fs = ecg.fs();
        int stOffsetSamples = Math.max(0, (int) Math.round(ST_OFFSET_MS / 1000.0 * fs));
        int stWindowSamples = Math.max(1, (int) Math.round(ST_WINDOW_MS / 1000.0 * fs));

        for (int leadIdx = 0; leadIdx < ecg.leadCount(); leadIdx++) {
            String leadName = ecg.leadNames()[leadIdx];
            double[] signal = ecg.samples()[leadIdx];

            List<Double> rAmps = new ArrayList<>();
            List<Double> sAmps = new ArrayList<>();
            List<Double> qDurations = new ArrayList<>();
            List<Double> qRatios = new ArrayList<>();
            List<Double> stDevs = new ArrayList<>();
            List<Double> tAmps = new ArrayList<>();

            for (BeatFiducials beat : beats) {
                int onset = beat.qrsOnset();
                int offset = beat.qrsOffset();
                if (onset < 0 || offset <= onset || offset >= signal.length) {
                    continue;
                }

                double baseline = mean(signal, Math.max(0, onset - BASELINE_LOOKBACK_SAMPLES), onset);

                double rMax = Double.NEGATIVE_INFINITY;
                double sMin = Double.POSITIVE_INFINITY;
                for (int i = onset; i <= offset; i++) {
                    double v = signal[i] - baseline;
                    rMax = Math.max(rMax, v);
                    sMin = Math.min(sMin, v);
                }
                double rAmplitude = Math.max(0, rMax);
                double sAmplitude = Math.max(0, -sMin);

                int qEnd = onset;
                while (qEnd <= offset && (signal[qEnd] - baseline) <= 0) {
                    qEnd++;
                }
                double qDuration = (qEnd - onset) * 1000.0 / fs;
                double qDepth = 0;
                for (int i = onset; i < qEnd && i <= offset; i++) {
                    qDepth = Math.max(qDepth, -(signal[i] - baseline));
                }
                double qRatio = rAmplitude > 1e-6 ? qDepth / rAmplitude : 0;

                int stFrom = Math.min(signal.length - 1, offset + stOffsetSamples);
                int stTo = Math.min(signal.length - 1, stFrom + stWindowSamples);
                double stLevel = mean(signal, stFrom, stTo) - baseline;

                double tAmp = 0;
                if (beat.tPeak() >= 0 && beat.tPeak() < signal.length) {
                    tAmp = signal[beat.tPeak()] - baseline;
                }

                rAmps.add(rAmplitude);
                sAmps.add(sAmplitude);
                qDurations.add(qDuration);
                qRatios.add(qRatio);
                stDevs.add(stLevel);
                tAmps.add(tAmp);
            }

            result.put(leadName, new LeadMorphology(
                    leadName,
                    median(rAmps),
                    median(sAmps),
                    median(qDurations),
                    median(qRatios),
                    median(stDevs),
                    median(tAmps)));
        }

        return result;
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

    private static double median(List<Double> values) {
        if (values.isEmpty()) {
            return 0;
        }
        List<Double> sorted = new ArrayList<>(values);
        sorted.sort(Double::compareTo);
        int n = sorted.size();
        return (n % 2 == 1) ? sorted.get(n / 2) : (sorted.get(n / 2 - 1) + sorted.get(n / 2)) / 2.0;
    }
}
