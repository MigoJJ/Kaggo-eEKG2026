package emr.ekg.features;

import emr.ekg.features.fiducial.BeatFiducials;
import emr.ekg.features.morphology.LeadMorphology;
import emr.ekg.signal.PreprocessedEcg;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * AHA/ACC/ESC 표준 12유도 판독 가이드(Systematic Interpretation)의 20개 핵심 임상 피처를 계산한다.
 * Rate/Rhythm·Axis·P·PR·QRS·ST·T·QT/QTc 8개 판독 영역을 대표하는 스칼라 값이며,
 * 정상범위는 가이드에 명시된 수치({@link ClinicalConstants})를 그대로 사용한다.
 */
public final class ClinicalFeatureExtractor {

    private static final int BASELINE_LOOKBACK_SAMPLES = 5;

    private ClinicalFeatureExtractor() {
    }

    public static List<ClinicalFeature> extract(
            PreprocessedEcg ecg,
            List<BeatFiducials> beats,
            Map<String, LeadMorphology> morphology,
            Sex sex,
            String referenceLeadName) {

        List<ClinicalFeature> features = new ArrayList<>();
        if (beats.isEmpty()) {
            return features;
        }

        int fs = ecg.fs();
        int referenceLeadIdx = indexOfLead(ecg.leadNames(), referenceLeadName);
        double[] referenceLead = ecg.samples()[referenceLeadIdx];

        double[] rrMs = rrIntervalsMs(beats, fs);
        double medianRrMs = median(rrMs);
        double hr = medianRrMs > 0 ? 60000.0 / medianRrMs : 0;
        double rrStd = stdDev(rrMs);

        double prMs = median(prIntervalsMs(beats, fs));
        double pDurationMs = median(pDurationsMs(beats, fs));
        double qrsDurationMs = median(qrsDurationsMs(beats, fs));
        double qtMs = median(qtIntervalsMs(beats, fs));
        double qtcMs = (medianRrMs > 0 && !Double.isNaN(qtMs))
                ? qtMs / Math.sqrt(medianRrMs / 1000.0)
                : Double.NaN;
        double pAmpMv = median(pAmplitudesMv(beats, referenceLead));

        LeadMorphology leadI = morphology.get("I");
        LeadMorphology leadAvf = lookup(morphology, "AVF", "aVF");
        LeadMorphology leadV1 = morphology.get("V1");
        LeadMorphology leadV2 = morphology.get("V2");
        LeadMorphology leadV3 = morphology.get("V3");
        LeadMorphology leadV5 = morphology.get("V5");
        LeadMorphology leadV6 = morphology.get("V6");
        LeadMorphology leadAvl = lookup(morphology, "AVL", "aVL");
        LeadMorphology leadII = lookup(morphology, "II", "MLII");

        double axisDeg = (leadI != null && leadAvf != null)
                ? Math.toDegrees(Math.atan2(leadAvf.netDeflectionMv(), leadI.netDeflectionMv()))
                : Double.NaN;

        double rV5 = leadV5 != null ? leadV5.rAmplitudeMv() : Double.NaN;
        double rV6 = leadV6 != null ? leadV6.rAmplitudeMv() : Double.NaN;
        double sV1 = leadV1 != null ? leadV1.sAmplitudeMv() : Double.NaN;
        double sokolowLyon = !Double.isNaN(sV1) ? sV1 + Math.max(nz(rV5), nz(rV6)) : Double.NaN;
        double rAvl = leadAvl != null ? leadAvl.rAmplitudeMv() : Double.NaN;

        double qDurationIi = leadII != null ? leadII.qDurationMs() : Double.NaN;
        double qRatioIi = leadII != null ? leadII.qDepthRatio() : Double.NaN;
        double stIi = leadII != null ? leadII.stDeviationMv() : Double.NaN;
        double stV2 = leadV2 != null ? leadV2.stDeviationMv() : Double.NaN;
        double stV3 = leadV3 != null ? leadV3.stDeviationMv() : Double.NaN;
        double tIi = leadII != null ? leadII.tAmplitudeMv() : Double.NaN;

        double qtcMin = sex == Sex.FEMALE ? ClinicalConstants.QTC_FEMALE_MIN_MS : ClinicalConstants.QTC_MALE_MIN_MS;
        double qtcMax = sex == Sex.FEMALE ? ClinicalConstants.QTC_FEMALE_MAX_MS : ClinicalConstants.QTC_MALE_MAX_MS;

        features.add(new ClinicalFeature(1, "HR", hr, "bpm",
                ClinicalConstants.HR_MIN_BPM, ClinicalConstants.HR_MAX_BPM));
        features.add(new ClinicalFeature(2, "PR_INTERVAL", prMs, "ms",
                ClinicalConstants.PR_MIN_MS, ClinicalConstants.PR_MAX_MS));
        features.add(new ClinicalFeature(3, "P_DURATION", pDurationMs, "ms",
                0, ClinicalConstants.P_DURATION_MAX_MS));
        features.add(new ClinicalFeature(4, "P_AMPLITUDE_II", pAmpMv, "mV",
                0, ClinicalConstants.P_AMPLITUDE_II_MAX_MV));
        features.add(new ClinicalFeature(5, "QRS_DURATION", qrsDurationMs, "ms",
                ClinicalConstants.QRS_MIN_MS, ClinicalConstants.QRS_MAX_MS));
        features.add(new ClinicalFeature(6, "QRS_AXIS", axisDeg, "deg",
                ClinicalConstants.AXIS_NORMAL_MIN_DEG, ClinicalConstants.AXIS_NORMAL_MAX_DEG));
        features.add(new ClinicalFeature(7, "R_AMPLITUDE_V5", rV5, "mV", 0, Double.MAX_VALUE));
        features.add(new ClinicalFeature(8, "R_AMPLITUDE_V6", rV6, "mV", 0, Double.MAX_VALUE));
        features.add(new ClinicalFeature(9, "S_AMPLITUDE_V1", sV1, "mV", 0, Double.MAX_VALUE));
        features.add(new ClinicalFeature(10, "SOKOLOW_LYON_INDEX", sokolowLyon, "mV",
                0, ClinicalConstants.SOKOLOW_LYON_LVH_MV));
        features.add(new ClinicalFeature(11, "R_AMPLITUDE_AVL", rAvl, "mV",
                0, ClinicalConstants.R_AVL_LVH_MV));
        features.add(new ClinicalFeature(12, "Q_DURATION_II", qDurationIi, "ms",
                0, ClinicalConstants.Q_PATHOLOGIC_DURATION_MS));
        features.add(new ClinicalFeature(13, "Q_DEPTH_RATIO_II", qRatioIi, "ratio",
                0, ClinicalConstants.Q_PATHOLOGIC_R_RATIO));
        features.add(new ClinicalFeature(14, "ST_DEVIATION_II", stIi, "mV",
                -ClinicalConstants.ST_DEPRESSION_MV, ClinicalConstants.ST_ELEVATION_GENERAL_MV));
        features.add(new ClinicalFeature(15, "ST_DEVIATION_V2", stV2, "mV",
                -ClinicalConstants.ST_DEPRESSION_MV, ClinicalConstants.ST_ELEVATION_V2V3_MV));
        features.add(new ClinicalFeature(16, "ST_DEVIATION_V3", stV3, "mV",
                -ClinicalConstants.ST_DEPRESSION_MV, ClinicalConstants.ST_ELEVATION_V2V3_MV));
        features.add(new ClinicalFeature(17, "T_AMPLITUDE_II", tIi, "mV",
                -Double.MAX_VALUE, Double.MAX_VALUE));
        features.add(new ClinicalFeature(18, "QT_INTERVAL", qtMs, "ms", 0, Double.MAX_VALUE));
        features.add(new ClinicalFeature(19, "QTC_BAZETT", qtcMs, "ms", qtcMin, qtcMax));
        features.add(new ClinicalFeature(20, "RR_VARIABILITY", rrStd, "ms", 0, Double.MAX_VALUE));

        return features;
    }

    private static double[] rrIntervalsMs(List<BeatFiducials> beats, int fs) {
        double[] out = new double[Math.max(0, beats.size() - 1)];
        for (int i = 1; i < beats.size(); i++) {
            out[i - 1] = (beats.get(i).rIndex() - beats.get(i - 1).rIndex()) * 1000.0 / fs;
        }
        return out;
    }

    private static double[] prIntervalsMs(List<BeatFiducials> beats, int fs) {
        List<Double> values = new ArrayList<>();
        for (BeatFiducials b : beats) {
            if (b.hasPWave()) {
                values.add((b.qrsOnset() - b.pOnset()) * 1000.0 / fs);
            }
        }
        return toArray(values);
    }

    private static double[] pDurationsMs(List<BeatFiducials> beats, int fs) {
        List<Double> values = new ArrayList<>();
        for (BeatFiducials b : beats) {
            if (b.hasPWave()) {
                values.add((b.pOffset() - b.pOnset()) * 1000.0 / fs);
            }
        }
        return toArray(values);
    }

    private static double[] qrsDurationsMs(List<BeatFiducials> beats, int fs) {
        double[] out = new double[beats.size()];
        for (int i = 0; i < beats.size(); i++) {
            out[i] = (beats.get(i).qrsOffset() - beats.get(i).qrsOnset()) * 1000.0 / fs;
        }
        return out;
    }

    private static double[] qtIntervalsMs(List<BeatFiducials> beats, int fs) {
        List<Double> values = new ArrayList<>();
        for (BeatFiducials b : beats) {
            if (b.tOffset() >= 0) {
                values.add((b.tOffset() - b.qrsOnset()) * 1000.0 / fs);
            }
        }
        return toArray(values);
    }

    private static double[] pAmplitudesMv(List<BeatFiducials> beats, double[] referenceLead) {
        List<Double> values = new ArrayList<>();
        for (BeatFiducials b : beats) {
            if (!b.hasPWave()) {
                continue;
            }
            int lookbackStart = Math.max(0, b.pOnset() - BASELINE_LOOKBACK_SAMPLES);
            double baseline = mean(referenceLead, lookbackStart, b.pOnset());
            values.add(referenceLead[b.pPeak()] - baseline);
        }
        return toArray(values);
    }

    private static LeadMorphology lookup(Map<String, LeadMorphology> map, String... names) {
        for (String n : names) {
            LeadMorphology m = map.get(n);
            if (m != null) {
                return m;
            }
        }
        return null;
    }

    private static int indexOfLead(String[] names, String name) {
        for (int i = 0; i < names.length; i++) {
            if (names[i].equalsIgnoreCase(name)) {
                return i;
            }
        }
        throw new IllegalArgumentException("리드를 찾을 수 없음: " + name);
    }

    private static double nz(double v) {
        return Double.isNaN(v) ? 0 : v;
    }

    private static double[] toArray(List<Double> list) {
        double[] out = new double[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = list.get(i);
        }
        return out;
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

    private static double median(double[] values) {
        if (values.length == 0) {
            return Double.NaN;
        }
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        int n = sorted.length;
        return (n % 2 == 1) ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    private static double stdDev(double[] values) {
        if (values.length == 0) {
            return Double.NaN;
        }
        double mean = 0;
        for (double v : values) {
            mean += v;
        }
        mean /= values.length;
        double variance = 0;
        for (double v : values) {
            variance += (v - mean) * (v - mean);
        }
        variance /= values.length;
        return Math.sqrt(variance);
    }
}
