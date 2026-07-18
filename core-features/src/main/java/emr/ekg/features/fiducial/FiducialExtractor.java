package emr.ekg.features.fiducial;

import emr.ekg.signal.PreprocessedEcg;

import java.util.ArrayList;
import java.util.List;

/**
 * 기준 리드(통상 Lead II, MIT-BIH는 MLII)에서 R-peak → QRS 경계 → P파 → T파 순서로
 * 전체 fiducial을 추출한다. 12리드는 동시기록으로 동일 시간축을 공유하므로, 여기서 산출한
 * 인덱스를 다른 리드의 진폭 측정(LeadMorphologyExtractor)에도 그대로 사용한다.
 */
public final class FiducialExtractor {

    private FiducialExtractor() {
    }

    public static List<BeatFiducials> extract(PreprocessedEcg ecg, String referenceLeadName) {
        int leadIdx = indexOfLead(ecg.leadNames(), referenceLeadName);
        double[] signal = ecg.samples()[leadIdx];
        int fs = ecg.fs();

        int[] rPeaks = RPeakDetector.detect(signal, fs);
        List<BeatFiducials> beats = new ArrayList<>();

        int previousQrsOffset = 0;
        for (int b = 0; b < rPeaks.length; b++) {
            int r = rPeaks[b];
            int[] qrs = QrsBoundaryDetector.onsetOffset(signal, r, fs);
            int qrsOnset = qrs[0];
            int qrsOffset = qrs[1];

            int[] p = PWaveDetector.detect(signal, qrsOnset, previousQrsOffset, fs);

            int nextBoundary = (b + 1 < rPeaks.length) ? rPeaks[b + 1] : signal.length - 1;
            int[] t = TWaveDetector.detect(signal, qrsOffset, nextBoundary, fs);

            beats.add(new BeatFiducials(r, qrsOnset, qrsOffset, p[0], p[1], p[2], t[0], t[1]));
            previousQrsOffset = qrsOffset;
        }

        return beats;
    }

    static int indexOfLead(String[] leadNames, String name) {
        for (int i = 0; i < leadNames.length; i++) {
            if (leadNames[i].equalsIgnoreCase(name)) {
                return i;
            }
        }
        throw new IllegalArgumentException("리드를 찾을 수 없음: " + name);
    }
}
