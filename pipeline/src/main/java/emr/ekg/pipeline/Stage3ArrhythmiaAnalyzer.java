package emr.ekg.pipeline;

import emr.ekg.features.fiducial.BeatFiducials;
import emr.ekg.inference.BeatArrhythmiaClassifier;
import emr.ekg.rules.Finding;
import emr.ekg.signal.PreprocessedEcg;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Stage3(비트 부정맥 검증) — 각 비트를 AAMI 5-클래스(N/S/V/F/Q)로 분류하고 이소성 부담
 * (ectopy burden)을 집계해 소견을 산출한다.
 *
 * ⚠️ Phase 4 backbone 단계다 — MIT-BIH로 학습했으나 정확도는 검증되지 않았다(학습 시 혼동행렬
 * 기준 N/F 혼동 다수 확인). 파이프라인 연동이 실제로 동작함을 증명하는 것이 목적이며,
 * Kaggle P100 재학습 후 정확도 보정 예정 — 그때까지 소견에 그 사실을 명시한다.
 */
final class Stage3ArrhythmiaAnalyzer {

    private static final int V_CLASS_INDEX = 2;
    private static final int S_CLASS_INDEX = 1;

    private Stage3ArrhythmiaAnalyzer() {
    }

    static List<Finding> analyze(PreprocessedEcg ecg, List<BeatFiducials> beats, String referenceLeadName,
            BeatArrhythmiaClassifier classifier, BeatArrhythmiaConfig config) {
        int leadIdx = indexOfLead(ecg.leadNames(), referenceLeadName);
        double[] signal = ecg.samples()[leadIdx];
        int half = BeatArrhythmiaClassifier.WINDOW_SAMPLES / 2;

        int[] counts = new int[BeatArrhythmiaClassifier.CLASS_NAMES.length];
        int classified = 0;
        for (BeatFiducials beat : beats) {
            int from = beat.rIndex() - half;
            int to = beat.rIndex() + half;
            if (from < 0 || to > signal.length) {
                continue;
            }
            double[] window = Arrays.copyOfRange(signal, from, to);
            try {
                String cls = classifier.classify(window);
                counts[indexOf(BeatArrhythmiaClassifier.CLASS_NAMES, cls)]++;
                classified++;
            } catch (Exception e) {
                // 개별 비트 분류 실패는 전체 리포트를 막지 않는다 — 해당 비트만 집계 제외.
            }
        }

        List<Finding> findings = new ArrayList<>();
        if (classified == 0) {
            return findings;
        }

        double vBurden = counts[V_CLASS_INDEX] / (double) classified;
        double sBurden = counts[S_CLASS_INDEX] / (double) classified;

        if (vBurden >= config.ventricularBurdenThreshold()) {
            findings.add(new Finding("VENTRICULAR_ECTOPY_BURDEN",
                    "심실 조기박동(PVC) 부담 증가 의심 [Stage3 backbone, 정확도 미검증]",
                    Finding.Severity.ABNORMAL,
                    "V-비트 %.1f%% (%d/%d)".formatted(vBurden * 100, counts[V_CLASS_INDEX], classified)));
        }
        if (sBurden >= config.supraventricularBurdenThreshold()) {
            findings.add(new Finding("SUPRAVENTRICULAR_ECTOPY_BURDEN",
                    "상심실 조기박동(PAC) 부담 증가 의심 [Stage3 backbone, 정확도 미검증]",
                    Finding.Severity.ABNORMAL,
                    "S-비트 %.1f%% (%d/%d)".formatted(sBurden * 100, counts[S_CLASS_INDEX], classified)));
        }
        return findings;
    }

    private static int indexOfLead(String[] names, String name) {
        for (int i = 0; i < names.length; i++) {
            if (names[i].equalsIgnoreCase(name)) {
                return i;
            }
        }
        throw new IllegalArgumentException("리드를 찾을 수 없음: " + name);
    }

    private static int indexOf(String[] arr, String value) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i].equals(value)) {
                return i;
            }
        }
        throw new IllegalStateException("알 수 없는 클래스: " + value);
    }
}
