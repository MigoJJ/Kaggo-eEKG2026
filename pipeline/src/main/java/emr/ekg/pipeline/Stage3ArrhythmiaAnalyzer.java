package emr.ekg.pipeline;

import emr.ekg.features.fiducial.BeatFiducials;
import emr.ekg.inference.BeatArrhythmiaClassifier;
import emr.ekg.rules.Finding;
import emr.ekg.rules.RuleThresholds;
import emr.ekg.signal.PreprocessedEcg;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

/**
 * Stage3(비트 부정맥 검증) — 각 비트를 AAMI 5-클래스(N/S/V/F/Q)로 분류하고 이소성 부담
 * (ectopy burden)을 집계해 소견을 산출한다. 연속 V-비트 런을 스캔해 VT(심실빈맥)도 판정한다
 * — VT는 비트분류 결과가 있어야 판정 가능해 {@code EmergencyRuleEngine}(E단계, 형태학/QTc만
 * 사용)의 범위 밖이므로 여기서 담당한다(Phase 10).
 *
 * ⚠️ Phase 4 backbone 단계다 — MIT-BIH로 학습했으나 정확도는 검증되지 않았다(학습 시 혼동행렬
 * 기준 N/F 혼동 다수 확인). 파이프라인 연동이 실제로 동작함을 증명하는 것이 목적이며,
 * Kaggle P100 재학습 후 정확도 보정 예정 — 그때까지 소견에 그 사실을 명시한다.
 */
final class Stage3ArrhythmiaAnalyzer {

    private static final int V_CLASS_INDEX = 2;
    private static final int S_CLASS_INDEX = 1;
    private static final String V_CLASS_LABEL = "V";
    private static final int VT_MIN_CONSECUTIVE_BEATS = 3;

    private Stage3ArrhythmiaAnalyzer() {
    }

    static List<Finding> analyze(PreprocessedEcg ecg, List<BeatFiducials> beats, String referenceLeadName,
            BeatArrhythmiaClassifier classifier, BeatArrhythmiaConfig config, RuleThresholds thresholds) {
        int leadIdx = indexOfLead(ecg.leadNames(), referenceLeadName);
        double[] signal = ecg.samples()[leadIdx];
        int half = BeatArrhythmiaClassifier.WINDOW_SAMPLES / 2;

        int[] counts = new int[BeatArrhythmiaClassifier.CLASS_NAMES.length];
        int classified = 0;
        // beats와 1:1로 맞춘 라벨(분류 실패/윈도우 범위밖은 null) — VT 런 검출이 "연속" 여부를
        // beats의 원래 시간순서 기준으로 정확히 판단할 수 있어야 하므로, 실패 비트를 건너뛰지
        // 않고 null로 자리를 채워 인덱스 정합을 유지한다.
        List<String> labels = new ArrayList<>(beats.size());
        for (BeatFiducials beat : beats) {
            int from = beat.rIndex() - half;
            int to = beat.rIndex() + half;
            if (from < 0 || to > signal.length) {
                labels.add(null);
                continue;
            }
            double[] window = Arrays.copyOfRange(signal, from, to);
            try {
                String cls = classifier.classify(window);
                counts[indexOf(BeatArrhythmiaClassifier.CLASS_NAMES, cls)]++;
                classified++;
                labels.add(cls);
            } catch (Exception e) {
                // 개별 비트 분류 실패는 전체 리포트를 막지 않는다 — 해당 비트만 집계 제외.
                labels.add(null);
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

        findings.addAll(detectVentricularTachycardiaRuns(beats, labels, ecg.fs(), thresholds));
        return findings;
    }

    /**
     * 연속 V-비트 런(≥3개)을 스캔해, 런 내 평균 심박수·평균 QRS폭이 임계치를 넘으면 VT로 판정한다.
     * 패키지 프라이빗 — 실제 ONNX 분류기 없이 합성 라벨 시퀀스로 직접 단위테스트하기 위함.
     */
    static List<Finding> detectVentricularTachycardiaRuns(List<BeatFiducials> beats, List<String> labels,
            int fs, RuleThresholds thresholds) {
        List<Finding> findings = new ArrayList<>();
        int runStart = -1;
        for (int i = 0; i <= beats.size(); i++) {
            boolean isV = i < beats.size() && V_CLASS_LABEL.equals(labels.get(i));
            if (isV) {
                if (runStart < 0) {
                    runStart = i;
                }
                continue;
            }
            if (runStart >= 0) {
                int runLength = i - runStart;
                if (runLength >= VT_MIN_CONSECUTIVE_BEATS) {
                    vtFindingForRun(beats, runStart, i, fs, thresholds).ifPresent(findings::add);
                }
                runStart = -1;
            }
        }
        return findings;
    }

    private static Optional<Finding> vtFindingForRun(List<BeatFiducials> beats, int runStart, int runEnd, int fs,
            RuleThresholds thresholds) {
        int runLength = runEnd - runStart;

        double totalRrSamples = 0;
        for (int i = runStart; i < runEnd - 1; i++) {
            totalRrSamples += beats.get(i + 1).rIndex() - beats.get(i).rIndex();
        }
        double avgRrSamples = totalRrSamples / (runLength - 1);
        double rateBpm = 60.0 * fs / avgRrSamples;

        double totalQrsMs = 0;
        for (int i = runStart; i < runEnd; i++) {
            BeatFiducials b = beats.get(i);
            totalQrsMs += (b.qrsOffset() - b.qrsOnset()) * 1000.0 / fs;
        }
        double avgQrsMs = totalQrsMs / runLength;

        if (rateBpm > thresholds.vtRateBpm() && avgQrsMs >= thresholds.vtQrsMs()) {
            return Optional.of(new Finding("VT",
                    "심실빈맥(VT) 의심 [Stage3 backbone, 정확도 미검증]",
                    Finding.Severity.CRITICAL,
                    "연속 V-비트 %d개, 평균 심박수 %.0fbpm(>%.0f), 평균 QRS폭 %.0fms(≥%.0f)"
                            .formatted(runLength, rateBpm, thresholds.vtRateBpm(), avgQrsMs, thresholds.vtQrsMs())));
        }
        return Optional.empty();
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
