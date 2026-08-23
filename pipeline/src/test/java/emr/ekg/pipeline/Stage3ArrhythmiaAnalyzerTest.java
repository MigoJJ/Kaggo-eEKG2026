package emr.ekg.pipeline;

import emr.ekg.features.fiducial.BeatFiducials;
import emr.ekg.rules.Finding;
import emr.ekg.rules.RuleThresholds;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * VT(심실빈맥) 런 검출 회귀 테스트 — 실제 ONNX 분류기 없이 합성 라벨 시퀀스로 검증한다.
 * fs=500Hz 기준: RR간격(샘플) = 60*fs/목표심박수, QRS폭(샘플) = 목표ms*fs/1000.
 */
class Stage3ArrhythmiaAnalyzerTest {

    private static final int FS = 500;

    @Test
    void triggersVtOnFourConsecutiveVBeatsAboveRateAndQrsThreshold() {
        // RR=200샘플(150bpm > 기본 임계 120bpm), QRS폭=70샘플(140ms ≥ 기본 임계 120ms)
        List<BeatFiducials> beats = List.of(
                beat(800), // N — 런 시작 전 경계
                beatWithQrs(1000, 70),
                beatWithQrs(1200, 70),
                beatWithQrs(1400, 70),
                beatWithQrs(1600, 70),
                beat(1800)); // N — 런 종료 후 경계
        List<String> labels = Arrays.asList("N", "V", "V", "V", "V", "N");

        List<Finding> findings = Stage3ArrhythmiaAnalyzer.detectVentricularTachycardiaRuns(
                beats, labels, FS, RuleThresholds.defaults());

        assertTrue(findings.stream().anyMatch(f -> f.code().equals("VT")
                && f.severity() == Finding.Severity.CRITICAL));
    }

    @Test
    void doesNotTriggerVtWithOnlyTwoConsecutiveVBeats() {
        List<BeatFiducials> beats = List.of(
                beatWithQrs(1000, 70),
                beatWithQrs(1200, 70));
        List<String> labels = List.of("V", "V");

        List<Finding> findings = Stage3ArrhythmiaAnalyzer.detectVentricularTachycardiaRuns(
                beats, labels, FS, RuleThresholds.defaults());

        assertFalse(findings.stream().anyMatch(f -> f.code().equals("VT")));
    }

    @Test
    void doesNotTriggerVtWhenRateBelowThreshold() {
        // RR=300샘플 => 100bpm, 기본 임계(120bpm) 미달
        List<BeatFiducials> beats = List.of(
                beatWithQrs(1000, 70),
                beatWithQrs(1300, 70),
                beatWithQrs(1600, 70));
        List<String> labels = List.of("V", "V", "V");

        List<Finding> findings = Stage3ArrhythmiaAnalyzer.detectVentricularTachycardiaRuns(
                beats, labels, FS, RuleThresholds.defaults());

        assertFalse(findings.stream().anyMatch(f -> f.code().equals("VT")));
    }

    @Test
    void doesNotTriggerVtWhenQrsNarrowerThanThreshold() {
        // 심박수는 충분히 빠르지만(150bpm) QRS폭이 60ms(30샘플)로 협소 — VT 정의(넓은 QRS) 미충족
        List<BeatFiducials> beats = List.of(
                beatWithQrs(1000, 30),
                beatWithQrs(1200, 30),
                beatWithQrs(1400, 30));
        List<String> labels = List.of("V", "V", "V");

        List<Finding> findings = Stage3ArrhythmiaAnalyzer.detectVentricularTachycardiaRuns(
                beats, labels, FS, RuleThresholds.defaults());

        assertFalse(findings.stream().anyMatch(f -> f.code().equals("VT")));
    }

    @Test
    void unclassifiedBeatBreaksAnOtherwiseQualifyingRun() {
        // 분류 실패(null)가 중간에 끼면 5비트 런이 아니라 2+2 런으로 쪼개져 어느 쪽도 3개 미만
        List<BeatFiducials> beats = List.of(
                beatWithQrs(1000, 70),
                beatWithQrs(1200, 70),
                beatWithQrs(1400, 70),
                beatWithQrs(1600, 70),
                beatWithQrs(1800, 70));
        List<String> labels = Arrays.asList("V", "V", null, "V", "V");

        List<Finding> findings = Stage3ArrhythmiaAnalyzer.detectVentricularTachycardiaRuns(
                beats, labels, FS, RuleThresholds.defaults());

        assertFalse(findings.stream().anyMatch(f -> f.code().equals("VT")));
    }

    private static BeatFiducials beatWithQrs(int rIndex, int qrsWidthSamples) {
        int half = qrsWidthSamples / 2;
        return new BeatFiducials(rIndex, rIndex - half, rIndex + half, -1, -1, -1, 0, 0);
    }

    private static BeatFiducials beat(int rIndex) {
        return beatWithQrs(rIndex, 70);
    }
}
