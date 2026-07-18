package emr.ekg.pipeline;

import ai.onnxruntime.OrtException;
import emr.ekg.features.ClinicalFeature;
import emr.ekg.features.ClinicalFeatureExtractor;
import emr.ekg.features.Sex;
import emr.ekg.features.fiducial.BeatFiducials;
import emr.ekg.features.fiducial.FiducialExtractor;
import emr.ekg.features.morphology.LeadMorphology;
import emr.ekg.features.morphology.LeadMorphologyExtractor;
import emr.ekg.inference.BeatArrhythmiaClassifier;
import emr.ekg.inference.FeatureBasedNormClassifier;
import emr.ekg.persistence.ReportStatus;
import emr.ekg.rules.ConductionRuleEngine;
import emr.ekg.rules.EmergencyRuleEngine;
import emr.ekg.rules.Finding;
import emr.ekg.rules.RuleThresholds;
import emr.ekg.signal.PreprocessedEcg;
import emr.ekg.signal.RawEcgSignal;
import emr.ekg.signal.preprocess.EcgPreprocessor;
import emr.ekg.signal.preprocess.PreprocessConfig;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 진단 파이프라인 오케스트레이터: 0단계 전처리 → 1단계 정상 우선순위 분류 → E단계 응급 →
 * 2단계 전도장애/축 → 3단계 비트 부정맥(모델 배치 시) → [4단계: Phase 4 이전에는 미구현] → 결과 합성.
 *
 * 1단계는 게이팅하지 않는다(ARCHITECTURE.md P2) — SQI 미달로 판독불가 반려되지 않는 한
 * 모든 ECG가 E→2(→3) 단계를 전부 거쳐 서명 큐(PENDING_SIGN)로 간다.
 *
 * 룰 임계치·모델경로는 {@link RuleThresholds}/{@link BeatArrhythmiaConfig}로 주입된다 —
 * {@link #fromConfig}로 생성하면 config/pipeline.yaml 값을 쓰고, 재보정 시 코드 변경 없이
 * YAML만 고치면 된다(backbone 목표). Stage3 모델이 배치되지 않았으면 조용히 건너뛰고
 * beatArrhythmiaAvailable=false로 명시한다.
 */
public final class EkgPipeline implements AutoCloseable {

    private final PreprocessConfig preprocessConfig;
    private final EcgPreprocessor preprocessor;
    private final FeatureBasedNormClassifier triageClassifier;
    private final String referenceLeadName;
    private final RuleThresholds ruleThresholds;
    private final BeatArrhythmiaConfig beatArrhythmiaConfig;
    private final BeatArrhythmiaClassifier beatClassifier;

    public EkgPipeline(Path triageModelJson, String referenceLeadName) throws IOException {
        this(PreprocessConfig.defaults(), triageModelJson, referenceLeadName, RuleThresholds.defaults(),
                BeatArrhythmiaConfig.unavailable());
    }

    public EkgPipeline(PreprocessConfig preprocessConfig, Path triageModelJson, String referenceLeadName,
            RuleThresholds ruleThresholds, BeatArrhythmiaConfig beatArrhythmiaConfig) throws IOException {
        this.preprocessConfig = preprocessConfig;
        this.preprocessor = new EcgPreprocessor(preprocessConfig);
        this.triageClassifier = FeatureBasedNormClassifier.loadFromJson(triageModelJson);
        this.referenceLeadName = referenceLeadName;
        this.ruleThresholds = ruleThresholds;
        this.beatArrhythmiaConfig = beatArrhythmiaConfig;

        if (beatArrhythmiaConfig.isAvailable()) {
            try {
                this.beatClassifier = new BeatArrhythmiaClassifier(beatArrhythmiaConfig.modelPath());
            } catch (OrtException e) {
                throw new IOException("Stage3 모델 로드 실패: " + beatArrhythmiaConfig.modelPath(), e);
            }
        } else {
            this.beatClassifier = null;
        }
    }

    /** config/pipeline.yaml로부터 파이프라인을 구성한다(재보정 시 YAML만 고치면 됨). */
    public static EkgPipeline fromConfig(PipelineConfig config) throws IOException {
        PreprocessConfig preprocessConfig = new PreprocessConfig(
                config.targetFs(), config.bandpassLowHz(), config.bandpassHighHz(),
                config.notchHz(), config.sqiThreshold());
        return new EkgPipeline(preprocessConfig, config.triageModelPath(), config.referenceLeadName(),
                config.ruleThresholds(), config.beatArrhythmiaConfig());
    }

    public DiagnosticReport process(String recordId, String source, RawEcgSignal raw, Sex sex) {
        return processWithSignal(recordId, source, raw, sex).report();
    }

    /** 리포트와 함께 렌더링용 전처리 신호도 반환한다(app-fx ECG 뷰어가 파형을 그려야 하므로). */
    public PipelineResult processWithSignal(String recordId, String source, RawEcgSignal raw, Sex sex) {
        PreprocessedEcg ecg = preprocessor.process(raw);

        if (!ecg.isInterpretable(preprocessConfig.sqiThreshold())) {
            DiagnosticReport rejected =
                    DiagnosticReport.rejected(recordId, source, ecg.fs(), ecg.sampleCount(), ecg.sqi());
            return new PipelineResult(rejected, ecg);
        }

        List<BeatFiducials> beats = FiducialExtractor.extract(ecg, referenceLeadName);
        Map<String, LeadMorphology> morphology = LeadMorphologyExtractor.extract(ecg, beats);
        List<ClinicalFeature> features =
                ClinicalFeatureExtractor.extract(ecg, beats, morphology, sex, referenceLeadName);

        double triageScore = triageClassifier.predictNormProbability(features);

        List<Finding> findings = new ArrayList<>();
        findings.addAll(EmergencyRuleEngine.evaluate(features, morphology, ruleThresholds));
        findings.addAll(ConductionRuleEngine.evaluate(features, morphology, ruleThresholds));

        boolean beatArrhythmiaAvailable = beatClassifier != null;
        if (beatArrhythmiaAvailable) {
            findings.addAll(Stage3ArrhythmiaAnalyzer.analyze(
                    ecg, beats, referenceLeadName, beatClassifier, beatArrhythmiaConfig));
        }

        DiagnosticReport report = new DiagnosticReport(
                recordId, source, Instant.now(), ecg.fs(), ecg.sampleCount(), ecg.sqi(), true,
                features, triageScore, triageClassifier.version(),
                findings, beatArrhythmiaAvailable, false, ReportStatus.PENDING_SIGN);
        return new PipelineResult(report, ecg);
    }

    @Override
    public void close() {
        if (beatClassifier != null) {
            try {
                beatClassifier.close();
            } catch (OrtException e) {
                // 종료 시 리소스 정리 실패는 무시 — 프로세스 종료를 막을 이유가 없다.
            }
        }
    }
}
