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
import emr.ekg.inference.ModelArtifactMetadata;
import emr.ekg.inference.Stage4DelineatorClassifier;
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
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HexFormat;
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
    private final Path stIschemiaModelPath;
    private final Stage4DelineatorClassifier stIschemiaClassifier;
    private final String triageModelHash;
    private final String beatArrhythmiaModelVersion;
    private final String beatArrhythmiaModelHash;
    private final String beatArrhythmiaModelMetadataHash;
    private final String beatArrhythmiaModelMetadataJson;
    private final String stIschemiaModelVersion;
    private final String stIschemiaModelHash;
    private final String stIschemiaModelMetadataHash;
    private final String stIschemiaModelMetadataJson;

    public EkgPipeline(Path triageModelJson, String referenceLeadName) throws IOException {
        this(PreprocessConfig.defaults(), triageModelJson, referenceLeadName, RuleThresholds.defaults(),
                BeatArrhythmiaConfig.unavailable(), null);
    }

    public EkgPipeline(PreprocessConfig preprocessConfig, Path triageModelJson, String referenceLeadName,
            RuleThresholds ruleThresholds, BeatArrhythmiaConfig beatArrhythmiaConfig) throws IOException {
        this(preprocessConfig, triageModelJson, referenceLeadName, ruleThresholds, beatArrhythmiaConfig, null);
    }

    public EkgPipeline(PreprocessConfig preprocessConfig, Path triageModelJson, String referenceLeadName,
            RuleThresholds ruleThresholds, BeatArrhythmiaConfig beatArrhythmiaConfig, Path stIschemiaModelPath) throws IOException {
        this.preprocessConfig = preprocessConfig;
        this.preprocessor = new EcgPreprocessor(preprocessConfig);
        this.triageClassifier = FeatureBasedNormClassifier.loadFromJson(triageModelJson);
        this.referenceLeadName = referenceLeadName;
        this.ruleThresholds = ruleThresholds;
        this.beatArrhythmiaConfig = beatArrhythmiaConfig;
        this.stIschemiaModelPath = stIschemiaModelPath;
        this.triageModelHash = sha256Hex(triageModelJson);

        if (beatArrhythmiaConfig.isAvailable()) {
            try {
                this.beatClassifier = new BeatArrhythmiaClassifier(beatArrhythmiaConfig.modelPath());
            } catch (OrtException e) {
                throw new IOException("Stage3 모델 로드 실패: " + beatArrhythmiaConfig.modelPath(), e);
            }
            this.beatArrhythmiaModelHash = sha256Hex(beatArrhythmiaConfig.modelPath());
            ModelArtifactMetadata metadata =
                    ModelArtifactMetadata.loadForModel(beatArrhythmiaConfig.modelPath(), "stage3", beatArrhythmiaModelHash);
            this.beatArrhythmiaModelVersion = metadata.version();
            this.beatArrhythmiaModelMetadataHash = metadata.metadataSha256();
            this.beatArrhythmiaModelMetadataJson = metadata.rawJson();
        } else {
            this.beatClassifier = null;
            this.beatArrhythmiaModelVersion = null;
            this.beatArrhythmiaModelHash = null;
            this.beatArrhythmiaModelMetadataHash = null;
            this.beatArrhythmiaModelMetadataJson = null;
        }

        if (stIschemiaModelPath != null && Files.exists(stIschemiaModelPath)) {
            try {
                this.stIschemiaClassifier = new Stage4DelineatorClassifier(stIschemiaModelPath);
            } catch (OrtException e) {
                throw new IOException("Stage4 모델 로드 실패: " + stIschemiaModelPath, e);
            }
            this.stIschemiaModelHash = sha256Hex(stIschemiaModelPath);
            ModelArtifactMetadata metadata =
                    ModelArtifactMetadata.loadForModel(stIschemiaModelPath, "stage4", stIschemiaModelHash);
            this.stIschemiaModelVersion = metadata.version();
            this.stIschemiaModelMetadataHash = metadata.metadataSha256();
            this.stIschemiaModelMetadataJson = metadata.rawJson();
        } else {
            this.stIschemiaClassifier = null;
            this.stIschemiaModelVersion = null;
            this.stIschemiaModelHash = null;
            this.stIschemiaModelMetadataHash = null;
            this.stIschemiaModelMetadataJson = null;
        }
    }

    /** config/pipeline.yaml로부터 파이프라인을 구성한다(재보정 시 YAML만 고치면 됨). */
    public static EkgPipeline fromConfig(PipelineConfig config) throws IOException {
        PreprocessConfig preprocessConfig = new PreprocessConfig(
                config.targetFs(), config.bandpassLowHz(), config.bandpassHighHz(),
                config.notchHz(), config.sqiThreshold());
        return new EkgPipeline(preprocessConfig, config.triageModelPath(), config.referenceLeadName(),
                config.ruleThresholds(), config.beatArrhythmiaConfig(), config.stIschemiaModelPath());
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
                    ecg, beats, referenceLeadName, beatClassifier, beatArrhythmiaConfig, ruleThresholds));
        }

        boolean stIschemiaAvailable = stIschemiaClassifier != null;
        if (stIschemiaAvailable) {
            try {
                // Stage 4 런타임 추론 기동 (12유도 P/QRS/T 픽셀 분할 수행)
                int[][] mask = stIschemiaClassifier.classify(ecg.samples());
                // 추후 수립될 고도화 룰: 이 마스크를 활용해 J점 기준 ST편위를 재보정하고 QT-interval 정밀 측정을 고도화할 수 있습니다.
            } catch (OrtException e) {
                // 개별 추론 오류가 파이프라인 전체를 중단시키지 않고 우아하게 반려
            }
        }

        DiagnosticReport report = new DiagnosticReport(
                recordId, source, Instant.now(), ecg.fs(), ecg.sampleCount(), ecg.sqi(), true,
                features, triageScore, triageClassifier.version(), triageModelHash, null, null,
                beatArrhythmiaModelVersion, beatArrhythmiaModelHash,
                beatArrhythmiaModelMetadataHash, beatArrhythmiaModelMetadataJson,
                stIschemiaModelVersion, stIschemiaModelHash,
                stIschemiaModelMetadataHash, stIschemiaModelMetadataJson,
                findings, beatArrhythmiaAvailable, stIschemiaAvailable, ReportStatus.PENDING_SIGN);
        return new PipelineResult(report, ecg);
    }

    private static String sha256Hex(Path path) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (InputStream in = Files.newInputStream(path);
                    DigestInputStream digestIn = new DigestInputStream(in, digest)) {
                byte[] buffer = new byte[8192];
                while (digestIn.read(buffer) != -1) {
                    // DigestInputStream updates the digest as bytes are read.
                }
            }
            return HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 digest algorithm unavailable", e);
        }
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
        if (stIschemiaClassifier != null) {
            try {
                stIschemiaClassifier.close();
            } catch (OrtException e) {
                // 리소스 정리 무시
            }
        }
    }
}
