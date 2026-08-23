package emr.ekg.pipeline;

import emr.ekg.rules.RuleThresholds;
import org.yaml.snakeyaml.Yaml;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * config/pipeline.yaml 로더. 임계치·모델경로를 코드 변경 없이 조정할 수 있게 한다(backbone 목표) —
 * Kaggle 학습·대규모 검증 후 재보정된 값은 이 YAML 파일만 고치면 EkgPipeline에 즉시 반영된다.
 */
public record PipelineConfig(
        int targetFs,
        double bandpassLowHz,
        double bandpassHighHz,
        double notchHz,
        double sqiThreshold,
        Path triageModelPath,
        RuleThresholds ruleThresholds,
        BeatArrhythmiaConfig beatArrhythmiaConfig,
        Path stIschemiaModelPath,
        String referenceLeadName) {

    private static final double UV_TO_MV = 1.0 / 1000.0;

    @SuppressWarnings("unchecked")
    public static PipelineConfig loadFromYaml(Path yamlPath) throws IOException {
        Map<String, Object> root;
        try (InputStream in = Files.newInputStream(yamlPath)) {
            root = new Yaml().load(in);
        }

        Map<String, Object> signal = (Map<String, Object>) root.get("signal");
        Map<String, Object> normalTriage = (Map<String, Object>) root.get("normal_triage");
        Map<String, Object> emergency = (Map<String, Object>) root.get("emergency");
        Map<String, Object> conduction = (Map<String, Object>) root.get("conduction");
        Map<String, Object> beatArrhythmia = (Map<String, Object>) root.get("beat_arrhythmia");
        Map<String, Object> stIschemia = (Map<String, Object>) root.get("st_ischemia");

        // config/pipeline.yaml은 <프로젝트루트>/config/에 위치하고, model_path 등 경로는
        // 프로젝트 루트 기준 상대경로다(paths: 섹션의 ./models, ./data 등과 동일 관례).
        Path projectRoot = yamlPath.toAbsolutePath().getParent().getParent();
        String modelPathStr = (String) normalTriage.get("model_path");
        Path modelPath = projectRoot.resolve(modelPathStr).normalize();

        BeatArrhythmiaConfig beatConfig = beatArrhythmia == null
                ? BeatArrhythmiaConfig.unavailable()
                : new BeatArrhythmiaConfig(
                        projectRoot.resolve((String) beatArrhythmia.get("model_path")).normalize(),
                        asDouble(beatArrhythmia, "ventricular_burden_threshold", 0.05),
                        asDouble(beatArrhythmia, "supraventricular_burden_threshold", 0.05));

        Path stIschemiaModelPath = stIschemia == null
                ? null
                : projectRoot.resolve((String) stIschemia.get("model_path")).normalize();

        RuleThresholds thresholds = new RuleThresholds(
                asDouble(emergency, "stemi_st_elevation_uv", 100) * UV_TO_MV,
                asDouble(emergency, "stemi_st_elevation_v2v3_uv", 150) * UV_TO_MV,
                asInt(emergency, "stemi_contiguous_leads", 2),
                asDouble(emergency, "ischemia_st_depression_uv", 50) * UV_TO_MV,
                asInt(emergency, "ischemia_contiguous_leads", 2),
                asDouble(emergency, "longqt_qtc_ms", 500),
                asDouble(emergency, "vt_rate_bpm", 120),
                asDouble(emergency, "vt_qrs_ms", 120),
                asDouble(conduction, "bbb_qrs_ms", 120),
                asDouble(conduction, "avb1_pr_ms", 200));

        return new PipelineConfig(
                asInt(signal, "target_fs", 500),
                asDouble(signal, "bandpass_low", 0.5),
                asDouble(signal, "bandpass_high", 40.0),
                asDouble(signal, "notch_hz", 60.0),
                asDouble(signal, "sqi_threshold", 0.6),
                modelPath,
                thresholds,
                beatConfig,
                stIschemiaModelPath,
                "II");
    }

    private static int asInt(Map<String, Object> map, String key, int fallback) {
        Object v = map.get(key);
        return v == null ? fallback : ((Number) v).intValue();
    }

    private static double asDouble(Map<String, Object> map, String key, double fallback) {
        Object v = map.get(key);
        return v == null ? fallback : ((Number) v).doubleValue();
    }
}
