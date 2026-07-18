package emr.ekg.pipeline;

import emr.ekg.features.Sex;
import emr.ekg.persistence.ReportStatus;
import emr.ekg.signal.RawEcgSignal;
import emr.ekg.signal.wfdb.WfdbRecordReader;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * config/pipeline.yaml → PipelineConfig → EkgPipeline.fromConfig 전체 경로 검증.
 * Kaggle 학습 후 재보정 값은 이 YAML만 고치면 반영되어야 한다(backbone 목표).
 */
class PipelineConfigTest {

    private static final Path CONFIG_YAML = Path.of("/home/migowj/git/EKGGDSEMR2026/config/pipeline.yaml");
    private static final Path PTBXL_HEADER = Path.of(
            "/mnt/t7/datasets/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
                    + "/records500/00000/00001_hr.hea");

    @Test
    void loadsRealConfigYaml() throws IOException {
        assumeTrue(Files.exists(CONFIG_YAML), "config/pipeline.yaml 미탑재 - 스킵");

        PipelineConfig config = PipelineConfig.loadFromYaml(CONFIG_YAML);

        assertEquals(500, config.targetFs());
        assertEquals(0.6, config.sqiThreshold(), 1e-9);
        assertTrue(Files.exists(config.triageModelPath()), "model_path가 실제 파일을 가리켜야 함: "
                + config.triageModelPath());
        assertEquals(0.1, config.ruleThresholds().stemiElevationGeneralMv(), 1e-9);
        assertEquals(0.05, config.ruleThresholds().ischemiaDepressionMv(), 1e-9);
        assertEquals(2, config.ruleThresholds().ischemiaContiguousLeads());
    }

    @Test
    void pipelineBuiltFromConfigProcessesRealRecord() throws IOException {
        assumeTrue(Files.exists(CONFIG_YAML), "config/pipeline.yaml 미탑재 - 스킵");
        assumeTrue(Files.exists(PTBXL_HEADER), "PTB-XL 데이터셋 미탑재 - 스킵");

        PipelineConfig config = PipelineConfig.loadFromYaml(CONFIG_YAML);
        EkgPipeline pipeline = EkgPipeline.fromConfig(config);

        RawEcgSignal raw = WfdbRecordReader.read(PTBXL_HEADER);
        DiagnosticReport report = pipeline.process("cfg-test-1", "ptbxl:1", raw, Sex.UNKNOWN);

        assertEquals(ReportStatus.PENDING_SIGN, report.status());
        assertEquals(20, report.features().size());
    }

    @Test
    void stage3BecomesAvailableWhenModelPresentInConfig() throws IOException {
        // Phase 4 backbone 검증: config에 stage3_beat.onnx 경로가 있고 파일이 실제로 있으면
        // beatArrhythmiaAvailable=true가 되어야 한다(정확도는 검증되지 않았어도 배선은 동작해야 함).
        assumeTrue(Files.exists(CONFIG_YAML), "config/pipeline.yaml 미탑재 - 스킵");
        assumeTrue(Files.exists(PTBXL_HEADER), "PTB-XL 데이터셋 미탑재 - 스킵");

        PipelineConfig config = PipelineConfig.loadFromYaml(CONFIG_YAML);
        assumeTrue(config.beatArrhythmiaConfig().isAvailable(), "Stage3 모델 미탑재 - 스킵");

        try (EkgPipeline pipeline = EkgPipeline.fromConfig(config)) {
            RawEcgSignal raw = WfdbRecordReader.read(PTBXL_HEADER);
            DiagnosticReport report = pipeline.process("cfg-test-2", "ptbxl:1", raw, Sex.UNKNOWN);

            assertTrue(report.beatArrhythmiaAvailable(), "Stage3 모델이 있으면 available=true여야 함");
        }
    }
}
