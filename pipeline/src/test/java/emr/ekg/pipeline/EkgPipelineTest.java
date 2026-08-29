package emr.ekg.pipeline;

import emr.ekg.features.Sex;
import emr.ekg.persistence.EmrDatabase;
import emr.ekg.persistence.ReportRepository;
import emr.ekg.persistence.ReportRow;
import emr.ekg.persistence.ReportStatus;
import emr.ekg.signal.RawEcgSignal;
import emr.ekg.signal.wfdb.WfdbRecordReader;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.SQLException;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class EkgPipelineTest {

    private static final Path MODEL_JSON = Path.of("/home/migojj/ittia/Kaggo-eEKG2026/models/stage1_logreg.json");
    private static final Path PTBXL_HEADER = Path.of(
            "/mnt/t7/datasets/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
                    + "/records500/00000/00001_hr.hea");

    @Test
    void processesRealRecordThroughToSignatureQueue() throws IOException {
        assumeTrue(Files.exists(MODEL_JSON), "학습된 1단계 모델 미탑재 - 스킵");
        assumeTrue(Files.exists(PTBXL_HEADER), "PTB-XL 데이터셋 미탑재 - 스킵");

        EkgPipeline pipeline = new EkgPipeline(MODEL_JSON, "II");
        RawEcgSignal raw = WfdbRecordReader.read(PTBXL_HEADER);

        DiagnosticReport report = pipeline.process("test-1", "ptbxl:1", raw, Sex.UNKNOWN);

        assertEquals(ReportStatus.PENDING_SIGN, report.status());
        assertTrue(report.interpretable());
        assertEquals(20, report.features().size());
        assertTrue(report.normTriageScore() >= 0.0 && report.normTriageScore() <= 1.0);
        assertEquals("stage1-logreg-v1", report.triageModelVersion());
        assertTrue(report.triageModelHash().matches("[0-9a-f]{64}"));
        assertFalse(report.beatArrhythmiaAvailable(), "Stage3는 Phase 4 이전에는 미구현이어야 함");
        assertFalse(report.stIschemiaAvailable(), "Stage4는 Phase 4 이전에는 미구현이어야 함");
    }

    @Test
    void rejectsFlatlineSignalOnSqi() throws IOException {
        assumeTrue(Files.exists(MODEL_JSON), "학습된 1단계 모델 미탑재 - 스킵");

        EkgPipeline pipeline = new EkgPipeline(MODEL_JSON, "II");

        double[][] flatline = new double[12][5000]; // 전부 0 -> flatline -> SQI 최저
        String[] leads = {"I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"};
        RawEcgSignal raw = new RawEcgSignal(flatline, 500, leads);

        DiagnosticReport report = pipeline.process("test-flat", "synthetic:flatline", raw, Sex.UNKNOWN);

        assertEquals(ReportStatus.REJECTED, report.status());
        assertFalse(report.interpretable());
        assertTrue(report.features().isEmpty());
    }

    @Test
    void persistsAndRoundTripsThroughSqlite() throws IOException, SQLException {
        assumeTrue(Files.exists(MODEL_JSON), "학습된 1단계 모델 미탑재 - 스킵");
        assumeTrue(Files.exists(PTBXL_HEADER), "PTB-XL 데이터셋 미탑재 - 스킵");

        EkgPipeline pipeline = new EkgPipeline(MODEL_JSON, "II");
        RawEcgSignal raw = WfdbRecordReader.read(PTBXL_HEADER);
        DiagnosticReport report = pipeline.process("test-2", "ptbxl:1", raw, Sex.UNKNOWN);

        try (EmrDatabase db = EmrDatabase.openInMemory()) {
            ReportRepository repository = new ReportRepository(db);
            new ReportPersister(repository).persist(report);

            Optional<ReportRow> found = repository.findReport("test-2");
            assertTrue(found.isPresent());
            assertEquals(ReportStatus.PENDING_SIGN, found.get().status());
            assertEquals(report.normTriageScore(), found.get().normTriageScore(), 1e-9);
            assertEquals(report.triageModelVersion(), found.get().triageModelVersion());
            assertEquals(report.triageModelHash(), found.get().triageModelHash());
            assertEquals(20, repository.findFeatures("test-2").size());
            assertEquals(1, repository.findAuditLogs("test-2").size());
            assertEquals(report.triageModelHash(), repository.findAuditLogs("test-2").get(0).modelHash());
        }
    }
}
