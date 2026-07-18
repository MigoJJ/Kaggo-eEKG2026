package emr.ekg.persistence;

import org.junit.jupiter.api.Test;

import java.sql.SQLException;
import java.time.Instant;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ReportRepositoryTest {

    @Test
    void savesAndRetrievesFullReport() throws SQLException {
        try (EmrDatabase db = EmrDatabase.openInMemory()) {
            ReportRepository repo = new ReportRepository(db);

            EcgRecordRow record = new EcgRecordRow("rec-1", "ptbxl:1", 500, 5000, 0.95, true, Instant.now());
            List<FeatureRow> features = List.of(
                    new FeatureRow("rec-1", 1, "HR", 72.0, "bpm", 60, 100),
                    new FeatureRow("rec-1", 5, "QRS_DURATION", 90.0, "ms", 70, 100));
            List<FindingRow> findings = List.of(
                    new FindingRow("rec-1", "AVB1", "1도 방실차단", "ABNORMAL", "PR 220ms"));
            ReportRow report = new ReportRow("rec-1", ReportStatus.PENDING_SIGN, 0.83, "logreg-v1",
                    "abc123", "meta123", "{\"version\":\"logreg-v1\"}",
                    null, null, null, null, null, null, null, null, false, false, null, null, Instant.now());

            repo.save(record, features, findings, report);

            Optional<ReportRow> found = repo.findReport("rec-1");
            assertTrue(found.isPresent());
            assertEquals(ReportStatus.PENDING_SIGN, found.get().status());
            assertEquals(0.83, found.get().normTriageScore(), 1e-9);
            assertEquals("logreg-v1", found.get().triageModelVersion());
            assertEquals("abc123", found.get().triageModelHash());

            assertEquals(2, repo.findFeatures("rec-1").size());
            assertEquals(1, repo.findFindings("rec-1").size());
            assertEquals("AVB1", repo.findFindings("rec-1").get(0).code());

            List<AuditLogRow> auditLogs = repo.findAuditLogs("rec-1");
            assertEquals(1, auditLogs.size());
            assertEquals("MODEL_USED", auditLogs.get(0).action());
            assertEquals("rec-1:stage1", auditLogs.get(0).target());
            assertEquals("logreg-v1", auditLogs.get(0).modelVersion());
            assertEquals("abc123", auditLogs.get(0).modelHash());
            assertEquals("meta123", auditLogs.get(0).modelMetadataHash());

            Optional<ModelArtifactRow> artifact = repo.findModelArtifact("abc123");
            assertTrue(artifact.isPresent());
            assertEquals("stage1", artifact.get().stage());
            assertEquals("logreg-v1", artifact.get().modelVersion());
            assertEquals("meta123", artifact.get().metadataHash());
            assertEquals("{\"version\":\"logreg-v1\"}", artifact.get().metadataJson());
        }
    }

    @Test
    void resavingReplacesFeaturesAndFindingsNotDuplicates() throws SQLException {
        try (EmrDatabase db = EmrDatabase.openInMemory()) {
            ReportRepository repo = new ReportRepository(db);
            EcgRecordRow record = new EcgRecordRow("rec-2", "ptbxl:2", 500, 5000, 0.95, true, Instant.now());
            ReportRow report = new ReportRow("rec-2", ReportStatus.PENDING_SIGN, 0.5, "logreg-v1",
                    "abc123", null, null, null, null, null, null, null, null, null, null,
                    false, false, null, null, Instant.now());

            repo.save(record, List.of(new FeatureRow("rec-2", 1, "HR", 70.0, "bpm", 60, 100)),
                    List.of(new FindingRow("rec-2", "AVB1", "1도 방실차단", "ABNORMAL", "PR 220ms")), report);
            repo.save(record, List.of(new FeatureRow("rec-2", 1, "HR", 75.0, "bpm", 60, 100)),
                    List.of(), report);

            assertEquals(1, repo.findFeatures("rec-2").size());
            assertEquals(75.0, repo.findFeatures("rec-2").get(0).value(), 1e-9);
            assertEquals(0, repo.findFindings("rec-2").size());
        }
    }

    @Test
    void signMovesReportFromPendingToSigned() throws SQLException {
        try (EmrDatabase db = EmrDatabase.openInMemory()) {
            ReportRepository repo = new ReportRepository(db);
            EcgRecordRow record = new EcgRecordRow("rec-3", "ptbxl:3", 500, 5000, 0.95, true, Instant.now());
            ReportRow report = new ReportRow("rec-3", ReportStatus.PENDING_SIGN, 0.9, "logreg-v1",
                    "abc123", null, null, null, null, null, null, null, null, null, null,
                    false, false, null, null, Instant.now());
            repo.save(record, List.of(), List.of(), report);

            assertEquals(1, repo.listPendingSignatureIds().size());

            repo.sign("rec-3", "dr.kim");

            assertEquals(0, repo.listPendingSignatureIds().size());
            Optional<ReportRow> signed = repo.findReport("rec-3");
            assertEquals(ReportStatus.SIGNED, signed.get().status());
            assertEquals("dr.kim", signed.get().signedBy());
        }
    }
}
