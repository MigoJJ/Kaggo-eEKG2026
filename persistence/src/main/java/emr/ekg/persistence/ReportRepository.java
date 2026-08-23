package emr.ekg.persistence;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * 진단 리포트 저장소. 레코드+피처+소견+리포트를 하나의 트랜잭션으로 저장한다.
 * 피처/소견은 레코드별로 전체 치환(delete-then-insert)한다 — 재판독 시 이전 결과가
 * 남아있지 않도록 하기 위함이다.
 */
public final class ReportRepository {

    private final Connection connection;

    public ReportRepository(EmrDatabase db) {
        this.connection = db.connection();
    }

    public void save(EcgRecordRow record, List<FeatureRow> features, List<FindingRow> findings, ReportRow report)
            throws SQLException {
        connection.setAutoCommit(false);
        try {
            upsertRecord(record);
            replaceFeatures(record.id(), features);
            replaceFindings(record.id(), findings);
            upsertReport(report);
            insertModelAuditLogs(report);
            connection.commit();
        } catch (SQLException e) {
            connection.rollback();
            throw e;
        } finally {
            connection.setAutoCommit(true);
        }
    }

    public Optional<ReportRow> findReport(String recordId) throws SQLException {
        try (PreparedStatement ps = connection.prepareStatement("SELECT * FROM report WHERE record_id=?")) {
            ps.setString(1, recordId);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next() ? Optional.of(mapReport(rs)) : Optional.empty();
            }
        }
    }

    public List<FeatureRow> findFeatures(String recordId) throws SQLException {
        List<FeatureRow> out = new ArrayList<>();
        try (PreparedStatement ps = connection.prepareStatement(
                "SELECT * FROM ecg_feature WHERE record_id=? ORDER BY rank")) {
            ps.setString(1, recordId);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    out.add(new FeatureRow(recordId, rs.getInt("rank"), rs.getString("name"),
                            rs.getDouble("value"), rs.getString("unit"),
                            rs.getDouble("normal_low"), rs.getDouble("normal_high")));
                }
            }
        }
        return out;
    }

    public List<FindingRow> findFindings(String recordId) throws SQLException {
        List<FindingRow> out = new ArrayList<>();
        try (PreparedStatement ps = connection.prepareStatement(
                "SELECT * FROM diagnosis WHERE record_id=?")) {
            ps.setString(1, recordId);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    out.add(new FindingRow(recordId, rs.getString("code"), rs.getString("label"),
                            rs.getString("severity"), rs.getString("evidence")));
                }
            }
        }
        return out;
    }

    public List<String> listPendingSignatureIds() throws SQLException {
        List<String> ids = new ArrayList<>();
        try (PreparedStatement ps = connection.prepareStatement("SELECT record_id FROM report WHERE status=?")) {
            ps.setString(1, ReportStatus.PENDING_SIGN.name());
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    ids.add(rs.getString("record_id"));
                }
            }
        }
        return ids;
    }

    public List<AuditLogRow> findAuditLogs(String targetPrefix) throws SQLException {
        List<AuditLogRow> out = new ArrayList<>();
        try (PreparedStatement ps = connection.prepareStatement(
                "SELECT * FROM audit_log WHERE target LIKE ? ORDER BY id")) {
            ps.setString(1, targetPrefix + "%");
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    out.add(new AuditLogRow(
                            rs.getLong("id"),
                            rs.getString("actor"),
                            rs.getString("action"),
                            rs.getString("target"),
                            rs.getString("model_version"),
                            rs.getString("model_hash"),
                            rs.getString("model_metadata_hash"),
                            Instant.parse(rs.getString("ts"))));
                }
            }
        }
        return out;
    }

    public Optional<ModelArtifactRow> findModelArtifact(String modelHash) throws SQLException {
        try (PreparedStatement ps = connection.prepareStatement("SELECT * FROM model_artifact WHERE model_hash=?")) {
            ps.setString(1, modelHash);
            try (ResultSet rs = ps.executeQuery()) {
                if (!rs.next()) {
                    return Optional.empty();
                }
                return Optional.of(new ModelArtifactRow(
                        rs.getString("model_hash"),
                        rs.getString("stage"),
                        rs.getString("model_version"),
                        rs.getString("metadata_hash"),
                        rs.getString("metadata_json"),
                        Instant.parse(rs.getString("registered_at"))));
            }
        }
    }

    public void sign(String recordId, String signedBy) throws SQLException {
        try (PreparedStatement ps = connection.prepareStatement(
                "UPDATE report SET status=?, signed_by=?, signed_at=? WHERE record_id=?")) {
            ps.setString(1, ReportStatus.SIGNED.name());
            ps.setString(2, signedBy);
            ps.setString(3, Instant.now().toString());
            ps.setString(4, recordId);
            ps.executeUpdate();
        }
    }

    private void upsertRecord(EcgRecordRow r) throws SQLException {
        try (PreparedStatement ps = connection.prepareStatement("""
                INSERT INTO ecg_record (id, source, fs, sample_count, sqi, interpretable, created_at)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET source=excluded.source, fs=excluded.fs,
                    sample_count=excluded.sample_count, sqi=excluded.sqi,
                    interpretable=excluded.interpretable, created_at=excluded.created_at
                """)) {
            ps.setString(1, r.id());
            ps.setString(2, r.source());
            ps.setInt(3, r.fs());
            ps.setInt(4, r.sampleCount());
            ps.setDouble(5, r.sqi());
            ps.setInt(6, r.interpretable() ? 1 : 0);
            ps.setString(7, r.createdAt().toString());
            ps.executeUpdate();
        }
    }

    private void replaceFeatures(String recordId, List<FeatureRow> features) throws SQLException {
        try (PreparedStatement del = connection.prepareStatement("DELETE FROM ecg_feature WHERE record_id=?")) {
            del.setString(1, recordId);
            del.executeUpdate();
        }
        try (PreparedStatement ps = connection.prepareStatement("""
                INSERT INTO ecg_feature (record_id, rank, name, value, unit, normal_low, normal_high)
                VALUES (?,?,?,?,?,?,?)
                """)) {
            for (FeatureRow f : features) {
                ps.setString(1, recordId);
                ps.setInt(2, f.rank());
                ps.setString(3, f.name());
                ps.setDouble(4, f.value());
                ps.setString(5, f.unit());
                ps.setDouble(6, f.normalLow());
                ps.setDouble(7, f.normalHigh());
                ps.addBatch();
            }
            ps.executeBatch();
        }
    }

    private void replaceFindings(String recordId, List<FindingRow> findings) throws SQLException {
        try (PreparedStatement del = connection.prepareStatement("DELETE FROM diagnosis WHERE record_id=?")) {
            del.setString(1, recordId);
            del.executeUpdate();
        }
        try (PreparedStatement ps = connection.prepareStatement("""
                INSERT INTO diagnosis (record_id, code, label, severity, evidence)
                VALUES (?,?,?,?,?)
                """)) {
            for (FindingRow f : findings) {
                ps.setString(1, recordId);
                ps.setString(2, f.code());
                ps.setString(3, f.label());
                ps.setString(4, f.severity());
                ps.setString(5, f.evidence());
                ps.addBatch();
            }
            ps.executeBatch();
        }
    }

    private void upsertReport(ReportRow r) throws SQLException {
        try (PreparedStatement ps = connection.prepareStatement("""
                INSERT INTO report (record_id, status, norm_triage_score, triage_model_version,
                    triage_model_hash, triage_model_metadata_hash,
                    beat_arrhythmia_model_version, beat_arrhythmia_model_hash, beat_arrhythmia_model_metadata_hash,
                    st_ischemia_model_version, st_ischemia_model_hash, st_ischemia_model_metadata_hash,
                    beat_arrhythmia_available, st_ischemia_available, signed_by, signed_at, created_at,
                    payload_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(record_id) DO UPDATE SET status=excluded.status,
                    norm_triage_score=excluded.norm_triage_score,
                    triage_model_version=excluded.triage_model_version,
                    triage_model_hash=excluded.triage_model_hash,
                    triage_model_metadata_hash=excluded.triage_model_metadata_hash,
                    beat_arrhythmia_model_version=excluded.beat_arrhythmia_model_version,
                    beat_arrhythmia_model_hash=excluded.beat_arrhythmia_model_hash,
                    beat_arrhythmia_model_metadata_hash=excluded.beat_arrhythmia_model_metadata_hash,
                    st_ischemia_model_version=excluded.st_ischemia_model_version,
                    st_ischemia_model_hash=excluded.st_ischemia_model_hash,
                    st_ischemia_model_metadata_hash=excluded.st_ischemia_model_metadata_hash,
                    beat_arrhythmia_available=excluded.beat_arrhythmia_available,
                    st_ischemia_available=excluded.st_ischemia_available,
                    signed_by=excluded.signed_by, signed_at=excluded.signed_at,
                    created_at=excluded.created_at, payload_json=excluded.payload_json
                """)) {
            ps.setString(1, r.recordId());
            ps.setString(2, r.status().name());
            ps.setDouble(3, r.normTriageScore());
            ps.setString(4, r.triageModelVersion());
            ps.setString(5, r.triageModelHash());
            ps.setString(6, r.triageModelMetadataHash());
            ps.setString(7, r.beatArrhythmiaModelVersion());
            ps.setString(8, r.beatArrhythmiaModelHash());
            ps.setString(9, r.beatArrhythmiaModelMetadataHash());
            ps.setString(10, r.stIschemiaModelVersion());
            ps.setString(11, r.stIschemiaModelHash());
            ps.setString(12, r.stIschemiaModelMetadataHash());
            ps.setInt(13, r.beatArrhythmiaAvailable() ? 1 : 0);
            ps.setInt(14, r.stIschemiaAvailable() ? 1 : 0);
            ps.setString(15, r.signedBy());
            ps.setString(16, r.signedAt() == null ? null : r.signedAt().toString());
            ps.setString(17, r.createdAt().toString());
            ps.setString(18, r.payloadJson());
            ps.executeUpdate();
        }
    }

    private void insertModelAuditLogs(ReportRow report) throws SQLException {
        insertModelAuditLog(report.recordId(), "stage1", report.triageModelVersion(), report.triageModelHash(),
                report.triageModelMetadataHash(), report.createdAt());
        upsertModelArtifact("stage1", report.triageModelVersion(), report.triageModelHash(),
                report.triageModelMetadataHash(), report.triageModelMetadataJson(), report.createdAt());
        insertModelAuditLog(report.recordId(), "stage3", report.beatArrhythmiaModelVersion(),
                report.beatArrhythmiaModelHash(), report.beatArrhythmiaModelMetadataHash(), report.createdAt());
        upsertModelArtifact("stage3", report.beatArrhythmiaModelVersion(), report.beatArrhythmiaModelHash(),
                report.beatArrhythmiaModelMetadataHash(), report.beatArrhythmiaModelMetadataJson(), report.createdAt());
        insertModelAuditLog(report.recordId(), "stage4", report.stIschemiaModelVersion(),
                report.stIschemiaModelHash(), report.stIschemiaModelMetadataHash(), report.createdAt());
        upsertModelArtifact("stage4", report.stIschemiaModelVersion(), report.stIschemiaModelHash(),
                report.stIschemiaModelMetadataHash(), report.stIschemiaModelMetadataJson(), report.createdAt());
    }

    private void insertModelAuditLog(String recordId, String stage, String modelVersion, String modelHash,
            String metadataHash, Instant ts) throws SQLException {
        if (modelHash == null || modelHash.isBlank()) {
            return;
        }
        try (PreparedStatement ps = connection.prepareStatement("""
                INSERT INTO audit_log (actor, action, target, model_version, model_hash, model_metadata_hash, ts)
                VALUES (?,?,?,?,?,?,?)
                """)) {
            ps.setString(1, "pipeline");
            ps.setString(2, "MODEL_USED");
            ps.setString(3, recordId + ":" + stage);
            ps.setString(4, modelVersion);
            ps.setString(5, modelHash);
            ps.setString(6, metadataHash);
            ps.setString(7, ts.toString());
            ps.executeUpdate();
        }
    }

    private void upsertModelArtifact(String stage, String modelVersion, String modelHash, String metadataHash,
            String metadataJson, Instant registeredAt) throws SQLException {
        if (modelHash == null || modelHash.isBlank()) {
            return;
        }
        try (PreparedStatement ps = connection.prepareStatement("""
                INSERT INTO model_artifact (model_hash, stage, model_version, metadata_hash, metadata_json, registered_at)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(model_hash) DO UPDATE SET stage=excluded.stage,
                    model_version=excluded.model_version,
                    metadata_hash=excluded.metadata_hash,
                    metadata_json=excluded.metadata_json,
                    registered_at=excluded.registered_at
                """)) {
            ps.setString(1, modelHash);
            ps.setString(2, stage);
            ps.setString(3, modelVersion);
            ps.setString(4, metadataHash);
            ps.setString(5, metadataJson);
            ps.setString(6, registeredAt.toString());
            ps.executeUpdate();
        }
    }

    private static ReportRow mapReport(ResultSet rs) throws SQLException {
        String signedAtStr = rs.getString("signed_at");
        return new ReportRow(
                rs.getString("record_id"),
                ReportStatus.valueOf(rs.getString("status")),
                rs.getDouble("norm_triage_score"),
                rs.getString("triage_model_version"),
                rs.getString("triage_model_hash"),
                rs.getString("triage_model_metadata_hash"),
                null,
                rs.getString("beat_arrhythmia_model_version"),
                rs.getString("beat_arrhythmia_model_hash"),
                rs.getString("beat_arrhythmia_model_metadata_hash"),
                null,
                rs.getString("st_ischemia_model_version"),
                rs.getString("st_ischemia_model_hash"),
                rs.getString("st_ischemia_model_metadata_hash"),
                null,
                rs.getInt("beat_arrhythmia_available") != 0,
                rs.getInt("st_ischemia_available") != 0,
                rs.getString("signed_by"),
                signedAtStr == null ? null : Instant.parse(signedAtStr),
                Instant.parse(rs.getString("created_at")),
                rs.getString("payload_json"));
    }
}
