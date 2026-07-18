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
                    beat_arrhythmia_available, st_ischemia_available, signed_by, signed_at, created_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(record_id) DO UPDATE SET status=excluded.status,
                    norm_triage_score=excluded.norm_triage_score,
                    triage_model_version=excluded.triage_model_version,
                    beat_arrhythmia_available=excluded.beat_arrhythmia_available,
                    st_ischemia_available=excluded.st_ischemia_available,
                    signed_by=excluded.signed_by, signed_at=excluded.signed_at,
                    created_at=excluded.created_at
                """)) {
            ps.setString(1, r.recordId());
            ps.setString(2, r.status().name());
            ps.setDouble(3, r.normTriageScore());
            ps.setString(4, r.triageModelVersion());
            ps.setInt(5, r.beatArrhythmiaAvailable() ? 1 : 0);
            ps.setInt(6, r.stIschemiaAvailable() ? 1 : 0);
            ps.setString(7, r.signedBy());
            ps.setString(8, r.signedAt() == null ? null : r.signedAt().toString());
            ps.setString(9, r.createdAt().toString());
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
                rs.getInt("beat_arrhythmia_available") != 0,
                rs.getInt("st_ischemia_available") != 0,
                rs.getString("signed_by"),
                signedAtStr == null ? null : Instant.parse(signedAtStr),
                Instant.parse(rs.getString("created_at")));
    }
}
