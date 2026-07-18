package emr.ekg.persistence;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

/** SQLite 스키마 마이그레이션. 멱등(idempotent) — 이미 존재하면 건너뛴다. */
final class SqliteSchema {

    private SqliteSchema() {
    }

    static void apply(Connection connection) throws SQLException {
        try (Statement stmt = connection.createStatement()) {
            stmt.execute("""
                    CREATE TABLE IF NOT EXISTS ecg_record (
                        id TEXT PRIMARY KEY,
                        source TEXT,
                        fs INTEGER,
                        sample_count INTEGER,
                        sqi REAL,
                        interpretable INTEGER,
                        created_at TEXT
                    )
                    """);

            stmt.execute("""
                    CREATE TABLE IF NOT EXISTS ecg_feature (
                        record_id TEXT NOT NULL,
                        rank INTEGER,
                        name TEXT,
                        value REAL,
                        unit TEXT,
                        normal_low REAL,
                        normal_high REAL,
                        FOREIGN KEY(record_id) REFERENCES ecg_record(id)
                    )
                    """);

            stmt.execute("""
                    CREATE TABLE IF NOT EXISTS diagnosis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        record_id TEXT NOT NULL,
                        code TEXT,
                        label TEXT,
                        severity TEXT,
                        evidence TEXT,
                        FOREIGN KEY(record_id) REFERENCES ecg_record(id)
                    )
                    """);

            stmt.execute("""
                    CREATE TABLE IF NOT EXISTS report (
                        record_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        norm_triage_score REAL,
                        triage_model_version TEXT,
                        triage_model_hash TEXT,
                        triage_model_metadata_hash TEXT,
                        beat_arrhythmia_model_version TEXT,
                        beat_arrhythmia_model_hash TEXT,
                        beat_arrhythmia_model_metadata_hash TEXT,
                        st_ischemia_model_version TEXT,
                        st_ischemia_model_hash TEXT,
                        st_ischemia_model_metadata_hash TEXT,
                        beat_arrhythmia_available INTEGER,
                        st_ischemia_available INTEGER,
                        signed_by TEXT,
                        signed_at TEXT,
                        created_at TEXT,
                        FOREIGN KEY(record_id) REFERENCES ecg_record(id)
                    )
                    """);

            stmt.execute("""
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        actor TEXT,
                        action TEXT,
                        target TEXT,
                        model_version TEXT,
                        model_hash TEXT,
                        model_metadata_hash TEXT,
                        ts TEXT
                    )
                    """);

            stmt.execute("""
                    CREATE TABLE IF NOT EXISTS model_artifact (
                        model_hash TEXT PRIMARY KEY,
                        stage TEXT,
                        model_version TEXT,
                        metadata_hash TEXT,
                        metadata_json TEXT,
                        registered_at TEXT
                    )
                    """);

            addColumnIfMissing(connection, "report", "triage_model_hash", "TEXT");
            addColumnIfMissing(connection, "report", "triage_model_metadata_hash", "TEXT");
            addColumnIfMissing(connection, "report", "beat_arrhythmia_model_version", "TEXT");
            addColumnIfMissing(connection, "report", "beat_arrhythmia_model_hash", "TEXT");
            addColumnIfMissing(connection, "report", "beat_arrhythmia_model_metadata_hash", "TEXT");
            addColumnIfMissing(connection, "report", "st_ischemia_model_version", "TEXT");
            addColumnIfMissing(connection, "report", "st_ischemia_model_hash", "TEXT");
            addColumnIfMissing(connection, "report", "st_ischemia_model_metadata_hash", "TEXT");
            addColumnIfMissing(connection, "audit_log", "model_version", "TEXT");
            addColumnIfMissing(connection, "audit_log", "model_hash", "TEXT");
            addColumnIfMissing(connection, "audit_log", "model_metadata_hash", "TEXT");
        }
    }

    private static void addColumnIfMissing(Connection connection, String table, String column, String type)
            throws SQLException {
        try (Statement stmt = connection.createStatement()) {
            try (ResultSet rs = stmt.executeQuery("PRAGMA table_info(" + table + ")")) {
                while (rs.next()) {
                    if (column.equals(rs.getString("name"))) {
                        return;
                    }
                }
            }
            stmt.execute("ALTER TABLE " + table + " ADD COLUMN " + column + " " + type);
        }
    }
}
