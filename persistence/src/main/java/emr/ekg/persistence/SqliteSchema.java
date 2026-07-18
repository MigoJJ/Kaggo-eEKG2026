package emr.ekg.persistence;

import java.sql.Connection;
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
                        ts TEXT
                    )
                    """);
        }
    }
}
