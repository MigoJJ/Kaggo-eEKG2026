package emr.ekg.persistence;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

/** SQLite 연결 + 스키마 적용 파사드. */
public final class EmrDatabase implements AutoCloseable {

    private final Connection connection;

    private EmrDatabase(Connection connection) {
        this.connection = connection;
    }

    public static EmrDatabase open(String jdbcUrl) throws SQLException {
        Connection connection = DriverManager.getConnection(jdbcUrl);
        SqliteSchema.apply(connection);
        return new EmrDatabase(connection);
    }

    public static EmrDatabase openFile(Path dbFile) throws SQLException {
        return open("jdbc:sqlite:" + dbFile.toAbsolutePath());
    }

    public static EmrDatabase openInMemory() throws SQLException {
        return open("jdbc:sqlite::memory:");
    }

    Connection connection() {
        return connection;
    }

    @Override
    public void close() throws SQLException {
        connection.close();
    }
}
