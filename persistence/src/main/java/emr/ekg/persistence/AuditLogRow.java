package emr.ekg.persistence;

import java.time.Instant;

public record AuditLogRow(
        long id,
        String actor,
        String action,
        String target,
        String modelVersion,
        String modelHash,
        String modelMetadataHash,
        Instant ts) {
}
