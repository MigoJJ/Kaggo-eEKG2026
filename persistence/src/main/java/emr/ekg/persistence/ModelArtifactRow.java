package emr.ekg.persistence;

import java.time.Instant;

public record ModelArtifactRow(
        String modelHash,
        String stage,
        String modelVersion,
        String metadataHash,
        String metadataJson,
        Instant registeredAt) {
}
