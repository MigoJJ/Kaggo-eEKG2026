package emr.ekg.inference;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

class ModelArtifactMetadataTest {

    @Test
    void loadsSidecarNextToOnnx(@TempDir Path tempDir) throws IOException {
        Path model = tempDir.resolve("stage3_beat.onnx");
        Files.writeString(model, "fake-onnx");
        Files.writeString(tempDir.resolve("stage3_beat.json"), """
                {
                  "stage": "stage3",
                  "version": "stage3-resnet-test",
                  "model_type": "resnet",
                  "artifact_sha256": "artifact-hash",
                  "environment": {"python": "3.11"}
                }
                """);

        ModelArtifactMetadata metadata = ModelArtifactMetadata.loadForModel(model, "stage3", "fallback-hash");

        assertEquals("stage3", metadata.stage());
        assertEquals("stage3-resnet-test", metadata.version());
        assertEquals("resnet", metadata.modelType());
        assertEquals("artifact-hash", metadata.artifactSha256());
        assertNotNull(metadata.metadataSha256());
        assertNotNull(metadata.rawJson());
    }

    @Test
    void fallsBackWhenSidecarIsMissing(@TempDir Path tempDir) throws IOException {
        Path model = tempDir.resolve("stage4_delineator.onnx");
        Files.writeString(model, "fake-onnx");

        ModelArtifactMetadata metadata = ModelArtifactMetadata.loadForModel(model, "stage4", "fallback-hash");

        assertEquals("stage4", metadata.stage());
        assertEquals("stage4_delineator.onnx", metadata.version());
        assertEquals("fallback-hash", metadata.artifactSha256());
        assertNull(metadata.metadataSha256());
        assertNull(metadata.rawJson());
    }
}
