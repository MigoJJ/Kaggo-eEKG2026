package emr.ekg.inference;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/**
 * ONNX 모델 옆의 sidecar JSON 메타데이터. 파일이 없으면 모델 파일명 기반 fallback을 쓴다.
 */
public record ModelArtifactMetadata(
        String stage,
        String version,
        String modelType,
        String artifactSha256,
        String metadataSha256,
        String rawJson) {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    public static ModelArtifactMetadata loadForModel(Path modelPath, String fallbackStage, String fallbackHash)
            throws IOException {
        Path metadataPath = sidecarPath(modelPath);
        if (!Files.exists(metadataPath)) {
            return new ModelArtifactMetadata(
                    fallbackStage, modelPath.getFileName().toString(), null, fallbackHash, null, null);
        }

        String rawJson = Files.readString(metadataPath);
        JsonNode root = MAPPER.readTree(rawJson);
        String version = textOr(root, "version", modelPath.getFileName().toString());
        String stage = textOr(root, "stage", fallbackStage);
        String modelType = textOr(root, "model_type", null);
        String artifactSha256 = textOr(root, "artifact_sha256", fallbackHash);

        return new ModelArtifactMetadata(
                stage, version, modelType, artifactSha256, sha256Hex(rawJson.getBytes()), rawJson);
    }

    private static Path sidecarPath(Path modelPath) {
        String fileName = modelPath.getFileName().toString();
        int dot = fileName.lastIndexOf('.');
        String sidecarName = dot < 0 ? fileName + ".json" : fileName.substring(0, dot) + ".json";
        Path parent = modelPath.getParent();
        return parent == null ? Path.of(sidecarName) : parent.resolve(sidecarName);
    }

    private static String textOr(JsonNode root, String field, String fallback) {
        JsonNode value = root.get(field);
        return value == null || value.isNull() ? fallback : value.asText();
    }

    private static String sha256Hex(byte[] bytes) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(digest.digest(bytes));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 digest algorithm unavailable", e);
        }
    }
}
