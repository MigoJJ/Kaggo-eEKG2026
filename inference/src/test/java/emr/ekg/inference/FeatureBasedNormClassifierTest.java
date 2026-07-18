package emr.ekg.inference;

import emr.ekg.features.ClinicalFeature;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FeatureBasedNormClassifierTest {

    @Test
    void computesProbabilityFromKnownWeights(@org.junit.jupiter.api.io.TempDir Path tempDir) throws IOException {
        // 2피처 합성 모델: HR(mean60,std10), QRS_DURATION(mean90,std10), weight=[1,-1], bias=0
        String json = """
                {
                  "model_type": "logistic_regression",
                  "feature_order": ["HR", "QRS_DURATION"],
                  "impute_median": [60.0, 90.0],
                  "standardize_mean": [60.0, 90.0],
                  "standardize_std": [10.0, 10.0],
                  "weights": [1.0, -1.0],
                  "bias": 0.0
                }
                """;
        Path jsonPath = tempDir.resolve("model.json");
        Files.writeString(jsonPath, json);

        FeatureBasedNormClassifier classifier = FeatureBasedNormClassifier.loadFromJson(jsonPath);

        // HR=70 -> z=1.0, QRS=90 -> z=0.0 => logit = 1*1.0 + (-1)*0.0 = 1.0 => sigmoid(1.0)
        List<ClinicalFeature> features = List.of(
                new ClinicalFeature(1, "HR", 70.0, "bpm", 60, 100),
                new ClinicalFeature(5, "QRS_DURATION", 90.0, "ms", 70, 100));

        double prob = classifier.predictNormProbability(features);
        double expected = 1.0 / (1.0 + Math.exp(-1.0));
        assertEquals(expected, prob, 1e-9);
    }

    @Test
    void missingFeatureFallsBackToImputeMedian(@org.junit.jupiter.api.io.TempDir Path tempDir) throws IOException {
        String json = """
                {
                  "model_type": "logistic_regression",
                  "feature_order": ["HR", "QRS_DURATION"],
                  "impute_median": [60.0, 90.0],
                  "standardize_mean": [60.0, 90.0],
                  "standardize_std": [10.0, 10.0],
                  "weights": [1.0, -1.0],
                  "bias": 0.0
                }
                """;
        Path jsonPath = tempDir.resolve("model.json");
        Files.writeString(jsonPath, json);
        FeatureBasedNormClassifier classifier = FeatureBasedNormClassifier.loadFromJson(jsonPath);

        // QRS_DURATION 누락 -> impute_median(90) 사용 -> z=0 -> logit = 1*z(HR=60)=0 + 0 = 0 => sigmoid(0)=0.5
        List<ClinicalFeature> features = List.of(new ClinicalFeature(1, "HR", 60.0, "bpm", 60, 100));

        double prob = classifier.predictNormProbability(features);
        assertEquals(0.5, prob, 1e-9);
    }

    @Test
    void explainContributionsMatchesLogitDecomposition(@org.junit.jupiter.api.io.TempDir Path tempDir)
            throws IOException {
        String json = """
                {
                  "model_type": "logistic_regression",
                  "feature_order": ["HR", "QRS_DURATION"],
                  "impute_median": [60.0, 90.0],
                  "standardize_mean": [60.0, 90.0],
                  "standardize_std": [10.0, 10.0],
                  "weights": [2.0, 3.0],
                  "bias": 0.5
                }
                """;
        Path jsonPath = tempDir.resolve("model.json");
        Files.writeString(jsonPath, json);
        FeatureBasedNormClassifier classifier = FeatureBasedNormClassifier.loadFromJson(jsonPath);

        List<ClinicalFeature> features = List.of(
                new ClinicalFeature(1, "HR", 70.0, "bpm", 60, 100),
                new ClinicalFeature(5, "QRS_DURATION", 100.0, "ms", 70, 100));

        Map<String, Double> contributions = classifier.explainContributions(features);
        // HR: z=1.0*w2=2.0, QRS: z=1.0*w3=3.0
        assertEquals(2.0, contributions.get("HR"), 1e-9);
        assertEquals(3.0, contributions.get("QRS_DURATION"), 1e-9);

        double sumPlusBias = contributions.values().stream().mapToDouble(Double::doubleValue).sum() + 0.5;
        double expectedProb = 1.0 / (1.0 + Math.exp(-sumPlusBias));
        assertTrue(Math.abs(classifier.predictNormProbability(features) - expectedProb) < 1e-9);
    }
}
