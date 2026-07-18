package emr.ekg.inference;

import ai.onnxruntime.OrtException;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class Stage4DelineatorClassifierTest {

    private static final Path MODEL_PATH = Path.of("/home/migowj/git/EKGGDSEMR2026/models/stage4_delineator.onnx");

    @Test
    void segments12LeadEcgIntoFourClasses() throws OrtException {
        assumeTrue(Files.exists(MODEL_PATH), "Stage4 모델 미탑재 - 스킵");

        try (Stage4DelineatorClassifier classifier = new Stage4DelineatorClassifier(MODEL_PATH)) {
            double[][] samples = new double[12][Stage4DelineatorClassifier.WINDOW_SAMPLES];
            for (int l = 0; l < 12; l++) {
                for (int i = 0; i < samples[l].length; i++) {
                    // 합성 사인/코사인 파동
                    samples[l][i] = Math.sin(2 * Math.PI * i / 100.0) + Math.cos(2 * Math.PI * i / 250.0);
                }
            }

            int[][] mask = classifier.classify(samples);
            assertNotNull(mask);
            assertEquals(12, mask.length);
            assertEquals(Stage4DelineatorClassifier.WINDOW_SAMPLES, mask[0].length);

            // 각 마스크가 4개 클래스 영역 안에 존재하는지 검증 (0: background, 1: p, 2: qrs, 3: t)
            for (int l = 0; l < 12; l++) {
                for (int t = 0; t < Stage4DelineatorClassifier.WINDOW_SAMPLES; t++) {
                    int cls = mask[l][t];
                    assertEquals(true, cls >= 0 && cls < 4, "알 수 없는 마스크 클래스 반환: " + cls);
                }
            }
        }
    }
}
