package emr.ekg.inference;

import ai.onnxruntime.OrtException;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class BeatArrhythmiaClassifierTest {

    private static final Path MODEL_PATH = Path.of("/home/migojj/ittia/Kaggo-eEKG2026/models/stage3_beat.onnx");

    @Test
    void classifiesSyntheticBeatWindowIntoKnownClass() throws OrtException {
        assumeTrue(Files.exists(MODEL_PATH), "Stage3 모델 미탑재 - 스킵");

        try (BeatArrhythmiaClassifier classifier = new BeatArrhythmiaClassifier(MODEL_PATH)) {
            double[] window = new double[BeatArrhythmiaClassifier.WINDOW_SAMPLES];
            for (int i = 0; i < window.length; i++) {
                // 대략적인 QRS 유사 스파이크가 있는 합성 신호
                window[i] = Math.exp(-Math.pow((i - window.length / 2.0) / 10.0, 2));
            }

            String result = classifier.classify(window);
            assertTrue(Arrays.asList(BeatArrhythmiaClassifier.CLASS_NAMES).contains(result),
                    "알 수 없는 클래스 반환: " + result);
        }
    }
}
