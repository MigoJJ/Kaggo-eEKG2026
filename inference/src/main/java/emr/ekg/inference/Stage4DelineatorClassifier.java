package emr.ekg.inference;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.nio.file.Path;
import java.util.Map;

/**
 * Stage4(ST-T 허혈 정밀 및 QT 분할) — LUDB+QTDB로 학습한 12유도 P/QRS/T Delineator (1D U-Net).
 */
public final class Stage4DelineatorClassifier implements AutoCloseable {

    public static final String[] CLASS_NAMES = {"background", "p", "qrs", "t"};
    public static final int WINDOW_SAMPLES = 5000; // 10초 500Hz 고정

    private final OrtEnvironment env;
    private final OrtSession session;

    public Stage4DelineatorClassifier(Path modelPath) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelPath.toString(), new OrtSession.SessionOptions());
    }

    /**
     * @param samples 12유도 preprocessed ECG 신호 [12][5000] (Z-정규화 이전, mV 단위)
     * @return 각 리드, 각 샘플별 분할 마스크 [12][5000] (0: background, 1: p, 2: qrs, 3: t)
     */
    public int[][] classify(double[][] samples) throws OrtException {
        if (samples.length != 12) {
            throw new IllegalArgumentException("리드 개수는 반드시 12개여야 합니다: " + samples.length);
        }
        if (samples[0].length != WINDOW_SAMPLES) {
            throw new IllegalArgumentException("샘플 길이는 반드시 " + WINDOW_SAMPLES + "이어야 합니다: " + samples[0].length);
        }

        // Z-정규화 수행 (train/serve parity 보장)
        float[][] normalized = ZNormalizer.normalize(samples);
        float[][][] batched = {normalized}; // shape: [1, 12, 5000]

        try (OnnxTensor input = OnnxTensor.createTensor(env, batched)) {
            try (OrtSession.Result result = session.run(Map.of("ecg", input))) {
                // logits shape: [1, 4, 12, 5000]
                float[][][][] logits = (float[][][][]) result.get(0).getValue();
                float[][][] batchLogits = logits[0]; // shape: [4, 12, 5000]

                int[][] mask = new int[12][WINDOW_SAMPLES];
                for (int l = 0; l < 12; l++) {
                    for (int t = 0; t < WINDOW_SAMPLES; t++) {
                        // 4개 클래스 중 argmax 탐색
                        int bestClass = 0;
                        float bestVal = batchLogits[0][l][t];
                        for (int c = 1; c < 4; c++) {
                            if (batchLogits[c][l][t] > bestVal) {
                                bestVal = batchLogits[c][l][t];
                                bestClass = c;
                            }
                        }
                        mask[l][t] = bestClass;
                    }
                }
                return mask;
            }
        }
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}
