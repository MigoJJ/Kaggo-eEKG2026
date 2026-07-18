package emr.ekg.inference;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.nio.file.Path;
import java.util.Map;

/**
 * Stage3(비트 부정맥 검증) — MIT-BIH로 학습한 AAMI 5-클래스(N/S/V/F/Q) 단일비트 분류기.
 *
 * ⚠️ Phase 4 backbone 단계 모델이다 — 파이프라인 연동(ONNX 로드→분류→소견 산출)이 실제로
 * 동작함을 증명하는 것이 목적이며, 정확도는 검증되지 않았다(학습 시 혼동행렬 기준 N/F 혼동
 * 심함). Kaggle P100 재학습 후 정확도 보정 예정.
 */
public final class BeatArrhythmiaClassifier implements AutoCloseable {

    public static final String[] CLASS_NAMES = {"N", "S", "V", "F", "Q"};
    public static final int WINDOW_SAMPLES = 200; // 500Hz 기준 R-peak 중심 ±200ms

    private final OrtEnvironment env;
    private final OrtSession session;

    public BeatArrhythmiaClassifier(Path modelPath) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelPath.toString(), new OrtSession.SessionOptions());
    }

    /** @param window 길이 {@link #WINDOW_SAMPLES}의 리드 II 비트 윈도우(R-peak 중심, 정규화 전) */
    public String classify(double[] window) throws OrtException {
        float[] normalized = zNormalize(window);
        float[][][] batched = {{normalized}};

        try (OnnxTensor input = OnnxTensor.createTensor(env, batched)) {
            try (OrtSession.Result result = session.run(Map.of("beat_window", input))) {
                float[][] logits = (float[][]) result.get(0).getValue();
                return CLASS_NAMES[argmax(logits[0])];
            }
        }
    }

    private static float[] zNormalize(double[] window) {
        double mean = 0;
        for (double v : window) {
            mean += v;
        }
        mean /= window.length;

        double variance = 0;
        for (double v : window) {
            variance += (v - mean) * (v - mean);
        }
        variance /= window.length;
        double std = Math.sqrt(variance);
        if (std < 1e-9) {
            std = 1.0;
        }

        float[] out = new float[window.length];
        for (int i = 0; i < window.length; i++) {
            out[i] = (float) ((window[i] - mean) / std);
        }
        return out;
    }

    private static int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}
