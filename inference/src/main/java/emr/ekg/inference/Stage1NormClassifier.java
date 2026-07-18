package emr.ekg.inference;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import emr.ekg.signal.PreprocessedEcg;

import java.util.Arrays;
import java.util.Map;

/**
 * 1단계(정상 사전선별) ONNX 모델 추론 래퍼. Python(ecgml.models.stage1_norm)에서 학습해
 * export한 모델을 ONNX Runtime으로 로드해 P(NORM)을 산출한다.
 *
 * {@link ModelSignature}에 명시된 fs·리드순서와 실제 입력이 다르면 즉시 예외를 던진다 —
 * train/serve 불일치를 조용히 넘기지 않기 위함이다(P1).
 */
public final class Stage1NormClassifier implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final ModelSignature signature;

    public Stage1NormClassifier(ModelSignature signature) throws OrtException {
        this.signature = signature;
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(signature.modelPath(), new OrtSession.SessionOptions());
    }

    /** @return P(NORM), 0.0~1.0 */
    public double predictNormProbability(PreprocessedEcg ecg) throws OrtException {
        validateSignature(ecg);

        float[][] normalized = ZNormalizer.normalize(ecg.samples());
        float[][][] batched = {normalized};

        try (OnnxTensor input = OnnxTensor.createTensor(env, batched)) {
            try (OrtSession.Result result = session.run(Map.of(signature.inputName(), input))) {
                float[] logits = (float[]) result.get(0).getValue();
                return sigmoid(logits[0]);
            }
        }
    }

    private void validateSignature(PreprocessedEcg ecg) {
        if (ecg.fs() != signature.expectedFs()) {
            throw new IllegalArgumentException(
                    "fs 불일치: 모델 기대 %d, 실제 %d".formatted(signature.expectedFs(), ecg.fs()));
        }
        if (!Arrays.equals(ecg.leadNames(), signature.expectedLeads())) {
            throw new IllegalArgumentException("리드 순서 불일치: 모델 기대 %s, 실제 %s".formatted(
                    Arrays.toString(signature.expectedLeads()), Arrays.toString(ecg.leadNames())));
        }
    }

    private static double sigmoid(double logit) {
        return 1.0 / (1.0 + Math.exp(-logit));
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}
