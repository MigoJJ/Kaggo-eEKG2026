package emr.ekg.inference;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import emr.ekg.features.ClinicalFeature;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 1단계(정상 사전선별) 20개 임상 피처 기반 로지스틱회귀 분류기.
 *
 * ONNX Runtime 없이 Python(train_stage1_features.py)에서 export된 JSON 가중치를 직접
 * 적용한다. raw waveform CNN/Transformer 대신 이 경로를 우선하는 이유는 완전한 감사
 * 가능성(P4)이다 — 결측치 대체값·정규화 통계·가중치가 전부 평문 JSON으로 보관되어
 * 검토·설명이 가능하고, {@link #explainContributions}로 피처별 기여도까지 리포트할 수 있다.
 */
public final class FeatureBasedNormClassifier {

    private final String version;
    private final List<String> featureOrder;
    private final double[] imputeMedian;
    private final double[] standardizeMean;
    private final double[] standardizeStd;
    private final double[] weights;
    private final double bias;

    private FeatureBasedNormClassifier(String version, List<String> featureOrder, double[] imputeMedian,
            double[] standardizeMean, double[] standardizeStd, double[] weights, double bias) {
        this.version = version;
        this.featureOrder = featureOrder;
        this.imputeMedian = imputeMedian;
        this.standardizeMean = standardizeMean;
        this.standardizeStd = standardizeStd;
        this.weights = weights;
        this.bias = bias;
    }

    public static FeatureBasedNormClassifier loadFromJson(Path jsonPath) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root = mapper.readTree(jsonPath.toFile());

        List<String> featureOrder = new ArrayList<>();
        root.get("feature_order").forEach(n -> featureOrder.add(n.asText()));
        int n = featureOrder.size();
        String version = root.has("version") ? root.get("version").asText() : "unknown";

        return new FeatureBasedNormClassifier(
                version,
                featureOrder,
                toArray(root.get("impute_median"), n),
                toArray(root.get("standardize_mean"), n),
                toArray(root.get("standardize_std"), n),
                toArray(root.get("weights"), n),
                root.get("bias").asDouble());
    }

    /** 감사로그·리포트에 동봉하는 모델 버전 (P6). */
    public String version() {
        return version;
    }

    private static double[] toArray(JsonNode arrayNode, int expectedLength) {
        double[] out = new double[expectedLength];
        for (int i = 0; i < expectedLength; i++) {
            out[i] = arrayNode.get(i).asDouble();
        }
        return out;
    }

    /** @return P(NORM), 0.0~1.0 */
    public double predictNormProbability(List<ClinicalFeature> features) {
        Map<String, Double> byName = valueByName(features);

        double logit = bias;
        for (int i = 0; i < featureOrder.size(); i++) {
            logit += weights[i] * normalized(byName, i);
        }
        return 1.0 / (1.0 + Math.exp(-logit));
    }

    /** 근거 리포트용: 각 피처가 최종 logit에 기여한 값(가중치×정규화값), 절대값 큰 순으로 설명 가능. */
    public Map<String, Double> explainContributions(List<ClinicalFeature> features) {
        Map<String, Double> byName = valueByName(features);

        Map<String, Double> contributions = new LinkedHashMap<>();
        for (int i = 0; i < featureOrder.size(); i++) {
            contributions.put(featureOrder.get(i), weights[i] * normalized(byName, i));
        }
        return contributions;
    }

    private double normalized(Map<String, Double> byName, int i) {
        Double raw = byName.get(featureOrder.get(i));
        double value = (raw == null || Double.isNaN(raw)) ? imputeMedian[i] : raw;
        return (value - standardizeMean[i]) / standardizeStd[i];
    }

    private static Map<String, Double> valueByName(List<ClinicalFeature> features) {
        Map<String, Double> byName = new HashMap<>();
        for (ClinicalFeature f : features) {
            byName.put(f.name(), f.value());
        }
        return byName;
    }
}
