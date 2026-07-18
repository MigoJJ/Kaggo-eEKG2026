package emr.ekg.rules;

import emr.ekg.features.ClinicalFeature;
import emr.ekg.features.morphology.LeadMorphology;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Stage 2 룰 로직에 대한 결정론적 검증. 실측 신호가 아닌 합성(hand-crafted) 값을 사용해
 * 각 규칙의 경계조건을 정확히 통제한다(FiducialExtractorTest 등은 실데이터 구조 검증 담당).
 */
class ConductionRuleEngineTest {

    @Test
    void detectsRbbbFromV1RsrPatternAndWideQrs() {
        List<ClinicalFeature> features = baselineNormalFeatures();
        features = withValue(features, "QRS_DURATION", 140);

        Map<String, LeadMorphology> morphology = new HashMap<>();
        morphology.put("V1", new LeadMorphology("V1", 0.3, 0.4, 0, 0, 0, 0));
        morphology.put("V6", new LeadMorphology("V6", 1.0, 0.2, 0, 0, 0, 0));

        List<Finding> findings = ConductionRuleEngine.evaluate(features, morphology);
        assertTrue(findings.stream().anyMatch(f -> f.code().equals("RBBB")));
    }

    @Test
    void detectsLbbbFromBroadRInLateralLeads() {
        List<ClinicalFeature> features = baselineNormalFeatures();
        features = withValue(features, "QRS_DURATION", 145);

        Map<String, LeadMorphology> morphology = new HashMap<>();
        morphology.put("V1", new LeadMorphology("V1", 0.1, 0.0, 0, 0, 0, 0)); // RBBB 패턴 불일치
        morphology.put("V5", new LeadMorphology("V5", 1.2, 0.05, 0, 0, 0, 0));
        morphology.put("V6", new LeadMorphology("V6", 1.3, 0.05, 0, 0, 0, 0));
        morphology.put("I", new LeadMorphology("I", 1.1, 0.05, 0, 0, 0, 0));

        List<Finding> findings = ConductionRuleEngine.evaluate(features, morphology);
        assertTrue(findings.stream().anyMatch(f -> f.code().equals("LBBB")));
    }

    @Test
    void detectsFirstDegreeAvBlockWhenPrExceeds200ms() {
        List<ClinicalFeature> features = withValue(baselineNormalFeatures(), "PR_INTERVAL", 220);
        List<Finding> findings = ConductionRuleEngine.evaluate(features, Map.of());
        assertTrue(findings.stream().anyMatch(f -> f.code().equals("AVB1")));
    }

    @Test
    void doesNotFlagAvBlockWhenPrIsNormal() {
        List<ClinicalFeature> features = withValue(baselineNormalFeatures(), "PR_INTERVAL", 160);
        List<Finding> findings = ConductionRuleEngine.evaluate(features, Map.of());
        assertFalse(findings.stream().anyMatch(f -> f.code().equals("AVB1")));
    }

    @Test
    void detectsLvhBySokolowLyonIndex() {
        List<ClinicalFeature> features = withValue(baselineNormalFeatures(), "SOKOLOW_LYON_INDEX", 4.0);
        List<Finding> findings = ConductionRuleEngine.evaluate(features, Map.of());
        assertTrue(findings.stream().anyMatch(f -> f.code().equals("LVH")));
    }

    @Test
    void flagsPathologicQInInferiorLeadButExcludesAvr() {
        Map<String, LeadMorphology> morphology = new HashMap<>();
        morphology.put("III", new LeadMorphology("III", 1.0, 0.0, 50, 0.3, 0, 0));
        morphology.put("AVR", new LeadMorphology("AVR", 0.0, 1.0, 50, 0.9, 0, 0));

        List<Finding> findings = ConductionRuleEngine.evaluate(baselineNormalFeatures(), morphology);

        assertTrue(findings.stream().anyMatch(f -> f.code().equals("PATHOLOGIC_Q") && f.evidence().contains("III")));
        assertFalse(findings.stream().anyMatch(f -> f.code().equals("PATHOLOGIC_Q") && f.evidence().contains("AVR")));
    }

    @Test
    void detectsLeftAxisDeviationFromLeadIAndAvfPolarity() {
        Map<String, LeadMorphology> morphology = new HashMap<>();
        morphology.put("I", new LeadMorphology("I", 1.0, 0.0, 0, 0, 0, 0));
        morphology.put("AVF", new LeadMorphology("AVF", 0.0, 1.0, 0, 0, 0, 0));

        List<Finding> findings = ConductionRuleEngine.evaluate(baselineNormalFeatures(), morphology);
        assertTrue(findings.stream().anyMatch(f -> f.code().equals("LAD")));
    }

    private static List<ClinicalFeature> baselineNormalFeatures() {
        return new java.util.ArrayList<>(List.of(
                feature("QRS_DURATION", 90),
                feature("PR_INTERVAL", 160),
                feature("P_DURATION", 90),
                feature("P_AMPLITUDE_II", 0.1),
                feature("SOKOLOW_LYON_INDEX", 2.0),
                feature("R_AMPLITUDE_AVL", 0.5)));
    }

    private static List<ClinicalFeature> withValue(List<ClinicalFeature> base, String name, double value) {
        List<ClinicalFeature> out = new java.util.ArrayList<>();
        for (ClinicalFeature f : base) {
            out.add(f.name().equals(name) ? feature(name, value) : f);
        }
        return out;
    }

    private static ClinicalFeature feature(String name, double value) {
        return new ClinicalFeature(0, name, value, "", 0, 0);
    }
}
