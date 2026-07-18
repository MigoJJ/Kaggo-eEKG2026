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
 * Stage E 응급 fast-path 룰 로직에 대한 결정론적 검증 (합성 데이터).
 */
class EmergencyRuleEngineTest {

    @Test
    void triggersStemiWhenTwoContiguousAnteriorLeadsElevated() {
        Map<String, LeadMorphology> morphology = new HashMap<>();
        morphology.put("V1", stLead("V1", 0.12));
        morphology.put("V2", stLead("V2", 0.20)); // V2/V3 임계(0.15) 초과

        List<Finding> findings = EmergencyRuleEngine.evaluate(List.of(), morphology);
        assertTrue(findings.stream().anyMatch(f -> f.code().equals("STEMI")
                && f.severity() == Finding.Severity.CRITICAL));
    }

    @Test
    void doesNotTriggerStemiWithOnlySingleLeadElevated() {
        Map<String, LeadMorphology> morphology = new HashMap<>();
        morphology.put("V1", stLead("V1", 0.20)); // 단일 리드만 상승 - 인접성 조건 미충족

        List<Finding> findings = EmergencyRuleEngine.evaluate(List.of(), morphology);
        assertFalse(findings.stream().anyMatch(f -> f.code().equals("STEMI")));
    }

    @Test
    void v2v3RequireHigherElevationThresholdThanOtherLeads() {
        // V2/V3 임계는 0.15mV, 일반 리드는 0.1mV — 0.12mV는 V2/V3 단독으론 미충족
        Map<String, LeadMorphology> morphology = new HashMap<>();
        morphology.put("V2", stLead("V2", 0.12));
        morphology.put("V3", stLead("V3", 0.12));

        List<Finding> findings = EmergencyRuleEngine.evaluate(List.of(), morphology);
        assertFalse(findings.stream().anyMatch(f -> f.code().equals("STEMI")));
    }

    @Test
    void triggersIschemiaOnStDepressionInTwoOrMoreLeads() {
        Map<String, LeadMorphology> morphology = new HashMap<>();
        morphology.put("V4", stLead("V4", -0.08));
        morphology.put("V5", stLead("V5", -0.08));

        List<Finding> findings = EmergencyRuleEngine.evaluate(List.of(), morphology);
        assertTrue(findings.stream().anyMatch(f -> f.code().equals("ISCHEMIA")
                && f.severity() == Finding.Severity.CRITICAL));
    }

    @Test
    void doesNotTriggerIschemiaOnSingleLeadDepression() {
        // 단일 리드 ST하강은 측정 잡음일 수 있어 트리거하지 않는다(STEMI와 동일한 인접성 요건).
        Map<String, LeadMorphology> morphology = new HashMap<>();
        morphology.put("V4", stLead("V4", -0.08));

        List<Finding> findings = EmergencyRuleEngine.evaluate(List.of(), morphology);
        assertFalse(findings.stream().anyMatch(f -> f.code().equals("ISCHEMIA")));
    }

    @Test
    void triggersCriticalLongQtAboveThreshold() {
        List<ClinicalFeature> features = List.of(new ClinicalFeature(19, "QTC_BAZETT", 520, "ms", 350, 450));
        List<Finding> findings = EmergencyRuleEngine.evaluate(features, Map.of());
        assertTrue(findings.stream().anyMatch(f -> f.code().equals("LONG_QT_CRITICAL")
                && f.severity() == Finding.Severity.CRITICAL));
    }

    @Test
    void doesNotTriggerLongQtWhenBelowCriticalThreshold() {
        List<ClinicalFeature> features = List.of(new ClinicalFeature(19, "QTC_BAZETT", 480, "ms", 350, 450));
        List<Finding> findings = EmergencyRuleEngine.evaluate(features, Map.of());
        assertFalse(findings.stream().anyMatch(f -> f.code().equals("LONG_QT_CRITICAL")));
    }

    @Test
    void customThresholdsOverrideDefaults() {
        // ischemiaContiguousLeads=1로 완화하면 단일 리드에서도 트리거되어야 한다 —
        // config/pipeline.yaml로 재보정 가능함을 보장하는 backbone 검증.
        Map<String, LeadMorphology> morphology = new HashMap<>();
        morphology.put("V4", stLead("V4", -0.08));

        RuleThresholds relaxed = new RuleThresholds(
                RuleThresholds.defaults().stemiElevationGeneralMv(),
                RuleThresholds.defaults().stemiElevationV2V3Mv(),
                RuleThresholds.defaults().stemiContiguousLeads(),
                0.05, 1,
                RuleThresholds.defaults().qtcCriticalMs(),
                RuleThresholds.defaults().bbbQrsMs(),
                RuleThresholds.defaults().avb1PrMs());

        List<Finding> findings = EmergencyRuleEngine.evaluate(List.of(), morphology, relaxed);
        assertTrue(findings.stream().anyMatch(f -> f.code().equals("ISCHEMIA")));
    }

    private static LeadMorphology stLead(String name, double stDeviationMv) {
        return new LeadMorphology(name, 1.0, 0.0, 0, 0, stDeviationMv, 0);
    }
}
