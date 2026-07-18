package emr.ekg.rules;

import emr.ekg.features.ClinicalFeature;
import emr.ekg.features.morphology.LeadMorphology;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * E단계 응급 fast-path. 1단계(정상 사전선별) 미통과군에 항상 먼저 적용되며,
 * 게이팅 없이 후속 2/3/4단계와 병행 진행된다.
 *
 * 가이드가 명시적 수치 기준을 제공하는 3개 소견만 다룬다: STEMI(ST상승, 인접 리드 규칙),
 * 허혈성 ST하강, Long QT(QTc&gt;500ms). VT/VF·완전방실차단은 비트 형태학/리듬 해리 분석이
 * 필요해 이 표준 판독 가이드의 범위 밖이며 미구현이다(3단계 비트분류 이후 별도 검증 필요).
 *
 * 임계치는 {@link RuleThresholds}로 조정 가능하다(config/pipeline.yaml 연동, backbone 목표) —
 * 인자 없는 {@link #evaluate(List, Map)}는 {@link RuleThresholds#defaults()}를 사용한다.
 */
public final class EmergencyRuleEngine {

    private EmergencyRuleEngine() {
    }

    public static List<Finding> evaluate(List<ClinicalFeature> features, Map<String, LeadMorphology> morphology) {
        return evaluate(features, morphology, RuleThresholds.defaults());
    }

    public static List<Finding> evaluate(List<ClinicalFeature> features, Map<String, LeadMorphology> morphology,
            RuleThresholds thresholds) {
        List<Finding> findings = new ArrayList<>();
        findings.addAll(detectStemi(morphology, thresholds));
        findings.addAll(detectIschemicDepression(morphology, thresholds));

        double qtc = value(features, "QTC_BAZETT");
        if (!Double.isNaN(qtc) && qtc > thresholds.qtcCriticalMs()) {
            findings.add(new Finding("LONG_QT_CRITICAL",
                    "고도 QT 연장 — 다형성 심실빈맥(Torsades de Pointes) 위험",
                    Finding.Severity.CRITICAL,
                    "QTc %.0fms (>%.0fms)".formatted(qtc, thresholds.qtcCriticalMs())));
        }

        return findings;
    }

    private static List<Finding> detectStemi(Map<String, LeadMorphology> morphology, RuleThresholds thresholds) {
        List<Finding> out = new ArrayList<>();
        Map<String, List<String>> groups = new LinkedHashMap<>();
        groups.put("전벽(Anterior)", LeadGroups.ANTERIOR);
        groups.put("하벽(Inferior)", LeadGroups.INFERIOR);
        groups.put("측벽(Lateral)", LeadGroups.LATERAL);

        for (Map.Entry<String, List<String>> group : groups.entrySet()) {
            List<String> elevatedLeads = new ArrayList<>();
            for (String lead : group.getValue()) {
                LeadMorphology m = morphology.get(lead);
                if (m == null) {
                    continue;
                }
                double threshold = (lead.equals("V2") || lead.equals("V3"))
                        ? thresholds.stemiElevationV2V3Mv()
                        : thresholds.stemiElevationGeneralMv();
                if (m.stDeviationMv() >= threshold) {
                    elevatedLeads.add(lead);
                }
            }
            if (elevatedLeads.size() >= thresholds.stemiContiguousLeads()) {
                out.add(new Finding("STEMI", "ST분절 상승 심근경색(STEMI) 의심 — " + group.getKey(),
                        Finding.Severity.CRITICAL, "인접 리드 ST상승: " + elevatedLeads));
            }
        }
        return out;
    }

    private static List<Finding> detectIschemicDepression(Map<String, LeadMorphology> morphology,
            RuleThresholds thresholds) {
        List<Finding> out = new ArrayList<>();
        List<String> depressedLeads = new ArrayList<>();
        for (Map.Entry<String, LeadMorphology> e : morphology.entrySet()) {
            if (e.getValue().stDeviationMv() <= -thresholds.ischemiaDepressionMv()) {
                depressedLeads.add(e.getKey());
            }
        }
        // STEMI와 동일하게 인접 리드 수를 요구한다 — 단일 리드의 측정 잡음/편향만으로
        // CRITICAL을 발생시키지 않기 위함(실측: 단일 리드 기준이 정상 레코드에서도 상시 발생).
        if (depressedLeads.size() >= thresholds.ischemiaContiguousLeads()) {
            out.add(new Finding("ISCHEMIA", "심근허혈 의심(ST분절 하강)", Finding.Severity.CRITICAL,
                    "ST하강 리드: " + depressedLeads));
        }
        return out;
    }

    private static double value(List<ClinicalFeature> features, String name) {
        for (ClinicalFeature f : features) {
            if (f.name().equals(name)) {
                return f.value();
            }
        }
        return Double.NaN;
    }
}
