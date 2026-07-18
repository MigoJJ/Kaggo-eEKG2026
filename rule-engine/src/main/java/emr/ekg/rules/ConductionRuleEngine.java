package emr.ekg.rules;

import emr.ekg.features.ClinicalConstants;
import emr.ekg.features.ClinicalFeature;
import emr.ekg.features.morphology.LeadMorphology;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 2단계 룰 기반 전도장애/축/비대 판정 (AHA/ACC/ESC 표준 가이드 기준).
 * QRS 폭·PR 간격·P파 등 20개 스칼라 피처와 리드별 형태학(R/S/Q 진폭)을 조합해
 * 심전기축·RBBB·LBBB·1도AVB·LVH(Sokolow-Lyon)·병적Q·RAE·LAE를 판정한다.
 *
 * RBBB/LBBB 판정은 rSR' 노치(notch) 형태를 직접 검출하지 않고 R/S 진폭·폭 기반으로
 * 근사한다 — 진성 노치 검출은 QRS 파형 내 국소 극값 탐색이 추가로 필요한 향후 정밀화 대상이다.
 */
public final class ConductionRuleEngine {

    private ConductionRuleEngine() {
    }

    public static List<Finding> evaluate(List<ClinicalFeature> features, Map<String, LeadMorphology> morphology) {
        return evaluate(features, morphology, RuleThresholds.defaults());
    }

    public static List<Finding> evaluate(List<ClinicalFeature> features, Map<String, LeadMorphology> morphology,
            RuleThresholds thresholds) {
        List<Finding> findings = new ArrayList<>();
        Map<String, ClinicalFeature> byName = new HashMap<>();
        for (ClinicalFeature f : features) {
            byName.put(f.name(), f);
        }

        double qrsDuration = value(byName, "QRS_DURATION");
        double prInterval = value(byName, "PR_INTERVAL");
        double pDuration = value(byName, "P_DURATION");
        double pAmplitude = value(byName, "P_AMPLITUDE_II");
        double sokolowLyon = value(byName, "SOKOLOW_LYON_INDEX");
        double rAvl = value(byName, "R_AMPLITUDE_AVL");

        LeadMorphology v1 = morphology.get("V1");
        LeadMorphology v5 = morphology.get("V5");
        LeadMorphology v6 = morphology.get("V6");
        LeadMorphology leadI = morphology.get("I");
        LeadMorphology avl = lookup(morphology, "AVL", "aVL");
        LeadMorphology avf = lookup(morphology, "AVF", "aVF");

        findings.addAll(bundleBranchBlockFindings(qrsDuration, v1, v5, v6, leadI, avl, thresholds));

        if (prInterval > thresholds.avb1PrMs()) {
            findings.add(new Finding("AVB1", "1도 방실차단(First-degree AV Block)", Finding.Severity.ABNORMAL,
                    "PR %.0fms (>%.0fms)".formatted(prInterval, thresholds.avb1PrMs())));
        }

        if (isLvh(sokolowLyon, rAvl)) {
            findings.add(new Finding("LVH", "좌심실비대 의심(Sokolow-Lyon 기준)", Finding.Severity.ABNORMAL,
                    "Sokolow-Lyon %.2fmV, R-aVL %.2fmV".formatted(sokolowLyon, rAvl)));
        }

        findings.addAll(pathologicQFindings(morphology));

        if (!Double.isNaN(pAmplitude) && pAmplitude >= ClinicalConstants.P_AMPLITUDE_II_MAX_MV) {
            findings.add(new Finding("RAE", "우심방확장 의심(P-pulmonale)", Finding.Severity.ABNORMAL,
                    "P-II 진폭 %.2fmV".formatted(pAmplitude)));
        }
        if (!Double.isNaN(pDuration) && pDuration >= ClinicalConstants.P_DURATION_MAX_MS) {
            findings.add(new Finding("LAE", "좌심방확장 의심(P-mitrale)", Finding.Severity.ABNORMAL,
                    "P-II 폭 %.0fms".formatted(pDuration)));
        }

        if (leadI != null && avf != null) {
            AxisClassifier.AxisCategory category =
                    AxisClassifier.classify(leadI.netDeflectionMv(), avf.netDeflectionMv());
            double axisDeg = Math.toDegrees(Math.atan2(avf.netDeflectionMv(), leadI.netDeflectionMv()));
            AxisClassifier.toFinding(category, axisDeg).ifPresent(findings::add);
        }

        return findings;
    }

    private static List<Finding> bundleBranchBlockFindings(
            double qrsDuration, LeadMorphology v1, LeadMorphology v5, LeadMorphology v6,
            LeadMorphology leadI, LeadMorphology avl, RuleThresholds thresholds) {

        List<Finding> out = new ArrayList<>();
        if (qrsDuration < thresholds.bbbQrsMs()) {
            return out;
        }

        if (isRbbbPattern(v1, v6)) {
            out.add(new Finding("RBBB", "우각차단(Right Bundle Branch Block)", Finding.Severity.ABNORMAL,
                    "QRS %.0fms, V1 R/S 병존 + V6 유의 S파".formatted(qrsDuration)));
        } else if (isLbbbPattern(v5, v6, leadI, avl)) {
            out.add(new Finding("LBBB", "좌각차단(Left Bundle Branch Block)", Finding.Severity.ABNORMAL,
                    "QRS %.0fms, V5/V6/I/aVL 중 3개 이상 광범위 R파".formatted(qrsDuration)));
        } else {
            out.add(new Finding("IVCD", "비특이적 심실내전도지연", Finding.Severity.ABNORMAL,
                    "QRS %.0fms (RBBB/LBBB 패턴 불일치)".formatted(qrsDuration)));
        }
        return out;
    }

    private static boolean isRbbbPattern(LeadMorphology v1, LeadMorphology v6) {
        if (v1 == null || v6 == null) {
            return false;
        }
        return v1.rAmplitudeMv() > 0 && v1.sAmplitudeMv() > 0 && v6.sAmplitudeMv() > 0.05;
    }

    private static boolean isLbbbPattern(LeadMorphology v5, LeadMorphology v6, LeadMorphology leadI, LeadMorphology avl) {
        int broadR = 0;
        for (LeadMorphology m : new LeadMorphology[] {v5, v6, leadI, avl}) {
            if (m != null && m.rAmplitudeMv() > 0.5 && m.sAmplitudeMv() < 0.1) {
                broadR++;
            }
        }
        return broadR >= 3;
    }

    private static boolean isLvh(double sokolowLyon, double rAvl) {
        boolean sokolowPositive = !Double.isNaN(sokolowLyon) && sokolowLyon >= ClinicalConstants.SOKOLOW_LYON_LVH_MV;
        boolean avlPositive = !Double.isNaN(rAvl) && rAvl >= ClinicalConstants.R_AVL_LVH_MV;
        return sokolowPositive || avlPositive;
    }

    private static List<Finding> pathologicQFindings(Map<String, LeadMorphology> morphology) {
        List<Finding> out = new ArrayList<>();
        for (Map.Entry<String, LeadMorphology> e : morphology.entrySet()) {
            String lead = e.getKey();
            if (lead.equalsIgnoreCase("AVR") || lead.equalsIgnoreCase("aVR")) {
                continue; // aVR은 정상적으로 QS 패턴이라 병적 Q 판정에서 제외
            }
            if (e.getValue().isPathologicQ()) {
                out.add(new Finding("PATHOLOGIC_Q", "병적 Q파 (진구성 심근경색 의심)", Finding.Severity.ABNORMAL,
                        "%s: Q폭 %.0fms, Q/R비 %.2f".formatted(lead, e.getValue().qDurationMs(), e.getValue().qDepthRatio())));
            }
        }
        return out;
    }

    private static double value(Map<String, ClinicalFeature> byName, String name) {
        ClinicalFeature f = byName.get(name);
        return f != null ? f.value() : Double.NaN;
    }

    private static LeadMorphology lookup(Map<String, LeadMorphology> map, String... names) {
        for (String n : names) {
            LeadMorphology m = map.get(n);
            if (m != null) {
                return m;
            }
        }
        return null;
    }
}
