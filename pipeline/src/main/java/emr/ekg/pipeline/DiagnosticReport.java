package emr.ekg.pipeline;

import emr.ekg.features.ClinicalFeature;
import emr.ekg.persistence.ReportStatus;
import emr.ekg.rules.Finding;

import java.time.Instant;
import java.util.List;

/**
 * 파이프라인 실행 결과. beatArrhythmiaAvailable/stIschemiaAvailable은 Stage3/4 모델이
 * 아직 없는 동안(Phase 4 이전) false로 고정되어, 리포트가 "이 항목은 아직 분석되지 않았다"를
 * 명시적으로 드러낸다 — 조용히 누락시키지 않기 위함이다(P4).
 *
 * normTriageScore는 게이팅에 쓰이지 않는다 — 의사 검토 큐의 우선순위 정렬용 메타데이터일
 * 뿐이며, status는 SQI 미달(REJECTED)이 아닌 한 항상 PENDING_SIGN이다(P3).
 */
public record DiagnosticReport(
        String recordId,
        String source,
        Instant createdAt,
        int fs,
        int sampleCount,
        double sqi,
        boolean interpretable,
        List<ClinicalFeature> features,
        double normTriageScore,
        String triageModelVersion,
        List<Finding> findings,
        boolean beatArrhythmiaAvailable,
        boolean stIschemiaAvailable,
        ReportStatus status) {

    public boolean hasCriticalFinding() {
        return findings.stream().anyMatch(Finding::isCritical);
    }

    public static DiagnosticReport rejected(String recordId, String source, int fs, int sampleCount, double sqi) {
        return new DiagnosticReport(recordId, source, Instant.now(), fs, sampleCount, sqi, false,
                List.of(), Double.NaN, null, List.of(), false, false, ReportStatus.REJECTED);
    }
}
