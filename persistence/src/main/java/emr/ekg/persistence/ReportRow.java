package emr.ekg.persistence;

import java.time.Instant;

/**
 * report 테이블 한 행. beatArrhythmiaAvailable/stIschemiaAvailable은 Stage3/4 모델이
 * 아직 없는 동안(Phase 4 이전) false로 저장되어, 리포트가 "이 항목은 아직 분석 안 됨"을
 * 명시적으로 알 수 있게 한다 — 조용히 누락시키지 않기 위함이다.
 */
public record ReportRow(
        String recordId,
        ReportStatus status,
        double normTriageScore,
        String triageModelVersion,
        String triageModelHash,
        String triageModelMetadataHash,
        String triageModelMetadataJson,
        String beatArrhythmiaModelVersion,
        String beatArrhythmiaModelHash,
        String beatArrhythmiaModelMetadataHash,
        String beatArrhythmiaModelMetadataJson,
        String stIschemiaModelVersion,
        String stIschemiaModelHash,
        String stIschemiaModelMetadataHash,
        String stIschemiaModelMetadataJson,
        boolean beatArrhythmiaAvailable,
        boolean stIschemiaAvailable,
        String signedBy,
        Instant signedAt,
        Instant createdAt,
        String payloadJson) {
}
