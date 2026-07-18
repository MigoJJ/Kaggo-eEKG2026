package emr.ekg.persistence;

/** diagnosis 테이블 한 행 (rule-engine Finding의 평면화된 표현). severity는 문자열로 저장. */
public record FindingRow(
        String recordId,
        String code,
        String label,
        String severity,
        String evidence) {
}
