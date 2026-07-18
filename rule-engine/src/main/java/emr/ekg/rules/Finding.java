package emr.ekg.rules;

/**
 * 룰엔진(E·2단계)이 산출하는 단일 소견. 근거(evidence)를 반드시 동봉하여
 * 설명가능성(P4)을 보장한다.
 *
 * @param code     소견 코드 (예: "STEMI", "VT", "RBBB", "LBBB", "AVB1", "LAD")
 * @param label    사람이 읽는 라벨
 * @param severity 심각도
 * @param evidence 판정 근거 (사용된 피처값·임계·리드)
 */
public record Finding(
        String code,
        String label,
        Severity severity,
        String evidence) {

    public enum Severity {
        /** 응급 fast-path — 즉시 CRITICAL 알람. */
        CRITICAL,
        /** 유의 소견. */
        ABNORMAL,
        /** 경계/참고. */
        BORDERLINE
    }

    public boolean isCritical() {
        return severity == Severity.CRITICAL;
    }
}
