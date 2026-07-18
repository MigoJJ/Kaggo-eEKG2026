package emr.ekg.persistence;

/**
 * 리포트 생명주기 상태. 자동 확정 금지(P3): 모든 리포트는 PENDING_SIGN에서
 * 시작하며 의사 서명으로만 SIGNED가 된다.
 */
public enum ReportStatus {
    /** 판독 완료, 의사 서명 대기. */
    PENDING_SIGN,
    /** 의사 서명 발행 완료. */
    SIGNED,
    /** SQI 불량 등으로 판독불가 반려. */
    REJECTED
}
