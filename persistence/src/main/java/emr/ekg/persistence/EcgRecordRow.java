package emr.ekg.persistence;

import java.time.Instant;

/** ecg_record 테이블 한 행. persistence 모듈은 core-features/rule-engine의 도메인 타입에
 * 의존하지 않고 평면(flat) 값만 다룬다 — 저장 계층을 임상 도메인과 분리하기 위함이다. */
public record EcgRecordRow(
        String id,
        String source,
        int fs,
        int sampleCount,
        double sqi,
        boolean interpretable,
        Instant createdAt) {
}
