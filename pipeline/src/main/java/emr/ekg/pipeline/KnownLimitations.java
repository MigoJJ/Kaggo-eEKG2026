package emr.ekg.pipeline;

/**
 * 자동판독 룰의 알려진 한계 — Phase 6/7 실측(ARCHITECTURE.md)에 근거한 정적 고지.
 * 특정 소견 발생 여부와 무관하게 항상 노출한다: RBBB/LBBB는 F1이 근접 0이라 미검출(위음성)도
 * 문제이므로, 해당 코드가 이번 리포트에 없더라도 고지가 필요하다.
 */
public final class KnownLimitations {

    public static final String DISCLAIMER_TEXT =
            "⚠ 알려진 한계: ISCHEMIA 룰은 ST 하강 임계값(0.05mV) 미보정으로 정상 소견에서도 과발화할 수 있음. "
                    + "RBBB/LBBB 룰은 외부검증 F1이 0에 가까움(미보정). 모든 자동 소견은 반드시 임상 확인 필요.";

    private KnownLimitations() {
    }

    public static String asAuditJson() {
        return "{\"known_limitations\":["
                + "\"ISCHEMIA rule over-fires on validated NORM records: ST depression threshold "
                + "(0.05mV) uncalibrated, see ARCHITECTURE.md Phase 6\","
                + "\"RBBB/LBBB rule: external validation F1 near zero, uncalibrated, see "
                + "ARCHITECTURE.md Phase 7 Track B\"]}";
    }
}
