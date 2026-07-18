package emr.ekg.features;

/**
 * AHA/ACC/ESC 표준 12유도 심전도 판독 가이드(Systematic Interpretation)의 정량 임계치.
 * 가이드는 표준 게인(10mm/mV) 기록지 기준 mm 단위이므로, mV 단위(물리 신호)로 환산해 상수화했다
 * (예: 1mm = 0.1mV).
 */
public final class ClinicalConstants {

    private ClinicalConstants() {
    }

    // 1단계 — 율동/박동수
    public static final double HR_MIN_BPM = 60;
    public static final double HR_MAX_BPM = 100;

    // P파
    public static final double P_DURATION_MAX_MS = 120;
    public static final double P_AMPLITUDE_II_MAX_MV = 0.25; // 2.5mm (RAE 기준)

    // PR 간격
    public static final double PR_MIN_MS = 120;
    public static final double PR_MAX_MS = 200;

    // QRS 폭
    public static final double QRS_MIN_MS = 70;
    public static final double QRS_MAX_MS = 100;
    public static final double QRS_INCOMPLETE_BBB_MS = 110;
    public static final double QRS_COMPLETE_BBB_MS = 120;

    // 좌심실비대(Sokolow-Lyon)
    public static final double SOKOLOW_LYON_LVH_MV = 3.5; // 35mm
    public static final double R_AVL_LVH_MV = 1.1; // 11mm

    // 병적 Q파
    public static final double Q_PATHOLOGIC_DURATION_MS = 40;
    public static final double Q_PATHOLOGIC_R_RATIO = 0.25;

    // ST 분절 (J-point 기준)
    public static final double ST_ELEVATION_GENERAL_MV = 0.1; // 1mm (V2/V3 제외 전 리드)
    public static final double ST_ELEVATION_V2V3_MV = 0.15; // 1.5mm (V2/V3)
    public static final double ST_DEPRESSION_MV = 0.05; // 0.5mm

    // QT/QTc (Bazett)
    public static final double QTC_MALE_MIN_MS = 350;
    public static final double QTC_MALE_MAX_MS = 450;
    public static final double QTC_FEMALE_MIN_MS = 360;
    public static final double QTC_FEMALE_MAX_MS = 460;
    public static final double QTC_CRITICAL_MS = 500; // Torsades de Pointes 위험 임계

    // 심전기축
    public static final double AXIS_NORMAL_MIN_DEG = -30;
    public static final double AXIS_NORMAL_MAX_DEG = 90;
}
