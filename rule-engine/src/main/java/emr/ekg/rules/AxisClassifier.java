package emr.ekg.rules;

import java.util.Optional;

/**
 * Lead I/aVF의 QRS 순전위(R-S 진폭차) 극성 기반 심전기축 판정
 * (AHA/ACC/ESC 표준 가이드 2단계: 심전기축 결정).
 */
public final class AxisClassifier {

    public enum AxisCategory {
        NORMAL, LEFT_DEVIATION, RIGHT_DEVIATION, EXTREME_DEVIATION
    }

    private AxisClassifier() {
    }

    public static AxisCategory classify(double netI, double netAvf) {
        boolean iPositive = netI >= 0;
        boolean avfPositive = netAvf >= 0;
        if (iPositive && avfPositive) {
            return AxisCategory.NORMAL;
        }
        if (iPositive) {
            return AxisCategory.LEFT_DEVIATION;
        }
        if (avfPositive) {
            return AxisCategory.RIGHT_DEVIATION;
        }
        return AxisCategory.EXTREME_DEVIATION;
    }

    /** 정상축(NORMAL)은 보고할 소견이 아니므로 Optional.empty()를 반환한다. */
    public static Optional<Finding> toFinding(AxisCategory category, double axisDeg) {
        String evidence = "QRS axis ≈ %.0f°".formatted(axisDeg);
        return switch (category) {
            case NORMAL -> Optional.empty();
            case LEFT_DEVIATION -> Optional.of(new Finding("LAD",
                    "좌축편위(Left Axis Deviation)", Finding.Severity.ABNORMAL, evidence));
            case RIGHT_DEVIATION -> Optional.of(new Finding("RAD",
                    "우축편위(Right Axis Deviation)", Finding.Severity.ABNORMAL, evidence));
            case EXTREME_DEVIATION -> Optional.of(new Finding("EXTREME_AXIS",
                    "극단적 축편위(Extreme Axis Deviation)", Finding.Severity.ABNORMAL, evidence));
        };
    }
}
