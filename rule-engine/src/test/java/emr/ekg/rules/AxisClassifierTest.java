package emr.ekg.rules;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AxisClassifierTest {

    @Test
    void iPositiveAvfPositiveIsNormal() {
        assertEquals(AxisClassifier.AxisCategory.NORMAL, AxisClassifier.classify(1.0, 0.5));
        assertTrue(AxisClassifier.toFinding(AxisClassifier.AxisCategory.NORMAL, 45).isEmpty());
    }

    @Test
    void iPositiveAvfNegativeIsLeftDeviation() {
        assertEquals(AxisClassifier.AxisCategory.LEFT_DEVIATION, AxisClassifier.classify(1.0, -0.5));
        assertEquals("LAD", AxisClassifier.toFinding(AxisClassifier.AxisCategory.LEFT_DEVIATION, -45)
                .orElseThrow().code());
    }

    @Test
    void iNegativeAvfPositiveIsRightDeviation() {
        assertEquals(AxisClassifier.AxisCategory.RIGHT_DEVIATION, AxisClassifier.classify(-1.0, 0.5));
        assertEquals("RAD", AxisClassifier.toFinding(AxisClassifier.AxisCategory.RIGHT_DEVIATION, 135)
                .orElseThrow().code());
    }

    @Test
    void iNegativeAvfNegativeIsExtremeDeviation() {
        assertEquals(AxisClassifier.AxisCategory.EXTREME_DEVIATION, AxisClassifier.classify(-1.0, -0.5));
        assertEquals("EXTREME_AXIS", AxisClassifier.toFinding(AxisClassifier.AxisCategory.EXTREME_DEVIATION, -135)
                .orElseThrow().code());
    }
}
