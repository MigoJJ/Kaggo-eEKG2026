package emr.ekg.signal.preprocess;

/**
 * RBJ Audio EQ Cookbook 계수 규약 기반 biquad 설계.
 * 0단계 대역통과(고역+저역)와 전원 노이즈 notch에 사용.
 */
final class BiquadDesign {

    private BiquadDesign() {
    }

    static Biquad lowpass(double fs, double cutoffHz, double q) {
        double w0 = 2 * Math.PI * cutoffHz / fs;
        double cosw0 = Math.cos(w0);
        double alpha = Math.sin(w0) / (2 * q);

        double b0 = (1 - cosw0) / 2;
        double b1 = 1 - cosw0;
        double b2 = (1 - cosw0) / 2;
        double a0 = 1 + alpha;
        double a1 = -2 * cosw0;
        double a2 = 1 - alpha;
        return new Biquad(b0, b1, b2, a0, a1, a2);
    }

    static Biquad highpass(double fs, double cutoffHz, double q) {
        double w0 = 2 * Math.PI * cutoffHz / fs;
        double cosw0 = Math.cos(w0);
        double alpha = Math.sin(w0) / (2 * q);

        double b0 = (1 + cosw0) / 2;
        double b1 = -(1 + cosw0);
        double b2 = (1 + cosw0) / 2;
        double a0 = 1 + alpha;
        double a1 = -2 * cosw0;
        double a2 = 1 - alpha;
        return new Biquad(b0, b1, b2, a0, a1, a2);
    }

    static Biquad notch(double fs, double centerHz, double q) {
        double w0 = 2 * Math.PI * centerHz / fs;
        double cosw0 = Math.cos(w0);
        double alpha = Math.sin(w0) / (2 * q);

        double b0 = 1.0;
        double b1 = -2 * cosw0;
        double b2 = 1.0;
        double a0 = 1 + alpha;
        double a1 = -2 * cosw0;
        double a2 = 1 - alpha;
        return new Biquad(b0, b1, b2, a0, a1, a2);
    }
}
