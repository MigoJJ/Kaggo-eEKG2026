package emr.ekg.signal.preprocess;

/** 2차 IIR biquad 필터 (Direct Form I). */
final class Biquad {

    private final double b0, b1, b2, a1, a2;
    private double x1, x2, y1, y2;

    Biquad(double b0, double b1, double b2, double a0, double a1, double a2) {
        this.b0 = b0 / a0;
        this.b1 = b1 / a0;
        this.b2 = b2 / a0;
        this.a1 = a1 / a0;
        this.a2 = a2 / a0;
    }

    private double process(double x0) {
        double y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
        return y0;
    }

    double[] processAll(double[] input) {
        double[] out = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            out[i] = process(input[i]);
        }
        return out;
    }
}
