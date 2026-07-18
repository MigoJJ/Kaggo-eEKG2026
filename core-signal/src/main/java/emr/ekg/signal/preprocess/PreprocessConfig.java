package emr.ekg.signal.preprocess;

/** 0단계 전처리 설정. config/pipeline.yaml의 signal: 블록과 값이 대응된다. */
public record PreprocessConfig(
        int targetFs,
        double bandpassLowHz,
        double bandpassHighHz,
        double notchHz,
        double sqiThreshold) {

    public static PreprocessConfig defaults() {
        return new PreprocessConfig(500, 0.5, 40.0, 60.0, 0.6);
    }
}
