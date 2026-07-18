package emr.ekg.signal.preprocess;

import emr.ekg.signal.PreprocessedEcg;
import emr.ekg.signal.RawEcgSignal;

import java.util.Arrays;

/**
 * 0단계 전처리 파이프라인: 리샘플 → 대역통과(0.5–40Hz)+notch → 2단 중앙값 기선보정 → SQI 산출.
 *
 * 출력은 물리단위(mV)를 유지한다 — 룰엔진(2/E단계)의 절대 임계치(mm/mV 기반 AHA/ACC/ESC 기준)가
 * 그대로 적용되어야 하기 때문이다. Z-정규화 등 모델 입력 전처리는 여기서 하지 않고,
 * 추론 직전(1/3/4단계, inference 모듈)에 별도로 적용한다.
 *
 * 모든 후속 단계(1: 정상 사전선별, E: 응급, 2/3/4: 정밀 판독)의 공유 프론트엔드다(P1).
 */
public final class EcgPreprocessor {

    private static final double BUTTERWORTH_Q = 1.0 / Math.sqrt(2.0);
    private static final double NOTCH_Q = 30.0;
    private static final double EDGE_PAD_SECONDS = 2.0;

    private final PreprocessConfig config;

    public EcgPreprocessor(PreprocessConfig config) {
        this.config = config;
    }

    public PreprocessedEcg process(RawEcgSignal raw) {
        int targetFs = config.targetFs();
        int nLeads = raw.leadCount();
        double[][] filtered = new double[nLeads][];
        int padSamples = (int) Math.round(EDGE_PAD_SECONDS * targetFs);

        for (int i = 0; i < nLeads; i++) {
            double[] resampled = Resampler.resample(raw.samples()[i], raw.fs(), targetFs);
            int pad = Math.min(padSamples, resampled.length - 1);
            double[] signal = mirrorPad(resampled, pad);

            signal = BiquadDesign.highpass(targetFs, config.bandpassLowHz(), BUTTERWORTH_Q)
                    .processAll(signal);
            signal = BiquadDesign.lowpass(targetFs, config.bandpassHighHz(), BUTTERWORTH_Q)
                    .processAll(signal);
            signal = BiquadDesign.notch(targetFs, config.notchHz(), NOTCH_Q)
                    .processAll(signal);
            signal = BaselineWanderRemover.remove(signal, targetFs);

            filtered[i] = Arrays.copyOfRange(signal, pad, pad + resampled.length);
        }

        double[] sqiScores = SignalQualityIndex.perLeadScore(filtered);
        double sqi = average(sqiScores);
        boolean[] leadOk = new boolean[nLeads];
        for (int i = 0; i < nLeads; i++) {
            leadOk[i] = sqiScores[i] >= config.sqiThreshold();
        }

        return new PreprocessedEcg(filtered, targetFs, raw.leadNames().clone(), sqi, leadOk);
    }

    /**
     * IIR 필터는 0 초기상태로 시작하므로, 신호 시작부의 DC 오프셋이 크면 워밍업 과도현상이
     * 진짜 QRS보다 큰 스파이크를 만들어 R-peak 검출을 무너뜨릴 수 있다(실측: PTB-XL 일부
     * 레코드에서 확인됨). 양끝을 거울반사로 패딩해 필터가 실제 구간 진입 전에 안정되게 한다.
     */
    private static double[] mirrorPad(double[] signal, int pad) {
        int n = signal.length;
        double[] out = new double[n + 2 * pad];
        for (int i = 0; i < pad; i++) {
            out[i] = signal[pad - i];
            out[pad + n + i] = signal[n - 2 - i];
        }
        System.arraycopy(signal, 0, out, pad, n);
        return out;
    }

    private static double average(double[] values) {
        if (values.length == 0) {
            return 0;
        }
        double sum = 0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.length;
    }
}
