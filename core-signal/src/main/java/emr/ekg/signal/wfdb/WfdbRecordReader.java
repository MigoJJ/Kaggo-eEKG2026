package emr.ekg.signal.wfdb;

import emr.ekg.signal.RawEcgSignal;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * WFDB 레코드(.hea + .dat/.mat) 리더. PTB-XL(format 16), MIT-BIH(format 212),
 * CinC2020류(format "16+24", MATLAB v4 flat .mat) 지원.
 * 각 신호(리드)를 게인·기선으로 물리단위(mV)까지 변환하여 반환한다.
 */
public final class WfdbRecordReader {

    private WfdbRecordReader() {
    }

    public static RawEcgSignal read(Path headerFile) throws IOException {
        WfdbHeader header = WfdbHeaderParser.parse(headerFile);
        Path dir = headerFile.toAbsolutePath().getParent();
        List<SignalSpec> signals = header.signals();

        String dataFile = signals.get(0).fileName();
        int format = signals.get(0).format();
        for (SignalSpec s : signals) {
            if (!s.fileName().equals(dataFile)) {
                throw new IOException("리드별 개별 .dat 분리 저장은 미지원: " + headerFile);
            }
            if (s.format() != format) {
                throw new IOException("혼합 포맷 미지원: " + headerFile);
            }
        }

        byte[] fileBytes = Files.readAllBytes(dir.resolve(dataFile));
        int byteOffset = signals.get(0).byteOffset();
        byte[] raw = byteOffset == 0 ? fileBytes : Arrays.copyOfRange(fileBytes, byteOffset, fileBytes.length);
        int nSignals = header.signalCount();
        int nSamples = header.sampleCount();
        int totalSamples = nSignals * nSamples;

        int[] flat = switch (format) {
            case 16 -> Format16Codec.decode(raw, totalSamples);
            case 212 -> Format212Codec.decode(raw, totalSamples);
            default -> throw new IOException("미지원 WFDB 포맷: " + format);
        };

        double[][] physical = new double[nSignals][nSamples];
        String[] leadNames = new String[nSignals];
        for (int ch = 0; ch < nSignals; ch++) {
            SignalSpec spec = signals.get(ch);
            leadNames[ch] = spec.description();
            double[] lead = physical[ch];
            for (int i = 0; i < nSamples; i++) {
                int rawVal = flat[i * nSignals + ch];
                lead[i] = (rawVal - spec.baseline()) / spec.gain();
            }
        }

        return new RawEcgSignal(physical, header.fs(), leadNames);
    }
}
