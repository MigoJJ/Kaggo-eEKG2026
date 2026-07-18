package emr.ekg.signal.wfdb;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * WFDB 텍스트 헤더(.hea) 파서. PTB-XL(format 16, 12-lead), MIT-BIH(format 212, 2-lead),
 * CinC2020류(format "16+24", MATLAB v4 flat .mat 컨테이너) 실제 헤더로 검증됨.
 *
 * 신호 라인 규격: {@code <file> <format>[+<byteOffset>] <gain>[(<baseline>)][/<units>] <adcRes>
 * <adcZero> <initValue> <checksum> <blockSize> <description>}.
 * gain 필드에 baseline이 생략된 경우(MIT-BIH 관행) adcZero를 baseline으로 대체한다.
 * format 필드의 "+N"은 데이터 파일 시작에서 N바이트를 건너뛰고 디코딩을 시작하라는 뜻이다.
 */
public final class WfdbHeaderParser {

    private static final Pattern GAIN_PATTERN =
            Pattern.compile("^([+-]?[0-9.]+)(?:\\((-?\\d+)\\))?(?:/(.+))?$");

    private WfdbHeaderParser() {
    }

    public static WfdbHeader parse(Path headerFile) throws IOException {
        List<String> lines = Files.readAllLines(headerFile);
        String recordLine = null;
        List<String> signalLines = new ArrayList<>();
        for (String rawLine : lines) {
            String line = rawLine.strip();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            if (recordLine == null) {
                recordLine = line;
            } else {
                signalLines.add(line);
            }
        }
        if (recordLine == null) {
            throw new IOException("빈 WFDB 헤더: " + headerFile);
        }

        String[] rt = recordLine.split("\\s+");
        String recordName = rt[0];
        int signalCount = Integer.parseInt(rt[1]);
        int fs = (int) Math.round(Double.parseDouble(rt[2].split("/")[0]));
        int sampleCount = Integer.parseInt(rt[3]);

        List<SignalSpec> signals = new ArrayList<>();
        for (int i = 0; i < signalCount && i < signalLines.size(); i++) {
            signals.add(parseSignalLine(signalLines.get(i)));
        }
        if (signals.size() != signalCount) {
            throw new IOException("헤더 signalCount(%d)와 실제 신호라인 수(%d) 불일치: %s"
                    .formatted(signalCount, signals.size(), headerFile));
        }

        return new WfdbHeader(recordName, signalCount, fs, sampleCount, signals);
    }

    private static SignalSpec parseSignalLine(String line) throws IOException {
        String[] t = line.split("\\s+", 9);
        if (t.length < 8) {
            throw new IOException("잘못된 WFDB 신호 라인: " + line);
        }
        String fileName = t[0];
        String formatField = t[1];
        int format = Integer.parseInt(formatField.split("[x:+]")[0]);
        int plusIdx = formatField.indexOf('+');
        int byteOffset = plusIdx < 0 ? 0 : Integer.parseInt(formatField.substring(plusIdx + 1));

        Matcher m = GAIN_PATTERN.matcher(t[2]);
        if (!m.matches()) {
            throw new IOException("잘못된 gain 필드: " + t[2]);
        }
        double gain = Double.parseDouble(m.group(1));
        int adcResolution = Integer.parseInt(t[3]);
        int adcZero = Integer.parseInt(t[4]);
        int initialValue = Integer.parseInt(t[5]);
        int checksum = Integer.parseInt(t[6]);
        int blockSize = Integer.parseInt(t[7]);
        String description = t.length > 8 ? t[8].strip() : "";

        double baseline = m.group(2) != null ? Double.parseDouble(m.group(2)) : adcZero;
        String units = m.group(3) != null ? m.group(3) : "mV";

        return new SignalSpec(fileName, format, byteOffset, gain, baseline, units,
                adcResolution, adcZero, initialValue, checksum, blockSize, description);
    }
}
