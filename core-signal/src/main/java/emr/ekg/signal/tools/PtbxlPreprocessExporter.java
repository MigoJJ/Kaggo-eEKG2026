package emr.ekg.signal.tools;

import emr.ekg.signal.PreprocessedEcg;
import emr.ekg.signal.RawEcgSignal;
import emr.ekg.signal.preprocess.EcgPreprocessor;
import emr.ekg.signal.preprocess.PreprocessConfig;
import emr.ekg.signal.wfdb.WfdbRecordReader;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * PTB-XL records500 전체(또는 지정한 개수)를 0단계 전처리(EcgPreprocessor)까지 거쳐
 * 리드별 float32 이진 파일(.f32, 12*5000 row-major mV)로 배치 내보내기하는 CLI 도구.
 *
 * Python(ml/) 학습 파이프라인이 이 산출물을 그대로 읽어 학습에 사용함으로써,
 * Java 추론 경로와 동일한 전처리를 보장한다(train/serve skew 방지, P1).
 *
 * 사용법: java -cp core-signal/build/classes/java/main emr.ekg.signal.tools.PtbxlPreprocessExporter
 *         <ptbxlRoot> <outputDir> [maxRecords]
 */
public final class PtbxlPreprocessExporter {

    private PtbxlPreprocessExporter() {
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("사용법: <ptbxlRoot> <outputDir> [maxRecords]");
            System.exit(1);
        }
        Path root = Paths.get(args[0]);
        Path outputDir = Paths.get(args[1]);
        int maxRecords = args.length >= 3 ? Integer.parseInt(args[2]) : Integer.MAX_VALUE;

        Files.createDirectories(outputDir);
        Path recordsDir = root.resolve("records500");

        List<Path> headers = listHeaders(recordsDir, maxRecords);
        System.out.println("대상 레코드 수: " + headers.size());

        EcgPreprocessor preprocessor = new EcgPreprocessor(PreprocessConfig.defaults());

        try (BufferedWriter manifest = Files.newBufferedWriter(outputDir.resolve("manifest.csv"),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            manifest.write("ecg_id,fs,lead_count,sample_count,sqi,interpretable\n");

            long start = System.currentTimeMillis();
            int done = 0;
            int failed = 0;
            for (Path header : headers) {
                String recordName = header.getFileName().toString().replace(".hea", "");
                String ecgId = recordName.replace("_hr", "").replaceFirst("^0+(?=\\d)", "");

                try {
                    RawEcgSignal raw = WfdbRecordReader.read(header);
                    PreprocessedEcg ecg = preprocessor.process(raw);

                    Path outFile = outputDir.resolve(ecgId + ".f32");
                    Float32EcgIo.write(outFile, ecg.samples());

                    manifest.write("%s,%d,%d,%d,%.4f,%b\n".formatted(
                            ecgId, ecg.fs(), ecg.leadCount(), ecg.sampleCount(),
                            ecg.sqi(), ecg.isInterpretable(PreprocessConfig.defaults().sqiThreshold())));
                } catch (Exception e) {
                    failed++;
                    System.err.println("실패: " + header + " -> " + e.getMessage());
                }

                done++;
                if (done % 1000 == 0) {
                    long elapsed = System.currentTimeMillis() - start;
                    System.out.printf("진행: %d/%d (%.1fs 경과, 실패 %d)%n",
                            done, headers.size(), elapsed / 1000.0, failed);
                }
            }
            long totalMs = System.currentTimeMillis() - start;
            System.out.printf("완료: %d개 처리, %d개 실패, %.1fs 소요%n", done, failed, totalMs / 1000.0);
        }
    }

    private static List<Path> listHeaders(Path recordsDir, int maxRecords) throws IOException {
        List<Path> headers = new ArrayList<>();
        try (Stream<Path> walk = Files.walk(recordsDir)) {
            walk.filter(p -> p.toString().endsWith("_hr.hea"))
                    .sorted()
                    .limit(maxRecords)
                    .forEach(headers::add);
        }
        return headers;
    }

}
