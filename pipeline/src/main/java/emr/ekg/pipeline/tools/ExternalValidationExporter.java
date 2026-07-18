package emr.ekg.pipeline.tools;

import emr.ekg.features.Sex;
import emr.ekg.pipeline.DiagnosticReport;
import emr.ekg.pipeline.EkgPipeline;
import emr.ekg.pipeline.PipelineConfig;
import emr.ekg.rules.Finding;
import emr.ekg.signal.RawEcgSignal;
import emr.ekg.signal.wfdb.WfdbRecordReader;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * CinC2020류 외부기관 데이터셋(CPSC2018, Chapman-Shaoxing/Ningbo 등)을 배치로 돌려
 * {@code config/pipeline.yaml} 그대로의 실서빙 경로({@link EkgPipeline#fromConfig})로 처리하고,
 * SNOMED CT 진단 코드(.hea의 {@code #Dx:} 줄)와 Stage1 triage score·findings를 CSV로 남긴다.
 *
 * 목적은 Phase 7 실측 검증(AUROC/F1)이지 재보정이 아니다 — 여기서 나온 CSV는 Python
 * (ml/evaluate_external_validation.py)에서 지표만 계산하고, 임계값은 이 도구가 건드리지 않는다.
 * PTB-XL 학습·서빙에 쓰는 {@link WfdbRecordReader}/{@link EkgPipeline}을 그대로 재사용하므로
 * train/serve parity가 그대로 유지된다.
 *
 * 사용법: <config-yaml> <dataset-dir> <source-label> <output-csv>
 */
public final class ExternalValidationExporter {

    private static final Pattern DX_LINE = Pattern.compile("^#Dx:\\s*(.+)$");

    private ExternalValidationExporter() {
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 4) {
            System.err.println("사용법: <config-yaml> <dataset-dir> <source-label> <output-csv>");
            System.exit(1);
        }
        Path configYaml = Path.of(args[0]);
        Path datasetDir = Path.of(args[1]);
        String sourceLabel = args[2];
        Path outputCsv = Path.of(args[3]);

        PipelineConfig config = PipelineConfig.loadFromYaml(configYaml);
        List<Path> headerFiles;
        try (Stream<Path> paths = Files.list(datasetDir)) {
            headerFiles = paths.filter(p -> p.getFileName().toString().endsWith(".hea")).sorted().toList();
        }
        System.out.println("대상 레코드 수(" + sourceLabel + "): " + headerFiles.size());

        try (EkgPipeline pipeline = EkgPipeline.fromConfig(config);
                BufferedWriter out = Files.newBufferedWriter(outputCsv,
                        StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            out.write("record_id,source_db,dx_codes,interpretable,triage_norm_prob,model_version,finding_codes\n");

            long start = System.currentTimeMillis();
            int done = 0;
            int failed = 0;
            for (Path headerFile : headerFiles) {
                try {
                    String recordId = stripExtension(headerFile.getFileName().toString());
                    String dxCodes = readDxCodes(headerFile);

                    RawEcgSignal raw = WfdbRecordReader.read(headerFile);
                    DiagnosticReport report = pipeline.process(recordId, sourceLabel, raw, Sex.UNKNOWN);

                    writeRow(out, recordId, sourceLabel, dxCodes, report);
                } catch (Exception e) {
                    failed++;
                    System.err.println("실패: " + headerFile.getFileName() + " -> " + e.getMessage());
                }

                done++;
                if (done % 1000 == 0) {
                    long elapsed = System.currentTimeMillis() - start;
                    System.out.printf("진행: %d/%d (%.1fs 경과, 실패 %d)%n",
                            done, headerFiles.size(), elapsed / 1000.0, failed);
                }
            }
            long totalMs = System.currentTimeMillis() - start;
            System.out.printf("완료: %d개 처리, %d개 실패, %.1fs 소요%n", done, failed, totalMs / 1000.0);
        }
    }

    private static void writeRow(BufferedWriter out, String recordId, String sourceLabel, String dxCodes,
            DiagnosticReport report) throws IOException {
        String findingCodes = String.join(";", report.findings().stream().map(Finding::code).toList());
        double triageProb = report.interpretable() ? report.normTriageScore() : Double.NaN;
        String modelVersion = report.triageModelVersion() == null ? "" : report.triageModelVersion();

        out.write(recordId + ","
                + sourceLabel + ","
                + dxCodes + ","
                + report.interpretable() + ","
                + (Double.isNaN(triageProb) ? "" : triageProb) + ","
                + modelVersion + ","
                + findingCodes + "\n");
    }

    private static String readDxCodes(Path headerFile) throws IOException {
        for (String line : Files.readAllLines(headerFile)) {
            Matcher m = DX_LINE.matcher(line.strip());
            if (m.matches()) {
                return m.group(1).strip().replace(',', ';');
            }
        }
        return "";
    }

    private static String stripExtension(String fileName) {
        int dot = fileName.lastIndexOf('.');
        return dot < 0 ? fileName : fileName.substring(0, dot);
    }
}
