package emr.ekg.features.tools;

import emr.ekg.features.ClinicalFeature;
import emr.ekg.features.ClinicalFeatureExtractor;
import emr.ekg.features.Sex;
import emr.ekg.features.fiducial.BeatFiducials;
import emr.ekg.features.fiducial.FiducialExtractor;
import emr.ekg.features.morphology.LeadMorphology;
import emr.ekg.features.morphology.LeadMorphologyExtractor;
import emr.ekg.signal.PreprocessedEcg;
import emr.ekg.signal.tools.Float32EcgIo;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;

/**
 * {@code PtbxlPreprocessExporter}가 만든 .f32 산출물(Stage-0 전처리 완료)을 읽어
 * 2단계 파이프라인(fiducial → morphology → 20 임상 피처)을 배치 실행하고,
 * 레코드별 20개 피처값을 CSV로 내보내는 도구.
 *
 * 이 CSV는 raw waveform 대신 해석 가능한 임상 피처로 Stage1 정상 사전선별 모델을
 * 학습하기 위한 입력이다(로지스틱회귀/그래디언트부스팅 등 감사 가능한 모델 우선 시도).
 *
 * 사용법: java -cp core-features/build/classes/java/main:core-signal/build/classes/java/main
 *         emr.ekg.features.tools.PtbxlFeatureExporter <exportDir> <outputCsv>
 */
public final class PtbxlFeatureExporter {

    private static final String[] PTBXL_LEAD_ORDER =
            {"I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"};
    private static final int N_LEADS = 12;
    private static final int N_SAMPLES = 5000;
    private static final int FS = 500;

    private PtbxlFeatureExporter() {
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("사용법: <exportDir> <outputCsv>");
            System.exit(1);
        }
        Path exportDir = Paths.get(args[0]);
        Path outputCsv = Paths.get(args[1]);

        List<String> manifestLines = readManifestInterpretableIds(exportDir.resolve("manifest.csv"));
        System.out.println("대상 레코드 수: " + manifestLines.size());

        try (BufferedWriter out = Files.newBufferedWriter(outputCsv,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            out.write("ecg_id");
            for (int rank = 1; rank <= 20; rank++) {
                out.write(",f" + rank);
            }
            out.write(",n_beats\n");

            long start = System.currentTimeMillis();
            int done = 0;
            int failed = 0;
            for (String ecgId : manifestLines) {
                try {
                    Path f32 = exportDir.resolve(ecgId + ".f32");
                    double[][] samples = Float32EcgIo.read(f32, N_LEADS, N_SAMPLES);
                    PreprocessedEcg ecg = new PreprocessedEcg(samples, FS, PTBXL_LEAD_ORDER, 1.0,
                            new boolean[N_LEADS]);

                    List<BeatFiducials> beats = FiducialExtractor.extract(ecg, "II");
                    Map<String, LeadMorphology> morphology = LeadMorphologyExtractor.extract(ecg, beats);
                    List<ClinicalFeature> features =
                            ClinicalFeatureExtractor.extract(ecg, beats, morphology, Sex.UNKNOWN, "II");

                    writeRow(out, ecgId, features, beats.size());
                } catch (Exception e) {
                    failed++;
                    System.err.println("실패: " + ecgId + " -> " + e.getMessage());
                }

                done++;
                if (done % 1000 == 0) {
                    long elapsed = System.currentTimeMillis() - start;
                    System.out.printf("진행: %d/%d (%.1fs 경과, 실패 %d)%n",
                            done, manifestLines.size(), elapsed / 1000.0, failed);
                }
            }
            long totalMs = System.currentTimeMillis() - start;
            System.out.printf("완료: %d개 처리, %d개 실패, %.1fs 소요%n", done, failed, totalMs / 1000.0);
        }
    }

    private static void writeRow(BufferedWriter out, String ecgId, List<ClinicalFeature> features, int nBeats)
            throws IOException {
        double[] byRank = new double[21];
        for (ClinicalFeature f : features) {
            byRank[f.rank()] = f.value();
        }
        StringBuilder sb = new StringBuilder(ecgId);
        for (int rank = 1; rank <= 20; rank++) {
            sb.append(',').append(features.isEmpty() ? "" : formatValue(byRank[rank]));
        }
        sb.append(',').append(nBeats).append('\n');
        out.write(sb.toString());
    }

    private static String formatValue(double v) {
        return Double.isNaN(v) ? "" : Double.toString(v);
    }

    private static List<String> readManifestInterpretableIds(Path manifestCsv) throws IOException {
        List<String> ids = new java.util.ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(manifestCsv)) {
            String header = reader.readLine();
            String[] cols = header.split(",");
            int idIdx = indexOf(cols, "ecg_id");
            int interpretableIdx = indexOf(cols, "interpretable");

            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                if (Boolean.parseBoolean(parts[interpretableIdx])) {
                    ids.add(parts[idIdx]);
                }
            }
        }
        return ids;
    }

    private static int indexOf(String[] arr, String name) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i].equals(name)) {
                return i;
            }
        }
        throw new IllegalStateException("CSV 컬럼을 찾을 수 없음: " + name);
    }
}
