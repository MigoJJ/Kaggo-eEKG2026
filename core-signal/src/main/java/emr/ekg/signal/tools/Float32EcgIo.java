package emr.ekg.signal.tools;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * 배치 도구 간 공유하는 .f32 이진 포맷(row-major [lead][sample], float32 little-endian) I/O.
 * {@code PtbxlPreprocessExporter}(쓰기)와 core-features의 배치 피처 추출 도구(읽기)가 공용한다.
 */
public final class Float32EcgIo {

    private Float32EcgIo() {
    }

    public static void write(Path outFile, double[][] samples) throws IOException {
        int nLeads = samples.length;
        int nSamples = samples[0].length;
        ByteBuffer buffer = ByteBuffer.allocate(nLeads * nSamples * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (double[] lead : samples) {
            for (double v : lead) {
                buffer.putFloat((float) v);
            }
        }
        Files.write(outFile, buffer.array(), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }

    public static double[][] read(Path inFile, int nLeads, int nSamples) throws IOException {
        byte[] bytes = Files.readAllBytes(inFile);
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        double[][] samples = new double[nLeads][nSamples];
        for (int lead = 0; lead < nLeads; lead++) {
            for (int i = 0; i < nSamples; i++) {
                samples[lead][i] = buffer.getFloat();
            }
        }
        return samples;
    }
}
