package emr.ekg.signal.preprocess;

import emr.ekg.signal.PreprocessedEcg;
import emr.ekg.signal.RawEcgSignal;
import emr.ekg.signal.wfdb.WfdbRecordReader;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class EcgPreprocessorTest {

    private static final Path PTBXL_HEADER = Path.of(
            "/mnt/t7/datasets/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
                    + "/records500/00000/00001_hr.hea");
    private static final Path MITBIH_HEADER =
            Path.of("/mnt/t7/datasets/mit-bih-arrhythmia-database-1.0.0/101.hea");

    @Test
    void preprocessesPtbXlRecordAt500Hz() throws IOException {
        assumeTrue(Files.exists(PTBXL_HEADER), "PTB-XL 데이터셋 미탑재 - 스킵");

        RawEcgSignal raw = WfdbRecordReader.read(PTBXL_HEADER);
        PreprocessedEcg pre = new EcgPreprocessor(PreprocessConfig.defaults()).process(raw);

        assertEquals(12, pre.leadCount());
        assertEquals(5000, pre.sampleCount()); // 이미 500Hz라 리샘플 no-op
        assertEquals(500, pre.fs());
        assertTrue(pre.sqi() > 0.5, "정상 PTB-XL 레코드는 SQI가 양호해야 함, 실제=" + pre.sqi());
        assertTrue(pre.isInterpretable(0.6));

        for (double[] lead : pre.samples()) {
            for (double v : lead) {
                assertFalse(Double.isNaN(v));
                assertFalse(Double.isInfinite(v));
            }
        }
    }

    @Test
    void resamplesMitBihFrom360To500Hz() throws IOException {
        assumeTrue(Files.exists(MITBIH_HEADER), "MIT-BIH 데이터셋 미탑재 - 스킵");

        RawEcgSignal full = WfdbRecordReader.read(MITBIH_HEADER);
        // 테스트 속도를 위해 앞 10초(3600 샘플 @360Hz)만 슬라이스
        int sliceLen = 10 * full.fs();
        double[][] sliced = new double[full.leadCount()][];
        for (int i = 0; i < full.leadCount(); i++) {
            sliced[i] = Arrays.copyOf(full.samples()[i], sliceLen);
        }
        RawEcgSignal tenSec = new RawEcgSignal(sliced, full.fs(), full.leadNames());

        PreprocessedEcg pre = new EcgPreprocessor(PreprocessConfig.defaults()).process(tenSec);

        assertEquals(2, pre.leadCount());
        assertEquals(500, pre.fs());
        assertEquals(5000, pre.sampleCount());
    }
}
