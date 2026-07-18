package emr.ekg.features.fiducial;

import emr.ekg.signal.PreprocessedEcg;
import emr.ekg.signal.RawEcgSignal;
import emr.ekg.signal.preprocess.EcgPreprocessor;
import emr.ekg.signal.preprocess.PreprocessConfig;
import emr.ekg.signal.wfdb.WfdbRecordReader;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class FiducialExtractorTest {

    private static final Path PTBXL_HEADER = Path.of(
            "/mnt/t7/datasets/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
                    + "/records500/00000/00001_hr.hea");

    @Test
    void extractsPlausibleBeatsFromRealPtbXlRecord() throws IOException {
        assumeTrue(Files.exists(PTBXL_HEADER), "PTB-XL 데이터셋 미탑재 - 스킵");

        RawEcgSignal raw = WfdbRecordReader.read(PTBXL_HEADER);
        PreprocessedEcg ecg = new EcgPreprocessor(PreprocessConfig.defaults()).process(raw);

        List<BeatFiducials> beats = FiducialExtractor.extract(ecg, "II");

        // 10초 스트립, 정상 성인 HR 가정 시 최소 5~20박 정도가 합리적 범위
        assertTrue(beats.size() >= 3, "검출된 심박 수가 너무 적음: " + beats.size());

        int previousR = -1;
        for (BeatFiducials b : beats) {
            assertTrue(b.rIndex() > previousR, "R-peak 인덱스는 단조 증가해야 함");
            assertTrue(b.qrsOnset() <= b.rIndex(), "QRS onset은 R-peak 이전이어야 함");
            assertTrue(b.qrsOffset() >= b.rIndex(), "QRS offset은 R-peak 이후여야 함");
            assertTrue(b.qrsOffset() < ecg.sampleCount(), "QRS offset이 신호 길이를 초과함");
            if (b.tOffset() >= 0) {
                assertTrue(b.tOffset() > b.qrsOffset(), "T-offset은 QRS offset 이후여야 함");
            }
            if (b.hasPWave()) {
                assertTrue(b.pOnset() <= b.pPeak() && b.pPeak() <= b.pOffset(),
                        "P파 onset<=peak<=offset 순서가 맞아야 함");
                assertTrue(b.pOffset() < b.qrsOnset(), "P파는 QRS onset 이전에 끝나야 함");
            }
            previousR = b.rIndex();
        }
    }
}
