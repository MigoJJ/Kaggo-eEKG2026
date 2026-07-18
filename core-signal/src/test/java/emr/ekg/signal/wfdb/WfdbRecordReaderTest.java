package emr.ekg.signal.wfdb;

import emr.ekg.signal.RawEcgSignal;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class WfdbRecordReaderTest {

    private static final Path PTBXL_HEADER = Path.of(
            "/mnt/t7/datasets/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
                    + "/records500/00000/00001_hr.hea");
    private static final Path MITBIH_HEADER =
            Path.of("/mnt/t7/datasets/mit-bih-arrhythmia-database-1.0.0/101.hea");
    private static final Path CPSC2018_HEADER = Path.of(
            "/mnt/t7/datasets/ecg_data/archive (1)/Training_WFDB/A0001.hea");
    private static final Path CHAPMAN_SHAOXING_HEADER = Path.of(
            "/mnt/t7/datasets/ecg_data/archive (2)/WFDB_ShaoxingUniv/JS00001.hea");

    @Test
    void readsPtbXlFormat16Record() throws IOException {
        assumeTrue(Files.exists(PTBXL_HEADER), "PTB-XL 데이터셋 미탑재 - 스킵");

        RawEcgSignal ecg = WfdbRecordReader.read(PTBXL_HEADER);

        assertEquals(12, ecg.leadCount());
        assertEquals(5000, ecg.sampleCount());
        assertEquals(500, ecg.fs());
        assertArrayEquals(
                new String[] {"I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"},
                ecg.leadNames());

        // 첫 샘플은 헤더의 initial value(-115)와 일치해야 함: (-115 - baseline0) / gain1000
        assertEquals(-0.115, ecg.samples()[0][0], 1e-9);

        for (double[] lead : ecg.samples()) {
            for (double v : lead) {
                assertFalse(Double.isNaN(v));
            }
        }
    }

    @Test
    void readsMitBihFormat212Record() throws IOException {
        assumeTrue(Files.exists(MITBIH_HEADER), "MIT-BIH 데이터셋 미탑재 - 스킵");

        RawEcgSignal ecg = WfdbRecordReader.read(MITBIH_HEADER);

        assertEquals(2, ecg.leadCount());
        assertEquals(650_000, ecg.sampleCount());
        assertEquals(360, ecg.fs());
        assertArrayEquals(new String[] {"MLII", "V1"}, ecg.leadNames());

        // 첫 샘플은 헤더의 initial value와 일치해야 함: (raw - adcZero1024) / gain200
        assertEquals((955 - 1024) / 200.0, ecg.samples()[0][0], 1e-9);
        assertEquals((992 - 1024) / 200.0, ecg.samples()[1][0], 1e-9);
    }

    @Test
    void readsCpsc2018MatFormat16WithByteOffset() throws IOException {
        assumeTrue(Files.exists(CPSC2018_HEADER), "CPSC2018 데이터셋 미탑재 - 스킵");

        RawEcgSignal ecg = WfdbRecordReader.read(CPSC2018_HEADER);

        assertEquals(12, ecg.leadCount());
        assertEquals(7500, ecg.sampleCount());
        assertEquals(500, ecg.fs());
        for (double[] lead : ecg.samples()) {
            for (double v : lead) {
                assertFalse(Double.isNaN(v));
            }
        }
    }

    @Test
    void cpsc2018ChecksumMatchesHeaderAfterByteOffsetSkip() throws IOException {
        assumeTrue(Files.exists(CPSC2018_HEADER), "CPSC2018 데이터셋 미탑재 - 스킵");
        assertChecksumMatchesHeader(CPSC2018_HEADER);
    }

    @Test
    void chapmanShaoxingChecksumMatchesHeaderAfterByteOffsetSkip() throws IOException {
        assumeTrue(Files.exists(CHAPMAN_SHAOXING_HEADER), "Chapman-Shaoxing 데이터셋 미탑재 - 스킵");
        assertChecksumMatchesHeader(CHAPMAN_SHAOXING_HEADER);
    }

    /**
     * .mat(MATLAB v4 flat) 컨테이너의 "16+24" byteOffset skip이 정확한지, 헤더에 기록된
     * 체크섬과 실제 디코딩 결과를 비교해 증명한다(MIT-BIH format 212 테스트와 동일 원리).
     */
    private static void assertChecksumMatchesHeader(Path headerFile) throws IOException {
        WfdbHeader header = WfdbHeaderParser.parse(headerFile);
        byte[] fileBytes = Files.readAllBytes(
                headerFile.toAbsolutePath().getParent().resolve(header.signals().get(0).fileName()));
        int byteOffset = header.signals().get(0).byteOffset();
        byte[] raw = java.util.Arrays.copyOfRange(fileBytes, byteOffset, fileBytes.length);

        int nSignals = header.signalCount();
        int nSamples = header.sampleCount();
        int[] flat = Format16Codec.decode(raw, nSignals * nSamples);

        for (int ch = 0; ch < nSignals; ch++) {
            int sum = 0;
            for (int i = 0; i < nSamples; i++) {
                sum += flat[i * nSignals + ch];
            }
            short checksum = (short) sum;
            assertEquals((short) header.signals().get(ch).checksum(), checksum,
                    "체크섬 불일치 - byteOffset(" + byteOffset + ") 또는 디코더 오류 의심 (리드 "
                            + header.signals().get(ch).description() + ")");
        }
    }

    @Test
    void mitBihChecksumMatchesHeader() throws IOException {
        assumeTrue(Files.exists(MITBIH_HEADER), "MIT-BIH 데이터셋 미탑재 - 스킵");

        WfdbHeader header = WfdbHeaderParser.parse(MITBIH_HEADER);
        byte[] raw = Files.readAllBytes(
                MITBIH_HEADER.toAbsolutePath().getParent().resolve(header.signals().get(0).fileName()));
        int nSignals = header.signalCount();
        int nSamples = header.sampleCount();
        int[] flat = Format212Codec.decode(raw, nSignals * nSamples);

        for (int ch = 0; ch < nSignals; ch++) {
            int sum = 0;
            for (int i = 0; i < nSamples; i++) {
                sum += flat[i * nSignals + ch];
            }
            short checksum = (short) sum;
            assertEquals((short) header.signals().get(ch).checksum(), checksum,
                    "체크섬 불일치 - 디코더 오류 의심 (리드 " + header.signals().get(ch).description() + ")");
        }
    }
}
