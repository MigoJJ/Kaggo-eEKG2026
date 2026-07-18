package emr.ekg.features;

import emr.ekg.features.fiducial.BeatFiducials;
import emr.ekg.features.fiducial.FiducialExtractor;
import emr.ekg.features.morphology.LeadMorphology;
import emr.ekg.features.morphology.LeadMorphologyExtractor;
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
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class ClinicalFeatureExtractorTest {

    private static final Path PTBXL_HEADER = Path.of(
            "/mnt/t7/datasets/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
                    + "/records500/00000/00001_hr.hea");

    @Test
    void extractsTwentyPlausibleFeaturesFromRealNormRecord() throws IOException {
        assumeTrue(Files.exists(PTBXL_HEADER), "PTB-XL 데이터셋 미탑재 - 스킵");

        // ecg_id=1 (00001_hr): scp_codes에 NORM 100.0 라벨 (PTB-XL 데이터베이스 기준)
        RawEcgSignal raw = WfdbRecordReader.read(PTBXL_HEADER);
        PreprocessedEcg ecg = new EcgPreprocessor(PreprocessConfig.defaults()).process(raw);

        List<BeatFiducials> beats = FiducialExtractor.extract(ecg, "II");
        Map<String, LeadMorphology> morphology = LeadMorphologyExtractor.extract(ecg, beats);
        List<ClinicalFeature> features =
                ClinicalFeatureExtractor.extract(ecg, beats, morphology, Sex.UNKNOWN, "II");

        assertEquals(20, features.size());

        double hr = valueOf(features, "HR");
        assertTrue(hr > 30 && hr < 220, "HR이 생리적으로 타당하지 않음: " + hr);

        double qrs = valueOf(features, "QRS_DURATION");
        assertTrue(qrs > 40 && qrs < 250, "QRS 폭이 생리적으로 타당하지 않음: " + qrs);

        double axis = valueOf(features, "QRS_AXIS");
        assertTrue(axis >= -180 && axis <= 180, "축 각도가 -180~180 범위를 벗어남: " + axis);
    }

    private static double valueOf(List<ClinicalFeature> features, String name) {
        for (ClinicalFeature f : features) {
            if (f.name().equals(name)) {
                return f.value();
            }
        }
        throw new AssertionError("피처를 찾을 수 없음: " + name);
    }
}
