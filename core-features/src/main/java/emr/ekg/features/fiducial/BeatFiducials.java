package emr.ekg.features.fiducial;

/**
 * 기준 리드(통상 Lead II) 1개 심박의 시간 fiducial(샘플 인덱스).
 * 12리드는 동시기록으로 동일 시간축을 공유하므로, 다른 리드의 진폭 측정에도
 * 이 인덱스를 그대로 적용할 수 있다(LeadMorphologyExtractor).
 *
 * P파가 검출되지 않으면(진폭이 노이즈 플로어 이하 등) pOnset/pOffset/pPeak는 -1이다.
 */
public record BeatFiducials(
        int rIndex,
        int qrsOnset,
        int qrsOffset,
        int pOnset,
        int pOffset,
        int pPeak,
        int tPeak,
        int tOffset) {

    public boolean hasPWave() {
        return pPeak >= 0;
    }
}
