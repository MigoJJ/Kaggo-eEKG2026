package emr.ekg.signal.wfdb;

/**
 * WFDB 헤더의 개별 신호(리드) 스펙 한 줄.
 *
 * @param fileName      데이터 파일명(.dat 또는 .mat)
 * @param format        WFDB 저장 포맷 (16, 212 등)
 * @param byteOffset    데이터 파일 시작에서 실제 신호 샘플까지 건너뛸 바이트 수(포맷 필드의
 *                      "+N" 표기, 예: "16+24" — CinC2020류 MATLAB v4 flat .mat 컨테이너가
 *                      24바이트 헤더 뒤에 format 16과 동일한 바이트 스트림을 담는 경우에 쓰임).
 *                      명시 없으면 0(.dat 표준 레코드).
 * @param gain          ADC gain (raw/units)
 * @param baseline      0 물리값에 대응하는 raw 값 (gain 필드에 명시 없으면 adcZero로 대체)
 * @param units         물리 단위 (기본 mV)
 * @param adcResolution ADC 해상도(비트)
 * @param adcZero       ADC 영점
 * @param initialValue  해당 신호 첫 샘플의 raw 값 (디코더 검증용)
 * @param checksum      전체 샘플 16비트 부호있는 합 (디코더 검증용)
 * @param blockSize     블록 크기
 * @param description   리드 명칭 (예: I, II, V1, MLII)
 */
public record SignalSpec(
        String fileName,
        int format,
        int byteOffset,
        double gain,
        double baseline,
        String units,
        int adcResolution,
        int adcZero,
        int initialValue,
        int checksum,
        int blockSize,
        String description) {
}
