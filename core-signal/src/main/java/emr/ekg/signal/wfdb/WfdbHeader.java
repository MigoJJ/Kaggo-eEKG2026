package emr.ekg.signal.wfdb;

import java.util.List;

/** WFDB 텍스트 헤더(.hea) 파싱 결과. */
public record WfdbHeader(
        String recordName,
        int signalCount,
        int fs,
        int sampleCount,
        List<SignalSpec> signals) {
}
