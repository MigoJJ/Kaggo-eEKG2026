package emr.ekg.pipeline;

import emr.ekg.signal.PreprocessedEcg;

/** DiagnosticReport와 함께 렌더링용 전처리 신호(PreprocessedEcg)를 반환한다(app-fx ECG 뷰어용). */
public record PipelineResult(DiagnosticReport report, PreprocessedEcg ecg) {
}
