package emr.ekg.pipeline;

import emr.ekg.persistence.EcgRecordRow;
import emr.ekg.persistence.FeatureRow;
import emr.ekg.persistence.FindingRow;
import emr.ekg.persistence.ReportRepository;
import emr.ekg.persistence.ReportRow;

import java.sql.SQLException;
import java.util.List;

/**
 * DiagnosticReport(도메인 타입)를 persistence의 평면 row 타입으로 매핑해 저장한다.
 * persistence 모듈은 core-features/rule-engine에 의존하지 않으므로 이 변환이 필요하다.
 */
public final class ReportPersister {

    private final ReportRepository repository;

    public ReportPersister(ReportRepository repository) {
        this.repository = repository;
    }

    public void persist(DiagnosticReport report) throws SQLException {
        EcgRecordRow record = new EcgRecordRow(
                report.recordId(), report.source(), report.fs(), report.sampleCount(),
                report.sqi(), report.interpretable(), report.createdAt());

        List<FeatureRow> features = report.features().stream()
                .map(f -> new FeatureRow(report.recordId(), f.rank(), f.name(), f.value(), f.unit(),
                        f.normalLow(), f.normalHigh()))
                .toList();

        List<FindingRow> findings = report.findings().stream()
                .map(f -> new FindingRow(report.recordId(), f.code(), f.label(), f.severity().name(), f.evidence()))
                .toList();

        ReportRow reportRow = new ReportRow(
                report.recordId(), report.status(), report.normTriageScore(), report.triageModelVersion(),
                report.beatArrhythmiaAvailable(), report.stIschemiaAvailable(), null, null, report.createdAt());

        repository.save(record, features, findings, reportRow);
    }
}
