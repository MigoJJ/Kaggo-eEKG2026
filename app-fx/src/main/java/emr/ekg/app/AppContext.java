package emr.ekg.app;

import emr.ekg.features.Sex;
import emr.ekg.persistence.EmrDatabase;
import emr.ekg.persistence.ReportRepository;
import emr.ekg.pipeline.EkgPipeline;
import emr.ekg.pipeline.PipelineConfig;
import emr.ekg.pipeline.PipelineResult;
import emr.ekg.pipeline.ReportPersister;
import emr.ekg.signal.RawEcgSignal;
import emr.ekg.signal.wfdb.WfdbRecordReader;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.SQLException;

/**
 * 파이프라인 + SQLite persistence 결선. UI 레이어가 직접 도메인 조립 로직을 알 필요 없게 한다.
 * config/pipeline.yaml로 구성되므로, 임계치·모델경로 재보정은 코드 변경 없이 YAML만 고치면 된다.
 */
public final class AppContext {

    private final EkgPipeline pipeline;
    private final ReportRepository repository;
    private final ReportPersister persister;

    public AppContext(Path configYaml, Path dbFile) throws IOException, SQLException {
        PipelineConfig config = PipelineConfig.loadFromYaml(configYaml);
        this.pipeline = EkgPipeline.fromConfig(config);
        Files.createDirectories(dbFile.toAbsolutePath().getParent());
        EmrDatabase db = EmrDatabase.openFile(dbFile);
        this.repository = new ReportRepository(db);
        this.persister = new ReportPersister(repository);
    }

    public PipelineResult loadWfdbRecord(String recordId, String source, Path headerPath, Sex sex)
            throws IOException, SQLException {
        RawEcgSignal raw = WfdbRecordReader.read(headerPath);
        PipelineResult result = pipeline.processWithSignal(recordId, source, raw, sex);
        persister.persist(result.report());
        return result;
    }

    public void sign(String recordId, String signedBy) throws SQLException {
        repository.sign(recordId, signedBy);
    }
}
