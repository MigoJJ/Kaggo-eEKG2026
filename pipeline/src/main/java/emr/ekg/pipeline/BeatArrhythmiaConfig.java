package emr.ekg.pipeline;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Stage3(비트 부정맥) 설정. model_path가 없거나 파일이 존재하지 않으면 {@link #isAvailable()}가
 * false를 반환해 EkgPipeline이 조용히 건너뛰고 {@code beatArrhythmiaAvailable=false}로
 * 명시한다 — Phase 4 이전/모델 미배치 환경에서도 파이프라인이 죽지 않게 하기 위함(backbone 목표).
 */
public record BeatArrhythmiaConfig(
        Path modelPath,
        double ventricularBurdenThreshold,
        double supraventricularBurdenThreshold) {

    public boolean isAvailable() {
        return modelPath != null && Files.exists(modelPath);
    }

    public static BeatArrhythmiaConfig unavailable() {
        return new BeatArrhythmiaConfig(null, 0.05, 0.05);
    }
}
