package emr.ekg.inference;

/**
 * Python(PyTorch) 학습과 Java(ONNX Runtime) 추론 사이의 유일한 계약.
 * train/serve skew 방지를 위해 전처리 규격(fs, 리드순서, 정규화)을 모델과 함께 고정한다.
 *
 * @param modelPath   .onnx 파일 경로
 * @param modelHash   재현성/감사용 해시 (예: sha256)
 * @param version     모델 버전 문자열
 * @param inputName   ONNX 입력 텐서명
 * @param outputName  ONNX 출력 텐서명
 * @param expectedFs  요구 샘플링레이트(Hz)
 * @param expectedLeads 요구 리드 순서
 */
public record ModelSignature(
        String modelPath,
        String modelHash,
        String version,
        String inputName,
        String outputName,
        int expectedFs,
        String[] expectedLeads) {
}
