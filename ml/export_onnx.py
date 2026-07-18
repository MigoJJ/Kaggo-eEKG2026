"""학습된 Stage1NormClassifier 체크포인트를 ONNX로 export한다.
Java `inference` 모듈이 ONNX Runtime으로 이 파일을 로드한다.
"""

import argparse
from pathlib import Path

import torch

from ecgml.models.stage1_norm import Stage1NormClassifier
from ecgml.models.stage1_norm_transformer import Stage1NormTransformer


def build_model(model_type: str) -> torch.nn.Module:
    if model_type == "cnn":
        return Stage1NormClassifier()
    if model_type == "transformer":
        return Stage1NormTransformer()
    raise ValueError(f"알 수 없는 model_type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", choices=["cnn", "transformer"], default="cnn")
    parser.add_argument("--out", default="models/stage1_norm.onnx")
    args = parser.parse_args()

    model = build_model(args.model_type)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.randn(1, 12, 5000)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["ecg_input"],
        output_names=["norm_logit"],
        dynamic_axes={"ecg_input": {0: "batch"}, "norm_logit": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # 신규 torch.export 기반 exporter는 onnxscript 의존성이 추가로 필요해 legacy 경로 사용
    )
    print(f"ONNX 모델 저장: {out_path}")

    import numpy as np
    import onnxruntime as ort

    with torch.no_grad():
        torch_out = model(dummy).numpy()
    sess = ort.InferenceSession(str(out_path))
    onnx_out = sess.run(None, {"ecg_input": dummy.numpy().astype(np.float32)})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    print(f"torch/onnxruntime 정합성 검증: max_abs_diff={max_diff:.2e}")
    if max_diff > 1e-3:
        raise RuntimeError(f"ONNX export 정합성 실패 (diff={max_diff})")


if __name__ == "__main__":
    main()
