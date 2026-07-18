"""학습된 Stage3 BeatClassifier 체크포인트를 ONNX로 export한다."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ecgml.data.mitbih_dataset import WINDOW_SAMPLES
from ecgml.models.stage3_beat import BeatClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="models/stage3_beat.onnx")
    args = parser.parse_args()

    model = BeatClassifier()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.randn(1, 1, WINDOW_SAMPLES)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["beat_window"], output_names=["class_logits"],
        dynamic_axes={"beat_window": {0: "batch"}, "class_logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    with torch.no_grad():
        torch_out = model(dummy).numpy()
    sess = ort.InferenceSession(str(out_path))
    onnx_out = sess.run(None, {"beat_window": dummy.numpy().astype(np.float32)})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    print(f"ONNX 저장: {out_path}, torch/onnxruntime 정합성 max_abs_diff={max_diff:.2e}")
    if max_diff > 1e-3:
        raise RuntimeError(f"ONNX export 정합성 실패 (diff={max_diff})")


if __name__ == "__main__":
    main()
