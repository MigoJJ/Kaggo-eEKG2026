"""Step 4: export Stage4 delineator checkpoint to ONNX."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ecgml.artifacts import write_sidecar
from ecgml.data.wfdb_delineation_dataset import CLASS_NAMES, WINDOW_SAMPLES
from ecgml.models.stage4_delineator import Stage4Delineator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="models/stage4_delineator.onnx")
    parser.add_argument("--version", help="sidecar JSON에 기록할 모델 버전")
    args = parser.parse_args()

    model = Stage4Delineator(n_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.randn(1, 12, WINDOW_SAMPLES)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(out),
        input_names=["ecg"], output_names=["wave_logits"],
        dynamic_axes={"ecg": {0: "batch"}, "wave_logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    torch_out = model(dummy).detach().numpy()
    sess = ort.InferenceSession(str(out))
    onnx_out = sess.run(None, {"ecg": dummy.numpy().astype(np.float32)})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    print(f"ONNX saved: {out}, max_abs_diff={max_diff:.2e}")
    if max_diff > 1e-3:
        raise RuntimeError(f"ONNX export mismatch: {max_diff}")

    sidecar = write_sidecar(out, {
        "stage": "stage4",
        "version": args.version or "stage4-delineator-ludb-qtdb",
        "model_type": "stage4_delineator",
        "opset_version": 17,
        "trained_on": {
            "datasets": ["LUDB", "QTDB"],
            "target_fs": 500,
            "window_samples": WINDOW_SAMPLES,
        },
        "input_signature": {
            "input_name": "ecg",
            "shape": ["batch", 12, WINDOW_SAMPLES],
            "dtype": "float32",
            "normalization": "per-lead z-normalization, std_floor=1e-9",
        },
        "output_signature": {
            "output_name": "wave_logits",
            "shape": ["batch", len(CLASS_NAMES), 12, WINDOW_SAMPLES],
            "classes": CLASS_NAMES,
        },
        "export_validation": {
            "max_abs_diff": max_diff,
            "threshold": 1e-3,
        },
    })
    print(f"sidecar saved: {sidecar}")


if __name__ == "__main__":
    main()
