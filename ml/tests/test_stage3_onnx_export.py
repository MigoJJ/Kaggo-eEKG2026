import subprocess
import sys
import tempfile
import unittest
import json
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = REPO_ROOT / "ml"
sys.path.insert(0, str(ML_ROOT))

from ecgml.models.stage3_beat import build_beat_classifier  # noqa: E402


class Stage3OnnxExportTest(unittest.TestCase):
    def test_all_stage3_backbones_export_to_onnx(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for model_type in ("cnn", "resnet", "inception"):
                with self.subTest(model_type=model_type):
                    checkpoint = tmp_dir / f"stage3_{model_type}.pt"
                    onnx_path = tmp_dir / f"stage3_{model_type}.onnx"
                    torch.save(build_beat_classifier(model_type).state_dict(), checkpoint)

                    subprocess.run(
                        [
                            sys.executable,
                            str(ML_ROOT / "export_stage3_onnx.py"),
                            "--checkpoint",
                            str(checkpoint),
                            "--model-type",
                            model_type,
                            "--out",
                            str(onnx_path),
                        ],
                        cwd=REPO_ROOT,
                        check=True,
                    )

                    self.assertTrue(onnx_path.exists(), f"ONNX export missing: {onnx_path}")
                    sidecar = onnx_path.with_suffix(".json")
                    self.assertTrue(sidecar.exists(), f"sidecar missing: {sidecar}")
                    payload = json.loads(sidecar.read_text())
                    self.assertEqual("stage3", payload["stage"])
                    self.assertEqual(model_type, payload["model_type"])
                    self.assertEqual(17, payload["opset_version"])
                    self.assertRegex(payload["artifact_sha256"], r"^[0-9a-f]{64}$")
                    self.assertIn("environment", payload)


if __name__ == "__main__":
    unittest.main()
