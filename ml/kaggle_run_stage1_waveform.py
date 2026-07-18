"""Kaggle entrypoint for Stage1 PTB-XL waveform training.

Requires Java-preprocessed PTB-XL artifacts: ``manifest.csv`` plus ``*.f32``.
The production Stage1 feature model remains unchanged; this runner produces
candidate waveform ONNX+JSON artifacts for comparison.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from ecgml.kaggle_paths import discover_kaggle_roots, require_path


def run(cmd: list[str], cwd: Path) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--working-dir", default="/kaggle/working")
    parser.add_argument("--export-dir", help="PtbxlPreprocessExporter output: manifest.csv + *.f32")
    parser.add_argument("--ptbxl-csv", help="ptbxl_database.csv")
    parser.add_argument("--model-type", choices=["cnn", "transformer"], default="transformer")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--version")
    args = parser.parse_args()

    ml_dir = Path(__file__).resolve().parent
    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_kaggle_roots(args.input_root)
    export_dir = Path(args.export_dir) if args.export_dir else require_path(
        discovered.ptbxl_preprocessed, "PTB-XL preprocessed"
    )
    ptbxl_csv = Path(args.ptbxl_csv) if args.ptbxl_csv else require_path(discovered.ptbxl_csv, "ptbxl_database.csv")

    resolved = {
        "created_at": datetime.now(UTC).isoformat(),
        "export_dir": str(export_dir),
        "ptbxl_csv": str(ptbxl_csv),
        "model_type": args.model_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    (working_dir / "stage1_waveform_resolved_paths.json").write_text(json.dumps(resolved, indent=2) + "\n")
    print(json.dumps(resolved, indent=2), flush=True)

    checkpoint = working_dir / f"stage1_norm_{args.model_type}.pt"
    onnx = working_dir / f"stage1_norm_{args.model_type}.onnx"
    version = args.version or datetime.now(UTC).strftime(f"stage1-waveform-{args.model_type}-ptbxl-%Y%m%d")

    if not args.skip_train:
        run([
            sys.executable,
            "train_stage1.py",
            "--export-dir",
            str(export_dir),
            "--ptbxl-csv",
            str(ptbxl_csv),
            "--model-type",
            args.model_type,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--out",
            str(checkpoint),
        ], cwd=ml_dir)

    run([
        sys.executable,
        "export_onnx.py",
        "--checkpoint",
        str(checkpoint),
        "--model-type",
        args.model_type,
        "--out",
        str(onnx),
        "--version",
        version,
    ], cwd=ml_dir)

    print("\nKaggle outputs:", flush=True)
    for path in sorted(working_dir.glob(f"stage1_norm_{args.model_type}.*")):
        print(path, flush=True)


if __name__ == "__main__":
    main()
