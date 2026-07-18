"""Kaggle entrypoint for Stage4 LUDB/QTDB training and European ST-T sweep.

This script keeps Kaggle execution reproducible while preserving the local
contract-first design: train/export on Kaggle, then copy ONNX+JSON artifacts
back to ``models/`` without Java code changes.
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
    parser.add_argument("--ludb-root")
    parser.add_argument("--qtdb-root")
    parser.add_argument("--edb-root")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-edb-sweep", action="store_true")
    parser.add_argument("--thresholds-uv", default="50,75,100,150,200")
    parser.add_argument("--version", default=None)
    args = parser.parse_args()

    ml_dir = Path(__file__).resolve().parent
    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_kaggle_roots(args.input_root)
    ludb_root = Path(args.ludb_root) if args.ludb_root else require_path(discovered.ludb, "LUDB")
    qtdb_root = Path(args.qtdb_root) if args.qtdb_root else require_path(discovered.qtdb, "QTDB")
    edb_root = Path(args.edb_root) if args.edb_root else discovered.european_stt

    resolved = {
        "created_at": datetime.now(UTC).isoformat(),
        "ludb_root": str(ludb_root),
        "qtdb_root": str(qtdb_root),
        "edb_root": None if edb_root is None else str(edb_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    (working_dir / "stage4_resolved_paths.json").write_text(json.dumps(resolved, indent=2) + "\n")
    print(json.dumps(resolved, indent=2), flush=True)

    checkpoint = working_dir / "stage4_delineator.pt"
    onnx = working_dir / "stage4_delineator.onnx"
    version = args.version or datetime.now(UTC).strftime("stage4-delineator-ludb-qtdb-%Y%m%d")

    if not args.skip_train:
        run([
            sys.executable,
            "train_stage4_delineator.py",
            "--ludb-root",
            str(ludb_root),
            "--qtdb-root",
            str(qtdb_root),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--out",
            str(checkpoint),
        ], cwd=ml_dir)

    if not args.skip_edb_sweep and edb_root is not None:
        run([
            sys.executable,
            "evaluate_european_stt_thresholds.py",
            "--edb-root",
            str(edb_root),
            "--thresholds-uv",
            args.thresholds_uv,
            "--out",
            str(working_dir / "european_stt_threshold_sweep.csv"),
        ], cwd=ml_dir)
    elif not args.skip_edb_sweep:
        print("European ST-T root를 찾지 못해 threshold sweep을 건너뜁니다.", flush=True)

    run([
        sys.executable,
        "export_stage4_delineator_onnx.py",
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(onnx),
        "--version",
        version,
    ], cwd=ml_dir)

    print("\nKaggle outputs:", flush=True)
    for path in sorted(working_dir.glob("stage4_delineator.*")):
        print(path, flush=True)
    sweep = working_dir / "european_stt_threshold_sweep.csv"
    if sweep.exists():
        print(sweep, flush=True)


if __name__ == "__main__":
    main()
