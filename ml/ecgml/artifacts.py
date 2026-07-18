"""Model artifact sidecar helpers."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def runtime_environment() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": package_version("torch"),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "onnx": package_version("onnx"),
        "onnxruntime": package_version("onnxruntime"),
        "scipy": package_version("scipy"),
        "wfdb": package_version("wfdb"),
    }


def write_sidecar(out_path: Path, payload: dict[str, Any]) -> Path:
    sidecar_path = out_path.with_suffix(".json")
    payload = dict(payload)
    payload["artifact"] = out_path.name
    payload["artifact_sha256"] = sha256_file(out_path)
    payload["metadata_created_at"] = datetime.now(UTC).isoformat()
    payload["environment"] = runtime_environment()
    sidecar_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return sidecar_path
