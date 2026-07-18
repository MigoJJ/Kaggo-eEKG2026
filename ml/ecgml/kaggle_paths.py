"""Kaggle input path discovery helpers.

Kaggle dataset mount paths depend on the uploader and slug. These helpers find
PhysioNet-style roots by file layout instead of hard-coding one path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KaggleDatasetRoots:
    ludb: Path | None = None
    qtdb: Path | None = None
    european_stt: Path | None = None
    ptbxl_csv: Path | None = None
    ptbxl_preprocessed: Path | None = None


def discover_kaggle_roots(input_root: str | Path = "/kaggle/input") -> KaggleDatasetRoots:
    root = Path(input_root)
    if not root.exists():
        return KaggleDatasetRoots()

    candidate_dirs = [p for p in root.rglob("*") if p.is_dir()]
    return KaggleDatasetRoots(
        ludb=_find_ludb_root(candidate_dirs),
        qtdb=_find_qtdb_root(candidate_dirs),
        european_stt=_find_european_stt_root(candidate_dirs),
        ptbxl_csv=_find_ptbxl_csv(root),
        ptbxl_preprocessed=_find_ptbxl_preprocessed_root(candidate_dirs),
    )


def require_path(path: Path | None, label: str) -> Path:
    if path is None:
        raise SystemExit(f"{label} root를 찾지 못했습니다. 명시적 인자를 전달하세요.")
    return path


def _find_ludb_root(candidate_dirs: list[Path]) -> Path | None:
    matches = []
    for path in candidate_dirs:
        if (path / "ludb.csv").exists() and (path / "RECORDS").exists():
            matches.append(path)
        elif path.name == "1.0.1" and "ludb" in str(path).lower() and (path / "RECORDS").exists():
            matches.append(path)
    return _shortest(matches)


def _find_qtdb_root(candidate_dirs: list[Path]) -> Path | None:
    matches = []
    for path in candidate_dirs:
        parts = {p.lower() for p in path.parts}
        if "ludb" in parts:
            continue
        if "qtdb" not in parts and "qt-database" not in parts:
            continue
        if (path / "RECORDS").exists() and any(path.rglob("*.hea")):
            matches.append(path)
    return _shortest(matches)


def _find_european_stt_root(candidate_dirs: list[Path]) -> Path | None:
    matches = []
    for path in candidate_dirs:
        lower = str(path).lower()
        if "european" not in lower and "edb" not in lower and "st-t" not in lower:
            continue
        if (path / "RECORDS").exists() and any(path.rglob("*.hea")):
            matches.append(path)
    return _shortest(matches)


def _find_ptbxl_csv(root: Path) -> Path | None:
    matches = sorted(root.rglob("ptbxl_database.csv"), key=lambda p: (len(p.parts), str(p)))
    return matches[0] if matches else None


def _find_ptbxl_preprocessed_root(candidate_dirs: list[Path]) -> Path | None:
    matches = []
    for path in candidate_dirs:
        manifest = path / "manifest.csv"
        if not manifest.exists():
            continue
        try:
            header = manifest.read_text(errors="ignore").splitlines()[0]
        except IndexError:
            continue
        if "ecg_id" in header and any(path.glob("*.f32")):
            matches.append(path)
    return _shortest(matches)


def _shortest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return sorted(paths, key=lambda p: (len(p.parts), str(p)))[0]
