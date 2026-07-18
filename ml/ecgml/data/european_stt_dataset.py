"""European ST-T Database helpers.

The first target for EDB is not a black-box model. We use it to measure ST
deviation thresholds and episode-level false alarms before enabling Stage4 in
the Java pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, resample_poly

TARGET_FS = 500


@dataclass(frozen=True)
class StEpisode:
    record: str
    start: int
    end: int
    kind: str


def list_records(root: Path) -> list[str]:
    return sorted(str(p.with_suffix("").relative_to(root)) for p in Path(root).rglob("*.hea"))


def load_record(root: Path, record_name: str) -> tuple[np.ndarray, int, list[str]]:
    rec = wfdb.rdrecord(str(Path(root) / record_name))
    x = rec.p_signal.T.astype(np.float32)
    fs = int(rec.fs)
    if fs > 2 * 40:
        nyq = fs / 2
        b, a = butter(2, [0.5 / nyq, 40.0 / nyq], btype="band")
        x = filtfilt(b, a, x, axis=-1).astype(np.float32)
    if fs != TARGET_FS:
        from math import gcd
        g = gcd(TARGET_FS, fs)
        x = resample_poly(x, TARGET_FS // g, fs // g, axis=-1).astype(np.float32)
        fs = TARGET_FS
    return x, fs, list(rec.sig_name)


def extract_st_episodes(root: Path, record_name: str) -> list[StEpisode]:
    """Extract coarse ST/T episode intervals from available annotation notes.

    EDB annotations contain rich beat, rhythm, ST, T, and quality markers. The
    exact marker text can vary by file; this parser deliberately keeps broad
    ST/T episode markers and reports counts for manual review.
    """
    base = Path(root) / record_name
    out: list[StEpisode] = []
    for annotator in ("st", "atr"):
        try:
            ann = wfdb.rdann(str(base), annotator)
        except Exception:
            continue
        active: tuple[str, int] | None = None
        for sample, note in zip(ann.sample, ann.aux_note):
            text = (note or "").lower()
            if "st" not in text and "ische" not in text and "t-wave" not in text and "t wave" not in text:
                continue
            kind = "st" if "st" in text or "ische" in text else "t"
            if "begin" in text or "start" in text or text.startswith("("):
                active = (kind, int(sample))
            elif ("end" in text or text.startswith(")")) and active is not None:
                k, start = active
                if sample > start:
                    out.append(StEpisode(record_name, start, int(sample), k))
                active = None
            else:
                # Point event fallback: keep a 30 s window around the marker.
                half = 15 * TARGET_FS
                out.append(StEpisode(record_name, max(0, int(sample) - half), int(sample) + half, kind))
        if out:
            break
    return out


def estimate_st_deviation_mv(x: np.ndarray, fs: int) -> np.ndarray:
    """Simple beat-agnostic ST proxy for threshold sweeps.

    This is intentionally conservative: baseline is a rolling lower-frequency
    median proxy, and ST deviation is summarized by per-lead robust percentile.
    The Java rule engine still remains the production path.
    """
    baseline = np.median(x, axis=1, keepdims=True)
    centered = x - baseline
    return np.percentile(centered, 10, axis=1)
