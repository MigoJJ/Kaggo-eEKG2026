"""WFDB delineation datasets for QTDB/LUDB.

This module builds supervised P/QRS/T segmentation windows from WFDB records
with waveform annotations. It is intentionally tolerant of the annotation
style because QTDB and LUDB use slightly different annotator/lead conventions.
Boundary markers are preferred when present; otherwise peak-centered fallback
windows are used so Kaggle runs can start before dataset-specific tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb
from scipy.signal import butter, filtfilt, resample_poly

from ecgml.preprocess import znormalize_per_lead

TARGET_FS = 500
TARGET_LEADS = 12
WINDOW_SAMPLES = 5000
CLASS_NAMES = ["background", "p", "qrs", "t"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class WaveSpan:
    lead: int
    start: int
    end: int
    label: str


def list_wfdb_records(root: Path) -> list[str]:
    root = Path(root)
    records = []
    for hea in sorted(root.rglob("*.hea")):
        records.append(str(hea.with_suffix("").relative_to(root)))
    return records


def _resample_signal(signal: np.ndarray, fs: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if fs > 2 * 40:
        nyq = fs / 2
        b, a = butter(2, [0.5 / nyq, 40.0 / nyq], btype="band")
        signal = filtfilt(b, a, signal, axis=-1).astype(np.float32)
    if fs != TARGET_FS:
        from math import gcd
        g = gcd(TARGET_FS, int(fs))
        signal = resample_poly(signal, TARGET_FS // g, int(fs) // g, axis=-1).astype(np.float32)
    return signal


def _label_from_note(note: str, symbol: str) -> str | None:
    text = (note or "").lower()
    symbol = (symbol or "").lower()
    if "qrs" in text or "n" in text or symbol in {"n", "l", "r"}:
        return "qrs"
    if "p" in text or symbol == "p":
        return "p"
    if "t" in text or symbol == "t":
        return "t"
    return None


def _annotation_candidates(record_base: Path, lead_names: Iterable[str]) -> list[tuple[str, int]]:
    """Return (annotator, lead_index) candidates.

    LUDB often stores lead-specific annotators such as ``ii``/``v1``. QTDB
    often stores global annotators such as ``q1c``/``q2c``. We try both.
    """
    lead_names = list(lead_names)
    candidates: list[tuple[str, int]] = []

    for i, lead in enumerate(lead_names):
        norm = lead.lower().replace("avr", "avr").replace("avl", "avl").replace("avf", "avf")
        candidates.append((norm, i))

    for annotator in ("q1c", "q2c", "qt1", "qt2", "pu", "atr"):
        candidates.append((annotator, 0))

    # Add every sidecar suffix as a last resort, excluding signal/header files.
    for path in record_base.parent.glob(record_base.name + ".*"):
        suffix = path.suffix[1:]
        if suffix not in {"hea", "dat", "mat", "xws"}:
            candidates.append((suffix, 0))

    seen = set()
    out = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_wave_spans(record_base: Path, fs: int, lead_names: list[str]) -> list[WaveSpan]:
    spans: list[WaveSpan] = []
    scale = TARGET_FS / fs

    for annotator, lead_idx in _annotation_candidates(record_base, lead_names):
        try:
            ann = wfdb.rdann(str(record_base), annotator)
        except Exception:
            continue

        active: tuple[str, int] | None = None
        for sample, symbol, note in zip(ann.sample, ann.symbol, ann.aux_note):
            label = _label_from_note(note, symbol)
            pos = int(round(sample * scale))
            sym = symbol.strip()
            note_l = (note or "").lower()

            if sym == "(" or note_l.startswith("("):
                if label is not None:
                    active = (label, pos)
                continue
            if sym == ")" or note_l.startswith(")"):
                if active is not None:
                    wave, start = active
                    if pos > start:
                        spans.append(WaveSpan(lead_idx, start, pos, wave))
                    active = None
                continue

            if label is not None:
                # Peak-only fallback. Boundaries are better, but this allows
                # early Kaggle runs even when annotations are not boundary-style.
                half_ms = {"p": 45, "qrs": 60, "t": 90}[label]
                half = int(round(half_ms * TARGET_FS / 1000))
                spans.append(WaveSpan(lead_idx, max(0, pos - half), pos + half, label))

        if spans:
            break

    return spans


def build_mask(n_leads: int, n_samples: int, spans: list[WaveSpan]) -> np.ndarray:
    mask = np.zeros((n_leads, n_samples), dtype=np.int64)
    for span in spans:
        if span.lead < 0 or span.lead >= n_leads:
            continue
        lo = max(0, span.start)
        hi = min(n_samples, span.end)
        if hi > lo and span.label in CLASS_TO_IDX:
            mask[span.lead, lo:hi] = CLASS_TO_IDX[span.label]
    return mask


def _center_crop_or_pad(x: np.ndarray, target_len: int) -> np.ndarray:
    n = x.shape[-1]
    if n == target_len:
        return x
    if n > target_len:
        start = (n - target_len) // 2
        return x[..., start:start + target_len]
    pad_left = (target_len - n) // 2
    pad_right = target_len - n - pad_left
    return np.pad(x, [(0, 0), (pad_left, pad_right)], mode="edge")


def _fit_leads(x: np.ndarray, target_leads: int, pad_value: float = 0.0) -> np.ndarray:
    n_leads = x.shape[0]
    if n_leads == target_leads:
        return x
    if n_leads > target_leads:
        return x[:target_leads]
    pad_shape = [(0, target_leads - n_leads), (0, 0)]
    return np.pad(x, pad_shape, mode="constant", constant_values=pad_value)


class WfdbDelineationDataset(Dataset):
    """Record-level 12-lead P/QRS/T segmentation dataset."""

    def __init__(self, root: str | Path, records: list[str] | None = None):
        self.root = Path(root)
        self.records = records if records is not None else list_wfdb_records(self.root)
        self.samples: list[tuple[np.ndarray, np.ndarray, str]] = []
        self._load()

    def _load(self) -> None:
        for rec in self.records:
            base = self.root / rec
            try:
                record = wfdb.rdrecord(str(base))
            except Exception as e:
                print(f"skip {rec}: record read failed: {e}")
                continue

            signal = record.p_signal.T.astype(np.float32)
            signal = _resample_signal(signal, int(record.fs))
            spans = extract_wave_spans(base, int(record.fs), list(record.sig_name))
            if not spans:
                print(f"skip {rec}: no usable delineation annotations")
                continue

            mask = build_mask(signal.shape[0], signal.shape[1], spans)
            signal = _center_crop_or_pad(signal, WINDOW_SAMPLES)
            mask = _center_crop_or_pad(mask, WINDOW_SAMPLES)
            signal = _fit_leads(signal, TARGET_LEADS, pad_value=0.0)
            mask = _fit_leads(mask, TARGET_LEADS, pad_value=0)
            signal = znormalize_per_lead(signal).astype(np.float32)
            self.samples.append((signal, mask.astype(np.int64), rec))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, _ = self.samples[idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).long()
