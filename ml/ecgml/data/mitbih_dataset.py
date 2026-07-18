"""MIT-BIH 심박(beat) 단위 AAMI 5-클래스(N/S/V/F/Q) 분류 데이터셋.

Stage3(비트 부정맥 검증)의 backbone 학습용. Phase 4는 backbone 우선 — 정밀 정확도 튜닝은
Kaggle P100 학습 이후로 미룬다. 이 초기 버전은 Java EcgPreprocessor와 완전히 동일하지
않은 Python 자체 전처리(scipy)를 쓴다 — Stage1(PTB-XL)에서처럼 Java 산출물을 그대로 쓰는
train/serve 완전일치 방식은 추후 재정비 과제로 남긴다(ARCHITECTURE.md에 명시).

AAMI EC57 매핑:
  N(정상)              : N, L, R, e, j
  S(심방/상심실 조기박동) : A, a, J, S
  V(심실 조기박동)       : V, E
  F(융합박동)           : F
  Q(분류불가/조율리듬)    : P, f, Q, /
비트가 아닌 주석(리듬/신호품질/코멘트 마커: ! " + [ ] x | ~)은 제외한다.

표준 inter-patient 분할(de Chazal et al. 2004)을 따른다: DS1=train, DS2=test.
"""

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, resample_poly
import torch
from torch.utils.data import Dataset

AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "P": "Q", "f": "Q", "Q": "Q", "/": "Q",
}
CLASS_NAMES = ["N", "S", "V", "F", "Q"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# de Chazal et al. 2004 표준 inter-patient 분할
DS1_RECORDS = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
               201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2_RECORDS = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
               213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

TARGET_FS = 500
WINDOW_MS = 400  # R-peak 중심 ±200ms
WINDOW_SAMPLES = int(WINDOW_MS / 1000 * TARGET_FS)


def _preprocess(signal: np.ndarray, fs: int) -> np.ndarray:
    """0.5-40Hz 대역통과 + 500Hz 리샘플. Java EcgPreprocessor와 근사(완전 동일은 아님)."""
    nyq = fs / 2
    b, a = butter(2, [0.5 / nyq, 40.0 / nyq], btype="band")
    filtered = filtfilt(b, a, signal)
    if fs != TARGET_FS:
        from math import gcd
        g = gcd(TARGET_FS, fs)
        filtered = resample_poly(filtered, TARGET_FS // g, fs // g)
    return filtered


def extract_beats(record_path: str) -> list[tuple[np.ndarray, str]]:
    """레코드 하나에서 (윈도우, AAMI라벨) 목록을 추출한다. MLII(채널0) 사용."""
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")
    fs = record.fs

    signal = record.p_signal[:, 0]
    processed = _preprocess(signal, fs)
    scale = TARGET_FS / fs

    beats = []
    half = WINDOW_SAMPLES // 2
    for sample, symbol in zip(ann.sample, ann.symbol):
        aami = AAMI_MAP.get(symbol)
        if aami is None:
            continue
        center = int(round(sample * scale))
        lo, hi = center - half, center + half
        if lo < 0 or hi > len(processed):
            continue
        window = processed[lo:hi].astype(np.float32)
        beats.append((window, aami))
    return beats


class MitbihBeatDataset(Dataset):
    def __init__(self, mitbih_root: str, record_ids: list[int]):
        self.samples: list[tuple[np.ndarray, int]] = []
        for rid in record_ids:
            record_path = f"{mitbih_root}/{rid}"
            for window, label in extract_beats(record_path):
                mean, std = window.mean(), window.std()
                std = std if std > 1e-9 else 1.0
                normalized = (window - mean) / std
                self.samples.append((normalized, CLASS_TO_IDX[label]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        window, label = self.samples[idx]
        return torch.from_numpy(window).unsqueeze(0).float(), torch.tensor(label, dtype=torch.long)
