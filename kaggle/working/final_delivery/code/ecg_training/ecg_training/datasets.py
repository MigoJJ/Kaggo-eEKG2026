import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset, WeightedRandomSampler

from .augmentations import ECGAugmenter


def normalize_signal(signal, mode):
    signal = np.asarray(signal, dtype=np.float32)
    if mode == "zscore_per_lead":
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)
        return (signal - mean) / np.clip(std, 1e-6, None)
    if mode == "global_zscore":
        mean = signal.mean()
        std = signal.std()
        return (signal - mean) / max(float(std), 1e-6)
    return signal


def load_ptbxl_metadata(dataset_dir, superclasses):
    dataset_dir = Path(dataset_dir)
    database = pd.read_csv(
        dataset_dir / "ptbxl_database.csv",
        index_col="ecg_id",
        converters={"scp_codes": ast.literal_eval},
    )
    statements = pd.read_csv(dataset_dir / "scp_statements.csv", index_col=0)
    diagnostic = statements[statements["diagnostic"] == 1]
    code_to_superclass = diagnostic["diagnostic_class"].dropna().to_dict()

    labels = []
    for codes in database["scp_codes"]:
        active = set()
        for code in codes:
            superclass = code_to_superclass.get(code)
            if superclass in superclasses:
                active.add(superclass)
        labels.append([1.0 if name in active else 0.0 for name in superclasses])

    database = database.copy()
    database["labels"] = labels
    database["num_positive_labels"] = [int(sum(label_row)) for label_row in labels]
    database = database[database["num_positive_labels"] > 0].reset_index()
    return database


def load_arrhythmia_metadata(dataset_dir):
    """
    PTB-XL 데이터셋에서 부정맥 6종(AFib, AFLT, SVPB, PVC, SVTA, VTA)을 정밀 추출합니다.
    """
    dataset_dir = Path(dataset_dir)
    database = pd.read_csv(
        dataset_dir / "ptbxl_database.csv",
        index_col="ecg_id",
        converters={"scp_codes": ast.literal_eval},
    )
    
    # 정밀 타격할 부정맥 레이블 정의 (SCP Codes)
    # AFIB: 심방세동, AFLT: 심방조동, SVPB: 상심실성 조기수축, 
    # PVC: 심실성 조기수축, SVTA: 상심실성 빈맥, VTA: 심실성 빈맥
    arrhythmia_classes = ["AFIB", "AFLT", "SVPB", "PVC", "SVTA", "VTA"]
    
    labels = []
    for codes in database["scp_codes"]:
        active = set()
        for code in codes:
            if code in arrhythmia_classes:
                active.add(code)
        # 6개 클래스에 대한 멀티 레이블 벡터 생성
        labels.append([1.0 if name in active else 0.0 for name in arrhythmia_classes])

    database = database.copy()
    database["labels"] = labels
    database["num_positive_labels"] = [int(sum(label_row)) for label_row in labels]
    
    # 해당 부정맥 중 하나라도 가진 데이터만 필터링
    database = database[database["num_positive_labels"] > 0].reset_index()
    return database, arrhythmia_classes


class PTBXLMultilabelDataset(Dataset):
    def __init__(self, frame, dataset_dir, sample_rate, duration_seconds, normalization, augmentation=None):
        self.frame = frame.reset_index(drop=True)
        self.dataset_dir = Path(dataset_dir)
        self.sample_rate = int(sample_rate)
        self.duration_seconds = int(duration_seconds)
        self.normalization = normalization
        self.augmentation = augmentation
        self.signal_length = self.sample_rate * self.duration_seconds
        self.record_column = "filename_lr" if self.sample_rate == 100 else "filename_hr"
        self.class_names = ["NORM", "MI", "STTC", "CD", "HYP"]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        record_path = self.dataset_dir / row[self.record_column]
        signal, _ = wfdb.rdsamp(str(record_path))
        signal = signal.T.astype(np.float32)
        signal = signal[:, : self.signal_length]
        if signal.shape[1] < self.signal_length:
            pad_width = self.signal_length - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad_width)), mode="constant")
        signal = normalize_signal(signal, self.normalization)
        if self.augmentation is not None:
            signal = self.augmentation(signal)
        labels = np.asarray(row["labels"], dtype=np.float32)
        return {
            "signal": torch.from_numpy(signal),
            "target": torch.from_numpy(labels),
            "ecg_id": int(row["ecg_id"]),
        }


def compute_multilabel_sample_weights(frame, class_names):
    labels = np.stack(frame["labels"].to_numpy()).astype(np.float32)
    class_frequency = labels.sum(axis=0)
    inverse = np.where(class_frequency > 0, len(frame) / class_frequency, 0.0)
    weights = []
    for row in labels:
        active = row > 0
        if active.any():
            weights.append(float(np.max(inverse[active])))
        else:
            weights.append(1.0)
    return np.asarray(weights, dtype=np.float64), class_frequency


def create_weighted_sampler(frame, class_names):
    sample_weights, _ = compute_multilabel_sample_weights(frame, class_names)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


AAMI_MAP = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "V": "V",
    "E": "V",
    "F": "F",
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}


def extract_mitbih_beats(dataset_dir, record_ids, left, right, max_beats_per_record=None):
    dataset_dir = Path(dataset_dir)
    beats = []
    labels = []
    beat_ids = []
    for record_id in record_ids:
        record = wfdb.rdrecord(str(dataset_dir / record_id))
        annotation = wfdb.rdann(str(dataset_dir / record_id), "atr")
        signal = record.p_signal.T.astype(np.float32)
        record_count = 0
        for idx, symbol in enumerate(annotation.symbol):
            mapped = AAMI_MAP.get(symbol)
            if mapped is None:
                continue
            center = int(annotation.sample[idx])
            start = center - left
            end = center + right
            if start < 0 or end > signal.shape[1]:
                continue
            beat = signal[:, start:end]
            beats.append(beat)
            labels.append(mapped)
            beat_ids.append(f"{record_id}:{center}")
            record_count += 1
            if max_beats_per_record is not None and record_count >= max_beats_per_record:
                break
    return beats, labels, beat_ids


class MITBIHBeatDataset(Dataset):
    def __init__(self, beats, labels, beat_ids, class_names, normalization="zscore_per_lead", augmentation=None):
        self.beats = beats
        self.labels = labels
        self.beat_ids = beat_ids
        self.class_names = class_names
        self.class_to_index = {name: idx for idx, name in enumerate(class_names)}
        self.normalization = normalization
        self.augmentation = augmentation

    def __len__(self):
        return len(self.beats)

    def __getitem__(self, index):
        beat = normalize_signal(self.beats[index], self.normalization)
        if self.augmentation is not None:
            beat = self.augmentation(beat)
        label_idx = self.class_to_index[self.labels[index]]
        return {
            "signal": torch.from_numpy(beat),
            "target": torch.tensor(label_idx, dtype=torch.long),
            "beat_id": self.beat_ids[index],
        }


def class_counts_from_labels(labels):
    return Counter(labels)
