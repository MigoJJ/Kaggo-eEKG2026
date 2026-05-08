import ast
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
import scipy.io
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


def get_labels_from_hea(hea_path):
    """Extracts labels from a PhysioNet .hea file."""
    labels = []
    if not os.path.exists(hea_path):
        return labels
    with open(hea_path, 'r') as f:
        for line in f:
            if line.startswith('#Dx:'):
                labels = line.split(': ')[1].strip().split(',')
                break
    return [l.strip() for l in labels]


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


class UnifiedPhysioNetDataset(Dataset):
    """
    A unified dataset for PhysioNet format datasets (.mat + .hea).
    Supports multi-label classification using SNOMED-CT codes.
    """
    def __init__(self, root_dirs, label_mapping, sample_rate=500, duration_seconds=10, 
                 normalization="zscore_per_lead", augmentation=None):
        self.records = []
        self.label_mapping = label_mapping
        self.class_names = sorted(list(set(label_mapping.values())))
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.signal_length = sample_rate * duration_seconds
        self.normalization = normalization
        self.augmentation = augmentation

        for root in root_dirs:
            root = Path(root)
            # Find all .hea files recursively
            hea_files = list(root.glob("**/*.hea"))
            for hea_path in hea_files:
                labels = get_labels_from_hea(hea_path)
                # Map labels to our target classes
                mapped_labels = set()
                for l in labels:
                    if l in label_mapping:
                        mapped_labels.add(label_mapping[l])
                
                if mapped_labels:
                    self.records.append({
                        "path": hea_path.with_suffix(""),
                        "labels": list(mapped_labels)
                    })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        record_path = str(record["path"])
        
        try:
            # Use scipy.io.loadmat as it's often faster for .mat files in these datasets
            data = scipy.io.loadmat(record_path + ".mat")
            signal = data.get("val", data.get("data", None))
            if signal is None:
                # Fallback to wfdb
                signal, _ = wfdb.rdsamp(record_path)
                signal = signal.T
            signal = signal.astype(np.float32)
        except Exception:
            signal = np.zeros((12, self.signal_length), dtype=np.float32)

        # Truncate or Pad
        if signal.shape[1] > self.signal_length:
            signal = signal[:, :self.signal_length]
        elif signal.shape[1] < self.signal_length:
            pad_width = self.signal_length - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad_width)), mode="constant")

        signal = normalize_signal(signal, self.normalization)
        if self.augmentation is not None:
            signal = self.augmentation(signal)

        # Multi-label vector
        label_vector = np.zeros(len(self.class_names), dtype=np.float32)
        for l in record["labels"]:
            label_vector[self.class_to_idx[l]] = 1.0

        return {
            "signal": torch.from_numpy(signal),
            "target": torch.from_numpy(label_vector),
            "path": record_path
        }


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
    # AFIB: 심방세동, AFLT: 심방조동, PVC: 심실성 조기수축
    # (SVPB, SVTA, VTA 등은 PTB-XL 데이터셋에 충분한 양성 샘플이 없어 제외)
    arrhythmia_classes = ["AFIB", "AFLT", "PVC"]
    
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


def load_chapman_metadata(dataset_dir):
    """
    Chapman (Shaoxing) dataset loader.
    Assumes a CSV with filename and rhythm labels.
    """
    dataset_dir = Path(dataset_dir)
    # This is a placeholder for the actual Chapman metadata format
    # Typically it has 'FileName' and 'Diagnosis' or 'Rhythm'
    csv_path = dataset_dir / "chapman_metadata.csv"
    if not csv_path.exists():
        return None, []
    
    df = pd.read_csv(csv_path)
    rhythm_classes = sorted(df["Rhythm"].unique().tolist())
    
    # Simple label encoding for now
    class_to_idx = {name: i for i, name in enumerate(rhythm_classes)}
    df["label"] = df["Rhythm"].map(class_to_idx)
    
    return df, rhythm_classes


class ChapmanRhythmDataset(Dataset):
    def __init__(self, frame, dataset_dir, sample_rate=500, duration_seconds=10, normalization="zscore_per_lead", augmentation=None):
        self.frame = frame.reset_index(drop=True)
        self.dataset_dir = Path(dataset_dir)
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.normalization = normalization
        self.augmentation = augmentation
        self.signal_length = self.sample_rate * self.duration_seconds

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        record_path = self.dataset_dir / row["FileName"]
        
        # Chapman/Shaoxing often uses .mat files (500Hz)
        try:
            import scipy.io
            data = scipy.io.loadmat(str(record_path))
            # The signal key might vary, common is 'val'
            signal = data.get("val", data.get("data", None))
            if signal is None:
                raise ValueError(f"No signal data found in {record_path}")
            signal = signal.astype(np.float32)
        except Exception as e:
            # Placeholder/Empty if load fails
            signal = np.zeros((12, self.signal_length), dtype=np.float32)
            
        signal = signal[:, :self.signal_length]
        if signal.shape[1] < self.signal_length:
            pad_width = self.signal_length - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad_width)), mode="constant")
            
        signal = normalize_signal(signal, self.normalization)
        if self.augmentation is not None:
            signal = self.augmentation(signal)
            
        return {
            "signal": torch.from_numpy(signal),
            "target": torch.tensor(row["label"], dtype=torch.long),
            "filename": row["FileName"]
        }


class CPSC2018Dataset(Dataset):
    def __init__(self, frame, dataset_dir, sample_rate=500, duration_seconds=10, normalization="zscore_per_lead", augmentation=None):
        self.frame = frame.reset_index(drop=True)
        self.dataset_dir = Path(dataset_dir)
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.normalization = normalization
        self.augmentation = augmentation
        self.signal_length = self.sample_rate * self.duration_seconds

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        record_path = self.dataset_dir / row["FileName"]
        
        # CPSC 2018 uses .mat files, often at 500Hz
        try:
            import scipy.io
            data = scipy.io.loadmat(str(record_path))
            signal = data["val"].astype(np.float32)
        except Exception:
            signal = np.zeros((12, self.signal_length), dtype=np.float32)
            
        signal = signal[:, :self.signal_length]
        if signal.shape[1] < self.signal_length:
            pad_width = self.signal_length - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad_width)), mode="constant")
            
        signal = normalize_signal(signal, self.normalization)
        if self.augmentation is not None:
            signal = self.augmentation(signal)
            
        return {
            "signal": torch.from_numpy(signal),
            "target": torch.tensor(row["label"], dtype=torch.long),
            "filename": row["FileName"]
        }


def get_unified_6class_mapper():
    """
    Returns a unified mapping for AFIB, AFLT, SVPB, PVC, SVTA, VTA.
    Uses SNOMED-CT or standard abbreviations.
    """
    return {
        "AFIB": 0, "AF": 0, "164889003": 0,
        "AFLT": 1, "AFL": 1, "164890007": 1,
        "SVPB": 2, "PAC": 2, "284470004": 2,
        "PVC": 3, "VPB": 3, "427172004": 3,
        "SVTA": 4, "SVT": 4, "426761007": 4,
        "VTA": 5, "VT": 5, "427084000": 5
    }
