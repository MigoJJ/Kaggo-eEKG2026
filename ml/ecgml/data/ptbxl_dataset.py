"""Java PtbxlPreprocessExporter가 만든 .f32 산출물(Stage-0 전처리 완료, mV)을 읽어
NORM 이진분류 학습셋을 구성한다. 라벨은 PTB-XL 공식 scp_codes에서 NORM 우도 100.0을 기준으로 한다.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ecgml.preprocess import znormalize_per_lead

N_LEADS = 12
N_SAMPLES = 5000


def load_labels(ptbxl_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(ptbxl_csv, index_col="ecg_id")

    def is_norm(scp_codes: str) -> bool:
        codes = ast.literal_eval(scp_codes)
        return codes.get("NORM", 0.0) >= 100.0

    df["is_norm"] = df["scp_codes"].apply(is_norm)
    return df[["is_norm", "strat_fold"]]


class PtbxlNormDataset(Dataset):
    def __init__(self, export_dir: Path, labels: pd.DataFrame, ecg_ids: list[int]):
        self.export_dir = Path(export_dir)
        self.labels = labels
        self.ecg_ids = ecg_ids

    def __len__(self) -> int:
        return len(self.ecg_ids)

    def __getitem__(self, idx: int):
        ecg_id = self.ecg_ids[idx]
        path = self.export_dir / f"{ecg_id}.f32"
        arr = np.fromfile(path, dtype="<f4").reshape(N_LEADS, N_SAMPLES)
        arr = znormalize_per_lead(arr)
        label = float(self.labels.loc[ecg_id, "is_norm"])
        return torch.from_numpy(arr.copy()).float(), torch.tensor(label, dtype=torch.float32)
