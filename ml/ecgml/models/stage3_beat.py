"""Stage3(비트 부정맥 검증) — MIT-BIH AAMI 5-클래스(N/S/V/F/Q) 단일비트 분류기.

Phase 4 backbone: 짧은 비트 윈도우(±200ms)를 분류하는 소형 1D-CNN. 로컬 CPU에서 빠르게
학습 가능한 크기로 유지했다 — 정밀 튜닝(계층 수·정규화 등)은 Kaggle P100 학습 이후 과제.
"""

import torch
import torch.nn as nn


class BeatClassifier(nn.Module):
    def __init__(self, n_classes: int = 5):
        super().__init__()
        # 입력: (batch, 1, 200)
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # (batch, 32, 100)
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # (batch, 64, 50)
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # (batch, 128, 25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),     # (batch, 128 * 25 = 3200)
            nn.Linear(128 * 25, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))  # logits (batch, n_classes)
