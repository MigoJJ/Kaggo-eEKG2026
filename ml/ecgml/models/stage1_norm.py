"""1단계(정상 사전선별) 분류기.

아키텍처 문서는 "CNN-Transformer"를 상정했으나, 이 개발 PC는 GPU가 없는 CPU 전용 환경
(Intel HD Graphics 530, CUDA 미지원)이라 실제 학습 가능한 범위로 축소한 소형 1D-CNN을 사용한다.
Transformer 인코더로의 업그레이드는 GPU 확보 후의 정밀화 과제로 남겨둔다.
"""

import torch
import torch.nn as nn


class Stage1NormClassifier(nn.Module):
    def __init__(self, n_leads: int = 12):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_leads, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x).squeeze(-1)  # logit (sigmoid는 손실함수/추론 시 적용)
