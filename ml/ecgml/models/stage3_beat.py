"""Stage3(비트 부정맥 검증) — MIT-BIH AAMI 5-클래스(N/S/V/F/Q) 단일비트 분류기.

기본 ``BeatClassifier``는 기존 체크포인트 호환을 위해 소형 1D-CNN으로 유지한다.
시계열 형태학을 더 강하게 학습할 때는 ``resnet`` 또는 ``inception`` 백본을 선택할 수 있다.
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


class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.main(x) + self.shortcut(x))


class BeatResNetClassifier(nn.Module):
    """짧은 비트 윈도우용 1D-ResNet.

    Residual connection으로 QRS 주변의 급격한 국소 패턴과 전후 문맥을 깊게 누적한다.
    입력/출력 계약은 ``BeatClassifier``와 동일하다: ``(batch, 1, 200) -> (batch, n_classes)``.
    """

    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock1d(32, 32),
            ResidualBlock1d(32, 64, stride=2),
            ResidualBlock1d(64, 64),
            ResidualBlock1d(64, 128, stride=2),
            ResidualBlock1d(128, 128),
            ResidualBlock1d(128, 192, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(192, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(self.stem(x)))


class InceptionBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_channels = out_channels // 4
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.branch3 = self._conv_branch(in_channels, branch_channels, kernel_size=3)
        self.branch7 = self._conv_branch(in_channels, branch_channels, kernel_size=7)
        self.branch15 = self._conv_branch(in_channels, branch_channels, kernel_size=15)
        self.project = nn.Sequential(
            nn.Conv1d(branch_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _conv_branch(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [self.branch1(x), self.branch3(x), self.branch7(x), self.branch15(x)]
        return self.project(torch.cat(branches, dim=1))


class BeatInceptionClassifier(nn.Module):
    """멀티스케일 1D-Inception 비트 분류기.

    3/7/15 샘플 커널을 병렬로 사용해 좁은 QRS, 넓은 QRS, 완만한 ST/T 변화를 함께 포착한다.
    """

    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            InceptionBlock1d(1, 64),
            nn.MaxPool1d(2),
            InceptionBlock1d(64, 128),
            nn.MaxPool1d(2),
            InceptionBlock1d(128, 192),
            nn.MaxPool1d(2),
            InceptionBlock1d(192, 256),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_beat_classifier(model_type: str = "cnn", n_classes: int = 5) -> nn.Module:
    if model_type == "cnn":
        return BeatClassifier(n_classes=n_classes)
    if model_type == "resnet":
        return BeatResNetClassifier(n_classes=n_classes)
    if model_type == "inception":
        return BeatInceptionClassifier(n_classes=n_classes)
    raise ValueError(f"알 수 없는 model_type: {model_type}")
