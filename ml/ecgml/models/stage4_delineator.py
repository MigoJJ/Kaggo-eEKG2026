"""Stage4 helper model: lightweight 12-lead P/QRS/T delineator."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv1d -> BatchNorm1d -> ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Stage4Delineator(nn.Module):
    def __init__(self, n_leads: int = 12, n_classes: int = 4):
        super().__init__()
        # 내부적으로 각 리드를 독립된 배치 아이템으로 취급하여 학습하므로, U-Net 입력 채널 수는 1개입니다.
        self.enc1 = DoubleConv(1, 32)
        self.pool1 = nn.MaxPool1d(2)  # 5000 -> 2500
        
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool1d(2)  # 2500 -> 1250
        
        # Bottleneck
        self.bottleneck = DoubleConv(64, 128)
        
        # Expanding Path (Decoder)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")  # 1250 -> 2500
        self.dec2 = DoubleConv(128 + 64, 64)
        
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")  # 2500 -> 5000
        self.dec1 = DoubleConv(64 + 32, 32)
        
        self.final_conv = nn.Conv1d(32, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_leads, seq_len) -> (batch, 12, 5000)
        batch_size, n_leads, seq_len = x.size()
        
        # 리드 차원을 배치 차원으로 변환하여 독립된 채널 분석 수행
        x_reshaped = x.view(batch_size * n_leads, 1, seq_len)  # (batch * 12, 1, 5000)
        
        # Encoder
        x1 = self.enc1(x_reshaped)
        p1 = self.pool1(x1)
        
        x2 = self.enc2(p1)
        p2 = self.pool2(x2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder with Skip Connections (Standard size-guaranteed concatenation)
        u2 = self.up2(b)
        concat2 = torch.cat([u2, x2], dim=1)
        d2 = self.dec2(concat2)
        
        u1 = self.up1(d2)
        concat1 = torch.cat([u1, x1], dim=1)
        d1 = self.dec1(concat1)
        
        out = self.final_conv(d1)  # (batch * 12, n_classes, seq_len) -> (batch * 12, 4, 5000)
        
        # 출력 차원을 원래의 리드별 구조인 (batch, n_classes, n_leads, seq_len)로 복원
        return out.view(batch_size, n_leads, 4, seq_len).transpose(1, 2)  # (batch, 4, 12, 5000)
