import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        return self.act(x)


class LeadEncoder1D(nn.Module):
    def __init__(self, embedding_dim=256, blocks=(2, 2, 2, 2), base_channels=32, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )
        channels = base_channels
        layers = []
        for stage_idx, depth in enumerate(blocks):
            out_channels = base_channels * (2 ** stage_idx)
            for block_idx in range(depth):
                stride = 2 if block_idx == 0 and stage_idx > 0 else 1
                layers.append(ResidualBlock1D(channels, out_channels, stride=stride, dropout=dropout))
                channels = out_channels
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(channels, embedding_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)


class LeadAttentionPooling(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.score = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(x * weights, dim=1)


class ECGBackbone(nn.Module):
    def __init__(self, input_leads, embedding_dim, blocks, base_channels, dropout):
        super().__init__()
        self.input_leads = int(input_leads)
        self.lead_encoder = LeadEncoder1D(
            embedding_dim=embedding_dim,
            blocks=tuple(blocks),
            base_channels=base_channels,
            dropout=dropout,
        )
        self.attention = LeadAttentionPooling(embedding_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, signal):
        batch_size, leads, length = signal.shape
        encoded = self.lead_encoder(signal.reshape(batch_size * leads, 1, length))
        encoded = encoded.reshape(batch_size, leads, -1)
        pooled_mean = encoded.mean(dim=1)
        pooled_attn = self.attention(encoded)
        return self.fusion(torch.cat([pooled_mean, pooled_attn], dim=1))


class PTBXLClassifier(nn.Module):
    def __init__(self, input_leads, num_classes, embedding_dim, blocks, base_channels, dropout):
        super().__init__()
        self.backbone = ECGBackbone(input_leads, embedding_dim, blocks, base_channels, dropout)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, signal):
        features = self.backbone(signal)
        return self.head(features)


class ArrhythmiaSpecialist(nn.Module):
    """
    부정맥(AFib, PVC 등) 정밀 진단을 위한 하이브리드 모델.
    기존 CNN 특징 추출기에 RNN(LSTM)을 결합하여 시간적 불규칙성을 포착합니다.
    """
    def __init__(self, backbone, embedding_dim=256, num_arrhythmia_classes=6, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        
        # 시간적 패턴 분석을 위한 LSTM (AFib 등 불규칙한 RR 간격 탐지용)
        # 특징 벡터를 시퀀스로 해석하거나 백본의 중간 출력을 시퀀스로 받을 수 있습니다.
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=128, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        
        self.specialized_head = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_arrhythmia_classes)
        )

    def forward(self, signal):
        # signal: (batch, leads, length)
        features = self.backbone(signal) # (batch, embedding_dim)
        
        # 특징 벡터를 시퀀스 데이터로 변환 (추후 백본 수정으로 시퀀스 출력을 직접 받을 수 있음)
        x = features.unsqueeze(1) # (batch, 1, embedding_dim)
        lstm_out, _ = self.lstm(x)
        logits = self.specialized_head(lstm_out[:, -1, :])
        return logits


class MITBIHBeatClassifier(nn.Module):
    def __init__(self, input_leads, num_classes, embedding_dim, blocks, base_channels, dropout):
        super().__init__()
        self.backbone = ECGBackbone(input_leads, embedding_dim, blocks, base_channels, dropout)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, signal):
        features = self.backbone(signal)
        return self.head(features)
