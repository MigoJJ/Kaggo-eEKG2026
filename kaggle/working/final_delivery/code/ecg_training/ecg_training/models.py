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

    def forward(self, x, return_sequence=False):
        x = self.stem(x)
        x = self.layers(x)
        if return_sequence:
            # (batch, channels, length) -> (batch, length, channels)
            return x.transpose(1, 2)
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

    def forward(self, signal, return_sequence=False):
        batch_size, leads, length = signal.shape
        if return_sequence:
            # 시퀀스 정보를 유지하며 인코딩
            # (batch * leads, 1, length) -> (batch * leads, seq_len, feat_dim)
            seq = self.lead_encoder(signal.reshape(batch_size * leads, 1, length), return_sequence=True)
            feat_dim = seq.shape[-1]
            # (batch, leads, seq_len, feat_dim) -> (batch, seq_len, leads * feat_dim) 또는 적절한 차원 축소
            # 여기서는 리드별 시퀀스를 평균하여 전체적인 시간 패턴을 유지합니다.
            seq = seq.reshape(batch_size, leads, -1, feat_dim)
            return seq.mean(dim=1) # (batch, seq_len, feat_dim)

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
    부정맥 정밀 진단을 위한 하이브리드 모델.
    1. Signal Backbone: 전체적인 리듬 특징 추출 (seq_len, feat_dim)
    2. Beat Features: 심박별 분류 결과(N, S, V, F, Q counts/probs)를 입력받아 융합
    """
    def __init__(self, backbone, feat_dim=256, num_beat_classes=5, num_arrhythmia_classes=3, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        
        # LSTM for rhythm pattern
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=128, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        
        # Beat feature projection (counts or aggregated probs of beats)
        self.beat_proj = nn.Sequential(
            nn.Linear(num_beat_classes, 64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.final_head = nn.Sequential(
            nn.Linear(128 * 2 + 64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_arrhythmia_classes)
        )

    def forward(self, signal, beat_features=None):
        # signal: (batch, leads, length)
        # beat_features: (batch, num_beat_classes) - e.g. counts or mean probs of beats in the signal
        
        # 1. Rhythm features from sequence
        seq_features = self.backbone(signal, return_sequence=True)
        lstm_out, _ = self.lstm(seq_features)
        rhythm_feat = lstm_out[:, -1, :] # (batch, 128 * 2)
        
        # 2. Beat features fusion
        if beat_features is None:
            # Placeholder if not provided
            beat_features = torch.zeros(signal.shape[0], 5).to(signal.device)
            
        beat_feat = self.beat_proj(beat_features)
        
        # 3. Combined classification
        combined = torch.cat([rhythm_feat, beat_feat], dim=1)
        logits = self.final_head(combined)
        return logits


class MITBIHBeatClassifier(nn.Module):
    def __init__(self, input_leads, num_classes, embedding_dim, blocks, base_channels, dropout):
        super().__init__()
        self.backbone = ECGBackbone(input_leads, embedding_dim, blocks, base_channels, dropout)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, signal):
        features = self.backbone(signal)
        return self.head(features)
