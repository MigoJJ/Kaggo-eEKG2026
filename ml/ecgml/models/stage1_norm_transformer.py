"""1단계(정상 사전선별) CNN-Transformer 분류기.

로컬 CPU(GPU 미지원)에서는 Stage1NormClassifier(소형 CNN)로 검증했고,
GPU(Kaggle P100 등) 확보 시 이 모듈로 학습한다. 시계열의 장거리 의존성을
Self-Attention으로 포착하기 위해 아키텍처 문서가 원래 상정한 "CNN-Transformer"를 구현한다.

입출력 인터페이스는 Stage1NormClassifier와 동일하다 — forward(x: (batch,12,5000)) -> logit (batch,).
train_stage1.py에서 --model-type transformer로 교체 가능하다.

구조:
  1) Conv1d 스템 4단(stride 2씩, 5000 -> 313 프레임)으로 국소 형태학(QRS 등) 특징을 d_model
     채널로 투영.
  2) [CLS] 토큰 + 학습형 위치 임베딩을 붙여 TransformerEncoder(N층, Pre-LN, GELU)에 통과.
  3) [CLS] 임베딩에 LayerNorm + Linear로 이진분류 logit 산출.
"""

import torch
import torch.nn as nn


class ConvStem(nn.Module):
    """국소 파형(QRS 등) 특징 추출 후 Transformer 입력 차원(d_model)으로 투영."""

    def __init__(self, n_leads: int, d_model: int):
        super().__init__()
        c1, c2 = d_model // 4, d_model // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_leads, c1, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.Conv1d(c1, c2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            nn.Conv1d(c2, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (batch, d_model, seq_len)


class Stage1NormTransformer(nn.Module):
    def __init__(
        self,
        n_leads: int = 12,
        n_samples: int = 5000,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = ConvStem(n_leads, d_model)
        seq_len = self._stem_output_length(n_samples)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    @staticmethod
    def _stem_output_length(n_samples: int) -> int:
        length = n_samples
        for kernel, stride, pad in ((7, 2, 3), (7, 2, 3), (5, 2, 2), (5, 2, 2)):
            length = (length + 2 * pad - kernel) // stride + 1
        return length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stem(x).transpose(1, 2)  # (batch, seq_len, d_model)

        batch_size = features.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, features], dim=1)
        tokens = tokens + self.pos_embedding[:, : tokens.size(1), :]

        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0, :]
        return self.head(cls_out).squeeze(-1)
