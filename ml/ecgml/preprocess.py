"""모델 입력 직전 정규화. Java EcgPreprocessor(0단계)는 mV 물리단위를 그대로 유지하므로
(rule-engine의 절대 임계치를 위해), Z-정규화는 여기서 별도로 적용한다.
Java `inference` 모듈의 ZNormalizer가 서빙 시 동일한 공식을 적용해야 train/serve가 일치한다.
"""

import numpy as np

STD_FLOOR = 1e-9


def znormalize_per_lead(x: np.ndarray) -> np.ndarray:
    """x: (n_leads, n_samples) mV 신호 -> 리드별 평균0/표준편차1 정규화."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.where(std < STD_FLOOR, 1.0, std)
    return (x - mean) / std
