# Stage3 CNN 샘플링 전략 실험 (2026-08-29)

로컬 CPU(migojj-VM), MIT-BIH DS1(train)/DS2(test), `model-type=cnn`, `epochs=20`, `batch-size=256`, `lr=1e-3` 고정.
목적: 재학습 계획([[ecg-project-roadmap]] 우선순위 #1)의 Phase 1 — Kaggle P100에 ResNet을 태우기 전, 클래스 불균형 보정 전략을 저렴하게 먼저 검증.

## 실행 결과 요약

| 실행 | 전략 | 전체 정확도 | N sens | S sens | V sens | F sens | Q sens | macro sens* |
|---|---|---|---|---|---|---|---|---|
| v1 | `--balanced-sampler` (완전 역수 1/count) | 79.59% | 82.3% | 27.2% | 81.3% | 4.4% | 0.0% | 39.0% |
| v2 | `--balanced-sampler` 수정 (제곱근 역수 1/sqrt(count)) | 83.59% | 86.8% | 21.6% | 84.8% | 0.8% | 0.0% | 38.8% |
| v3 | 샘플러 없음, `CrossEntropyLoss(weight=sqrt역수)` | 진행중 (epoch1 88.07%) | — | — | — | — | — | — |

*macro sens = 5개 클래스 sensitivity 단순평균 (전체 정확도보다 임상적으로 중요 — N/V 다수 클래스에 휘둘리지 않음)

## 핵심 발견

1. **완전 역수 → 제곱근 역수 샘플러 전환은 실패**: 전체 정확도는 79.59%→83.59%로 올랐지만, 이는 N/V(다수 클래스)가 더 좋아졌기 때문. **S sensitivity는 오히려 악화**(27.2%→21.6%), **F sensitivity는 사실상 소멸**(4.4%→0.8%). macro sensitivity는 39.0%→38.8%로 **개선 없음**.
   - 원인: 샘플러를 부드럽게 하면서 소수 클래스(S/F) 노출 빈도가 줄어, 모델이 그 패턴을 덜 학습.
   - 교훈: **전체 정확도만으로 판단하면 안 됨** — 항상 macro sensitivity와 클래스별 breakdown을 함께 봐야 함.

2. **Q 클래스(train 8개 샘플)는 어떤 샘플러/손실함수 조정으로도 해결 불가** — 데이터 부족 문제. ARCHITECTURE.md P4(Explainable) 원칙에 따라 "알려진 제한사항"으로 리포트에 명시하는 방향이 맞음. 재학습으로 고칠 대상이 아님.

3. **v3(재샘플링 없이 손실함수 가중치만)이 유망한 신호**: epoch 1부터 88.07% — v1(60.0%)·v2(75.1%)보다 훨씬 빠르고 안정적인 수렴. 재샘플링이 유발하는 "같은 샘플 반복 노출로 인한 왜곡"이 없어서로 추정. **단, 최종 혼동행렬(S/F sensitivity, macro 평균)로 검증 전까지는 결론 보류.**

## 코드 변경 (커밋 예정)

`ml/train_stage3_beat.py` — `--balanced-sampler` 옵션의 샘플 가중치를 완전 역수(`1.0/count`)에서 제곱근 역수(`1.0/sqrt(count)`)로 변경. (참고: 이 변경 자체는 v2 결과에서 보듯 macro sensitivity를 개선하지 못했음 — 향후 세션에서 되돌리거나 focal loss로 교체 검토 필요.)

## 다음 단계 (재개 시 여기서 시작)

1. v3(20 epoch) 완료까지 지켜보고 최종 혼동행렬 확인 — S/F sensitivity, macro sensitivity가 v1/v2보다 실제로 나은지 검증.
2. v3도 macro sensitivity 개선이 없다면 → **focal loss 도입** 검토 (코드 변경 필요, gamma 하이퍼파라미터 튜닝).
3. 로컬 CPU CNN 실험이 만족스러운 전략(샘플러 or 손실함수 or focal loss)을 찾으면, 그 전략을 그대로 Kaggle P100 + ResNet(Phase 2)에 적용.
4. Q 클래스는 학습 목표에서 제외하고 "알려진 제한사항"으로 문서화 — 4-클래스(N/S/V/F) 관점 macro sensitivity를 주 지표로 전환하는 것도 고려.
5. Kaggle 측: `/kaggle/input/datasets/migojjkoh/mit-bih-arrhythmia-database-1-0-0/mit-bih-arrhythmia-database-1.0.0`에 데이터 마운트 확인됨, `wfdb` 패키지만 네트워크 문제로 미설치 상태 — Kaggle Docker 이미지에 사전 포함된 버전이 있는지, 또는 offline wheel 방법 재시도 필요.
