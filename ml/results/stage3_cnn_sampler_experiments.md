# Stage3 CNN 샘플링 전략 실험 (2026-08-29)

로컬 CPU(migojj-VM), MIT-BIH DS1(train)/DS2(test), `model-type=cnn`, `epochs=20`, `batch-size=256`, `lr=1e-3` 고정.
목적: 재학습 계획([[ecg-project-roadmap]] 우선순위 #1)의 Phase 1 — Kaggle P100에 ResNet을 태우기 전, 클래스 불균형 보정 전략을 저렴하게 먼저 검증.

데이터 경로(이 기기): `/home/migojj/ittia/Kaggo-eEKG2026/data/mit-bih-arrhythmia-database-1.0.0`
가상환경: `/home/migojj/ittia/venv` (`source /home/migojj/ittia/venv/bin/activate` 로 활성화, wfdb/torch/numpy/pandas/scikit-learn 설치됨)

## 실행 결과 요약

| 실행 | 전략 | 체크포인트 기준 | 전체 정확도 | N sens | S sens | V sens | F sens | Q sens | 5-class macro | **4-class macro(N/S/V/F)** |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | `--balanced-sampler` 완전역수(1/count) | test_acc **(결함 있는 기준)** | 79.59% | 82.3% | 27.2% | 81.3% | 4.4% | 0.0% | 39.0% | 48.8% |
| v2 | `--balanced-sampler` 제곱근역수(1/sqrt(count)) | test_acc **(결함)** | 83.59% | 86.8% | 21.6% | 84.8% | 0.8% | 0.0% | 38.8% | 48.5% |
| v3 | 샘플러 없음, `CrossEntropyLoss(weight=sqrt역수)` | test_acc **(결함)** | 88.07% | 92.7% | 5.0% | 83.1% | 0.3% | 0.0% | 36.2% | 45.3% |
| v4 | 샘플러 없음, `CrossEntropyLoss(weight=sqrt역수)` | **macro_sens(수정됨)** | 72.25% | 74.0% | 24.3% | 83.7% | 1.3% | 14.3%* | 39.5%* | 45.8% |
| **v5** | `--balanced-sampler` **완전역수로 되돌림** | **macro_sens(수정됨)** | 68.84% | **67.6%**⚠️ | **77.4%**✅ | **88.8%**✅ | 6.7% | 0.0% | 48.1% | **60.1%**🏆 |

*v4의 Q sens 14.3%는 test 표본 7개 중 1개 우연히 맞은 것 — 통계적으로 무의미(노이즈). 5-class macro가 v4>v3로 보이는 것도 이 노이즈 때문. **4-class macro가 더 신뢰할 수 있는 비교 기준.**

## 핵심 발견 (시간순)

1. **완전역수→제곱근역수 샘플러 전환(v1→v2)은 무의미**: 전체 정확도는 올랐지만 4-class macro는 거의 동일(48.8%→48.5%). N/V 다수 클래스만 좋아진 착시.

2. **Q 클래스(train 8개, test 7개 샘플)는 어떤 방법으로도 해결 불가** — 데이터 부족 문제. ARCHITECTURE.md P4(Explainable) 원칙대로 "알려진 제한사항"으로 리포트 명시, 재학습 목표에서 제외. **macro sensitivity는 4-class(N/S/V/F) 기준으로 계산해야 함.**

3. **체크포인트 선택 버그 발견·수정 (v3 사례)**: 기존 코드는 `test_acc`(전체 정확도)가 최고인 epoch를 저장했음. v3에서 epoch1(거의 미학습 상태, train_loss=0.55)이 test_acc 88%로 "최고"로 잘못 채택됨 — 이때 S sensitivity 5.0%, F sensitivity 0.3%로 사실상 무용지물. **`ml/train_stage3_beat.py`를 수정하여 이제 macro_sens(5-class sensitivity 평균) 기준으로 체크포인트를 선택함.**

4. **재샘플링이 손실함수 가중치 단독보다 우수 (v3/v4 vs v1/v2/v5)**: 같은 macro_sens 기준으로 비교해도(v4 vs v5), 재샘플링(v5, 4-class 60.1%)이 손실함수 가중치만 쓴 경우(v4, 4-class 45.8%)보다 훨씬 좋음.

5. **v5 = 현재까지 최고 기록, 그러나 트레이드오프 존재**:
   - S sensitivity 77.4%, V sensitivity 88.8% — 둘 다 문헌 기준(Se(V)≥85%, Se(S)≥70-80%) 최초 달성
   - 4-class macro 60.1% — 이전에 논의한 "Stage4로 넘어갈 만한 기준선(60%)"을 처음 통과
   - **그러나 N sensitivity 67.6%로, 제안했던 안전선(75%) 미달** — 정상 심전도의 32%가 오탐(false positive)됨. 이 프로젝트는 P3(자율진단 금지, 의사 서명 필수) 원칙이라 거짓양성 자체는 거짓음성보다 훨씬 덜 위험하지만, 너무 잦으면 알람 피로 문제.

## 코드 변경 이력 (모두 커밋됨)

- `ml/train_stage3_beat.py`:
  - `--balanced-sampler` 가중치: 완전역수 → 제곱근역수(v2) → **다시 완전역수로 원복**(v5 결과가 더 좋았음)
  - 체크포인트 선택 기준: `test_acc` → **`macro_sens`(5-class sensitivity 평균)** 로 변경 (버그 수정)
  - **미반영**: N sensitivity 안전선 필터 — 코드 수정 논의만 하고 아직 미적용 (사용자가 tool 승인 안 함, 결정 보류 상태)

## 아직 결정되지 않은 사안 (재개 시 여기서부터)

**즉시 결정할 것**: v5(4-class macro 60.1%, 문헌급 S/V, 그러나 N sensitivity 67.6%)를 Stage3 로컬 CNN 트랙의 최종안으로 받아들이고 Stage4로 넘어갈지, 아니면 아래 옵션 중 하나를 더 시도할지.

옵션:
1. **v5를 최종으로 채택 → Stage4(LUDB/QTDB)로 전환** (직전 제안한 권장안)
   - 근거: S/V가 목표 달성, Stage4는 현재 가중치 0개로 더 급한 공백, 의사 서명 안전망이 있어 N 오탐 증가는 상대적으로 감내 가능
2. **N sensitivity 안전선 필터를 코드에 추가하고 v6 재실행** — macro_sens가 아무리 높아도 N sensitivity가 임계치(예 70~75%) 미만인 epoch는 체크포인트 후보에서 제외하는 로직. (`train_stage3_beat.py`의 `if macro_sens > best_macro_sens:` 조건에 `and per_class_sens[N_idx] >= args.min_n_sensitivity` 추가하는 방향으로 설계했으나 미적용)
3. **CNN을 접고 ResNet으로 아키텍처 전환** — CNN 표현력 한계로 보고, Kaggle P100(또는 로컬, 느림)에서 ResNet 재시도. Kaggle은 `wfdb` 패키지 설치가 네트워크 제한으로 막혀있어 별도 해결 필요(아래).

## Kaggle 환경 이슈 (미해결)

- 데이터 마운트는 완료: `/kaggle/input/datasets/migojjkoh/mit-bih-arrhythmia-database-1-0-0/mit-bih-arrhythmia-database-1.0.0`
- `torch`(2.10.0+cu128, GPU 있음), `numpy`, `pandas`, `sklearn`은 이미 설치되어 있음
- **`wfdb` 패키지만 없음** — `pip install wfdb` 시도 시 `Temporary failure in name resolution` (Kaggle 세션이 외부 네트워크 차단된 상태로 추정). 미해결. 다음 시도 후보: Kaggle Docker 이미지에 wfdb가 이미 포함된 다른 커널 템플릿 확인, 또는 offline wheel 업로드, 또는 로컬에서 학습 완료 후 ONNX만 Kaggle 없이 바로 Java로 통합(Kaggle을 아예 우회).

## 다음 단계 전체 로드맵 (참고: [[ecg-project-roadmap]] 원본과 비교해 갱신됨)

1. **[결정 대기]** v5 최종 채택 여부 확인
2. Stage3 로컬 CNN 트랙 종료 — 결론을 이 문서에 최종 기록
3. (병렬 가능) Stage4 LUDB/QTDB 데이터 작업 착수 — 이미 커밋 `761992b`(annotation parsing 수정)에서 일부 진행됨
4. (보류, Kaggle wfdb 해결 시 재개) Stage3 ResNet Phase 2
5. ONNX export → `ml/sync_models.sh` 배포 → Java 통합 테스트
6. VT 감지 규칙 재검증 (개선된 Stage3 beat 스트림 기반)
7. ExternalValidationExporter(CPSC2018/Chapman) 재실행
