Resume this session with:
claude --resume cb0897d4-bd73-491c-a6d1-8ce3abfbef41
To continue this session, run 
codex resume 019f6d0a-788e-75f0-adbc-7b05d869ceab
----------------------------

# EKG-GDS EMR 2026 — 다단계 심전도 진단 파이프라인 아키텍처

> 12-Lead ECG 자동 판독 + JavaFX EMR. 안전(safety-first)·전처리 선행(denoise-first)·응급 병렬(parallel-emergency) 원칙 기반.

---

## 1. 설계 원칙 (Design Principles)

| # | 원칙 | 의미 |
|---|------|------|
| P1 | **Denoise-first** | 디노이징/기선보정/신호품질(SQI)은 모든 판독의 최전방 공유 전처리(0단계). |
| P2 | **Triage, not gate** | 1단계는 모든 ECG를 예외 없이 E→2→3→4→서명 큐로 통과시키며, AI P(NORM) 점수는 의사 검토 우선순위 정렬용 메타데이터일 뿐 게이팅에 쓰지 않는다. **근거**: PTB-XL 실측에서 20개 요약 피처 기반 AND게이트(P(NORM)≥0.98 ∧ 전 피처 정상범위)로 자동확정을 시도했더니, 통과시킨 소수 레코드의 79%가 실제로는 비정상이었다 — 요약 통계 몇 개로는 "정상"을 안전하게 확정할 정보량이 부족함을 확인(2026-07 실험, ml/train_stage1_features.py). |
| P3 | **Safety-first (no autonomous Dx)** | "정상 자동 확정" 금지. 정상은 *사전선별*일 뿐, 최종 발행은 의사 서명 필수. |
| P4 | **Explainable** | 모든 판정은 근거(피처값·룰조건·모델확률)를 리포트에 동봉. 블랙박스 단독 판정 불가. |
| P5 | **Dataset-honest** | 각 데이터셋의 라벨 공간·리드 구성·샘플링레이트 차이를 존중. MIT-BIH로 전도장애(RBBB/LBBB) 교차검증 금지 → 부정맥 비트에만 사용. |
| P6 | **Auditable** | 모든 진단 결과는 모델 해시·버전·타임스탬프와 함께 감사로그 적재(재현성). |

---

## 2. 시스템 컨텍스트 (하이브리드)

```
┌─────────────────────────────┐        ┌──────────────────────────────┐
│   Python (PyTorch) — 학습     │  ONNX  │   Java / JavaFX — 런타임        │
│  · Stage1 NORM 분류기         │ ─────► │  · 신호 전처리 (0단계)          │
│  · Stage3 Beat 분류기         │ export │  · 룰엔진 (2단계, 응급)          │
│  · Stage4 ST 분할/디노이저     │  .onnx │  · 피처 추출 (20종)             │
│  (오프라인, /mnt/t7 데이터)   │        │  · ONNX Runtime 추론 (1/3/4)   │
└─────────────────────────────┘        │  · SQLite 저장 · 리포트 · UI    │
                                        └──────────────────────────────┘
```

- **학습(Python)**: `/mnt/t7/datasets`의 PTB-XL·MIT-BIH·challenge_2021을 배치 스트리밍하여 학습, `models/*.onnx`로 export.
- **런타임(Java)**: 학습된 `.onnx`를 ONNX Runtime(Java)으로 로드해 추론. 전처리·룰·피처는 순수 Java.
- 두 세계의 유일한 계약(contract) = **ONNX 모델 시그니처 + 전처리 규격(JSON)**. 전처리 규격을 양쪽이 공유해 train/serve skew 방지.

---

## 3. 진단 흐름 (수정 아키텍처)

```
              [Raw 12-Lead ECG 입력]
                       │
        ┌──────────────▼───────────────┐
        │ 0. 전처리 (공유 프론트엔드)    │  디노이징·기선보정·리샘플·SQI (mV 물리단위 유지)
        └──────────────┬───────────────┘
                       │  SQI 불량 → [판독불가 반려]
                       │
        ┌──────────────▼───────────────┐
        │ 1. 정상 우선순위 분류 (Triage) │  AI P(NORM) 연속점수로 의사 검토 큐 정렬(자동확정 아님)
        └──────────────┬───────────────┘  모든 ECG는 예외 없이 E→2→3→4→서명 큐로 계속 진행
                       │  (P(NORM) 점수는 우선순위 메타데이터로 리포트에 동봉될 뿐, 게이팅 없음)
        ┌──────────────▼───────────────┐
        │ E. 응급 fast-path              │  STEMI(ST상승)·Ischemia(ST하강)·Long QT(QTc>500)
        └──────────────┬───────────────┘  양성 → [즉시 CRITICAL 알람] (게이팅 안 함, 계속 진행)
                       │  (VT/VF·완전AVB는 리듬/비트 형태학 기반이라 3단계 이후 별도 검증 필요 — 미구현)
        ┌──────────────▼───────────────┐
        │ 2. 룰 기반 전도장애/축         │  RBBB·LBBB·AVB·LAD/RAD (결정트리)
        └──────────────┬───────────────┘
        ┌──────────────▼───────────────┐
        │ 3. Beat 부정맥 검증           │  MIT-BIH 전이 → N/S/V/F/Q 비트 통계
        └──────────────┬───────────────┘  (전도장애 아님 — 이소성/조기수축 검증 전용)
        ┌──────────────▼───────────────┐
        │ 4. ST-T 허혈 정밀             │  European ST-T + PTB-XL STTC, μV급 ST 편위
        └──────────────┬───────────────┘
        ┌──────────────▼───────────────┐
        │ R. 결과 합성 → 다중 진단 리포트 │  근거 동봉 → [의사 확인 후 발행]
        └──────────────────────────────┘
```

---

## 4. 단계별 상세 규격

### 0단계 — 전처리 (core-signal, Java)
- **입력**: 12-lead raw (WFDB / CSV / 수치배열), 임의 샘플레이트.
- **처리**:
  1. 리샘플 → 500Hz 표준화.
  2. 대역통과 0.5–40Hz (진단 대역), 전원 노이즈 50/60Hz notch.
  3. 기선보정: median-filter(200ms+600ms) 또는 wavelet detrend.
  4. (선택) Stage4 디노이징 오토인코더로 근전도 노이즈 억제.
  5. Z-정규화(리드별) → 추론 입력 규격 통일.
- **SQI (신호품질지수)**: flatline·포화·과잉노이즈·리드탈락 검출 → 임계 미달 시 **판독불가 반려** (오판독 방지).
- **출력**: `PreprocessedEcg { double[12][5000], fs=500, sqi, leadsOk[12] }`.

### 1단계 — 정상 우선순위 분류 (features + inference, Java) — *가장 먼저, 게이팅 없음*
- **모델**: 20개 임상 피처(HR, PR, QRS폭, QTc, ST편위, 축 …) 기반 로지스틱회귀(`FeatureBasedNormClassifier`, ONNX 없이 JSON 가중치 직접 적용 — 완전 감사 가능).
- **역할**: `P(NORM)` 연속 점수를 산출해 의사 검토 큐의 **우선순위 정렬 메타데이터**로만 사용한다. 어떤 임계값을 넘어도 이후 단계를 건너뛰지 않는다 — 모든 ECG는 예외 없이 E→2→3→4를 거쳐 서명 큐로 간다.
- **왜 게이팅을 포기했는가**: 초기 설계는 `P(NORM)≥0.98 ∧ 20피처 전부 정상범위`를 만족하면 자동 사전선별하려 했으나, PTB-XL 21,799개 실측에서 이 AND게이트를 통과한 29건 중 23건(79%)이 실제로는 비정상이었다. 10여 개의 스트립-전체 요약 통계로는 "정상"을 안전하게 확정할 정보량이 부족하다는 것을 실측으로 확인했다(raw waveform 기반 CNN/Transformer로 재시도하거나, 피처를 12리드 개별 단위까지 대폭 확장하는 것이 향후 개선 방향).
- **성능(test set, PTB-XL fold 10)**: acc 73.6%, sensitivity(NORM recall) 83.8%, specificity 68.6% — 우선순위 정렬 신호로는 유의미하지만, 자동확정 게이트로 쓰기엔 specificity가 불충분.
- ⚠️ 자동 확정 아님 — 모든 리포트는 의사 서명 큐로 이동(P3).

### E단계 — 응급 fast-path (rule-engine, Java) — *전체 ECG 대상, 항상 실행*
- 1단계가 더 이상 게이팅하지 않으므로, **모든 ECG**에 대해 실행되어 CRITICAL 알람 여부를 판정한다.
- 검출: **STEMI**(연속 리드 ST elevation ≥ 기준), **VT/VF**(광폭 빈맥·무질서, 미구현), **완전 방실차단**(P-QRS 해리, 미구현), **Long QT**(QTc 임계 초과).
- 2·3·4단계보다 우선하는 상단 알람 — 후속 정밀 판독을 게이팅하지 않고 병행.

### 2단계 — 룰 기반 전도장애/축 (rule-engine, Java)
- **RBBB**: QRS ≥ 120ms ∧ V1 rSR′ ∧ V6 넓은 S.
- **LBBB**: QRS ≥ 120ms ∧ V5/V6/I 넓고 뭉툭한 R ∧ V1 QS/rS.
- **1도 AVB**: PR > 200ms. **2·3도**: 전도비/해리.
- **축편위**: I·aVF·II 극성으로 LAD/RAD 산출.
- 각 판정은 사용된 피처값·임계를 근거로 기록.

### 3단계 — Beat 부정맥 검증 (inference, Java + Python 학습)
- **MIT-BIH 전이학습** CNN-Transformer 백본, R-peak 기준 비트 슬라이싱.
- 비트 분류 **N/S/V/F/Q** (AAMI) 통계 → PVC·PAC·이소성 부담 정량.
- ❗ **전도장애 교차검증 아님**(P5). RBBB/LBBB 확정은 2단계 소관.
- 리드 구성(2→12)·샘플레이트(360→500) 차이는 학습 시 어댑터로 흡수.

### 4단계 — ST-T 허혈 정밀 (inference, Java + Python 학습)
- **디노이징 오토인코더**(Long-Term ST 학습)로 기선/EMG 잔여 노이즈 제거(0단계와 연계).
- **ST 분할 모델**(European ST-T + PTB-XL STTC)로 J점 기준 ST elevation/depression을 μV급 검출.
- 리드별 ST 편위 맵 → 허혈 영역(전벽/하벽/측벽) 추정.

### R단계 — 결과 합성 & 리포트 (pipeline + app-fx)
- 2·3·4단계 소견 병합, 응급 알람 상단 고정.
- **근거 동봉 리포트**: 각 진단의 피처값·룰조건·모델확률·모델버전.
- 상태: `PENDING_SIGN` → 의사 확인 → `SIGNED` 발행. SQLite 감사로그 적재.

---

## 5. 모듈 구조 (Gradle 멀티모듈 + Python)

```
EKGGDSEMR2026/
├── settings.gradle.kts            # 멀티모듈 등록
├── build.gradle.kts               # 루트 공통 (Java 25 toolchain)
├── gradle/libs.versions.toml      # 버전 카탈로그
├── config/pipeline.yaml           # 임계값·룰 파라미터·경로
├── ARCHITECTURE.md
│
├── core-signal/     # 0단계: WFDB 리더, 전처리, SQI          (순수 Java)
├── core-features/   # 피델셜 검출 + 20 임상 피처               (순수 Java)
├── rule-engine/     # E·2단계 룰 (STEMI/VT/RBBB/LBBB/AVB/축)   (순수 Java)
├── inference/       # 1단계 FeatureBasedNormClassifier(JSON, 감사가능)
│                    # + ONNX Runtime 래퍼(Stage1NormClassifier 등 raw waveform 모델 대안, 3/4단계용)
├── pipeline/        # 오케스트레이터 0→E→1→2→3→4→R            (Java)
├── persistence/     # SQLite DAO, 감사로그, 피처 적재           (Java + sqlite-jdbc)
├── app-fx/          # JavaFX UI, ECG 뷰어, 리포트, 서명 큐      (JavaFX 25)
│
├── ml/              # ── Python (Gradle 밖) ──
│   ├── pyproject.toml
│   ├── ecgml/data/ptbxl_dataset.py     # Java 전처리 산출물(.f32) 로더 + NORM 라벨
│   ├── ecgml/preprocess.py             # 모델 입력 Z-정규화 (Java ZNormalizer와 동일 공식)
│   ├── ecgml/models/stage1_norm.py             # 로컬 CPU용 소형 CNN (raw waveform, 대안 경로)
│   ├── ecgml/models/stage1_norm_transformer.py # GPU(Kaggle P100)용 CNN-Transformer (raw waveform, 대안 경로)
│   ├── train_stage1.py           # raw waveform CNN/Transformer 학습 (--model-type)
│   ├── train_stage1_features.py  # 20피처 로지스틱회귀/GBM 학습 (현재 채택된 1단계 모델)
│   └── export_onnx.py            # raw waveform 모델 → ONNX export
│
└── models/          # 학습 산출물: stage1_logreg.json(채택), *.onnx(대안, 대용량은 /mnt/t7 심볼릭)
```

의존 방향: `app-fx → pipeline → {core-signal, core-features, rule-engine, inference, persistence}`. core-* 는 서로 독립, 순환 없음.

---

## 6. 데이터 저장

### SQLite (런타임, `.emr/` 또는 프로젝트 `data/`)
- `patient(id, ...)` — 환자 매핑.
- `ecg_record(id, patient_id, acquired_at, source, fs, path)` — 원신호 참조.
- `ecg_feature(record_id, feature_rank, name, value, unit, normal_low, normal_high, in_range)` — 20 피처 수치.
- `diagnosis(id, record_id, stage, code, label, confidence, evidence_json, model_version, created_at)` — 단계별 소견.
- `report(id, record_id, status[PENDING_SIGN|SIGNED], signed_by, signed_at, payload_json)`.
- `audit_log(id, actor, action, target, model_hash, ts)` — 감사추적(dka-emr-audit.db 연계 가능).

### /mnt/t7 (대용량 Raw & 학습)
- `datasets/` PTB-XL(records500)·MIT-BIH·challenge_2021 원본 유지.
- 학습 입력은 레코드별 **NPZ 샤딩** 또는 WebDataset(거대 H5 랜덤액세스 병목 회피).
- `models/` 학습 체크포인트·ONNX 산출물 보관, 프로젝트 `models/`에 심볼릭.

---

## 7. 기술 스택

| 레이어 | 선택 | 근거 |
|--------|------|------|
| 언어(앱) | Java 25 (Temurin) | 설치 확인, LTS |
| 빌드 | Gradle 9.3.1 (Kotlin DSL) | 설치 확인, 멀티모듈 |
| UI | JavaFX 25.0.1 | SDK 확인 |
| 추론 | ONNX Runtime (Java) | PyTorch→ONNX 표준 서빙 |
| DB | SQLite (sqlite-jdbc) | 단일머신 EMR, 감사로그 |
| 학습 | Python 3 + PyTorch | 데이터셋·모델 생태계 |
| 신호 IO | wfdb(py), 자체 WFDB-16 리더(Java) | PTB-XL/MIT-BIH 포맷 |

---

## 8. 단계별 로드맵 (Phased Roadmap)

- **Phase 0** ✅ — 스캐폴드: Gradle 멀티모듈 골격 + 빌드 검증 + 설정/문서.
- **Phase 1** ✅ — core-signal: WFDB 리더 + 전처리(0단계) + SQI. PTB-XL/MIT-BIH 실데이터 로드 검증.
- **Phase 2** ✅ — core-features + rule-engine: 피델셜 검출, 20 피처, 2단계 룰, E 응급.
- **Phase 3** ✅ — ml(Python): PTB-XL 21,799개 배치 전처리 + Stage1 정상 우선순위 로지스틱회귀 학습
  → `models/stage1_logreg.json`. inference 모듈(`FeatureBasedNormClassifier`) Java 연동.
  (raw waveform CNN/Transformer는 대안 경로로 보관, Kaggle GPU 학습용)
- **Phase 5** ✅ — pipeline 오케스트레이터(`EkgPipeline`) + persistence(SQLite, `ReportRepository`) 결선.
  Stage3/4는 `beatArrhythmiaAvailable`/`stIschemiaAvailable=false`로 명시된 플레이스홀더.
- **Phase 6** ✅ — app-fx: 표준 12유도 파형 뷰어(임상 그리드, 25mm/s·10mm/mV), 진단 리포트 화면,
  서명 대기 큐, SQLite 연동까지 실제 PTB-XL 레코드로 구동 검증.
  실사용 중 발견·수정한 버그 3건: (1) 필터 워밍업 과도현상이 R-peak 검출을 무너뜨리던 문제
  (mirror-padding으로 해결, Phase 1 코드), (2) ST 측정 윈도우가 QRS 말단 잔여신호를 포함해
  계통적 음성 편향을 만들던 문제(J+20ms로 조정), (3) T파 종료 미검출 시 탐색구간 끝으로
  기본값을 채워 QT가 비정상적으로 길게 측정되던 문제(미검출 시 해당 비트 제외로 수정).
  ⚠️ **알려진 미해결 항목**: 위 수정 후에도 ISCHEMIA 룰이 검증된 NORM 라벨 레코드(4개 중 3개)에서
  여전히 과발화한다 — 임계값(0.05mV)이나 공유 리드-II 타이밍을 각 리드에 그대로 적용하는 설계
  자체의 한계일 수 있음. 대규모 검증셋 없이 파라미터를 임의로 더 조정하지 않기로 함 — Phase 7
  교차검증 및 추가 학습 데이터 확보 후 데이터 기반으로 재보정 예정.
- **Phase 7** ✅ **backbone 구축 + 실제 교차데이터 검증 완료**:
  - **backbone**: `rule-engine`에 `RuleThresholds` 도입(STEMI/ISCHEMIA/LongQT/BBB/AVB1 임계치를
    값으로 주입). 기존 인자없는 `evaluate(...)`는 `RuleThresholds.defaults()`로 위임해 기존 13개
    테스트 무변경. `pipeline`에 `PipelineConfig.loadFromYaml(...)` + `EkgPipeline.fromConfig(...)`
    추가 — `config/pipeline.yaml` 한 파일만 고치면 SQI/ST/QTc 임계치·모델경로가 전부 반영됨
    (재컴파일 불요). `AppContext`/`EmrApp`도 config 파일 기반으로 전환, 기존 하드코딩 값과 동일
    동작 회귀 검증 완료(37→40 테스트).
  - **`.mat` 포맷 지원 추가**: challenge_2021 데이터셋은 다운로드가 타임아웃으로 실패했지만, 로컬에
    이미 받아져 있던 CPSC2018/Chapman-Shaoxing+Ningbo(CinC2020 재배포)를 바이트 단위로 직접 조사한
    결과, 압축 없는 MATLAB v4 flat 포맷(`.hea`에 `16+24`로 명시)임을 확인 — 헤더 24바이트를 건너뛰면
    그 뒤는 기존 WFDB format 16과 완전히 동일한 바이트 스트림이다. 새 코덱 없이 `WfdbHeaderParser`가
    format 필드의 `+N` skip을 파싱해 `SignalSpec.byteOffset`으로 넘기고, `WfdbRecordReader`가 그만큼
    건너뛰고 기존 `Format16Codec`을 그대로 재사용하도록 한 줄만 추가(`core-signal`). 헤더에 기록된
    체크섬과 실제 디코딩 결과가 일치함을 실제 레코드로 검증(`WfdbRecordReaderTest`).
  - **성능 수정(부수 발견)**: 대량 배치를 돌리다 레코드당 처리시간이 약 0.5~0.8초로 비정상적으로
    느림을 발견 — 원인은 Stage-0 전처리의 2단 중앙값 기선보정(`MedianFilter`)이 매 샘플마다 윈도우를
    통째로 복사·재정렬(O(n·w·log w))하던 것. 빠지는/들어오는 원소만 정렬 버퍼에 이진탐색으로
    삽입·삭제하는 O(n·w) 증분 방식으로 교체(`core-signal`) — 레코드당 780ms → 28ms(약 28배).
    같은 값 집합에서 같은 순위를 뽑는 것이라 산술 연산이 없어 결과는 원래 구현과 비트 단위로
    동일함을 브루트포스 대조 테스트(`MedianFilterTest`, 무작위/경계/중복값 케이스)로 증명했고,
    판독 결과·임계값에는 영향이 없다(순수 속도 개선).
  - **외부기관 교차검증 실행**: `pipeline`에 `ExternalValidationExporter` 도구를 추가해
    `EkgPipeline.fromConfig`(실서빙 경로 그대로)로 CPSC2018(6,877건, 중국)과
    Chapman-Shaoxing+Ningbo(45,152건, 중국)를 배치 처리, `.hea`의 `#Dx:` SNOMED CT 코드와 Stage1
    triage score·rule-engine findings를 CSV로 남긴 뒤 `ml/evaluate_external_validation.py`로
    AUROC/F1을 측정했다(재보정 아님 — 측정만, 결과는 `ml/results/external_validation_report.md`).
    - **Track A(Stage1 P(NORM), 신뢰도 높음)**: 전체 51,034건 중 SNOMED 426783006(Sinus rhythm)
      단독 레코드를 NORM으로 보고 AUROC=0.717, F1@0.5=0.348. 데이터셋별로는 CPSC2018 AUROC=0.780,
      Chapman-Shaoxing/Ningbo AUROC=0.707 — PTB-XL 밖 데이터에서도 NORM 사전선별이 어느 정도
      일반화되지만(AUROC 0.7대), PTB-XL 자체 검증보다는 낮다. Kaggle 재학습 시 참고할 baseline.
    - **Track B(개별 finding, 참고용 — P5 dataset-honest)**: RBBB F1=0.077, LBBB F1=0.016,
      AVB1 F1=0.200(recall은 0.86으로 높으나 precision이 매우 낮음) — 진폭 휴리스틱 기반
      전도장애 룰이 미보정 상태임을 실측으로 재확인. 라벨 세분화 수준이 다르므로 이 수치로
      임계값을 자동 조정하지 않는다.
  - **범위 밖(의도적 보류)**: Georgia(SNOMED E##### prefix)는 로컬 미보유, 재다운로드 시도 안 함 —
    받아지면 `config/pipeline.yaml`의 `paths:`에 경로만 추가하면 됨. Stage3(beat) 외부검증은
    CPSC2018/Chapman에 비트 단위 어노테이션이 없어 불가(MIT-BIH DS1/DS2가 유일한 경로로 유지).
    RuleThresholds 실제 재조정은 이번엔 하지 않음 — Kaggle P100 학습 이후로 유지.
- **Phase 4 (Stage3 backbone 완료, Stage4는 데이터 대기)** — "backbone 우선, 정확도는 Kaggle P100
  학습 후 보정" 원칙으로 진행:
  - **Stage3(Beat 부정맥)**: MIT-BIH(DS1=train/DS2=test, de Chazal 표준 분할)로 AAMI 5-클래스
    (N/S/V/F/Q) CNN(`ecgml/models/stage3_beat.py`)을 학습·ONNX export·Java `BeatArrhythmiaClassifier`
    연동까지 **엔드투엔드로 실증 완료**. `Stage3ArrhythmiaAnalyzer`가 비트별 분류를 집계해 이소성
    부담(ectopy burden) 소견을 산출하고, `EkgPipeline`이 실제 PTB-XL 레코드에서
    `beatArrhythmiaAvailable=true`를 반환함을 UI 스크린샷으로 확인.
    ⚠️ 정확도 미검증(학습 시 혼동행렬 기준 N/F 혼동 다수, test acc ~50%대) — 배관 증명이 목적이며
    재학습은 Kaggle P100에서 진행 예정. `BeatArrhythmiaConfig`가 모델 부재 시 조용히
    `beatArrhythmiaAvailable=false`로 넘어가도록 설계되어 있어(Phase 7과 동일 backbone 패턴),
    재학습된 모델을 `models/stage3_beat.onnx`에 덮어쓰기만 하면 코드 변경 없이 반영된다.
  - **Stage4(ST-T 허혈 정밀)**: European ST-T 데이터셋이 로컬에 없어(다운로드 필요) 학습 보류.
    모델 배치 시 Stage3와 동일한 backbone 패턴(config 경로 존재 확인 → available 플래그)을
    재사용할 수 있도록 구조만 설계해둠.
  - Python 환경에서 `onnx` 설치 시 numpy가 2.x로 업그레이드되며 `wfdb` 어노테이션 파서가
    깨지는 회귀를 발견·수정(`numpy<2`로 고정). 향후 이 환경에 패키지 추가 시 유의.

각 Phase는 빌드·테스트 통과를 게이트로 다음으로 진행.
