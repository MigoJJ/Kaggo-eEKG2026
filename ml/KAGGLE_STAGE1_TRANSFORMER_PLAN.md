# Kaggle 학습 절차: Stage1 Waveform Transformer

목표:

1. PTB-XL raw waveform 전처리 산출물로 `Stage1NormTransformer`를 학습한다.
2. 기존 20피처 로지스틱회귀 Stage1과 별도 대안 경로로 성능을 측정한다.
3. ONNX로 export해 Java inference 계약 `(batch, 12, 5000) -> (batch)`을 검증한다.

현재 배포 우선 Stage1은 `models/stage1_logreg.json` 기반의 해석 가능한 로지스틱회귀다. Waveform
Transformer는 자동 확정 게이트가 아니라 triage score 개선 후보이며, 대규모 검증 전에는
`config/pipeline.yaml`의 운영 Stage1을 교체하지 않는다.

## 0. Kaggle Notebook 준비

Kaggle Notebook에서 GPU(P100)를 켜고 아래 패키지를 설치한다.

```bash
pip install -q pandas numpy scipy onnx onnxruntime
```

프로젝트 소스와 Java 전처리 산출물(`manifest.csv`, `*.f32`)을 Kaggle Dataset으로 올리거나
`/kaggle/working/EKGGDSEMR2026`에 둔다.

```bash
cd /kaggle/working/EKGGDSEMR2026/ml
```

먼저 Kaggle input 자동 탐색 결과를 확인한다.

```bash
python kaggle_discover_inputs.py --input-root /kaggle/input
```

## 1. 데이터셋 구조 확인

Stage1 waveform 학습은 Java `PtbxlPreprocessExporter`가 만든 파일을 입력으로 쓴다. Kaggle에서
아래 파일이 보이는지 먼저 확인한다.

```bash
ls -lh /kaggle/input/ptbxl-preprocessed
head /kaggle/input/ptbxl-preprocessed/manifest.csv
ls /kaggle/input/ptbxl-preprocessed/*.f32 | head
```

확인할 것:

- `manifest.csv`에 `ecg_id`, `interpretable`, `strat_fold` 결합에 필요한 ID가 있는지.
- 각 `.f32`가 12 x 5000 float32 row-major인지.
- `ptbxl_database.csv` 경로가 별도로 준비되어 있는지.

## 2. Transformer 학습

```bash
python kaggle_run_stage1_waveform.py \
  --input-root /kaggle/input \
  --model-type transformer \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-4 \
  --working-dir /kaggle/working
```

위 runner는 PTB-XL preprocessed root(`manifest.csv` + `*.f32`)와 `ptbxl_database.csv`를 자동 탐색한 뒤
학습 → ONNX export + sidecar JSON 생성을 순서대로 실행한다. root를 직접 지정하려면 아래처럼 실행한다.

```bash
python train_stage1.py \
  --export-dir /kaggle/input/ptbxl-preprocessed \
  --ptbxl-csv /kaggle/input/ptbxl/ptbxl_database.csv \
  --model-type transformer \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-4 \
  --out /kaggle/working/stage1_norm_transformer.pt
```

산출물:

- `/kaggle/working/stage1_norm_transformer.pt`

중간 판단 기준:

- PTB-XL 공식 fold 관례를 유지한다: fold 1-8 train, 9 val, 10 test.
- `val_loss` 기준 best checkpoint를 저장한다.
- `TEST`의 sensitivity/specificity는 triage 후보 지표일 뿐 자동 정상 확정 근거가 아니다.

## 3. CNN baseline 재확인

동일한 split과 데이터로 소형 CNN도 한 번 재학습해 Transformer 개선폭을 비교한다.

```bash
python train_stage1.py \
  --export-dir /kaggle/input/ptbxl-preprocessed \
  --ptbxl-csv /kaggle/input/ptbxl/ptbxl_database.csv \
  --model-type cnn \
  --epochs 30 \
  --batch-size 64 \
  --lr 1e-3 \
  --out /kaggle/working/stage1_norm_cnn.pt
```

산출물:

- `/kaggle/working/stage1_norm_cnn.pt`

## 4. ONNX export

```bash
python export_onnx.py \
  --checkpoint /kaggle/working/stage1_norm_transformer.pt \
  --model-type transformer \
  --out /kaggle/working/stage1_norm_transformer.onnx

python export_onnx.py \
  --checkpoint /kaggle/working/stage1_norm_cnn.pt \
  --model-type cnn \
  --out /kaggle/working/stage1_norm_cnn.onnx
```

산출물:

- `/kaggle/working/stage1_norm_transformer.onnx`
- `/kaggle/working/stage1_norm_transformer.json`
- `/kaggle/working/stage1_norm_cnn.onnx`
- `/kaggle/working/stage1_norm_cnn.json`

`max_abs_diff <= 1e-3`이면 PyTorch/ONNX Runtime 정합성은 통과다. `export_onnx.py`는 Java ONNX
Runtime 호환을 위해 `opset_version=17`로 고정한다. export 스크립트는 ONNX 옆 sidecar JSON도
생성하며, 여기에는 artifact SHA-256, opset, 입출력 시그니처, Python/package 환경 버전이 포함된다.

## 5. Threshold sweep 리포트

Stage1은 게이팅하지 않는다. 그래도 triage score로서 threshold별 특성을 기록한다.

```text
model_type,threshold,cleared,yield_rate,false_normal,false_normal_rate,sensitivity,specificity
transformer,0.50,,,,,,,
transformer,0.90,,,,,,,
transformer,0.95,,,,,,,
transformer,0.98,,,,,,,
transformer,0.99,,,,,,,
cnn,0.50,,,,,,,
cnn,0.90,,,,,,,
cnn,0.95,,,,,,,
cnn,0.98,,,,,,,
cnn,0.99,,,,,,,
```

정책:

- threshold sweep 결과만으로 자동 정상 확정 게이트를 부활시키지 않는다.
- 기존 feature-based Stage1보다 triage AUROC/정렬 품질이 좋아도 의사 서명 큐 정책은 유지한다.
- 운영 반영은 별도 PR/커밋에서 Java inference wrapper와 audit 기록까지 함께 검토한다.

## 6. 로컬 프로젝트 반영

Kaggle output에서 아래 파일을 내려받아 후보 산출물로 둔다.

```text
models/stage1_norm_transformer.onnx
models/stage1_norm_transformer.json
reports/stage1_waveform_threshold_sweep.csv
```

그 다음 로컬에서 실행한다.

```bash
python -m unittest discover -s ml/tests
./gradlew :inference:test
./gradlew :pipeline:test
```

기대 결과:

- ONNX export 정합성 통과.
- Java `ZNormalizer`와 Python `znormalize_per_lead` 계약 테스트 통과.
- sidecar JSON이 있으면 Java audit log와 `model_artifact` 테이블에 metadata hash/raw JSON이 저장된다.
- 운영 Stage1인 `models/stage1_logreg.json` 경로는 유지된다.
- Waveform Stage1을 운영 경로로 승격할지는 threshold sweep, audit metadata, Java wrapper 검증 후 결정한다.
