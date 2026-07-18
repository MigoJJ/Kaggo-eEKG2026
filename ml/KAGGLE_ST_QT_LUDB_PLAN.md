# Kaggle 학습 절차: European ST-T + QTDB + LUDB

목표:

1. LUDB/QTDB로 P/QRS/T delineation 모델을 학습한다.
2. European ST-T Database로 ST depression threshold 과발화를 측정한다.
3. ONNX로 export해 Java Stage4 연동 준비물을 만든다.

## 0. Kaggle Notebook 준비

Kaggle Notebook에서 GPU를 켜고 아래 패키지를 설치한다.

```bash
pip install -q wfdb scipy onnx onnxruntime
```

프로젝트 소스는 Kaggle Dataset으로 올리거나 GitHub/압축 파일로 `/kaggle/working/EKGGDSEMR2026`에 둔다.

```bash
cd /kaggle/working/EKGGDSEMR2026/ml
```

## 1. 데이터셋 구조 확인

Kaggle input 경로는 업로드 이름에 따라 달라진다. 먼저 실제 record 수, annotation 확장자, symbol을 확인한다.

```bash
python inspect_wfdb_dataset.py \
  --root /kaggle/input/ludb/ludb/1.0.1 \
  --limit 5

python inspect_wfdb_dataset.py \
  --root /kaggle/input/qt-database/qtdb/1.0.0 \
  --limit 5

python inspect_wfdb_dataset.py \
  --root /kaggle/input/european-st-t-database-100/european-st-t-database-1.0.0 \
  --limit 5
```

확인할 것:

- LUDB usable annotation: lead별 `i`, `ii`, `v1` 등.
- QTDB usable annotation: `q1c`, `q2c`, `pu` 등.
- European ST-T annotation: `atr` 또는 `st`, aux note에 ST/T episode 정보가 있는지.

## 2. LUDB + QTDB delineator 학습

QTDB는 2유도인 경우가 많아 12채널 zero-padding으로 학습에 포함된다. LUDB는 12유도 boundary 학습의 주 데이터로 둔다.

```bash
python train_stage4_delineator.py \
  --ludb-root /kaggle/input/ludb/ludb/1.0.1 \
  --qtdb-root /kaggle/input/qt-database/qtdb/1.0.0 \
  --epochs 30 \
  --batch-size 8 \
  --out /kaggle/working/stage4_delineator.pt
```

산출물:

- `/kaggle/working/stage4_delineator.pt`
- `/kaggle/working/stage4_delineator.json`

중간 판단 기준:

- `val_macro_wave_f1`가 올라가는지 확인한다.
- `p`, `qrs`, `t` 각각의 F1을 본다. `background` F1은 중요 지표가 아니다.

## 3. European ST-T threshold sweep

현재 Java rule-engine의 `ISCHEMIA` 과발화 문제를 먼저 수치화한다.

```bash
python evaluate_european_stt_thresholds.py \
  --edb-root /kaggle/input/european-st-t-database-100/european-st-t-database-1.0.0 \
  --thresholds-uv 50,75,100,150,200 \
  --out /kaggle/working/european_stt_threshold_sweep.csv
```

산출물:

- `/kaggle/working/european_stt_threshold_sweep.csv`

해석:

- `threshold_uv=50`에서 record-level fire rate가 높으면 현재 `config/pipeline.yaml`의 50uV 설정은 과민할 가능성이 크다.
- 75/100/150uV에서 episode coverage와 false alarm tradeoff를 비교한다.
- 이 결과만으로 Java 설정을 바로 바꾸지 말고, PTB-XL NORM false positive와 함께 본다.

## 4. ONNX export

```bash
python export_stage4_delineator_onnx.py \
  --checkpoint /kaggle/working/stage4_delineator.pt \
  --out /kaggle/working/stage4_delineator.onnx
```

산출물:

- `/kaggle/working/stage4_delineator.onnx`

`max_abs_diff <= 1e-3`이면 PyTorch/ONNX Runtime 정합성은 통과다.

## 5. 로컬 프로젝트 반영

Kaggle output에서 아래 파일을 내려받아 프로젝트에 배치한다.

```text
models/stage4_delineator.onnx
models/stage4_delineator.json
reports/european_stt_threshold_sweep.csv
```

그 다음 Java 쪽에서 할 일:

1. `inference`에 Stage4 ONNX wrapper 추가.
2. `pipeline`에 `stIschemiaAvailable=true` 연결.
3. `config/pipeline.yaml`에 Stage4 모델 경로 추가.
4. PTB-XL NORM false positive와 European ST-T sensitivity를 함께 보고 ST 임계값 재보정.

