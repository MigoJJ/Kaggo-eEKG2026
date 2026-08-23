# Kaggle 학습 절차: Stage3 Beat Arrhythmia

목표:

1. MIT-BIH DS1/DS2 표준 inter-patient 분할로 AAMI N/S/V/F/Q 비트 분류기를 학습한다.
2. `cnn`, `resnet`, `inception` 백본을 같은 데이터/지표로 비교한다.
3. ONNX로 export해 Java `BeatArrhythmiaClassifier` 계약을 유지한 채 `models/stage3_beat.onnx`만 교체한다.

로컬 CPU 결과는 배관 검증용이다. 최종 성능 주장은 Kaggle P100 학습 산출물의 confusion matrix,
class sensitivity, ONNX 정합성 리포트가 있을 때만 한다.

## 0. Kaggle Notebook 준비

Kaggle Notebook에서 GPU(P100)를 켜고 아래 패키지를 설치한다.

```bash
pip install -q wfdb scipy onnx onnxruntime
```

프로젝트 소스는 Kaggle Dataset으로 올리거나 GitHub/압축 파일로 `/kaggle/working/EKGGDSEMR2026`에 둔다.

```bash
cd /kaggle/working/EKGGDSEMR2026/ml
```

## 1. 데이터셋 구조 확인

Kaggle input 경로는 업로드 이름에 따라 달라진다. 먼저 실제 record 수와 annotation symbol을 확인한다.

```bash
python inspect_wfdb_dataset.py \
  --root /kaggle/input/mit-bih-arrhythmia-database/mit-bih-arrhythmia-database-1.0.0 \
  --limit 5
```

확인할 것:

- `.hea`, `.dat`, annotation 파일이 같은 디렉터리에 있는지.
- beat annotation symbol이 `N`, `A`, `V`, `F` 등 AAMI 매핑 가능한 형태인지.
- Kaggle input root가 스크립트의 `--mitbih-root`로 바로 전달 가능한지.

## 2. 백본별 학습

동일한 epoch/batch/lr로 먼저 비교한다. 클래스 불균형 때문에 단순 accuracy보다 S/V/F sensitivity를 우선한다.

```bash
python train_stage3_beat.py \
  --mitbih-root /kaggle/input/mit-bih-arrhythmia-database/mit-bih-arrhythmia-database-1.0.0 \
  --model-type cnn \
  --epochs 30 \
  --batch-size 256 \
  --out /kaggle/working/stage3_beat_cnn.pt

python train_stage3_beat.py \
  --mitbih-root /kaggle/input/mit-bih-arrhythmia-database/mit-bih-arrhythmia-database-1.0.0 \
  --model-type resnet \
  --epochs 30 \
  --batch-size 256 \
  --out /kaggle/working/stage3_beat_resnet.pt

python train_stage3_beat.py \
  --mitbih-root /kaggle/input/mit-bih-arrhythmia-database/mit-bih-arrhythmia-database-1.0.0 \
  --model-type inception \
  --epochs 30 \
  --batch-size 256 \
  --out /kaggle/working/stage3_beat_inception.pt
```

산출물:

- `/kaggle/working/stage3_beat_cnn.pt`
- `/kaggle/working/stage3_beat_resnet.pt`
- `/kaggle/working/stage3_beat_inception.pt`

중간 판단 기준:

- `test_acc`는 참고 지표다.
- N 클래스가 압도적으로 많으므로 S/V/F sensitivity가 무너지면 채택하지 않는다.
- 혼동행렬에서 N/F, S/N 혼동을 따로 기록한다.

## 3. ONNX export

학습한 백본과 같은 `--model-type`으로 export해야 한다.

```bash
python export_stage3_onnx.py \
  --checkpoint /kaggle/working/stage3_beat_cnn.pt \
  --model-type cnn \
  --out /kaggle/working/stage3_beat_cnn.onnx

python export_stage3_onnx.py \
  --checkpoint /kaggle/working/stage3_beat_resnet.pt \
  --model-type resnet \
  --out /kaggle/working/stage3_beat_resnet.onnx

python export_stage3_onnx.py \
  --checkpoint /kaggle/working/stage3_beat_inception.pt \
  --model-type inception \
  --out /kaggle/working/stage3_beat_inception.onnx
```

산출물:

- `/kaggle/working/stage3_beat_cnn.onnx`
- `/kaggle/working/stage3_beat_cnn.json`
- `/kaggle/working/stage3_beat_resnet.onnx`
- `/kaggle/working/stage3_beat_resnet.json`
- `/kaggle/working/stage3_beat_inception.onnx`
- `/kaggle/working/stage3_beat_inception.json`

`max_abs_diff <= 1e-3`이면 PyTorch/ONNX Runtime 정합성은 통과다. `export_stage3_onnx.py`는 Java
ONNX Runtime 호환을 위해 `opset_version=17`로 고정한다. export 스크립트는 ONNX 옆 sidecar JSON도
생성하며, 여기에는 artifact SHA-256, opset, 입출력 시그니처, Python/package 환경 버전이 포함된다.

## 4. 모델 선택 리포트

Kaggle output log를 기반으로 아래 표를 남긴다.

```text
model_type,test_acc,N_sens,S_sens,V_sens,F_sens,Q_sens,selected,reason
cnn,,,,,,,false,
resnet,,,,,,,false,
inception,,,,,,,false,
```

선택 원칙:

- S/V/F sensitivity를 우선한다.
- accuracy가 높아도 minority class sensitivity가 낮으면 탈락시킨다.
- Java rule threshold는 이 단계에서 바꾸지 않는다.

## 5. 로컬 프로젝트 반영

선택된 ONNX를 내려받아 git 추적 대상인 `ml/models/`(source of truth)에 배치한다.

```text
ml/models/stage3_beat.onnx
ml/models/stage3_beat.json
reports/stage3_model_comparison.csv
```

`ml/sync_models.sh`로 Java가 실제로 읽는 루트 `models/`에 동기화한 뒤 로컬에서 실행한다.

```bash
./ml/sync_models.sh
python -m unittest discover -s ml/tests
./gradlew :pipeline:test --tests emr.ekg.pipeline.PipelineConfigTest
```

기대 결과:

- Python ONNX export smoke test 통과.
- `config/pipeline.yaml`의 `beat_arrhythmia.model_path`가 실제 파일을 가리키면
  `beatArrhythmiaAvailable=true`.
- 모델 파일이 없으면 `beatArrhythmiaAvailable=false`로 우아하게 넘어간다.
- sidecar JSON이 있으면 Java audit log와 `model_artifact` 테이블에 metadata hash/raw JSON이 저장된다.
- Java inference/pipeline 코드는 수정하지 않는다.
