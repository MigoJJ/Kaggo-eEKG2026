"""1단계(정상 사전선별) — 20개 임상 피처 기반 해석 가능한 모델 학습.

Java PtbxlFeatureExporter가 만든 features.csv(f1~f20 = ClinicalFeature rank 1~20)를 입력으로,
로지스틱회귀(주 후보, 완전 감사 가능)와 HistGradientBoosting(성능 비교용)을 학습·비교한다.
raw waveform CNN/Transformer 대신 이 방향을 우선하는 이유: Stage 1은 "의심 여지 없는 정상만
배제하는 안전 게이트"이므로, 왜 정상으로 판단했는지 설명 가능해야 한다(ARCHITECTURE.md P4).

로지스틱회귀는 (feature_order, impute_median, standardize_mean/std, weights, bias)를
JSON으로 export하여 Java `inference` 모듈이 ONNX 없이 직접 재현한다(가중치를 코드/설정처럼
그대로 감사할 수 있다).

사용법:
  python train_stage1_features.py --features-csv <PtbxlFeatureExporter 출력 CSV> \
      --manifest-csv <PtbxlPreprocessExporter manifest.csv> --ptbxl-csv <ptbxl_database.csv> \
      --out models/stage1_logreg.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from ecgml.data.ptbxl_dataset import load_labels

FEATURE_ORDER = [
    "HR", "PR_INTERVAL", "P_DURATION", "P_AMPLITUDE_II", "QRS_DURATION", "QRS_AXIS",
    "R_AMPLITUDE_V5", "R_AMPLITUDE_V6", "S_AMPLITUDE_V1", "SOKOLOW_LYON_INDEX",
    "R_AMPLITUDE_AVL", "Q_DURATION_II", "Q_DEPTH_RATIO_II", "ST_DEVIATION_II",
    "ST_DEVIATION_V2", "ST_DEVIATION_V3", "T_AMPLITUDE_II", "QT_INTERVAL",
    "QTC_BAZETT", "RR_VARIABILITY",
]
STD_FLOOR = 1e-9


def evaluate(model, x: np.ndarray, y: np.ndarray) -> dict:
    preds = model.predict(x)
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(1, len(y))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {"acc": acc, "sensitivity": sensitivity, "specificity": specificity,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--ptbxl-csv", required=True)
    parser.add_argument("--out", default="models/stage1_logreg.json")
    args = parser.parse_args()

    features_df = pd.read_csv(args.features_csv, index_col="ecg_id")
    # PtbxlFeatureExporter는 ClinicalFeature rank 순서로 f1..f20 컬럼명을 쓴다 -> 의미 이름으로 매핑
    rank_to_name = {f"f{i + 1}": name for i, name in enumerate(FEATURE_ORDER)}
    features_df = features_df.rename(columns=rank_to_name)

    manifest = pd.read_csv(args.manifest_csv, index_col="ecg_id")
    manifest = manifest[manifest["interpretable"]]
    labels = load_labels(Path(args.ptbxl_csv))

    joined = features_df.join(manifest[[]], how="inner").join(labels, how="inner")
    joined = joined.dropna(subset=FEATURE_ORDER, how="all")  # 20개 전부 NaN(비트 미검출)인 레코드만 제외

    # R-peak 검출기 한계로 인한 명백한 오검출(HR 25~250bpm 범위 밖, 10초 스트립에 5비트 미만)
    # 레코드를 제외한다. 실측 21,799개 중 약 1.17%(256개) 해당 — 실제 서빙에서는 이런 케이스가
    # SQI/피델셜 신뢰도 게이트에서 걸러져 수동판독으로 넘어가야 한다(추후 정밀화 과제).
    implausible = (joined["HR"] < 25) | (joined["HR"] > 250) | (joined["n_beats"] < 5)
    print(f"R-peak 오검출 의심으로 제외: {implausible.sum()}개 ({implausible.mean()*100:.2f}%)")
    joined = joined[~implausible]

    train = joined[joined.strat_fold <= 8]
    val = joined[joined.strat_fold == 9]
    test = joined[joined.strat_fold == 10]
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    x_train_raw = train[FEATURE_ORDER].to_numpy(dtype=float)
    impute_median = np.nanmedian(x_train_raw, axis=0)

    def prepare(df: pd.DataFrame) -> np.ndarray:
        x = df[FEATURE_ORDER].to_numpy(dtype=float)
        inds = np.where(np.isnan(x))
        x[inds] = np.take(impute_median, inds[1])
        return x

    x_train = prepare(train)
    x_val = prepare(val)
    x_test = prepare(test)
    y_train = train["is_norm"].to_numpy(dtype=int)
    y_val = val["is_norm"].to_numpy(dtype=int)
    y_test = test["is_norm"].to_numpy(dtype=int)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < STD_FLOOR, 1.0, std)

    def standardize(x: np.ndarray) -> np.ndarray:
        return (x - mean) / std

    xs_train, xs_val, xs_test = standardize(x_train), standardize(x_val), standardize(x_test)

    logreg = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    logreg.fit(xs_train, y_train)
    print("LogisticRegression  val:", evaluate(logreg, xs_val, y_val))
    print("LogisticRegression test:", evaluate(logreg, xs_test, y_test))

    gbm = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced")
    gbm.fit(x_train, y_train)  # 트리 기반이라 표준화 불필요
    print("HistGradientBoosting  val:", evaluate(gbm, x_val, y_val))
    print("HistGradientBoosting test:", evaluate(gbm, x_test, y_test))

    for name, coef in sorted(zip(FEATURE_ORDER, logreg.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {name:20s} weight={coef:+.4f}")

    # 0.5 임계값 정확도는 실제 배포 로직과 다르다 — 실제로는 P(NORM) >= 임계치(0.98)를 요구한다.
    # Stage 1의 안전성 핵심 지표는 "그 엄격한 임계값에서 비정상을 정상으로 오판하는 비율"과
    # "그만큼 엄격하게 걸렀을 때 자동 통과되는 비율(수율)"이다.
    print("\n=== 엄격 임계값에서의 안전성 지표 (test set) ===")
    proba_test = logreg.predict_proba(xs_test)[:, 1]
    for threshold in (0.5, 0.9, 0.95, 0.98, 0.99):
        predicted_norm = proba_test >= threshold
        false_normal = int((predicted_norm & (y_test == 0)).sum())  # 비정상인데 정상으로 오판
        cleared = int(predicted_norm.sum())
        false_normal_rate = false_normal / max(1, cleared)
        yield_rate = cleared / len(y_test)
        print(f"  threshold={threshold:.2f}: 자동통과 수율={yield_rate*100:5.1f}% "
              f"({cleared}건) / 그중 오판(비정상→정상)={false_normal}건 ({false_normal_rate*100:.2f}%)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "logistic_regression",
        "version": "stage1-logreg-v1",
        "trained_on": f"ptbxl fold1-8, n={len(train)}",
        "feature_order": FEATURE_ORDER,
        "impute_median": impute_median.tolist(),
        "standardize_mean": mean.tolist(),
        "standardize_std": std.tolist(),
        "weights": logreg.coef_[0].tolist(),
        "bias": float(logreg.intercept_[0]),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"로지스틱회귀 모델 저장: {out_path}")


if __name__ == "__main__":
    main()
