"""Phase 7 외부기관 교차검증: CPSC2018 / Chapman-Shaoxing+Ningbo(CinC2020 재배포) 데이터로
Stage1(P(NORM)) triage score의 AUROC/F1과, 일부 rule-engine finding의 F1을 측정한다.

이 스크립트는 재보정(threshold tuning)을 하지 않는다 — 측정만 한다. RuleThresholds 등은
Kaggle P100 재학습·대규모 검증 이후 데이터 기반으로 반영할 예정(ARCHITECTURE.md 원칙).

Track A(주 지표, 신뢰도 높음): SNOMED CT 426783006(Sinus rhythm)만 단독으로 있는 레코드를
NORM=1, 다른 진단 코드가 하나라도 같이 있으면 NORM=0으로 보고 Stage1 triage_norm_prob 대비
AUROC/F1을 계산한다.

Track B(보조 지표, 라벨공간 불일치로 참고용— P5 dataset-honest 원칙): RBBB/LBBB/AVB1처럼
비교적 명확한 SNOMED 코드가 있는 항목만, rule-engine의 대응 finding 코드와 presence/absence
F1을 계산한다. AFib 등은 rule-engine에 해당 룰이 아직 없어 제외.

사용법: python3 ml/evaluate_external_validation.py
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV_FILES = [
    RESULTS_DIR / "cpsc2018_validation.csv",
    RESULTS_DIR / "chapman_shaoxing_ningbo_validation.csv",
]

NORM_CODE = "426783006"

# Track B: (finding_code, {harmonized SNOMED CT codes})
TRACK_B_TARGETS = {
    "RBBB": {"59118001"},
    "LBBB": {"733534002", "164909002"},
    "AVB1": {"270492004"},
}


def load_data() -> pd.DataFrame:
    frames = []
    for path in CSV_FILES:
        if not path.exists():
            print(f"경고: {path} 없음 - 스킵")
            continue
        frames.append(pd.read_csv(path, dtype={"dx_codes": str, "finding_codes": str}))
    if not frames:
        raise SystemExit("검증 CSV가 하나도 없습니다. ExternalValidationExporter를 먼저 실행하세요.")
    df = pd.concat(frames, ignore_index=True)
    df["dx_codes"] = df["dx_codes"].fillna("")
    df["finding_codes"] = df["finding_codes"].fillna("")
    df["dx_set"] = df["dx_codes"].apply(lambda s: set(c for c in s.split(";") if c))
    df["finding_set"] = df["finding_codes"].apply(lambda s: set(c for c in s.split(";") if c))
    return df


def best_f1_over_thresholds(y_true, y_score) -> tuple[float, float]:
    best_f1, best_t = 0.0, 0.5
    for t in [i / 100 for i in range(1, 100)]:
        pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t


def track_a(df: pd.DataFrame, label: str) -> dict:
    sub = df[df["interpretable"]].dropna(subset=["triage_norm_prob"])
    sub = sub[sub["dx_set"].apply(len) > 0]
    if sub.empty:
        return {"source": label, "n": 0}

    y_true = sub["dx_set"].apply(lambda s: 1 if s == {NORM_CODE} else 0).to_numpy()
    y_score = sub["triage_norm_prob"].to_numpy()

    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        auroc = float("nan")
    else:
        auroc = roc_auc_score(y_true, y_score)

    f1_at_05 = f1_score(y_true, (y_score >= 0.5).astype(int), zero_division=0)
    best_f1, best_t = best_f1_over_thresholds(y_true, y_score)

    return {
        "source": label,
        "n": len(y_true),
        "n_norm": n_pos,
        "n_abnormal": len(y_true) - n_pos,
        "auroc": auroc,
        "f1_at_0.5": f1_at_05,
        "best_f1": best_f1,
        "best_f1_threshold": best_t,
    }


def track_b(df: pd.DataFrame, label: str) -> list[dict]:
    sub = df[df["interpretable"]]
    rows = []
    for finding_code, snomed_codes in TRACK_B_TARGETS.items():
        y_true = sub["dx_set"].apply(lambda s: 1 if s & snomed_codes else 0).to_numpy()
        y_pred = sub["finding_set"].apply(lambda s: 1 if finding_code in s else 0).to_numpy()
        n_pos = int(y_true.sum())
        if n_pos == 0:
            rows.append({"source": label, "finding": finding_code, "n_pos": 0})
            continue
        rows.append({
            "source": label,
            "finding": finding_code,
            "n_pos": n_pos,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })
    return rows


def fmt(v) -> str:
    if isinstance(v, float):
        return "-" if pd.isna(v) else f"{v:.3f}"
    return str(v)


def main() -> None:
    df = load_data()
    print(f"전체 레코드: {len(df)} (소스: {sorted(df['source_db'].unique())})")

    a_rows = [track_a(df, "전체")]
    for src, g in df.groupby("source_db"):
        a_rows.append(track_a(g, src))

    b_rows = []
    b_rows.extend(track_b(df, "전체"))
    for src, g in df.groupby("source_db"):
        b_rows.extend(track_b(g, src))

    lines = []
    lines.append("# Phase 7 외부기관 교차검증 결과 (CinC2020: CPSC2018, Chapman-Shaoxing+Ningbo)\n")
    lines.append(
        "측정만 수행 — 재보정 없음. RuleThresholds/Stage1 임계값은 이 결과로 자동 변경되지 않음.\n"
    )

    lines.append("\n## Track A: Stage1 P(NORM) vs SNOMED CT 426783006(Sinus rhythm) 단독\n")
    lines.append("| 소스 | n | NORM수 | 비정상수 | AUROC | F1@0.5 | best F1 | best F1 threshold |")
    lines.append("|---|---|---|---|---|---|---|---|")
    print("\n=== Track A: Stage1 NORM AUROC/F1 ===")
    for r in a_rows:
        if r.get("n", 0) == 0:
            print(f"{r['source']}: 데이터 없음")
            continue
        row = (f"| {r['source']} | {r['n']} | {r['n_norm']} | {r['n_abnormal']} | "
               f"{fmt(r['auroc'])} | {fmt(r['f1_at_0.5'])} | {fmt(r['best_f1'])} | {fmt(r['best_f1_threshold'])} |")
        lines.append(row)
        print(f"{r['source']}: n={r['n']} NORM={r['n_norm']} 비정상={r['n_abnormal']} "
              f"AUROC={fmt(r['auroc'])} F1@0.5={fmt(r['f1_at_0.5'])} "
              f"bestF1={fmt(r['best_f1'])}(threshold={fmt(r['best_f1_threshold'])})")

    lines.append(
        "\n## Track B: 개별 finding 참고용 F1 (⚠️ 진단 세분화 수준 차이로 참고용, 재보정에 쓰지 않음)\n"
    )
    lines.append("| 소스 | finding | 양성수 | precision | recall | F1 |")
    lines.append("|---|---|---|---|---|---|")
    print("\n=== Track B: 개별 finding 참고용 F1 ===")
    for r in b_rows:
        if r.get("n_pos", 0) == 0:
            print(f"{r['source']}/{r['finding']}: 양성 라벨 없음 - 스킵")
            continue
        row = (f"| {r['source']} | {r['finding']} | {r['n_pos']} | "
               f"{fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} |")
        lines.append(row)
        print(f"{r['source']}/{r['finding']}: n_pos={r['n_pos']} "
              f"precision={fmt(r['precision'])} recall={fmt(r['recall'])} f1={fmt(r['f1'])}")

    out_path = RESULTS_DIR / "external_validation_report.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\n마크다운 저장: {out_path}")


if __name__ == "__main__":
    main()
