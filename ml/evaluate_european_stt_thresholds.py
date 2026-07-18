"""Step 3: threshold sweep on European ST-T Database.

This script produces a conservative first report for the current Java rule
issue: ischemic ST depression over-firing. It counts coarse ST/T episode
markers and measures how often robust per-lead ST-depression proxies exceed
candidate thresholds.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from ecgml.data.european_stt_dataset import (
    estimate_st_deviation_mv,
    extract_st_episodes,
    list_records,
    load_record,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edb-root", required=True)
    parser.add_argument("--out", default="reports/european_stt_threshold_sweep.csv")
    parser.add_argument("--thresholds-uv", default="50,75,100,150,200")
    args = parser.parse_args()

    root = Path(args.edb_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    thresholds_mv = [float(x) / 1000.0 for x in args.thresholds_uv.split(",")]

    rows = []
    records = list_records(root)
    print(f"EDB records={len(records)}")
    for rec in records:
        try:
            x, fs, leads = load_record(root, rec)
        except Exception as e:
            print(f"skip {rec}: {e}")
            continue
        episodes = extract_st_episodes(root, rec)
        st_proxy = estimate_st_deviation_mv(x, fs)
        for th in thresholds_mv:
            depressed = [lead for lead, v in zip(leads, st_proxy) if v <= -th]
            rows.append({
                "record": rec,
                "n_leads": len(leads),
                "n_st_t_episodes": len(episodes),
                "threshold_uv": int(round(th * 1000)),
                "depressed_lead_count": len(depressed),
                "depressed_leads": ";".join(depressed),
                "min_st_proxy_mv": float(np.min(st_proxy)),
            })

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "record", "n_leads", "n_st_t_episodes", "threshold_uv",
            "depressed_lead_count", "depressed_leads", "min_st_proxy_mv",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {out}")
    by_threshold = {}
    for th in sorted({r["threshold_uv"] for r in rows}):
        sub = [r for r in rows if r["threshold_uv"] == th]
        fired = sum(1 for r in sub if r["depressed_lead_count"] >= 2)
        by_threshold[th] = fired / max(1, len(sub))
    print("record-level fire rate when >=2 leads depressed:", by_threshold)


if __name__ == "__main__":
    main()
