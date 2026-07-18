"""Step 1: inspect a WFDB dataset on Kaggle.

Examples:
  python inspect_wfdb_dataset.py --root /kaggle/input/ludb/ludb/1.0.1
  python inspect_wfdb_dataset.py --root /kaggle/input/qt-database/qtdb/1.0.0
  python inspect_wfdb_dataset.py --root /kaggle/input/european-st-t-database-100/european-st-t-database-1.0.0
"""

import argparse
from collections import Counter
from pathlib import Path

import wfdb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root)
    records = sorted(p.with_suffix("").relative_to(root) for p in root.rglob("*.hea"))
    print(f"root={root}")
    print(f"records={len(records)}")

    suffixes = Counter(p.suffix for p in root.rglob("*") if p.is_file())
    print("file suffixes:", dict(sorted(suffixes.items())))

    for rel in records[:args.limit]:
        base = root / rel
        rec = wfdb.rdrecord(str(base))
        print(f"\n{rel}: fs={rec.fs} n_sig={rec.n_sig} sig_len={rec.sig_len} leads={rec.sig_name}")
        anns = []
        for p in base.parent.glob(base.name + ".*"):
            ext = p.suffix[1:]
            if ext in {"hea", "dat", "mat", "xws"}:
                continue
            try:
                ann = wfdb.rdann(str(base), ext)
                symbols = Counter(ann.symbol)
                notes = Counter(n for n in ann.aux_note if n)
                anns.append((ext, len(ann.sample), symbols.most_common(8), notes.most_common(8)))
            except Exception:
                pass
        for ext, n, symbols, notes in anns:
            print(f"  ann={ext} n={n} symbols={symbols} notes={notes}")


if __name__ == "__main__":
    main()
