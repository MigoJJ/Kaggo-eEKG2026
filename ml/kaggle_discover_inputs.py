"""Print auto-discovered Kaggle dataset roots."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ecgml.kaggle_paths import discover_kaggle_roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/kaggle/input")
    args = parser.parse_args()

    roots = discover_kaggle_roots(args.input_root)
    payload = {k: None if v is None else str(v) for k, v in asdict(roots).items()}
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    for label, value in payload.items():
        if value is None:
            continue
        path = Path(value)
        hea_count = sum(1 for _ in path.rglob("*.hea")) if path.is_dir() else 0
        print(f"{label}: {path} hea_count={hea_count}")


if __name__ == "__main__":
    main()
