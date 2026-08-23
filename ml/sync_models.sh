#!/usr/bin/env bash
# Copies validated model artifacts from ml/models/ (git-tracked source of truth)
# to the root models/ directory (gitignored, actually read by the Java pipeline).
# Run this after every export step, before running the Java app or tests.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/models"
DEST_DIR="${SCRIPT_DIR}/../models"

if [ ! -d "$SRC_DIR" ]; then
  echo "error: source dir not found: $SRC_DIR" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

copied=0
skipped=0
for src in "$SRC_DIR"/*.onnx "$SRC_DIR"/*.json "$SRC_DIR"/*.pt; do
  [ -e "$src" ] || continue
  base="$(basename "$src")"
  dest="$DEST_DIR/$base"

  if [ -f "$dest" ] && cmp -s "$src" "$dest"; then
    skipped=$((skipped + 1))
    continue
  fi

  if [ -f "$dest" ]; then
    echo "warn: $base differs from deployed copy, overwriting (src is source of truth)" >&2
  fi
  cp -p "$src" "$dest"
  echo "synced: $base"
  copied=$((copied + 1))
done

echo "done: ${copied} copied, ${skipped} already up to date"
