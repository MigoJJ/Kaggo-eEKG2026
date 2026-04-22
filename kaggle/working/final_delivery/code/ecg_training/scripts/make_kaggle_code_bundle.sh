#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUNDLE_DIR="${ROOT_DIR}/dist"
BUNDLE_PATH="${BUNDLE_DIR}/ecg_training_kaggle_bundle.zip"

mkdir -p "${BUNDLE_DIR}"
rm -f "${BUNDLE_PATH}"

cd "${ROOT_DIR}"
zip -r "${BUNDLE_PATH}" \
  ecg_training \
  -x "ecg_training/.venv/*" \
  -x "ecg_training/**/__pycache__/*" \
  -x "runs/*" \
  -x "data/*" \
  -x "*.pyc"

echo "Created ${BUNDLE_PATH}"
