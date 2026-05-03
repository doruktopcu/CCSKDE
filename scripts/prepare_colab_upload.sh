#!/usr/bin/env bash
# Creates zip files to upload to Google Drive for Colab training.
# Run from the project root: bash scripts/prepare_colab_upload.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=== Creating CCSKDE upload zips ==="
echo "Project root: ${PROJECT_ROOT}"

# ── Zip 1: Code + poses + GT + YOLO weights (~2.7 GB) ──
echo ""
echo "[1/3] Zipping code + poses + GT + yolo weights..."
zip -r ccskde_code_and_poses.zip \
    seeker/ \
    ccskde/ \
    scripts/ \
    requirements.txt \
    yolo11n.pt \
    data/ShanghaiTech/pose/ \
    data/ShanghaiTech/gt/ \
    -x "*.pyc" "*__pycache__*" "*.DS_Store" ".git/*" ".venv/*"

echo "  → $(du -sh ccskde_code_and_poses.zip | cut -f1)"

# ── Zip 2: Test frames (~4.3 GB) ──
echo ""
echo "[2/3] Zipping test frames..."
zip -r ccskde_test_frames.zip \
    data/ShanghaiTech/shanghaitech/testing/frames/ \
    -x "*.DS_Store"

echo "  → $(du -sh ccskde_test_frames.zip | cut -f1)"

# ── Zip 3: Training videos (~2.3 GB) ──
echo ""
echo "[3/3] Zipping training videos..."
zip -r ccskde_train_videos.zip \
    data/ShanghaiTech/shanghaitech/training/videos/ \
    -x "*.DS_Store"

echo "  → $(du -sh ccskde_train_videos.zip | cut -f1)"

echo ""
echo "=== Done! Upload these 3 files to Google Drive under a folder called CCSKDE/ ==="
echo "  1. ccskde_code_and_poses.zip"
echo "  2. ccskde_test_frames.zip"
echo "  3. ccskde_train_videos.zip"
echo ""
echo "Then open the Colab notebook (scripts/ccskde_colab.ipynb) and run all cells."
