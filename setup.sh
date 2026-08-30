#!/bin/bash
# Cài môi trường VSPS — chạy từ thư mục gốc artifact. Yêu cầu: Python 3.10+, JDK (cho VnCoreNLP).
set -e
# Python 3.10-3.12 required: the pinned torch-cu121 wheel line stops at cp312,
# and the transformers/trl pins are only tested there.
PYV=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "$PYV" in
  3.10|3.11|3.12) ;;
  *) echo "ERROR: Python $PYV is unsupported. Use Python 3.10-3.12 (e.g. install python3.12 and python3.12-venv)."; exit 1 ;;
esac
export NCS_DATA=${NCS_DATA:-$PWD/ncs-data}
mkdir -p "$NCS_DATA/datasets/ktvic/images" "$NCS_DATA/results" "$NCS_DATA/runs" "$NCS_DATA/vncorenlp"
python3 -m venv .venv && . .venv/bin/activate
# some distros ship venv without pip (needs the python3-venv package) - bootstrap it
python -m pip --version >/dev/null 2>&1 || python -m ensurepip --upgrade || {
  echo "ERROR: venv has no pip and ensurepip is unavailable - install the python3-venv (or python3.12-venv) package and re-run."; exit 1; }
python -m pip install -q --upgrade pip
# torch cu121 TRƯỚC (driver >=12.1; bản mặc định cu128 chết trên driver cũ)
pip install -q --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q -r requirements.txt
# VnCoreNLP + smoke tách từ (bắt buộc trước mọi lần chấm điểm)
python - <<'PY'
import py_vncorenlp, os
py_vncorenlp.download_model(save_dir=os.environ["NCS_DATA"]+"/vncorenlp")
seg = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=os.environ["NCS_DATA"]+"/vncorenlp")
assert any("phụ_nữ" in s for s in seg.word_segment("người phụ nữ")), "VnCoreNLP smoke FAILED"
print("VnCoreNLP smoke OK")
PY
echo "=== SETUP XONG. Tiếp theo: bash get_data.sh rồi python scripts/smoke_test.py ==="
