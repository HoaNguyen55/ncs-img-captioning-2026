#!/bin/bash
# Cài môi trường VSPS — chạy từ thư mục gốc artifact. Yêu cầu: Python 3.10+, JDK (cho VnCoreNLP).
set -e
export NCS_DATA=${NCS_DATA:-$PWD/ncs-data}
mkdir -p "$NCS_DATA/datasets/ktvic/images" "$NCS_DATA/results" "$NCS_DATA/runs" "$NCS_DATA/vncorenlp"
python3 -m venv .venv && . .venv/bin/activate && pip install -q --upgrade pip
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
