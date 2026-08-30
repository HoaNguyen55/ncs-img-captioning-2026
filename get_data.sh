#!/bin/bash
# Tải dữ liệu: KTVIC (annotation từ repo gốc + ảnh từ Google Drive chính thức của bộ dữ liệu).
set -e
export NCS_DATA=${NCS_DATA:-$PWD/ncs-data}
KT=$NCS_DATA/datasets/ktvic
# Annotation: theo hướng dẫn repo KTVIC (github.com/anhtu293/ktvic hoặc trang bài báo KTVIC FAIR'23)
echo ">> Đặt train_data.json + test_data.json vào $KT/ (tải từ repo chính thức KTVIC)"
# Ảnh (train ~563MB + test 77MB) — Google Drive folder chính thức của KTVIC:
python -m gdown --folder 16e8cd3AKusPS1H-h55JModiINM0og-ke -O /tmp/ktvic_zip || true
python - <<'PY'
import zipfile, glob, os
dst=os.environ["NCS_DATA"]+"/datasets/ktvic/images"
for z in glob.glob("/tmp/ktvic_zip/*.zip"):
    with zipfile.ZipFile(z) as f:
        for m in f.namelist():
            if m.lower().endswith((".jpg",".jpeg",".png")):
                open(os.path.join(dst, os.path.basename(m)),"wb").write(f.read(m))
print("ảnh KTVIC:", len(os.listdir(dst)))
PY
# COCO probe (chỉ cần cho thí nghiệm xuyên ngôn ngữ):
# python scripts/download_coco_images.py --out $NCS_DATA/coco_images
# + instances_val2014.json & captions_val2014.json (trang COCO) + dataset_coco.json (Karpathy)
