#!/bin/bash
# Download the data: KTVIC (annotations from the original repo + images from the dataset's official Google Drive).
set -e
export NCS_DATA=${NCS_DATA:-$PWD/ncs-data}
KT=$NCS_DATA/datasets/ktvic
# Annotations: per the KTVIC repo instructions (github.com/anhtu293/ktvic or the KTVIC FAIR'23 paper page)
# annotations (train_data.json/test_data.json) come with the same Drive folder and are copied below
# Images (train ~563MB + test 77MB) — KTVIC's official Google Drive folder:
python -m gdown --folder 16e8cd3AKusPS1H-h55JModiINM0og-ke -O /tmp/ktvic_zip || true
python - <<'PY'
import zipfile, glob, os
dst=os.environ["NCS_DATA"]+"/datasets/ktvic/images"
for z in glob.glob("/tmp/ktvic_zip/*.zip"):
    with zipfile.ZipFile(z) as f:
        for m in f.namelist():
            if m.lower().endswith((".jpg",".jpeg",".png")):
                open(os.path.join(dst, os.path.basename(m)),"wb").write(f.read(m))
print("KTVIC images:", len(os.listdir(dst)))
for j in ("train_data.json", "test_data.json"):
    src = os.path.join("/tmp/ktvic_zip", j)
    if os.path.exists(src):
        import shutil; shutil.copy(src, os.path.join(os.path.dirname(dst), j))
print("annotation:", [f for f in os.listdir(os.path.dirname(dst)) if f.endswith(".json")])

PY
# COCO probe (only needed for the cross-lingual experiment):
# python scripts/download_coco_images.py --out $NCS_DATA/coco_images
# + instances_val2014.json & captions_val2014.json (COCO site) + dataset_coco.json (Karpathy)
