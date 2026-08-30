"""Mark the 100 images that get two annotators, for Cohen's κ .

Chosen with a fixed seed and written into the manifest, not left to whoever
starts first: if the double-annotated set differs per person there is nothing to
compare, and κ needs the SAME images seen independently.

Stratified across the difficulty bands, because agreement on easy images says
nothing about the guideline. A κ computed only on the easy third would be the
most flattering number available and the least informative.
"""
import json, random, sys
from collections import defaultdict
from pathlib import Path

KT = Path("/root/ncs-data/datasets/ktvic")
m = json.loads((KT / "human_eval_manifest.json").read_text(encoding="utf-8"))
images = m["images"]
rng = random.Random(20260818)

bands = defaultdict(list)
for img in images:
    bands[img.get("difficulty", "?")].append(img)

target, picked = 100, []
for band, group in sorted(bands.items()):
    share = round(target * len(group) / len(images))
    rng.shuffle(group)
    picked.extend(group[:share])
picked = picked[:target]
ids = {i["image_id"] for i in picked}
for img in images:
    img["double_annotated"] = img["image_id"] in ids

m["double_annotated"] = sorted(ids)
m["double_seed"] = 20260818
(KT / "human_eval_manifest.json").write_text(
    json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

from collections import Counter
print(f"  đánh dấu {len(ids)} ảnh chấm đôi / {len(images)}")
print(f"  phân tầng: {dict(Counter(i['difficulty'] for i in picked))}")
print(f"  toàn tập : {dict(Counter(i['difficulty'] for i in images))}")
