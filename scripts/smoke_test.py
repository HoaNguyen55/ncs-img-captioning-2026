#!/usr/bin/env python
"""Kiểm nhanh môi trường KHÔNG cần GPU — chạy sau setup.sh.

    python scripts/smoke_test.py

PASS đủ 4 mục nghĩa là: import thư viện lõi OK, phiên bản ghim đúng,
kết xuất khuôn luật chạy được trên bản ghi mẫu, và bộ đếm CHAIR-vi hoạt động.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ok = 0
# 1) phiên bản ghim
import transformers
assert transformers.__version__ == "4.51.3", f"transformers {transformers.__version__} != 4.51.3 (BẮT BUỘC)"
print("1) pin transformers 4.51.3 OK"); ok += 1
# 2) import lõi
from rescap.pipeline.realize import RealizeConfig, realize
from rescap.pipeline.select import SelectionConfig, select
from rescap.pipeline.verify import verdict_name
from rescap.chair import chair, objects_in
print("2) import rescap OK"); ok += 1
# 3) kết xuất khuôn luật trên bản ghi mẫu (CPU)
rec = json.loads((ROOT/"data/stage1_records/sample_00000000833.json").read_text(encoding="utf-8"))
props = [p for p in rec["propositions"] if verdict_name(p) == "SUPPORTED"]
sel = select(props, rec["entities"], config=SelectionConfig(budget=9))
keep = {str(i) for i in sel.selected_ids}
cap = realize([p for p in props if str(p.get("id")) in keep], rec["entities"],
              config=RealizeConfig(strategy="template"))
text = cap.caption.get("text_vi") if isinstance(cap.caption, dict) else cap.caption.text_vi
assert text and len(text.split()) > 5, "kết xuất rỗng"
print(f"3) kết xuất mẫu OK: {text[:80]}…"); ok += 1
# 4) CHAIR-vi
res = chair({"1": text}, {"1": ["một cái cửa màu nâu", "một ngôi nhà"]}, strict_ids=False)
print(f"4) CHAIR-vi OK (CHAIR_s={res.chair_s})"); ok += 1
print(f"=== SMOKE TEST PASS {ok}/4 ===")
