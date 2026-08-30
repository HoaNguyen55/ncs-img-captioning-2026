#!/usr/bin/env python
""" (nhật ký NC) audit mode — NGƯỜI phán quyết trực tiếp mệnh đề máy-SUPPORTED.

    ~/ncs-data/venv/bin/python scripts/audit_props.py --annotator hoa
    # người thứ hai, cùng lúc:  --annotator <tên> --port 7861

Khác annotate.py (người viết mệnh đề tự do), tool này ĐƯA SẴN mệnh đề mà máy
đã chấm ĐƯỢC ỦNG HỘ và chỉ hỏi một câu: nhìn ảnh, mệnh đề này đúng không?
Mục đích (góp ý phương pháp nội bộ): người viết tự do hiếm khi tự
nghĩ ra câu sai để bác, nên ô nguy-hiểm (máy ỦNG HỘ / người BÁC) bị thiếu
phơi nhiễm; đưa thẳng danh sách của máy cho người chấm là cách đo không
thiên lệch. Danh sách mệnh đề cho sẵn là HỢP LỆ — thứ bắt buộc phải là người
thật là PHÁN QUYẾT (bài học (nhật ký NC): phán quyết do AI sinh bị cấm tuyệt đối).

Autosave mỗi lần bấm + backup xoay vòng, cùng kỷ luật với annotate.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

DATA = Path(os.environ.get("NCS_DATA", Path.home() / "ncs-data"))
KTVIC = DATA / "datasets" / "ktvic"
SAMPLE = Path(__file__).resolve().parents[1] / "paper" / "data" / "annotations" / "audit_supported_sample.json"
OUT_ROOT = DATA / "annotations" / "audit"
BACKUP_ROOT = DATA / "annotations-backup" / "audit"

VERDICTS = ["SUPPORTED — nhìn ảnh thấy đúng",
            "UNCERTAIN — ảnh không đủ để chắc",
            "REJECTED — ảnh cho thấy SAI"]

RULES = """### Một câu hỏi duy nhất: *nhìn ảnh, mệnh đề này có đúng không?*
- Thấy rõ là đúng → **SUPPORTED** · Không thể kết luận từ ảnh → **UNCERTAIN** · Ảnh cho thấy sai → **REJECTED**
- Đừng suy đoán hộ máy; đừng nhớ lại ảnh khác; chỉ ảnh đang hiện.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--sample", default=None,
                    help="đường dẫn mẫu khác (vd audit_round2_sample.json (nhật ký NC))")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    import gradio as gr

    sample_path = Path(args.sample) if args.sample else SAMPLE
    sample = json.loads(sample_path.read_text(encoding="utf-8"))["items"]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / f"{args.annotator}.json"
    data = (json.loads(path.read_text(encoding="utf-8"))
            if path.exists() else {"annotator": args.annotator, "verdicts": {}})

    def key(it) -> str:
        return f"{it['image_id']}:{it['prop_id']}"

    def save():
        data["updated_utc"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        shutil.copy2(path, BACKUP_ROOT / f"{args.annotator}-{stamp}.json")
        for old in sorted(BACKUP_ROOT.glob(f"{args.annotator}-*.json"))[:-20]:
            old.unlink(missing_ok=True)

    def first_pending() -> int:
        for i, it in enumerate(sample):
            if key(it) not in data["verdicts"]:
                return i
        return len(sample) - 1

    def view(i: int):
        i = max(0, min(i, len(sample) - 1))
        it = sample[i]
        done = len(data["verdicts"])
        prev = data["verdicts"].get(key(it), {}).get("verdict")
        chosen = next((v for v in VERDICTS if v.startswith(prev or "\x00")), None)
        return (i, str(KTVIC / "images" / it["file_name"]),
                f"### {i+1}/{len(sample)} · `{it['file_name']}` · loại: {it['type']}\n# «{it['text_vi']}»",
                chosen, f"**đã chấm {done}/{len(sample)}**")

    def judge(i: int, verdict: str | None):
        if verdict:
            it = sample[int(i)]
            data["verdicts"][key(it)] = {
                "image_id": it["image_id"], "prop_id": it["prop_id"],
                "text_vi": it["text_vi"], "type": it["type"],
                "verdict": verdict.split(" ")[0],
                "at_utc": datetime.now(timezone.utc).isoformat(),
            }
            save()
        return view(int(i) + 1)

    with gr.Blocks(title=f"Kiểm toán — {args.annotator}") as app:
        gr.Markdown(f"# Kiểm toán mệnh đề máy · **{args.annotator}**")
        gr.Markdown(RULES)
        idx = gr.Number(visible=False)
        with gr.Row():
            img = gr.Image(type="filepath", height=520, show_label=False)
            with gr.Column():
                header = gr.Markdown()
                verdict = gr.Radio(VERDICTS, label="Phán quyết")
                nxt = gr.Button("Lưu & tiếp →", variant="primary")
                with gr.Row():
                    back = gr.Button("← Quay lại")
                    skip = gr.Button("Bỏ qua →")
                progress = gr.Markdown()
        outs = [idx, img, header, verdict, progress]
        app.load(lambda: view(first_pending()), outputs=outs)
        nxt.click(judge, [idx, verdict], outs)
        back.click(lambda i: view(int(i) - 1), idx, outs)
        skip.click(lambda i: view(int(i) + 1), idx, outs)

    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share,
               allowed_paths=[str(KTVIC)])


if __name__ == "__main__":
    main()
