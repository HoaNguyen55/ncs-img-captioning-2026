#!/usr/bin/env python
"""Annotation tool — image beside the input fields, output in SVP schema.

    python scripts/annotate.py --annotator A --split pilot
    python scripts/annotate.py --annotator B --share      # public link

  type `SUPORTED` and only fail hours later during conversion. Here the verdict
  is a radio button and the type is a dropdown, so the invalid state is not
  reachable.
* **Autosave every 30 s** plus a rotating backup, because the server shares a
  machine with training runs and may be restarted.
* Output is written straight into `configs/proposition_schema.json` shape, so
  nothing has to be converted later.

Everything the annotator has to remember is on screen: the guideline's seven
rules, the three-way verdict criteria, and the `xanh` rule.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA = Path(os.environ.get("NCS_DATA", Path.home() / "ncs-data"))
KTVIC = DATA / "datasets" / "ktvic"
OUT_ROOT = DATA / "annotations"
BACKUP_ROOT = DATA / "annotations-backup"

PROP_TYPES = [
    ("entity", "Đối tượng — có gì trong ảnh"),
    ("attribute", "Thuộc tính — màu, kích thước, chất liệu, trạng thái"),
    ("action", "Hành động — đang làm gì"),
    ("relation", "Quan hệ — cầm, mặc, đẩy…"),
    ("spatial_relation", "Vị trí — trên, dưới, bên cạnh, phía sau"),
    ("counting", "Số lượng"),
    ("scene", "Bối cảnh — nơi chốn, thời gian"),
    ("interaction", "Tương tác — hai bên cùng tham gia"),
]

VERDICTS = ["SUPPORTED", "UNCERTAIN", "REJECTED"]

REJECT_REASONS = [
    "mâu thuẫn với ảnh",
    "không thể xác định từ ảnh",
]

ADVERSARIAL_KINDS = [
    "bịa đối tượng", "sai thuộc tính", "nhầm xanh dương/xanh lá",
    "sai quan hệ", "sai vị trí", "sai số lượng",
    "bịa giới tính", "suy đoán mục đích", "bịa địa điểm",
]

RULES_MD = """
### Bảy nguyên tắc

1. **Chỉ ghi cái NHÌN THẤY.** "đang cười" ✓ · "đang vui" ✗ (nội tâm)
   "mặc áo blouse trắng" ✓ · "là bác sĩ" ✗ (nghề nghiệp)
2. **Không đoán giới tính** — không rõ mặt thì dùng **"người"**
3. **`xanh` phải rõ** → `xanh dương` / `xanh lá`; không phân biệt được → UNCERTAIN
4. **Loại từ đúng** — con chó, chiếc xe đạp, cái bàn, người đàn ông
5. **Tính từ SAU danh từ** — "áo đỏ" ✓ · "đỏ áo" ✗
6. **Mỗi mệnh đề một thông tin** — tách "mặc áo đỏ đang đạp xe" thành 2
7. **Không chắc → UNCERTAIN.** "Không xác định được" ≠ "sai"

### Ba nhãn

| | Khi nào | Tự hỏi |
|---|---|---|
| **SUPPORTED** | Nhìn ảnh là thấy rõ | *Chỉ tay vào ảnh chứng minh được không?* |
| **UNCERTAIN** | Có thể đúng, ảnh không đủ rõ | *Tôi có phải đoán không?* |
| **REJECTED** | Ảnh ngược lại, HOẶC vốn không nhìn thấy được | *Mâu thuẫn, hay vốn không quan sát được?* |
"""


def load_manifest(split: str) -> list[dict[str, Any]]:
    """Image list for this split, written by select_pilot.py."""
    path = KTVIC / f"{split}_manifest.json"
    if not path.exists():
        raise SystemExit(
            f"No manifest at {path}\n"
            f"Create it first:  python scripts/select_pilot.py --split {split}"
        )
    return json.loads(path.read_text(encoding="utf-8"))["images"]


class Store:
    """Per-annotator JSON store with autosave and rotating backups."""

    def __init__(self, annotator: str, split: str, autosave_s: int = 30):
        self.annotator = annotator
        self.split = split
        self.autosave_s = autosave_s
        self.dir = OUT_ROOT / split
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{annotator}.json"
        self.data: dict[str, Any] = (
            json.loads(self.path.read_text(encoding="utf-8"))
            if self.path.exists()
            else {"annotator": annotator, "split": split, "images": {}}
        )
        self._last_save = 0.0

    def reload(self):
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass

    def get(self, image_id: str) -> dict[str, Any]:
        self.reload()
        return self.data["images"].setdefault(
            image_id, {"propositions": [], "adversarial": [], "notes": "", "done": False}
        )

    def save(self, force: bool = False) -> str:
        now = time.time()
        if not force and now - self._last_save < self.autosave_s:
            return ""
        self.data["updated_utc"] = datetime.now(timezone.utc).isoformat()
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Rotating backup: the server shares a box with training runs, and a
        # single file is one bad restart away from losing a day of annotation.
        backup_dir = BACKUP_ROOT / self.split
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        shutil.copy2(self.path, backup_dir / f"{self.annotator}-{stamp}.json")
        backups = sorted(backup_dir.glob(f"{self.annotator}-*.json"))
        for old in backups[:-20]:  # keep the last 20
            old.unlink(missing_ok=True)
        self._last_save = now
        return f"đã lưu {datetime.now():%H:%M:%S}"

    def progress(self, total: int) -> str:
        done = sum(1 for v in self.data["images"].values() if v.get("done"))
        props = sum(len(v["propositions"]) for v in self.data["images"].values())
        return f"**{done}/{total} ảnh xong** · {props} mệnh đề"


def build_ui(annotator: str, split: str, autosave_s: int):
    import gradio as gr

    images = load_manifest(split)
    store = Store(annotator, split, autosave_s)
    index = {"i": 0}

    def image_path(i: int) -> str:
        return str(KTVIC / "images" / images[i]["file_name"])

    def load_view(i: int):
        i = max(0, min(i, len(images) - 1))
        index["i"] = i
        rec = store.get(images[i]["image_id"])
        meta = images[i]
        header = (
            f"### Ảnh {i + 1}/{len(images)} · `{meta['file_name']}`"
            f"  ·  độ khó ước lượng: **{meta.get('difficulty', '?')}**"
        )
        return (
            image_path(i),
            header,
            prop_table(rec["propositions"]),
            adv_table(rec["adversarial"]),
            rec.get("notes", ""),
            rec.get("done", False),
            store.progress(len(images)),
        )

    def prop_table(props: list[dict]) -> list[list]:
        return [
            [p["type"], p["text_vi"], p["verdict"], p.get("reject_reason", "")]
            for p in props
        ]

    def adv_table(advs: list[dict]) -> list[list]:
        return [[a["kind"], a["text_vi"]] for a in advs]

    def add_prop(ptype, text, verdict, reason):
        text = (text or "").strip()
        if not text:
            return prop_table(store.get(images[index["i"]]["image_id"])["propositions"]), \
                   "⚠ chưa nhập nội dung mệnh đề", ""
        if verdict == "REJECTED" and not reason:
            return prop_table(store.get(images[index["i"]]["image_id"])["propositions"]), \
                   "⚠ REJECTED phải chọn lý do", text
        rec = store.get(images[index["i"]]["image_id"])
        rec["propositions"].append(
            {
                "type": ptype.split(" —")[0],
                "text_vi": text,
                "verdict": verdict,
                "reject_reason": reason if verdict == "REJECTED" else "",
                "annotator": annotator,
                "added_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        msg = store.save() or "đã thêm"
        return prop_table(rec["propositions"]), msg, ""

    def add_adv(kind, text):
        text = (text or "").strip()
        rec = store.get(images[index["i"]]["image_id"])
        if not text:
            return adv_table(rec["adversarial"]), "⚠ chưa nhập nội dung", ""
        rec["adversarial"].append(
            {"kind": kind, "text_vi": text, "adversarial": True, "annotator": annotator}
        )
        msg = store.save() or "đã thêm mệnh đề đối kháng"
        return adv_table(rec["adversarial"]), msg, ""

    def undo_prop():
        rec = store.get(images[index["i"]]["image_id"])
        if rec["propositions"]:
            rec["propositions"].pop()
            store.save(force=True)
        return prop_table(rec["propositions"]), "đã xoá mệnh đề cuối"

    def save_meta(notes, done):
        rec = store.get(images[index["i"]]["image_id"])
        rec["notes"] = notes
        rec["done"] = bool(done)
        store.save(force=True)
        return store.progress(len(images)), "đã lưu"

    def nav(delta):
        return load_view(index["i"] + delta)

    with gr.Blocks(title=f"Gán nhãn — {annotator}") as app:
        gr.Markdown(f"# Gán nhãn mệnh đề · **{annotator}** · tập `{split}`")

        with gr.Row():
            with gr.Column(scale=5):
                img = gr.Image(type="filepath", height=560, show_label=False)
                header = gr.Markdown()
                with gr.Row():
                    prev_b = gr.Button("← Ảnh trước")
                    next_b = gr.Button("Ảnh sau →", variant="primary")
                progress = gr.Markdown()
                with gr.Accordion("Nhắc lại quy tắc", open=False):
                    gr.Markdown(RULES_MD)

            with gr.Column(scale=5):
                status = gr.Markdown("")

                gr.Markdown("#### Thêm mệnh đề")
                ptype = gr.Dropdown(
                    [f"{k} — {v}" for k, v in PROP_TYPES],
                    value=f"{PROP_TYPES[0][0]} — {PROP_TYPES[0][1]}",
                    label="Loại",
                )
                text = gr.Textbox(
                    label="Nội dung (tiếng Việt)",
                    placeholder="ví dụ: người đàn ông mặc áo đỏ",
                    lines=2,
                )
                verdict = gr.Radio(
                    VERDICTS, value="SUPPORTED", label="Kiểm chứng"
                )
                reason = gr.Dropdown(
                    REJECT_REASONS, label="Lý do (bắt buộc khi REJECTED)",
                    value=None, visible=False,
                )
                verdict.change(
                    lambda v: gr.update(visible=(v == "REJECTED")),
                    verdict, reason,
                )
                with gr.Row():
                    add_b = gr.Button("Thêm mệnh đề", variant="primary")
                    undo_b = gr.Button("Xoá cái cuối")
                props = gr.Dataframe(
                    headers=["loại", "nội dung", "nhãn", "lý do"],
                    label="Mệnh đề đã nhập", interactive=False, wrap=True,
                )

                gr.Markdown("#### Mệnh đề đối kháng (cố ý sai, nghe hợp lý) — ~2/ảnh")
                adv_kind = gr.Dropdown(ADVERSARIAL_KINDS, value=ADVERSARIAL_KINDS[0], label="Loại lỗi")
                adv_text = gr.Textbox(label="Nội dung", lines=1)
                adv_b = gr.Button("Thêm")
                advs = gr.Dataframe(
                    headers=["loại", "nội dung"], label="Đối kháng",
                    interactive=False, wrap=True,
                )

                notes = gr.Textbox(label="Ghi chú (trường hợp không có trong hướng dẫn)", lines=2)
                done = gr.Checkbox(label="Ảnh này đã xong")
                save_b = gr.Button("Lưu ảnh này", variant="primary")

        outs = [img, header, props, advs, notes, done, progress]
        app.load(lambda: load_view(0), outputs=outs)
        prev_b.click(lambda: nav(-1), outputs=outs)
        next_b.click(lambda: nav(1), outputs=outs)
        add_b.click(add_prop, [ptype, text, verdict, reason], [props, status, text])
        undo_b.click(undo_prop, outputs=[props, status])
        adv_b.click(add_adv, [adv_kind, adv_text], [advs, status, adv_text])
        save_b.click(save_meta, [notes, done], [progress, status])

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator", required=True, help="your name or initial")
    parser.add_argument("--split", default="pilot")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="public gradio link")
    parser.add_argument("--autosave", type=int, default=30, help="seconds")
    args = parser.parse_args()

    try:
        import gradio  # noqa: F401
    except ImportError:
        raise SystemExit(
            "gradio not installed:\n"
            "  uv pip install --python $NCS_VENV/bin/python gradio"
        )

    app = build_ui(args.annotator, args.split, args.autosave)
    print(f"\nlưu vào : {OUT_ROOT / args.split / (args.annotator + '.json')}")
    print(f"backup  : {BACKUP_ROOT / args.split}/  (giữ 20 bản gần nhất)")
    # gradio ≥4 chặn file ngoài cwd — không có allowed_paths thì ô ảnh trắng trơn
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share,
               allowed_paths=[str(KTVIC)])


if __name__ == "__main__":
    main()
