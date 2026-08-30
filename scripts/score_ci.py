#!/usr/bin/env python
"""Khoảng tin cậy bootstrap + kiểm định cặp cho CIDEr và CHAIR_i.

    python scripts/score_ci.py \\
        --a ~/ncs-data/results/zeroshot-short.preds.json --name-a "zero-shot" \\
        --b ~/ncs-data/results/chungcat-short.preds.json --name-b "chưng cất"

Vì sao tồn tại : chênh vài điểm CIDEr trên 558 ảnh mà không có
khoảng tin cậy là chỗ phản biện IEEE soi đầu tiên. Mọi kết luận "hơn/kém"
trong bài phải có ±CI và kiểm định đi kèm.

Cách làm — bootstrap CẶP trên ảnh (B=5000, seed cố định):

* CIDEr: IDF tính MỘT lần trên toàn bộ tham chiếu (đúng như số công bố), rồi
  tái lấy mẫu vector điểm-từng-ảnh. Điểm corpus của pycocoevalcap là trung
  bình điểm từng ảnh nên cách này tái tạo đúng con số gốc.
* CHAIR_i: tái lấy mẫu cặp (số vật thể bịa, số vật thể nhắc) từng ảnh, lấy
  tỷ số trên mẫu — KHÔNG lấy trung bình các tỷ số từng ảnh (ảnh ít vật thể
  sẽ bị phóng đại).
* Cặp: hai hệ thống dùng CHUNG chỉ số tái lấy mẫu — đúng phép kiểm cho câu
  hỏi "trên cùng những ảnh này, A có hơn B không".
* p hai phía = 2·min(P(hiệu ≤ 0), P(hiệu ≥ 0)), chặn dưới 2/B — bootstrap
  không bao giờ cho p đúng bằng 0.

Chạy một hệ thống (chỉ --a) thì ra CI của riêng nó, không có kiểm định.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

B_DEFAULT = 5000
SEED = 42
SCALE = 100.0  # thang công bố, như evaluate.py


def load_preds(path: str) -> dict[str, list[str]]:
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "predictions" in raw:  # file kết quả đầy đủ
        raw = raw["predictions"]
    return {str(k): (v if isinstance(v, list) else [v]) for k, v in raw.items()}


def per_image_cider(gt: dict, preds: dict, ids: list[str], segmenter: str):
    """Vector điểm CIDEr từng ảnh, IDF trên toàn tập — cùng đường tiền xử lý
    với evaluate.py (align + tách từ) để con số khớp bảng chính."""
    from pycocoevalcap.cider.cider import Cider

    from rescap.metrics import CaptionMetrics

    scorer = CaptionMetrics(language="vi", tokenize=True, segmenter=segmenter)
    gts, res = scorer._align(gt, {i: preds[i] for i in ids})
    gts, res = scorer._tokenize(gts, res)
    # Bộ tách rơi cấp là chết, không phải cảnh báo: số âm tiết lệch số mức từ
    # ~7 điểm CIDEr mà không báo lỗi nào — đúng lớp bug đã cắn ta cả tuần.
    fallback = [w for w in scorer.warnings if "falling back" in w or "unavailable" in w]
    if fallback:
        raise SystemExit(
            "⛔ bộ tách từ rơi về chế độ khác — số CIDEr sẽ KHÔNG so được với "
            "bảng chính. Sửa môi trường trước, không có cờ để bỏ qua.\n  "
            + "\n  ".join(fallback)
        )
    for w in scorer.warnings:
        print(f"  ⚠ {w}")
    order = list(gts.keys())
    corpus, per_image = Cider().compute_score(gts, res)
    if len(order) != len(per_image):
        raise RuntimeError("số điểm từng ảnh không khớp số ảnh — không tin được")
    return float(corpus), dict(zip(order, [float(s) for s in per_image]))


def per_image_chair(gt: dict, preds: dict, ids: list[str]):
    """{image_id: (số bịa, số nhắc)} từ cùng bộ đếm với bảng chính."""
    from rescap.chair import chair

    result = chair({i: preds[i][0] for i in ids}, gt, strict_ids=False)
    out = {}
    for row in result.per_caption:
        out[str(row["image_id"])] = (len(row["hallucinated"]), row["n_mentions"])
    return result, out


def bootstrap(ids, samplers, b, rng):
    """samplers: {tên: hàm(ids đã tái lấy mẫu) -> giá trị}. Trả về
    {tên: [giá trị mỗi vòng]} — mọi hệ thống dùng CHUNG một mẫu (cặp)."""
    import numpy as np

    ids = list(ids)
    draws = {name: np.empty(b) for name in samplers}
    for round_i in range(b):
        sample = [ids[j] for j in rng.integers(0, len(ids), size=len(ids))]
        for name, fn in samplers.items():
            draws[name][round_i] = fn(sample)
    return draws


def ci95(values):
    import numpy as np

    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def p_two_sided(diffs) -> float:
    import numpy as np

    b = len(diffs)
    lo = float(np.mean(diffs <= 0))
    hi = float(np.mean(diffs >= 0))
    return max(2 * min(lo, hi), 2 / b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="file dự đoán hệ A")
    parser.add_argument("--b", default=None, help="file dự đoán hệ B (so cặp)")
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--split", default="test")
    parser.add_argument("--segmenter", default="rdrsegmenter")
    parser.add_argument("--rounds", type=int, default=B_DEFAULT)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--allow-subset", action="store_true",
        help="chấm phần giao thay vì từ chối — CHỈ để thử hạ tầng; so cặp "
             "trên tập con vẫn đúng phép kiểm cho chính các ảnh đó, nhưng "
             "con số KHÔNG so được với bảng 558 ảnh",
    )
    args = parser.parse_args()

    import numpy as np

    from evaluate import references  # cùng nguồn tham chiếu với bảng chính

    refs = references(args.split)
    systems = {args.name_a: load_preds(args.a)}
    if args.b:
        systems[args.name_b] = load_preds(args.b)

    # Giao tập ảnh: chỉ chấm ảnh mọi hệ đều có — và phải là TOÀN BỘ split,
    # cùng quy tắc từ chối tập con của evaluate.py.
    ids = sorted(set(refs))
    for name, preds in systems.items():
        missing = [i for i in ids if i not in preds]
        if missing and not args.allow_subset:
            raise SystemExit(
                f"hệ '{name}' thiếu {len(missing)}/{len(ids)} ảnh của split "
                f"(vd {missing[:3]}) — không chấm tập con, số sẽ không so được"
            )
        if missing:
            ids = [i for i in ids if i in preds]
    if args.allow_subset and len(ids) < len(refs):
        print(f"  ⚠⚠ TẬP CON {len(ids)}/{len(refs)} ảnh (--allow-subset) — "
              f"số dưới đây KHÔNG so được với bảng {len(refs)} ảnh ⚠⚠")
    gt = {i: list(refs[i]) for i in ids}

    print(f"  {len(ids)} ảnh · {args.rounds} vòng bootstrap · seed {SEED}\n")
    cider_pi, chair_pi, point = {}, {}, {}
    for name, preds in systems.items():
        corpus, pi = per_image_cider(gt, preds, ids, args.segmenter)
        chair_res, ch = per_image_chair(gt, preds, ids)
        cider_pi[name], chair_pi[name] = pi, ch
        point[name] = {"CIDEr": corpus * SCALE,
                       "CHAIR_i": (chair_res.chair_i or 0.0) * 100}

    rng = np.random.default_rng(SEED)
    samplers = {}
    for name in systems:
        pi, ch = cider_pi[name], chair_pi[name]
        samplers[f"CIDEr/{name}"] = (
            lambda sample, pi=pi: sum(pi[i] for i in sample) / len(sample) * SCALE)
        samplers[f"CHAIR_i/{name}"] = (
            lambda sample, ch=ch:
            (lambda h, m: h / m * 100 if m else 0.0)(
                sum(ch.get(i, (0, 0))[0] for i in sample),
                sum(ch.get(i, (0, 0))[1] for i in sample)))
    draws = bootstrap(ids, samplers, args.rounds, rng)

    report = {"n_images": len(ids), "rounds": args.rounds, "seed": SEED,
              "split": args.split, "metrics": {}}
    for metric in ("CIDEr", "CHAIR_i"):
        print(f"  === {metric} (thang ×100) ===")
        for name in systems:
            lo, hi = ci95(draws[f"{metric}/{name}"])
            print(f"    {name:<22} {point[name][metric]:>7.1f}  "
                  f"[95% CI {lo:.1f} – {hi:.1f}]")
            report["metrics"].setdefault(metric, {})[name] = {
                "point": point[name][metric], "ci95": [lo, hi]}
        if args.b:
            d = (draws[f"{metric}/{args.name_b}"]
                 - draws[f"{metric}/{args.name_a}"])
            lo, hi = ci95(d)
            p = p_two_sided(d)
            delta = point[args.name_b][metric] - point[args.name_a][metric]
            verdict = "CÓ Ý NGHĨA" if (lo > 0 or hi < 0) else "KHÔNG kết luận được"
            print(f"    hiệu (B−A){'':<12} {delta:>+7.1f}  "
                  f"[95% CI {lo:+.1f} – {hi:+.1f}]  p≈{p:.4f}  → {verdict}")
            report["metrics"][metric]["diff_b_minus_a"] = {
                "point": delta, "ci95": [lo, hi], "p_two_sided": p}
        print()

    if args.b:
        print("  Đọc kết quả: 'CÓ Ý NGHĨA' nghĩa là CI 95% của hiệu không chứa 0")
        print("  trên chính tập ảnh này — không phải khẳng định cho mọi tập ảnh.")
    out = Path(args.out or Path(args.a).expanduser().parent / "score_ci.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  đã ghi {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
