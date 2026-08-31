#!/usr/bin/env python
"""Bootstrap confidence intervals + paired tests for CIDEr and CHAIR_i.

    python scripts/score_ci.py \\
        --a ~/ncs-data/results/zeroshot-short.preds.json --name-a "zero-shot" \\
        --b ~/ncs-data/results/chungcat-short.preds.json --name-b "distilled"

Why this exists: a few CIDEr points of difference over 558 images without a
confidence interval is the first thing an IEEE reviewer pokes at. Every
"better/worse" conclusion in the paper must come with a ±CI and a test.

How — PAIRED bootstrap over images (B=5000, fixed seed):

* CIDEr: IDF is computed ONCE over all references (exactly as the published
  number), then the per-image score vector is resampled. pycocoevalcap's corpus
  score is the mean of per-image scores, so this reproduces the original figure.
* CHAIR_i: resample the per-image pair (invented-object count, mentioned-object
  count) and take the ratio over the sample — NOT the mean of per-image ratios
  (images with few objects would be blown up).
* Paired: both systems use the SAME resampled indices — the right test for the
  question "on these same images, is A better than B".
* Two-sided p = 2·min(P(diff ≤ 0), P(diff ≥ 0)), floored at 2/B — the bootstrap
  never yields a p of exactly 0.

Running one system (only --a) gives its own CI, with no test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

B_DEFAULT = 5000
SEED = 42
SCALE = 100.0  # the published scale, as in evaluate.py


def load_preds(path: str) -> dict[str, list[str]]:
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "predictions" in raw:  # a full results file
        raw = raw["predictions"]
    return {str(k): (v if isinstance(v, list) else [v]) for k, v in raw.items()}


def per_image_cider(gt: dict, preds: dict, ids: list[str], segmenter: str):
    """Per-image CIDEr score vector, IDF over the full set — same preprocessing
    path as evaluate.py (align + word segmentation) so the figure matches the main table."""
    from pycocoevalcap.cider.cider import Cider

    from rescap.metrics import CaptionMetrics

    scorer = CaptionMetrics(language="vi", tokenize=True, segmenter=segmenter)
    gts, res = scorer._align(gt, {i: preds[i] for i in ids})
    gts, res = scorer._tokenize(gts, res)
    # A segmenter falling back is fatal, not a warning: syllable-level numbers
    # drift ~7 CIDEr points from word-level without any error — exactly the class
    # of bug that bit us for a week.
    fallback = [w for w in scorer.warnings if "falling back" in w or "unavailable" in w]
    if fallback:
        raise SystemExit(
            "⛔ the word segmenter fell back to another mode — CIDEr numbers will "
            "NOT be comparable to the main table. Fix the environment first; there "
            "is no flag to skip this.\n  "
            + "\n  ".join(fallback)
        )
    for w in scorer.warnings:
        print(f"  ⚠ {w}")
    order = list(gts.keys())
    corpus, per_image = Cider().compute_score(gts, res)
    if len(order) != len(per_image):
        raise RuntimeError("per-image score count does not match image count — untrustworthy")
    return float(corpus), dict(zip(order, [float(s) for s in per_image]))


def per_image_chair(gt: dict, preds: dict, ids: list[str]):
    """{image_id: (invented count, mention count)} from the same counter as the main table."""
    from rescap.chair import chair

    result = chair({i: preds[i][0] for i in ids}, gt, strict_ids=False)
    out = {}
    for row in result.per_caption:
        out[str(row["image_id"])] = (len(row["hallucinated"]), row["n_mentions"])
    return result, out


def bootstrap(ids, samplers, b, rng):
    """samplers: {name: fn(resampled ids) -> value}. Returns
    {name: [value per round]} — every system uses the SAME sample (paired)."""
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
    parser.add_argument("--a", required=True, help="predictions file for system A")
    parser.add_argument("--b", default=None, help="predictions file for system B (paired comparison)")
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--split", default="test_data.json")
    parser.add_argument("--segmenter", default="rdrsegmenter")
    parser.add_argument("--rounds", type=int, default=B_DEFAULT)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--allow-subset", action="store_true",
        help="score the intersection instead of refusing — infrastructure testing "
             "ONLY; a paired comparison on a subset is still a valid test for those "
             "very images, but the number is NOT comparable to the 558-image table",
    )
    args = parser.parse_args()

    import numpy as np

    from evaluate import references  # same reference source as the main table

    refs = references(args.split)
    systems = {args.name_a: load_preds(args.a)}
    if args.b:
        systems[args.name_b] = load_preds(args.b)

    # Image-set intersection: only score images every system has — and it must be
    # the WHOLE split, same subset-refusal rule as evaluate.py.
    ids = sorted(set(refs))
    for name, preds in systems.items():
        missing = [i for i in ids if i not in preds]
        if missing and not args.allow_subset:
            raise SystemExit(
                f"system '{name}' is missing {len(missing)}/{len(ids)} images of the split "
                f"(e.g. {missing[:3]}) — no subset scoring, the number would not be comparable"
            )
        if missing:
            ids = [i for i in ids if i in preds]
    if args.allow_subset and len(ids) < len(refs):
        print(f"  ⚠⚠ SUBSET {len(ids)}/{len(refs)} images (--allow-subset) — "
              f"the numbers below are NOT comparable to the {len(refs)}-image table ⚠⚠")
    gt = {i: list(refs[i]) for i in ids}

    print(f"  {len(ids)} images · {args.rounds} bootstrap rounds · seed {SEED}\n")
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
        print(f"  === {metric} (×100 scale) ===")
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
            verdict = "SIGNIFICANT" if (lo > 0 or hi < 0) else "INCONCLUSIVE"
            print(f"    diff (B−A){'':<12} {delta:>+7.1f}  "
                  f"[95% CI {lo:+.1f} – {hi:+.1f}]  p≈{p:.4f}  → {verdict}")
            report["metrics"][metric]["diff_b_minus_a"] = {
                "point": delta, "ci95": [lo, hi], "p_two_sided": p}
        print()

    if args.b:
        print("  Reading the result: 'SIGNIFICANT' means the 95% CI of the difference excludes 0")
        print("  on this very image set — not a claim about every image set.")
    out = Path(args.out or Path(args.a).expanduser().parent / "score_ci.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
