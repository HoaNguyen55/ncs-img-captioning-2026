#!/usr/bin/env python
"""Regenerate every figure for a training run.

    python scripts/plot_run.py <run_dir>
    python scripts/plot_run.py <run_dir> --outdir research/figures

Reads `metrics.jsonl` written by `rescap.ExperimentRun.log_metrics` and emits
loss curves and caption-metric curves as PDF + PNG. Every figure in the paper
must be reproducible by a script — never edited by hand — so this is the only
sanctioned way to produce them.

Figure names are prefixed with the experiment name from `run_metadata.json`, so
figures from different runs do not overwrite each other.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rescap.viz import plot_loss_curves, plot_metric_curves, read_metrics  # noqa: E402

CAPTION_METRICS = ("BLEU-4", "CIDEr", "METEOR", "ROUGE-L", "SPICE", "BLEU-1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="logs/<timestamp>_<name>")
    parser.add_argument("--outdir", default=None, help="default: research/figures")
    parser.add_argument("--prefix", default=None, help="override the figure name prefix")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise SystemExit(f"no metrics.jsonl in {run_dir}")

    rows = read_metrics(metrics_path)
    if not rows:
        raise SystemExit(f"{metrics_path} is empty — the run logged no epochs")

    prefix = args.prefix
    if prefix is None:
        meta_path = run_dir / "run_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        prefix = meta.get("experiment", run_dir.name)

    print(f"[plot] {len(rows)} epochs from {run_dir.name}")

    written = [plot_loss_curves(rows, name=f"{prefix}_loss", outdir=args.outdir)]

    present = [m for m in CAPTION_METRICS if any(r.get(m) is not None for r in rows)]
    if present:
        written.append(
            plot_metric_curves(rows, metrics=present, name=f"{prefix}_metrics", outdir=args.outdir)
        )
        print(f"[plot] caption metrics plotted: {', '.join(present)}")
    else:
        # Not an error: metrics are only computed every `every_n_epochs`, so a
        # short run may legitimately have none yet.
        print("[plot] no caption metrics logged yet — skipping the metric panel")

    # A compact text summary beside the figures, so a run's headline numbers do
    # not have to be re-derived by eye from a plot.
    evaluated = [r for r in rows if any(r.get(m) is not None for m in CAPTION_METRICS)]
    if evaluated:
        best = max(evaluated, key=lambda r: r.get("CIDEr") or float("-inf"))
        print("\n  best epoch by CIDEr:")
        for key in ("epoch", "train_loss", "val_loss", *present):
            value = best.get(key)
            if value is not None:
                shown = f"{value:.4f}" if isinstance(value, float) else value
                print(f"    {key:<12}{shown}")

    print("\n[plot] wrote:")
    for path in written:
        print(f"  {path}  (+ .png)")


if __name__ == "__main__":
    main()
