"""Figure generation (PHASE 15).

Every figure in the paper must be regenerable by a script -- never edited by
hand.  Each function here takes data (or a path to a run's `metrics.jsonl`) and
writes a PDF + PNG pair into `research/figures/`.

    PDF -> for LaTeX inclusion (vector, no resampling)
    PNG -> for quick viewing / slides
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

FIGURE_DIR = Path(__file__).resolve().parents[1] / "figures"


def _style() -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: no display in WSL
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (5.5, 3.6),
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
        }
    )


def _save(fig, name: str, outdir: str | Path | None = None) -> Path:
    outdir = Path(outdir) if outdir else FIGURE_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(outdir / f"{name}.png")
    print(f"[viz] wrote {pdf} (+ .png)")
    return pdf


def read_metrics(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Read a run's `metrics.jsonl` written by `ExperimentRun.log_metrics`."""
    rows = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
def plot_loss_curves(
    rows: Sequence[dict[str, Any]],
    name: str = "loss_curves",
    outdir: str | Path | None = None,
):
    """Training vs. validation loss.  The first figure of any experiment."""
    _style()
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in rows if "epoch" in r]
    fig, ax = plt.subplots()
    for key, label in (("train_loss", "train"), ("val_loss", "validation")):
        values = [r.get(key) for r in rows if "epoch" in r]
        if any(v is not None for v in values):
            ax.plot(epochs, values, marker="o", markersize=3, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training dynamics")
    ax.legend()
    return _save(fig, name, outdir)


def plot_metric_curves(
    rows: Sequence[dict[str, Any]],
    metrics: Sequence[str] = ("BLEU-4", "CIDEr", "METEOR", "ROUGE-L"),
    name: str = "metric_curves",
    outdir: str | Path | None = None,
):
    """Validation captioning metrics over epochs, one panel each."""
    _style()
    import matplotlib.pyplot as plt

    present = [m for m in metrics if any(r.get(m) is not None for r in rows)]
    if not present:
        raise ValueError(f"none of {metrics} found in the metric rows")

    cols = min(2, len(present))
    figrows = (len(present) + cols - 1) // cols
    fig, axes = plt.subplots(figrows, cols, figsize=(5.5 * cols, 3.2 * figrows), squeeze=False)

    for ax, metric in zip(axes.flat, present):
        pairs = [(r["epoch"], r[metric]) for r in rows if r.get(metric) is not None]
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", markersize=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(metric)
    for ax in axes.flat[len(present) :]:
        ax.set_visible(False)

    fig.tight_layout()
    return _save(fig, name, outdir)


def plot_ablation(
    results: dict[str, dict[str, float]],
    metric: str = "CIDEr",
    name: str = "ablation",
    outdir: str | Path | None = None,
):
    """Horizontal bar chart of an ablation ladder (PHASE 16).

    `results` maps variant id -> metric dict, e.g.
        {"A0 baseline": {"CIDEr": 0.52}, "A1 +attention": {"CIDEr": 0.61}}
    """
    _style()
    import matplotlib.pyplot as plt

    labels = list(results)
    values = [results[k].get(metric, 0.0) for k in labels]

    fig, ax = plt.subplots(figsize=(5.5, 0.45 * len(labels) + 1.2))
    bars = ax.barh(labels, values, color="#4C72B0", height=0.6)
    bars[int(max(range(len(values)), key=values.__getitem__))].set_color("#C44E52")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel(metric)
    ax.set_xlim(0, max(values) * 1.15 if values else 1)
    ax.invert_yaxis()
    ax.set_title(f"Ablation study ({metric})")
    ax.grid(axis="y", visible=False)
    return _save(fig, name, outdir)


def plot_qualitative(
    samples: Sequence[dict[str, Any]],
    name: str = "qualitative",
    outdir: str | Path | None = None,
    ncols: int = 3,
):
    """Grid of image + ground-truth + prediction.

    `samples`: [{"image": path_or_array, "gt": str, "pred": str}, ...]
    Used for both the qualitative results figure and the error-analysis
    appendix (PHASE 17).
    """
    _style()
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(samples)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 4.0 * nrows), squeeze=False)

    for ax, sample in zip(axes.flat, samples):
        image = sample["image"]
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        ax.imshow(image)
        ax.axis("off")
        caption = f"GT:   {sample.get('gt', '')}\nPred: {sample.get('pred', '')}"
        if sample.get("error_category"):
            caption += f"\n[{sample['error_category']}]"
        ax.set_title(caption, fontsize=7, loc="left", wrap=True)
    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.tight_layout()
    return _save(fig, name, outdir)


def plot_error_distribution(
    counts: dict[str, int],
    name: str = "error_distribution",
    outdir: str | Path | None = None,
):
    """Bar chart of error categories (PHASE 17)."""
    _style()
    import matplotlib.pyplot as plt

    items = sorted(counts.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    total = sum(values) or 1

    fig, ax = plt.subplots(figsize=(5.5, 0.4 * len(labels) + 1.2))
    ax.barh(labels, values, color="#DD8452", height=0.6)
    for i, value in enumerate(values):
        ax.text(value + total * 0.005, i, f"{value} ({value / total:.0%})", va="center", fontsize=8)
    ax.set_xlabel("number of failure cases")
    ax.invert_yaxis()
    ax.set_title("Error categories")
    ax.grid(axis="y", visible=False)
    return _save(fig, name, outdir)
