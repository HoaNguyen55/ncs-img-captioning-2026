#!/usr/bin/env python
"""Verify the research environment.

Runs a battery of import / capability checks and prints a table plus a JSON
summary.  Exit code is 0 even when optional components are missing -- the point
is to *report* the environment truthfully, not to gate on it.

Usage:
    python scripts/verify_environment.py
    python scripts/verify_environment.py --json results/env_check.json
    python scripts/verify_environment.py --heavy   # also runs a timm forward pass
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Check:
    name: str
    group: str
    status: str  # OK | MISSING | FAIL | SKIP
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


CHECKS: list[Check] = []


def record(name: str, group: str, status: str, detail: str = "", **extra) -> None:
    CHECKS.append(Check(name=name, group=group, status=status, detail=detail, extra=extra))


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------
def check_system() -> None:
    record("os", "system", "OK", platform.platform())
    record("python", "system", "OK", sys.version.split()[0], executable=sys.executable)

    try:
        import multiprocessing

        cpus = multiprocessing.cpu_count()
    except Exception:  # pragma: no cover
        cpus = -1
    record("cpu_count", "system", "OK", str(cpus))

    # RAM (Linux only)
    try:
        with open("/proc/meminfo") as fh:
            total_kb = int(next(l for l in fh if l.startswith("MemTotal")).split()[1])
        record("ram_total_gb", "system", "OK", f"{total_kb / 1024 / 1024:.1f}")
    except Exception as exc:
        record("ram_total_gb", "system", "SKIP", str(exc))

    for path in ("/", os.path.expanduser("~"), "/mnt/c"):
        try:
            usage = shutil.disk_usage(path)
            record(
                f"disk:{path}",
                "system",
                "OK",
                f"{usage.free / 1e9:.1f} GB free / {usage.total / 1e9:.1f} GB",
            )
        except Exception:
            record(f"disk:{path}", "system", "SKIP", "not mounted")


def check_binaries() -> None:
    required = {
        "git": "version control",
        "nvidia-smi": "NVIDIA GPU driver",
        "nvcc": "CUDA toolkit",
        "git-lfs": "large file storage",
        "java": "required by METEOR + SPICE metrics",
        "pdflatex": "LaTeX paper build",
        "bibtex": "BibTeX",
        "pandoc": "document conversion",
        "gcc": "C compiler (needed to build mmcv etc.)",
        "cmake": "build system",
        "docker": "containers (optional)",
    }
    for binary, purpose in required.items():
        path = shutil.which(binary)
        if path:
            try:
                out = subprocess.run(
                    [binary, "--version"], capture_output=True, text=True, timeout=15
                )
                ver = (out.stdout or out.stderr).strip().splitlines()[0][:70]
            except Exception:
                ver = "found"
            record(binary, "binaries", "OK", ver, path=path, purpose=purpose)
        else:
            record(binary, "binaries", "MISSING", purpose)


# ---------------------------------------------------------------------------
# Deep learning stack
# ---------------------------------------------------------------------------
def check_torch(heavy: bool = False) -> None:
    try:
        import torch
    except Exception as exc:
        record("torch", "deep-learning", "FAIL", f"{type(exc).__name__}: {exc}")
        return

    record(
        "torch",
        "deep-learning",
        "OK",
        torch.__version__,
        threads=torch.get_num_threads(),
    )

    cuda = torch.cuda.is_available()
    if cuda:
        record(
            "cuda",
            "deep-learning",
            "OK",
            f"{torch.version.cuda} | {torch.cuda.get_device_name(0)}",
            device_count=torch.cuda.device_count(),
            vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        )
    else:
        record("cuda", "deep-learning", "MISSING", "no CUDA device -- CPU-only environment")

    # Real forward pass, not just an import.
    try:
        import torchvision
        import torchvision.models as models

        record("torchvision", "deep-learning", "OK", torchvision.__version__)
        net = models.resnet18(weights=None).eval()
        with torch.no_grad():
            out = net(torch.randn(1, 3, 224, 224))
        record("torch:forward", "deep-learning", "OK", f"resnet18 -> {tuple(out.shape)}")
    except Exception as exc:
        record("torchvision", "deep-learning", "FAIL", f"{type(exc).__name__}: {exc}")

    if heavy:
        try:
            import timm

            model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
            model.eval()
            with torch.no_grad():
                out = model(torch.randn(1, 3, 224, 224))
            record("timm:forward", "deep-learning", "OK", f"vit_tiny -> {tuple(out.shape)}")
        except Exception as exc:
            record("timm:forward", "deep-learning", "FAIL", f"{type(exc).__name__}: {exc}")


def check_packages() -> None:
    groups = {
        "scientific": ["numpy", "scipy", "pandas", "sklearn", "matplotlib", "seaborn"],
        "vision": ["cv2", "PIL", "timm"],
        "vision-language": ["transformers", "safetensors", "accelerate", "huggingface_hub"],
        "tooling": ["tqdm", "yaml", "omegaconf", "einops", "h5py", "tensorboard", "jupyterlab"],
        "metrics": ["pycocotools", "pycocoevalcap", "nltk"],
        "optional": ["wandb", "mlflow", "hydra", "albumentations", "pyarrow", "lavis"],
    }
    for group, mods in groups.items():
        for mod in mods:
            try:
                imported = __import__(mod)
                ver = getattr(imported, "__version__", "installed")
                record(mod, group, "OK", str(ver))
            except Exception as exc:
                status = "MISSING" if group == "optional" else "FAIL"
                record(mod, group, status, f"{type(exc).__name__}: {exc}"[:90])


def check_metrics() -> None:
    """The captioning metric suite -- METEOR and SPICE need a JVM."""
    java = shutil.which("java")
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge

        gts = {"1": ["a man riding a horse on a beach"]}
        res = {"1": ["a man rides a horse on the beach"]}
        bleu, _ = Bleu(4).compute_score(gts, res)
        rouge, _ = Rouge().compute_score(gts, res)
        record("BLEU-1..4", "captioning-metrics", "OK", ", ".join(f"{b:.3f}" for b in bleu))
        record("ROUGE-L", "captioning-metrics", "OK", f"{rouge:.3f}")
        # CIDEr is corpus-level: a single sentence gives a degenerate score, so we
        # only assert that it runs.
        Cider().compute_score(gts, res)
        record("CIDEr", "captioning-metrics", "OK", "runs (corpus-level: needs full test set)")
    except Exception as exc:
        record("pycocoevalcap", "captioning-metrics", "FAIL", f"{type(exc).__name__}: {exc}"[:90])

    if java:
        record("METEOR", "captioning-metrics", "OK", "JVM present")
    else:
        record(
            "METEOR", "captioning-metrics", "MISSING",
            "needs a JVM: sudo apt-get install -y default-jre",
        )

    # SPICE 1.0 (2016) serialises via FST, which the Java module system blocks
    # from JDK 16 onward -- a modern JVM alone is not enough.
    try:
        sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
        from rescap.metrics import legacy_java_home

        home = legacy_java_home()
    except Exception:
        home = None

    if home:
        record("SPICE", "captioning-metrics", "OK", f"legacy JVM: {home}")
    elif java:
        record(
            "SPICE", "captioning-metrics", "MISSING",
            "JVM is too new (FST is blocked from JDK 16); "
            "sudo apt-get install -y openjdk-11-jre-headless",
        )
    else:
        record(
            "SPICE", "captioning-metrics", "MISSING",
            "needs a JDK <= 15: sudo apt-get install -y openjdk-11-jre-headless",
        )

    # Vietnamese word segmentation -- mandatory before any n-gram metric on
    # Vietnamese (formulation/04-EVALUATION-FRAMEWORK.md section 0).
    for name in ("pyvi", "underthesea"):
        try:
            __import__(name)
            record(name, "vietnamese-nlp", "OK", "word segmenter available")
        except Exception:
            record(name, "vietnamese-nlp", "MISSING", "uv pip install " + name)
    for name in ("bert_score", "py_vncorenlp"):
        try:
            __import__(name)
            record(name, "vietnamese-nlp", "OK", "installed")
        except Exception:
            record(name, "vietnamese-nlp", "MISSING", "optional")


# ---------------------------------------------------------------------------
# Research tooling / repositories
# ---------------------------------------------------------------------------
def check_repositories() -> None:
    root = os.path.join(os.path.dirname(__file__), "..", "repositories")
    root = os.path.realpath(root)
    if not os.path.isdir(root):
        record("repositories", "research-repos", "MISSING", root)
        return

    # Walk one level of grouping too (e.g. repositories/tooling/*), matching
    # repo_manifest.py's discover(). Without this the count here disagrees with
    # configs/repositories.yaml, which reads as the manifest overstating.
    checkouts = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if os.path.isdir(os.path.join(path, ".git")):
            checkouts.append((name, path))
        else:
            for sub in sorted(os.listdir(path)):
                subpath = os.path.join(path, sub)
                if os.path.isdir(os.path.join(subpath, ".git")):
                    checkouts.append((f"{name}/{sub}", subpath))

    for name, path in checkouts:
        try:
            sha = subprocess.run(
                ["git", "-C", path, "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout.strip()
        except Exception:
            sha = "?"
        record(name, "research-repos", "OK", f"@{sha}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
SYMBOL = {"OK": "✓", "MISSING": "○", "FAIL": "✗", "SKIP": "-"}


def report(json_path: str | None) -> None:
    current = None
    for check in CHECKS:
        if check.group != current:
            current = check.group
            print(f"\n\033[1m{current.upper()}\033[0m")
        print(f"  {SYMBOL.get(check.status, '?')} {check.name:<22} {check.status:<8} {check.detail}")

    counts: dict[str, int] = {}
    for check in CHECKS:
        counts[check.status] = counts.get(check.status, 0) + 1
    print("\n" + "-" * 70)
    print("  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))

    failures = [c.name for c in CHECKS if c.status == "FAIL"]
    if failures:
        print(f"\nFAILURES: {', '.join(failures)}")

    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w") as fh:
            json.dump(
                {"checks": [asdict(c) for c in CHECKS], "summary": counts},
                fh,
                indent=2,
            )
        print(f"\nJSON written to {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path", default=None, help="write a JSON summary here")
    parser.add_argument(
        "--heavy", action="store_true", help="also build a timm ViT and run a forward pass"
    )
    args = parser.parse_args()

    check_system()
    check_binaries()
    check_torch(heavy=args.heavy)
    check_packages()
    check_metrics()
    check_repositories()
    report(args.json_path)


if __name__ == "__main__":
    main()
