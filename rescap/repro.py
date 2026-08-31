"""Reproducibility utilities.

Every result must trace back to:

    Result -> Experiment -> Config -> Code commit -> Dataset -> Checkpoint

`ExperimentRun` writes that provenance chain to `<run_dir>/run_metadata.json`
before training starts, so a result directory is self-describing even if the
working tree has moved on.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> int:
    """Seed every RNG we might touch.

    `deterministic=True` also disables cuDNN autotuning.  On CPU this costs
    nothing; on GPU it trades a few percent throughput for run-to-run
    reproducibility, which is the right default for a research ladder.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass

    return seed


# ---------------------------------------------------------------------------
# Provenance capture
# ---------------------------------------------------------------------------
def git_commit(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Return the current commit, branch and dirty-state of the research repo."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    info: dict[str, Any] = {"repo_root": str(root)}

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    info["commit"] = run("rev-parse", "HEAD")
    info["branch"] = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain")
    info["dirty"] = bool(status)
    if status:
        # Truncated: enough to spot uncommitted experiment code, not a full diff.
        info["dirty_files"] = status.splitlines()[:20]
    return info


def snapshot_environment() -> dict[str, Any]:
    """Capture everything needed to re-create this run's software stack."""
    env: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }

    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_version"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        env["num_threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        else:
            env["gpu"] = None
    except ImportError:
        env["torch"] = None

    for mod in ("torchvision", "transformers", "timm", "numpy"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None

    env["git"] = git_commit()
    return env


def pip_freeze() -> list[str]:
    """Exact dependency set, for `requirements.lock.txt` next to the run."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return out.stdout.strip().splitlines()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------
@dataclass
class ExperimentRun:
    """A single, fully-traceable training/eval run.

    Usage (not a doctest -- it writes to disk, and `>>>` here made `doctest`
    try to run it and fail on an undefined `cfg`)::

        run = ExperimentRun(name="A0_baseline", config=cfg, root="logs")
        run.start()                     # writes run_metadata.json
        run.log_metrics(epoch=1, loss=3.2)
        run.finish(status="completed")
    """

    name: str
    config: dict[str, Any]
    root: str | Path = "logs"
    seed: int = 42
    notes: str = ""
    run_dir: Path = field(init=False)
    _started: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(self.root) / f"{stamp}_{self.name}"

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "ExperimentRun":
        import time

        self._started = time.time()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "results").mkdir(exist_ok=True)

        set_seed(self.seed)

        metadata = {
            "experiment": self.name,
            "seed": self.seed,
            "notes": self.notes,
            "config": self.config,
            "environment": snapshot_environment(),
            "status": "running",
        }
        self._write("run_metadata.json", metadata)

        frozen = pip_freeze()
        if frozen:
            (self.run_dir / "requirements.lock.txt").write_text("\n".join(frozen) + "\n")

        print(f"[rescap] run directory: {self.run_dir}")
        return self

    def log_metrics(self, **kwargs: Any) -> None:
        """Append one row to `metrics.jsonl` (one JSON object per line)."""
        row = {"wall_time": datetime.now(timezone.utc).isoformat(), **kwargs}
        with (self.run_dir / "metrics.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    def save_results(self, results: dict[str, Any], filename: str = "final_metrics.json") -> None:
        self._write(Path("results") / filename, results)

    def finish(self, status: str = "completed", **extra: Any) -> None:
        import time

        path = self.run_dir / "run_metadata.json"
        metadata = json.loads(path.read_text())
        metadata["status"] = status
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if self._started is not None:
            metadata["duration_seconds"] = round(time.time() - self._started, 1)
        metadata.update(extra)
        self._write("run_metadata.json", metadata)
        print(f"[rescap] run {status}: {self.run_dir}")

    # -- internals ---------------------------------------------------------
    def _write(self, relative: str | Path, payload: dict[str, Any]) -> None:
        target = self.run_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, default=str))


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config, apply `key.sub=value` command-line overrides.

    No hyper-parameter may be hard-coded in an experiment's source; everything
    flows through here.
    """
    import yaml

    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override must be key=value, got {override!r}")
        key, raw = override.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        try:
            value = yaml.safe_load(raw)
        except Exception:
            value = raw
        node[parts[-1]] = value

    return cfg
