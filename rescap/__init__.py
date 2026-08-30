"""rescap -- shared research utilities for the image-captioning lab.

Deliberately small.  Anything an experiment needs *and* that must behave
identically across experiments (seeding, metric computation, environment
capture, figure style) lives here.  Anything experiment-specific stays in the
experiment folder.

Import from an experiment with:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from rescap import set_seed, snapshot_environment, CaptionMetrics
"""

from .metrics import CaptionMetrics, evaluate_captions
from .repro import ExperimentRun, git_commit, set_seed, snapshot_environment
from .vocab import Vocabulary

__all__ = [
    "CaptionMetrics",
    "evaluate_captions",
    "ExperimentRun",
    "git_commit",
    "set_seed",
    "snapshot_environment",
    "Vocabulary",
]

__version__ = "0.1.0"
