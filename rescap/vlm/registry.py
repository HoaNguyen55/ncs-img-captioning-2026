"""Name → adapter, so a backbone is a config string.

This is what makes `MODEL-SELECTION.md` §4's ablation table a configuration
change rather than a code fork:

    generator:    qwen2.5-vl-7b
    verifier:     vintern-1b
    ...
    generator:    vintern-3b       # H5: Vietnamese-specialised vs general
    verifier:     qwen2.5-vl-7b    # same-model check for self-verification bias

Everything is lazy: importing this module pulls in neither torch nor
transformers, so `rescap.vlm` stays importable on a machine with no GPU stack.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import VLM

#: Short name -> (factory, default kwargs). Short names are what appear in
#: configs and in the paper's tables.
_REGISTRY: dict[str, tuple[str, dict[str, Any]]] = {
    # --- generators -------------------------------------------------------
    "qwen2.5-vl-7b": ("qwen", {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct"}),
    "qwen2.5-vl-3b": ("qwen", {"model_id": "Qwen/Qwen2.5-VL-3B-Instruct"}),
    "qwen2-vl-7b": ("qwen", {"model_id": "Qwen/Qwen2-VL-7B-Instruct"}),
    "vintern-3b": ("vintern", {"model_id": "5CD-AI/Vintern-3B-R-beta", "max_tiles": 6}),
    # --- verifiers --------------------------------------------------------
    "vintern-1b": ("vintern", {"model_id": "5CD-AI/Vintern-1B-v3_5", "max_tiles": 4}),
    "vintern-1b-v2": ("vintern", {"model_id": "5CD-AI/Vintern-1B-v2", "max_tiles": 4}),
    # --- development ------------------------------------------------------
    "mock": ("mock", {}),
}

#: Roles the pipeline asks for, and what they default to. Stated here so the
#: defaults are one edit, not scattered through the pipeline.
DEFAULT_ROLES: dict[str, str] = {
    "generator": "qwen2.5-vl-7b",
    "verifier": "vintern-1b",
    "realiser": "qwen2.5-vl-7b",
}


def _factory(kind: str) -> Callable[..., VLM]:
    if kind == "qwen":
        from .qwen_vl import QwenVLAdapter

        return QwenVLAdapter
    if kind == "vintern":
        from .vintern import VinternAdapter

        return VinternAdapter
    if kind == "mock":
        from .mock import MockVLM

        return MockVLM
    raise ValueError(f"unknown adapter kind: {kind!r}")


def available() -> list[str]:
    """Registered short names."""
    return sorted(_REGISTRY)


def get_vlm(name: str, **overrides: Any) -> VLM:
    """Build a backbone by short name, without loading weights.

    Weights load on first use (or on an explicit `.load()`), so constructing a
    pipeline is cheap and a config typo fails immediately rather than after a
    15 GB download.

    >>> get_vlm("mock").name
    'mock'
    """
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"unknown backbone {name!r}. Available: {', '.join(available())}\n"
            "Add new ones to rescap/vlm/registry.py so they can be named in a config."
        )
    kind, defaults = _REGISTRY[key]
    return _factory(kind)(**{**defaults, **overrides})


def register(name: str, kind: str, **defaults: Any) -> None:
    """Register an extra backbone at runtime (for experiments, not for the paper)."""
    _REGISTRY[name.strip().lower()] = (kind, defaults)


def build_roles(config: dict[str, Any] | None = None) -> dict[str, VLM]:
    """Build the role → backbone map the pipeline runs on.

    Reuses one instance when two roles name the same backbone, so the
    same-model condition (`MODEL-SELECTION.md` §4) does not load 15 GB twice.

    >>> models = build_roles({"generator": "mock", "verifier": "mock"})
    >>> models["generator"] is models["verifier"]
    True
    """
    config = {**DEFAULT_ROLES, **(config or {})}
    built: dict[str, VLM] = {}
    by_name: dict[str, VLM] = {}
    for role, name in config.items():
        key = str(name).strip().lower()
        if key not in by_name:
            by_name[key] = get_vlm(key)
        built[role] = by_name[key]
    return built
