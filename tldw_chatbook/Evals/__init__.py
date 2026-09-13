"""Public service exports, resolved lazily for dependency-light recovery."""

from importlib import import_module
from pathlib import Path

_EXPORTS = {
    "EvaluationOrchestrator": "eval_orchestrator",
    "TaskLoader": "task_loader",
    "TaskLoadError": "task_loader",
}
__all__ = ["EvaluationOrchestrator", "TaskLoader", "TaskLoadError"]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("." + module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


def _default_config_path() -> Path:
    """Canonical installed evaluation definition path, without runtime imports."""
    return Path(__file__).parent / "config" / "eval_config.yaml"
