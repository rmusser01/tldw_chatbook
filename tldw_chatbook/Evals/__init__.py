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


def _override_config_path(config_selector: Path | None = None) -> Path:
    """Select private Eval overrides beside the effective CLI configuration.

    Recovery supplies its already-selected config path to avoid runtime imports.
    """
    if config_selector is None:
        from ..config import get_cli_config_path

        config_selector = get_cli_config_path()
    from ..Backup_Recovery.profile_paths import lexical_path

    return lexical_path(config_selector).parent / "eval_overrides.yaml"
