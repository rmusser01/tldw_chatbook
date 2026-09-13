# Evals package
# Description: LLM Evaluation framework for tldw_chatbook

__all__ = ["EvaluationOrchestrator", "TaskLoader", "TaskLoadError"]


def __getattr__(name: str) -> type:
    """Load public runners only when requested, not for every Evals subpackage.

    Args:
        name: Public evaluation class name to resolve.

    Returns:
        The requested class, cached in this package's namespace.

    Raises:
        AttributeError: If the name is not a public lazy export.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name = (
        "eval_orchestrator" if name == "EvaluationOrchestrator" else "task_loader"
    )
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
