"""Guard the complete speculative-voice Python lint inventory."""

from __future__ import annotations

from pathlib import Path
import re


_ROOT = Path(__file__).resolve().parents[2]
_PLANS = (
    _ROOT
    / "Docs/superpowers/plans/2026-08-28-low-latency-speculative-duplex-voice-pipeline.md",
    _ROOT / "Docs/superpowers/plans/2026-09-05-speculative-voice-process-isolation.md",
)
_SOURCE_PATHS = _ROOT / "Packaging/speculative_voice_source_paths.txt"
_PYTHON_PATHS = _ROOT / "Packaging/speculative_voice_python_paths.txt"
_PLAN_PYTHON_PATH = re.compile(
    r"^- (?:Create|Inspect|Modify|Test): `([^`]+\.py)`(?: .*)?$"
)
_PROCESS_PLAN = _PLANS[1]
_PROCESS_PLAN_PYTHON_COUNT = 78
_PROCESS_PLAN_INSPECT_SCOPE = {"tldw_chatbook/Chat/console_voice_worker.py"}
_LIVE_QUALIFICATION_PYTHON_SCOPE = {
    "Packaging/physical_voice_runner.py",
    "Tests/Audio/test_acoustic_isolation.py",
    "Tests/Packaging/test_physical_voice_runner.py",
    "tldw_chatbook/Audio/acoustic_isolation.py",
}


def _listed_paths(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_speculative_voice_python_lint_scope_is_complete_and_safe() -> None:
    listed = _listed_paths(_PYTHON_PATHS)
    listed_set = set(listed)
    source_python = {
        path for path in _listed_paths(_SOURCE_PATHS) if path.endswith(".py")
    }
    plan_python_by_path = {
        plan: {
            match.group(1)
            for line in plan.read_text(encoding="utf-8").splitlines()
            if (match := _PLAN_PYTHON_PATH.fullmatch(line)) is not None
        }
        for plan in _PLANS
    }
    plan_python = set().union(*plan_python_by_path.values())

    assert listed == sorted(listed_set)
    assert listed_set == source_python
    assert plan_python <= listed_set
    assert len(plan_python_by_path[_PROCESS_PLAN]) == _PROCESS_PLAN_PYTHON_COUNT
    assert _PROCESS_PLAN_INSPECT_SCOPE <= plan_python_by_path[_PROCESS_PLAN]
    assert _LIVE_QUALIFICATION_PYTHON_SCOPE <= listed_set
    assert not {
        path
        for path in listed
        if not path.endswith(".py")
        or not (_ROOT / path).is_file()
        or (_ROOT / path).is_symlink()
    }
