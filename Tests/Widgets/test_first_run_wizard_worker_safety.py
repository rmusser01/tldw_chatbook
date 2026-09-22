"""TASK-32892 item 6: two first-run wizard workers could take the app down.

`ProtectKeysStep._apply_password_worker` and `SummaryStep._render_rows` both
run as `run_worker(<coroutine>)` with the DEFAULT `exit_on_error=True`, and
both reach for the DOM after an await. Dismiss the wizard (or advance past
the step) while the await is in flight and the `query_one` raises out of the
worker -- which, with `exit_on_error` left at its default, exits the whole
application during first-run setup.

Gate-free: the steps are allocated with `__new__`; nothing loads settings.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from tldw_chatbook.UI.Wizards import FirstRunSetupWizard as wizard_module
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import ProtectKeysStep, SummaryStep

#: The two coroutine workers this task hardened, by the name `run_worker`
#: is handed. Located by AST rather than by line, so the pin survives edits
#: above them.
_HARDENED_WORKERS = {"_apply_password_worker", "_render_rows"}


def _run_worker_flags() -> dict[str, dict[str, bool]]:
    tree = ast.parse(Path(wizard_module.__file__).read_text(encoding="utf-8"))
    found: dict[str, dict[str, bool]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "run_worker"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Call):
            continue
        inner = node.args[0].func
        name = inner.attr if isinstance(inner, ast.Attribute) else None
        if name in _HARDENED_WORKERS:
            found[name] = {
                kw.arg: getattr(kw.value, "value", None)
                for kw in node.keywords
                if kw.arg
            }
    return found


def test_both_workers_refuse_to_exit_the_app_on_error():
    flags = _run_worker_flags()

    assert set(flags) == _HARDENED_WORKERS, f"worker call sites moved: {flags}"
    for name, kwargs in flags.items():
        assert kwargs.get("exit_on_error") is False, (
            f"{name} still runs with exit_on_error defaulting to True"
        )


def test_password_worker_returns_quietly_when_the_step_is_gone():
    step = ProtectKeysStep.__new__(ProtectKeysStep)
    step._is_mounted = False

    async def _apply(_password: str) -> bool:
        return True

    step.apply_password = _apply

    asyncio.run(step._apply_password_worker("hunter2"))


def test_summary_render_returns_quietly_when_the_step_is_gone():
    step = SummaryStep.__new__(SummaryStep)
    step._is_mounted = False
    step._load_config = lambda: {}
    step._rag_deps_installed = lambda: False
    step._speech_installed = lambda: False
    step._speech_runtime_installed = lambda: False

    asyncio.run(step._render_rows())


@pytest.mark.parametrize("step_cls", [ProtectKeysStep, SummaryStep])
def test_the_guard_reads_the_real_textual_property(step_cls):
    """`is_mounted` is a plain `_is_mounted` read, so the fakes above are
    exercising the production guard rather than a stand-in."""
    step = step_cls.__new__(step_cls)
    step._is_mounted = False
    assert step.is_mounted is False
