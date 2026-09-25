"""TASK-32892 item 6: two first-run wizard workers could take the app down.

`ProtectKeysStep._apply_password_worker` and `SummaryStep._render_rows` both
run as `run_worker(<coroutine>)` with the DEFAULT `exit_on_error=True`, and
both reach for the DOM after an await. Dismiss the wizard (or advance past
the step) while the await is in flight and the `query_one` raises out of the
worker -- which, with `exit_on_error` left at its default, exits the whole
application during first-run setup.

The unit tests are gate-free: the steps are allocated with `__new__` and
nothing loads settings. The mounted journeys at the bottom (Qodo review of
PR #2799, "Wizard crashes lack ui test coverage") run both workers through a
real Textual app so the worker lifecycle and `exit_on_error` are exercised
for real rather than asserted about by AST.
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


class _DetachedProtectKeysStep(ProtectKeysStep):
    """`is_attached` is a read-only Textual property and a `__new__`-allocated
    step has no app, so it is already False -- overriding it states that
    intent instead of relying on a `NoActiveAppError` falling through."""

    is_attached = False


class _DetachedSummaryStep(SummaryStep):
    is_attached = False


def test_password_worker_returns_quietly_when_the_step_is_gone():
    step = _DetachedProtectKeysStep.__new__(_DetachedProtectKeysStep)

    async def _apply(_password: str) -> bool:
        return True

    step.apply_password = _apply

    asyncio.run(step._apply_password_worker("hunter2"))


def test_summary_render_returns_quietly_when_the_step_is_gone():
    step = _DetachedSummaryStep.__new__(_DetachedSummaryStep)
    step._load_config = lambda: {}
    step._rag_deps_installed = lambda: False
    step._speech_installed = lambda: False
    step._speech_runtime_installed = lambda: False

    asyncio.run(step._render_rows())


# (The "does the fake exercise the real property?" check that used to sit here
# cannot be written against a `__new__` step: `is_attached` walks
# `_MessagePump__parent`, which `__new__` never creates, so it raises
# AttributeError rather than returning False. The mounted journeys at the
# bottom of this file assert `is_attached is False` on a genuinely removed
# step, which is the same guarantee against the real Textual property.)


# --------------------------------------------------------------------------
# Qodo review of #2799, "Wizard crashes lack ui test coverage": everything
# above either reads the source with `ast` or calls the coroutine directly on
# a `__new__` step, so no Textual worker was ever started, no step was ever
# unmounted, and `exit_on_error=False` was never exercised. These mount the
# steps, launch each worker through its own handler, tear the step down while
# the awaited work is blocked, then release it.
#
# They are also what caught the original guard being inert: it read
# `is_mounted`, and Textual 8.2.8 assigns `_is_mounted = True` exactly once
# and never clears it -- a removed step still reports True. Setting
# `_is_mounted = False` by hand, as this file used to, asserts on a state
# production never reaches.
#
# Measured while writing these: removing a step makes Textual cancel that
# node's workers (`Widget._on_unmount` -> `WorkerManager.cancel_node`), and
# the cancellation lands ON the await -- so the post-await DOM work never
# runs at all on the dismiss path. The teardown journeys below therefore pin
# SURVIVAL, not `exit_on_error`; the AST test at the top of this file is what
# holds `exit_on_error=False` in place (verified by mutation: dropping the
# kwarg reddens that test and leaves these green). The two "while the step is
# up" journeys are the ones that catch an over-firing guard, which is the
# regression `is_attached` could plausibly introduce.
# --------------------------------------------------------------------------

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from textual.app import ComposeResult
from textual.widgets import Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig


class _StepHost(ConsolidatedCSSApp):
    def __init__(self, step):
        super().__init__()
        self._step = step

    def compose(self) -> ComposeResult:
        yield self._step


def _wizard():
    return SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        wizard_data={},
        key_entered=True,
        rerun=False,
    )


def _protect_step(enable_encryption):
    return ProtectKeysStep(
        wizard=_wizard(),
        config=WizardStepConfig(id="protect", title="Protect", step_number=8),
        enable_encryption=enable_encryption,
    )


def _summary_step(load_config):
    return SummaryStep(
        wizard=_wizard(),
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=load_config,
        rag_deps_installed=lambda: False,
        speech_installed=lambda: False,
        speech_runtime_installed=lambda: False,
    )


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_password_worker_survives_the_step_being_torn_down_mid_apply():
    release = threading.Event()
    step = _protect_step(lambda _password: (release.wait(5), True)[1])
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # The production handler -- what PasswordDialog's dismissal calls.
        step._on_password_result("hunter2")
        await pilot.pause()

        # The user dismisses the wizard while enable_config_encryption runs.
        await step.remove()
        await pilot.pause()
        release.set()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert step.is_mounted is True  # the Textual fact; see the note above
        assert step.is_attached is False
        assert pilot.app.is_running, "the password worker took the app down"


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_password_worker_still_reports_success_while_the_step_is_up():
    """The guard must not eat the normal case -- an `is_attached` that read
    False on a live step would silently drop every status update."""
    step = _protect_step(lambda _password: True)
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step._on_password_result("hunter2")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        status = step.query_one("#setup-protect-status", Static)
        assert "Encryption enabled" in str(status.renderable)
        assert step.encryption_enabled is True


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_summary_render_survives_the_step_being_torn_down_mid_load():
    release = threading.Event()

    def _blocking_load():
        release.wait(5)
        return {}

    step = _summary_step(_blocking_load)
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        # on_show launches the render worker; it blocks in the executor.
        await pilot.pause()
        await step.remove()
        await pilot.pause()
        release.set()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert step.is_attached is False
        assert pilot.app.is_running, "the summary worker took the app down"


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_summary_render_still_fills_the_matrix_while_the_step_is_up():
    step = _summary_step(lambda: {})
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        rows = step.query_one("#setup-summary-rows", Static)
        assert str(rows.renderable).strip(), "the read-back matrix stayed empty"
