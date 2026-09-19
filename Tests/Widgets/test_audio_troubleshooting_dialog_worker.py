"""The audio troubleshooting dialog must not kill the app on open.

TASK-32800.3. ``_initialize_audio`` enumerated devices with
``self.run_worker(self._get_devices_safe)``. ``_get_devices_safe`` is a plain
``def``, so Textual's ``Worker._run_async`` raises
``WorkerError("Request to run a non-async function as an async worker")``;
``Worker._run`` catches that generically and ``exit_on_error`` defaults to
``True``, so the whole app exits. The dialog is reachable from the shipped
``stts`` route via the dictation window's "troubleshoot" action.

The regression is the *worker declaration*, so that is what is asserted here:
a non-coroutine target must be run with ``thread=True``. Asserting it through
a mounted app would need the real audio stack, which is exactly what the
dialog exists to report on when it is missing.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from tldw_chatbook.Widgets.audio_troubleshooting_dialog import (
    AudioTroubleshootingDialog,
)

MODULE = Path(inspect.getfile(AudioTroubleshootingDialog))


def _run_worker_calls() -> list[ast.Call]:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_worker"
    ]


def test_dialog_has_run_worker_call_sites() -> None:
    """Guard the guard: if the calls move, the assertions below are vacuous."""
    assert _run_worker_calls(), "expected run_worker call sites in the dialog"


@pytest.mark.parametrize("call", _run_worker_calls(), ids=lambda c: f"line{c.lineno}")
def test_sync_worker_targets_declare_thread(call: ast.Call) -> None:
    """A non-coroutine run_worker target must declare ``thread=True``."""
    if not call.args:
        pytest.skip("run_worker called without a positional target")
    target = call.args[0]

    # ``run_worker(self._method())`` passes an already-created coroutine or
    # Worker; only a bare attribute reference names a callable to be run.
    if not isinstance(target, ast.Attribute) or not isinstance(target.value, ast.Name):
        pytest.skip("target is not a bare `self.<method>` reference")
    if target.value.id != "self":
        pytest.skip("target is not a method of this widget")

    method = getattr(AudioTroubleshootingDialog, target.attr, None)
    assert method is not None, f"run_worker names a missing method: {target.attr}"

    unwrapped = inspect.unwrap(method)
    if inspect.iscoroutinefunction(unwrapped):
        return  # an async target is valid without thread=True

    threaded = any(
        kw.arg == "thread"
        and isinstance(kw.value, ast.Constant)
        and kw.value.value is True
        for kw in call.keywords
    )
    assert threaded, (
        f"{MODULE.name}:{call.lineno} runs the synchronous method "
        f"{target.attr!r} as an async worker. Textual raises WorkerError and "
        f"exit_on_error defaults to True, so this exits the app. "
        f"Pass thread=True."
    )
