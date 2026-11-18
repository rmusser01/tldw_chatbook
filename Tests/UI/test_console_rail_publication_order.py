"""Resident scheduler ordering only; these tests issue no native authority."""

from functools import partial
from types import SimpleNamespace

import pytest
from textual.worker import Worker, WorkerState

from tldw_chatbook.UI.Console_Modules import execution_decisions


@pytest.mark.parametrize("state", list(WorkerState))
def test_only_original_pending_or_running_rail_worker_precedes_modal(monkeypatch, state):
    screen = SimpleNamespace(post_message=lambda _message: None)
    original = execution_decisions.ChatScreen._save_console_rail_preferences.__wrapped__
    worker = Worker(screen, partial(original, screen, "scope", {}), thread=True)
    worker._state = state
    screen.workers = (worker,)
    # Check resident scheduling identities independently of the existing full
    # Screen source guard; no config setter or backend receipt is substituted.
    monkeypatch.setattr(execution_decisions, "_require_methods", lambda *_: None)
    assert execution_decisions._rail_save_precedes_modal(screen) is (
        state in (WorkerState.PENDING, WorkerState.RUNNING)
    )


@pytest.mark.parametrize("changed", ["node", "function", "argument", "thread", "other_worker"])
def test_unrelated_work_does_not_hold_project_modal(monkeypatch, changed):
    screen = SimpleNamespace(post_message=lambda _message: None)
    original = execution_decisions.ChatScreen._save_console_rail_preferences.__wrapped__
    worker = Worker(screen, partial(original, screen, "scope", {}), thread=True)
    if changed == "node":
        worker._node = object()
    elif changed == "function":
        worker._work = partial(lambda *_: None, screen)
    elif changed == "argument":
        worker._work = partial(original, object(), "scope", {})
    elif changed == "thread":
        worker._thread_worker = False
    else:
        worker = SimpleNamespace(**vars(worker))
    screen.workers = (worker,)
    monkeypatch.setattr(execution_decisions, "_require_methods", lambda *_: None)
    assert not execution_decisions._rail_save_precedes_modal(screen)


def test_pending_rail_modal_gate_retains_original_source_check(monkeypatch):
    error = RuntimeError("Original screen source changed")

    def refuse(*_):
        raise error

    monkeypatch.setattr(execution_decisions, "_require_methods", refuse)
    with pytest.raises(RuntimeError) as caught:
        execution_decisions._rail_save_precedes_modal(SimpleNamespace(workers=()))
    assert caught.value is error
