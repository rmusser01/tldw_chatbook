"""Passive draft spend refreshes use the existing selected display retry owner."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from Tests.UI.test_console_selected_refresh_coverage import _cover, _visit
from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost
from tldw_chatbook.UI.Console_Modules import execution_state, wiring
from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
    ConsoleDraftSpendRefresh,
)


def _refresh(monkeypatch):
    # Orchestration only: original timer/shared refresh functions, no native
    # receiver or retirement result is simulated by these UI call recorders.
    screen, client, app = _visit(monkeypatch)
    calls, workers = [], []
    screen._console_derivation_scope = nullcontext

    def unrelated_controls(*_):
        raise AssertionError("A fresh spend timer rendered unrelated controls")

    monkeypatch.setattr(execution_state, "sync_selected_controls", unrelated_controls)
    screen._sync_console_settings_summary = lambda: calls.append("summary")
    screen._sync_console_cost_chip = lambda: calls.append("cost")

    def schedule(work, **_):
        workers.append(work)
        return SimpleNamespace(is_finished=False)

    screen.run_worker = schedule
    refresh = ConsoleDraftSpendRefresh(
        schedule_timer=lambda *_: None,
        sync_settings_summary=lambda: calls.append("local summary"),
        sync_cost_chip=lambda: calls.append("local cost"),
        sync_selected_display=lambda: wiring._sync_selected_draft_spend(screen),
    )
    return screen, client, app, refresh, calls, workers


def test_expired_pair_coalesces_and_fresh_retry_renders_both_original_surfaces(
    monkeypatch,
):
    screen, client, _, refresh, calls, workers = _refresh(monkeypatch)
    client._readiness_deadline = 0
    refresh.timer = object()
    refresh.refresh()
    refresh.refresh()
    assert refresh.timer is None
    assert calls == []
    assert len(workers) == 1
    assert workers[0].func is execution_state.reconcile_selected_view
    assert workers[0].args == (screen,)
    # Model only completion of the shared refresh's wait; its native admission
    # and physical retirement remain covered by the actual process tests.
    screen._console_attach_reconcile_worker = None
    client._readiness_deadline = float("inf")
    refresh.refresh()
    assert calls == ["summary", "cost"]


@pytest.mark.parametrize("surface", ["summary", "cost"])
def test_expiry_within_either_surface_joins_shared_retry(monkeypatch, surface):
    screen, client, _, refresh, calls, workers = _refresh(monkeypatch)

    def expired():
        client._readiness_deadline = 0
        raise execution_state.ConsoleDisplayExpired("display expired")

    setattr(
        screen,
        f"_sync_console_{'settings_summary' if surface == 'summary' else 'cost_chip'}",
        expired,
    )
    refresh.refresh()
    assert len(workers) == 1
    assert "local summary" not in calls and "local cost" not in calls


@pytest.mark.parametrize("state", ["covered", "revoked", "teardown"])
def test_obsolete_passive_pair_does_not_paint_or_schedule(monkeypatch, state):
    screen, client, app, refresh, calls, workers = _refresh(monkeypatch)
    if state == "covered":
        _cover(screen, app)
    elif state == "revoked":
        client._visit += 1
    else:
        screen._console_chat_tearing_down = True
    client._readiness_deadline = 0
    refresh.refresh()
    assert calls == workers == []


@pytest.mark.parametrize("failure", ["source", "render"])
def test_genuine_active_failure_is_not_treated_as_expiry(monkeypatch, failure):
    screen, _, _, refresh, _, workers = _refresh(monkeypatch)
    error = ConsoleChannelLost("active failure")

    def failed(*_):
        raise error

    if failure == "source":
        monkeypatch.setattr(execution_state, "selected_client", failed)
    else:
        screen._sync_console_settings_summary = failed
    with pytest.raises(ConsoleChannelLost) as caught:
        refresh.refresh()
    assert caught.value is error
    assert workers == []


def test_local_pair_keeps_original_order(monkeypatch):
    _, _, app, refresh, calls, workers = _refresh(monkeypatch)
    app.console_execution_client = None
    refresh.refresh()
    assert calls == ["local summary", "local cost"]
    assert workers == []
