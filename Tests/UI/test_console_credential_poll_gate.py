"""The Console credential poll skips its readiness build when signals are idle."""

import contextlib
from types import SimpleNamespace

import pytest

from tldw_chatbook.LLM_Calls import anthropic_subscription as subscription
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = pytest.mark.bootstrap_profile


class _PollScreen(ChatScreen):
    """Bare screen shell: the poll's collaborators are stubbed per test."""

    is_attached = True
    is_current = True

    def __init__(self) -> None:  # skip the heavy real screen init
        pass


def _screen(monkeypatch, *, status="ready"):
    monkeypatch.setattr(
        chat_screen_module,
        "shared_connection_evidence",
        lambda app: SimpleNamespace(version=0),
    )
    revision = [subscription.subscription_readiness_revision()]
    monkeypatch.setattr(
        subscription, "subscription_readiness_revision", lambda: revision[0]
    )
    state = {"status": status}
    monkeypatch.setattr(
        subscription,
        "subscription_credential_status",
        lambda *, background=False: state["status"],
    )

    screen = _PollScreen()
    screen._console_attach_reconciled = True
    screen._console_credential_snapshot = None
    screen._console_credential_pre_gate = None
    screen._console_subscription_selected = True  # A chat on the subscription.
    screen._console_derivation_scope = contextlib.nullcontext
    builds = []
    screen._active_console_settings_readiness = lambda: (
        builds.append(None),
        (SimpleNamespace(), SimpleNamespace(subscription_status=state["status"])),
    )[1]
    syncs = []
    screen._sync_console_settings_summary = lambda: syncs.append("summary")
    rendered = [True]  # False: the control bar deferred to its replay.
    screen._sync_console_control_bar = lambda: syncs.append("bar") or rendered[0]
    screen.rendered = rendered
    return screen, builds, syncs, state, revision


def test_idle_ticks_skip_the_readiness_build(monkeypatch):
    screen, builds, syncs, _state, _rev = _screen(monkeypatch)

    screen._poll_console_credential_readiness()
    assert builds and syncs, "first tick must build and refresh controls"

    screen._poll_console_credential_readiness()
    screen._poll_console_credential_readiness()
    assert len(builds) == 1, "unchanged signals must skip the derivation build"
    assert len(syncs) == 2, "no extra control refreshes on idle ticks"


def test_a_revision_bump_rebuilds(monkeypatch):
    screen, builds, _syncs, _state, revision = _screen(monkeypatch)

    screen._poll_console_credential_readiness()
    revision[0] += 1
    screen._poll_console_credential_readiness()
    assert len(builds) == 2, "background completion must rebuild readiness"


def test_a_status_transition_without_revision_rebuilds(monkeypatch):
    screen, builds, _syncs, state, _rev = _screen(monkeypatch, status="pending")

    screen._poll_console_credential_readiness()
    state["status"] = "expired"  # time-based expiry: no revision bump
    screen._poll_console_credential_readiness()
    assert len(builds) == 2, "expiry must rebuild even without a revision bump"


def test_a_deferred_control_refresh_leaves_the_gate_open(monkeypatch):
    """TASK-42478 review: a pass whose control-bar sync deferred commits
    neither the snapshot nor the pre-gate, so the next tick retries (the
    TASK-36511.4 contract); once one renders, idle ticks go quiet again."""
    screen, builds, syncs, _state, _rev = _screen(monkeypatch)
    screen.rendered[0] = False

    screen._poll_console_credential_readiness()
    screen._poll_console_credential_readiness()
    assert syncs == ["summary", "bar"] * 2, "a deferred refresh must be retried"
    assert screen._console_credential_pre_gate is None

    screen.rendered[0] = True
    for _ in range(3):
        screen._poll_console_credential_readiness()
    assert len(builds) == 3
    assert syncs == ["summary", "bar"] * 3
