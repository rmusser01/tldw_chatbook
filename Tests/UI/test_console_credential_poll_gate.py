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
    state = {"status": status, "evidence_version": 0}
    monkeypatch.setattr(
        chat_screen_module,
        "shared_connection_evidence",
        lambda app: SimpleNamespace(version=state["evidence_version"]),
    )
    revision = [subscription.subscription_readiness_revision()]
    monkeypatch.setattr(
        subscription, "subscription_readiness_revision", lambda: revision[0]
    )
    monkeypatch.setattr(
        subscription,
        "subscription_credential_status",
        lambda *, background=False: state["status"],
    )

    screen = _PollScreen()
    screen._console_attach_reconciled = True
    screen._console_credential_snapshot = None
    screen._console_credential_pre_gate = None
    screen._console_derivation_scope = contextlib.nullcontext
    builds = []
    screen._active_console_settings_readiness = lambda: (
        builds.append(None),
        (SimpleNamespace(), SimpleNamespace(subscription_status=state["status"])),
    )[1]
    syncs = []
    screen._sync_console_settings_summary = lambda: syncs.append("summary")

    def render_controls():
        syncs.append("bar")
        return True

    screen._sync_console_control_bar = render_controls
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


def test_a_deferred_control_render_retries_the_same_evidence(monkeypatch):
    screen, builds, _syncs, state, _revision = _screen(monkeypatch)
    screen._poll_console_credential_readiness()
    previous_snapshot = screen._console_credential_snapshot
    previous_gate = screen._console_credential_pre_gate
    render_controls = screen._sync_console_control_bar
    state["evidence_version"] = 1

    monkeypatch.setattr(screen, "_sync_console_control_bar", lambda: False)
    screen._poll_console_credential_readiness()
    assert screen._console_credential_snapshot == previous_snapshot
    assert screen._console_credential_pre_gate == previous_gate

    monkeypatch.setattr(screen, "_sync_console_control_bar", render_controls)
    screen._poll_console_credential_readiness()
    assert screen._console_credential_snapshot[2] == 1
    assert screen._console_credential_pre_gate[2] == 1
    screen._poll_console_credential_readiness()
    assert len(builds) == 3, "a successful retry restores the idle fast path"
