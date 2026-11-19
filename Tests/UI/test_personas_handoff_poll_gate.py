"""The Roleplay handoff readiness poll resolves only when its inputs change."""

from types import SimpleNamespace

import pytest

from Tests.UI.test_personas_workbench import PersonasScreen
from tldw_chatbook.LLM_Calls import anthropic_subscription as subscription

pytestmark = pytest.mark.bootstrap_profile


class _PollScreen(PersonasScreen):
    """Bare screen shell: the poll's collaborators are stubbed per test."""

    is_mounted = True
    is_active = True

    def __init__(self) -> None:  # skip the heavy real screen init
        pass


def _screen() -> _PollScreen:
    screen = _PollScreen()
    screen.state = SimpleNamespace(selected_entity_kind="character")
    screen.app_instance = SimpleNamespace(
        app_config={"chat_defaults": {"provider": "anthropic", "model": "m1"}}
    )
    screen._console_action_allowed = lambda: True
    screen._gather_header_inputs = lambda: ("header",)
    screen._header_inputs = ("header",)
    screen._console_readiness_block_reason = None
    screen._console_handoff_readiness_gate = None
    resolutions = []
    screen._provider_send_block_reason = lambda: (resolutions.append(1), None)[1]
    return screen, resolutions


def test_poll_resolves_once_then_skips_unchanged_ticks(monkeypatch):
    revision = [subscription.subscription_readiness_revision()]
    monkeypatch.setattr(
        subscription, "subscription_readiness_revision", lambda: revision[0]
    )
    screen, resolutions = _screen()

    screen._poll_console_handoff_readiness()
    assert len(resolutions) == 1, "first poll must resolve handoff readiness"

    screen._poll_console_handoff_readiness()
    screen._poll_console_handoff_readiness()
    assert len(resolutions) == 1, "unchanged gate must skip provider resolution"


def test_poll_re_resolves_when_the_revision_bumps(monkeypatch):
    revision = [subscription.subscription_readiness_revision()]
    monkeypatch.setattr(
        subscription, "subscription_readiness_revision", lambda: revision[0]
    )
    screen, resolutions = _screen()

    screen._poll_console_handoff_readiness()
    revision[0] += 1
    screen._poll_console_handoff_readiness()
    assert len(resolutions) == 2, "a revision bump must re-resolve"


def test_poll_re_resolves_when_the_selection_changes(monkeypatch):
    revision = [subscription.subscription_readiness_revision()]
    monkeypatch.setattr(
        subscription, "subscription_readiness_revision", lambda: revision[0]
    )
    screen, resolutions = _screen()

    screen._poll_console_handoff_readiness()
    screen.state.selected_entity_kind = "persona"
    screen._poll_console_handoff_readiness()
    assert len(resolutions) == 2, "an entity-kind change must re-resolve"
