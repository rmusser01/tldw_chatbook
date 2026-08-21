"""Pure state and ownership contracts for the app-owned Persona Buddy."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
from tldw_chatbook.Persona_Buddy.preferences import PersonaBuddySelection


def test_priority_is_exact_for_overlapping_sources() -> None:
    controller = PersonaBuddyController()
    offline = controller.acquire_state(
        source="network", owner="profile", state="offline"
    )
    live_idle = controller.acquire_state(source="voice", owner="session", state="idle")
    wake = controller.acquire_state(source="wake", owner="listener", state="wake_armed")
    live = controller.acquire_state(source="voice", owner="turn", state="speaking")
    tool = controller.acquire_state(
        source="tool", owner="run:call", state="tool_running"
    )
    authored = controller.set_authored_trigger(
        owner="pack:trigger", state="reaction:happy"
    )
    timed = controller.set_timed_state(
        owner="explicit:preview", state="preview_pose", ttl_seconds=60
    )
    approval = controller.acquire_state(
        source="approval", owner="session:round", state="approval_needed"
    )
    error = controller.acquire_state(
        source="runtime", owner="session:error", state="error"
    )

    assert controller.snapshot().state == "error"
    assert controller.release_state(token=error) is True
    assert controller.snapshot().state == "approval_needed"
    assert controller.release_state(token=approval) is True
    assert controller.snapshot().state == "preview_pose"
    assert controller.release_state(token=timed) is True
    assert controller.snapshot().state == "reaction:happy"
    assert controller.release_state(token=authored) is True
    assert controller.snapshot().state == "tool_running"
    assert controller.release_state(token=tool) is True
    assert controller.snapshot().state == "speaking"
    assert controller.release_state(token=live) is True
    assert controller.snapshot().state == "wake_armed"
    assert controller.release_state(token=wake) is True
    assert controller.snapshot().state == "idle"
    assert controller.release_state(token=live_idle) is True
    assert controller.snapshot().state == "offline"
    assert controller.release_state(token=offline) is True
    assert controller.snapshot().state == "idle"


def test_release_requires_exact_source_owner() -> None:
    controller = PersonaBuddyController()
    first = controller.acquire_state(
        source="approval", owner="session:first", state="approval_needed"
    )
    stale = controller.acquire_state(
        source="approval", owner="session:second", state="approval_needed"
    )

    assert controller.release_state(source="approval", owner="session:wrong") is False
    assert controller.release_state(token=first) is True
    assert controller.snapshot().state == "approval_needed"

    replacement = controller.acquire_state(
        source="approval", owner="session:second", state="approval_needed"
    )
    assert replacement != stale
    assert controller.release_state(token=stale) is False
    assert controller.snapshot().state == "approval_needed"
    assert controller.release_state(source="approval", owner="session:second") is True
    assert controller.snapshot().state == "idle"


def test_wake_armed_yields_to_non_idle_live_voice() -> None:
    controller = PersonaBuddyController()
    wake = controller.acquire_state(source="wake", owner="listener", state="wake_armed")
    idle = controller.acquire_state(source="voice", owner="idle", state="idle")

    assert controller.snapshot().state == "wake_armed"

    listening = controller.acquire_state(
        source="voice", owner="turn", state="listening"
    )
    assert controller.snapshot().state == "listening"
    assert controller.release_state(token=listening) is True
    assert controller.snapshot().state == "wake_armed"

    assert controller.release_state(token=idle) is True
    assert controller.release_state(token=wake) is True


def test_timed_custom_state_expires() -> None:
    now = [100.0]
    controller = PersonaBuddyController(clock=lambda: now[0])

    token = controller.set_timed_state(
        owner="preview", state="mood_calm", ttl_seconds=2.0
    )

    active = controller.snapshot()
    assert active.state == "mood_calm"
    assert not hasattr(active, "__dict__")
    with pytest.raises(FrozenInstanceError):
        active.state = "idle"  # type: ignore[misc]

    now[0] = 102.0
    assert controller.snapshot().state == "idle"
    assert controller.release_state(token=token) is False

    with pytest.raises(ValueError, match="^persona_buddy_state_invalid$"):
        controller.set_timed_state(
            owner="preview", state="file:private", ttl_seconds=1.0
        )


def test_expiration_does_not_promote_operational_priority() -> None:
    controller = PersonaBuddyController(clock=lambda: 100.0)
    authored = controller.set_authored_trigger(owner="pack", state="mood_calm")
    controller.acquire_state(
        source="tool",
        owner="call",
        state="tool_running",
        expires_at=200.0,
    )
    controller.acquire_state(
        source="network",
        owner="profile",
        state="offline",
        expires_at=200.0,
    )

    assert controller.snapshot().state == "mood_calm"
    assert controller.release_state(token=authored) is True
    assert controller.snapshot().state == "tool_running"


def test_indefinite_explicit_custom_state_is_rejected() -> None:
    controller = PersonaBuddyController(clock=lambda: 100.0)

    with pytest.raises(ValueError, match="^persona_buddy_state_invalid$"):
        controller.acquire_state(
            source="explicit",
            owner="preview",
            state="mood_calm",
        )


def test_arbitrary_source_custom_state_is_rejected() -> None:
    controller = PersonaBuddyController(clock=lambda: 100.0)

    with pytest.raises(ValueError, match="^persona_buddy_state_invalid$"):
        controller.acquire_state(
            source="authroed",
            owner="pack",
            state="mood_calm",
            expires_at=200.0,
        )


@pytest.mark.parametrize("expires_at", (float("nan"), float("inf"), 100.0, 99.0))
def test_explicit_custom_state_requires_finite_future_expiry(
    expires_at: float,
) -> None:
    controller = PersonaBuddyController(clock=lambda: 100.0)

    with pytest.raises(ValueError, match="^persona_buddy_state_invalid$"):
        controller.acquire_state(
            source="explicit",
            owner="preview",
            state="mood_calm",
            expires_at=expires_at,
        )


def test_authored_and_explicit_timed_custom_states_are_valid() -> None:
    controller = PersonaBuddyController(clock=lambda: 100.0)
    authored = controller.set_authored_trigger(owner="pack", state="mood_calm")
    explicit = controller.acquire_state(
        source="explicit",
        owner="preview",
        state="reaction:happy",
        expires_at=101.0,
    )

    assert authored.state == "mood_calm"
    assert explicit.state == "reaction:happy"
    assert controller.snapshot().state == "reaction:happy"


def test_selection_never_changes_from_observed_persona() -> None:
    controller = PersonaBuddyController()
    selected_generation = controller.select_local_persona("p-1")

    assert controller.snapshot().selection == PersonaBuddySelection("local", "p-1")
    assert (
        controller.observe_persona(source="workbench", persona_id="p-2")
        == selected_generation
    )
    assert (
        controller.observe_persona(source="console", persona_id="p-3")
        == selected_generation
    )
    assert controller.snapshot().selection == PersonaBuddySelection("local", "p-1")


@pytest.mark.parametrize("contract_name", ("lease_token", "snapshot"))
def test_controller_public_contracts_are_exactly_frozen_and_slotted(
    contract_name: str,
) -> None:
    controller = PersonaBuddyController()
    token = controller.acquire_state(source="tool", owner="call", state="tool_running")
    value = token if contract_name == "lease_token" else controller.snapshot()

    assert type(value).__dataclass_params__.frozen is True
    assert "__slots__" in vars(type(value))
    assert not hasattr(value, "__dict__")
    with pytest.raises(FrozenInstanceError):
        value.state = "idle"  # type: ignore[misc]
