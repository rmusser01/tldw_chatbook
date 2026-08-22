"""Trusted, content-free Console lifecycle adapters for Persona Buddy."""

from __future__ import annotations

import time
from dataclasses import fields

import pytest

from tldw_chatbook.Persona_Buddy.console_adapter import (
    BuddyLifecycleEvent,
    PersonaBuddyConsoleAdapter,
)
from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController


@pytest.mark.unit
def test_lifecycle_event_is_frozen_slotted_and_content_free() -> None:
    event = BuddyLifecycleEvent(
        source="console-run",
        owner="run:a1",
        state="thinking",
        terminal=False,
        expires_at=None,
    )

    assert event.__slots__ == (
        "source",
        "owner",
        "state",
        "terminal",
        "expires_at",
    )
    assert tuple(field.name for field in fields(event)) == event.__slots__
    with pytest.raises(AttributeError):
        event.state = "speaking"  # type: ignore[misc]
    with pytest.raises(TypeError):
        BuddyLifecycleEvent(  # type: ignore[call-arg]
            source="console-run",
            owner="run:a1",
            state="thinking",
            terminal=False,
            expires_at=None,
            prompt="private",
        )


@pytest.mark.unit
def test_run_mapping_replaces_exact_generation_and_releases_terminal_states() -> None:
    controller = PersonaBuddyController()
    adapter = PersonaBuddyConsoleAdapter(controller)

    first = adapter.run_state("session-a", "validating")
    assert controller.snapshot().state == "thinking"
    assert controller.snapshot().state_source == "console-run"
    adapter.run_state("session-b", "checking_citations")
    assert adapter.active_owner_count("console-run") == 2

    replacement = adapter.run_state("session-a", "validating")
    assert replacement != first
    assert adapter.active_owner_count("console-run") == 2
    adapter.run_state("session-a", "streaming")
    assert controller.snapshot().state == "speaking"
    adapter.run_state("session-a", "completed")
    assert adapter.active_owner_count("console-run") == 1
    adapter.run_state("session-b", "blocked")
    assert controller.snapshot().state == "idle"


@pytest.mark.unit
def test_failed_run_maps_to_error_until_idle_or_replacement() -> None:
    controller = PersonaBuddyController()
    adapter = PersonaBuddyConsoleAdapter(controller)

    adapter.run_state("session-a", "validating")
    adapter.run_state("session-a", "failed")
    assert controller.snapshot().state == "error"

    adapter.run_state("session-a", "idle")
    assert controller.snapshot().state == "idle"


@pytest.mark.unit
def test_approval_rounds_overlap_and_release_only_exact_owner() -> None:
    controller = PersonaBuddyController()
    adapter = PersonaBuddyConsoleAdapter(controller)

    adapter.approval_round("session-a", "round-1", pending=True)
    adapter.approval_round("session-a", "round-2", pending=True)
    adapter.approval_round("session-b", "round-3", pending=True)
    assert adapter.active_owner_count("approval") == 3
    assert controller.snapshot().state == "approval_needed"

    adapter.approval_round("session-a", "round-1", pending=False)
    assert adapter.active_owner_count("approval") == 2
    adapter.approval_round("session-a", "round-1", pending=False)
    assert adapter.active_owner_count("approval") == 2
    assert controller.snapshot().state == "approval_needed"
    adapter.release_session("session-a", sources={"approval"})
    assert adapter.active_owner_count("approval") == 1


@pytest.mark.unit
def test_tool_steps_pair_by_run_and_sequence_and_cleanup_terminal_run() -> None:
    controller = PersonaBuddyController()
    adapter = PersonaBuddyConsoleAdapter(controller)

    adapter.tool_step("run-a", 1, "tool_call")
    adapter.tool_step("run-a", 2, "tool_call")
    adapter.tool_step("run-b", 1, "tool_call")
    assert adapter.active_owner_count("tool") == 3
    assert controller.snapshot().state == "tool_running"

    adapter.tool_step("run-a", 1, "tool_result")
    assert adapter.active_owner_count("tool") == 2
    adapter.release_run("run-a")
    assert adapter.active_owner_count("tool") == 1
    adapter.release_run("run-b")
    assert controller.snapshot().state == "idle"


@pytest.mark.unit
def test_wake_membership_overlaps_and_yields_to_live_voice() -> None:
    controller = PersonaBuddyController()
    adapter = PersonaBuddyConsoleAdapter(controller)

    adapter.wake("conversation-a", "run-1", active=True)
    adapter.wake("conversation-a", "run-2", active=True)
    assert adapter.active_owner_count("wake") == 2
    assert controller.snapshot().state == "wake_armed"

    adapter.voice_state("session-a", 1, "listening")
    assert controller.snapshot().state == "listening"
    adapter.voice_state("session-a", 1, "idle")
    assert controller.snapshot().state == "wake_armed"

    adapter.wake("conversation-a", "run-1", active=False)
    assert adapter.active_owner_count("wake") == 1
    adapter.clear_wakes("conversation-a")
    assert controller.snapshot().state == "idle"


@pytest.mark.unit
def test_voice_generation_replacement_and_terminal_release_are_exact() -> None:
    controller = PersonaBuddyController()
    adapter = PersonaBuddyConsoleAdapter(controller)

    adapter.voice_state("session-a", 1, "connecting")
    assert controller.snapshot().state == "offline"
    adapter.voice_state("session-a", 1, "live")
    assert controller.snapshot().state == "listening"
    adapter.voice_state("session-b", 1, "thinking")
    assert adapter.active_owner_count("voice") == 2

    adapter.voice_state("session-a", 2, "speaking")
    assert adapter.active_owner_count("voice") == 2
    adapter.voice_state("session-a", 1, "idle")
    assert adapter.active_owner_count("voice") == 2
    adapter.voice_state("session-a", 2, "idle")
    assert adapter.active_owner_count("voice") == 1
    adapter.release_voice("session-b", 1)
    assert controller.snapshot().state == "idle"


@pytest.mark.unit
def test_no_controller_sink_is_a_noop() -> None:
    adapter = PersonaBuddyConsoleAdapter(None)

    adapter.run_state("session-a", "streaming")
    adapter.approval_round("session-a", "round-1", pending=True)
    adapter.tool_step("run-a", 1, "tool_call")
    adapter.wake("conversation-a", "run-a", active=True)
    adapter.voice_state("session-a", 1, "speaking")

    assert adapter.active_owner_count() == 0


@pytest.mark.unit
def test_trusted_public_events_require_safe_state_and_future_expiry() -> None:
    controller = PersonaBuddyController()
    adapter = PersonaBuddyConsoleAdapter(controller)
    owner = "trusted:operation"

    adapter.publish(
        BuddyLifecycleEvent(
            source="explicit",
            owner=owner,
            state="celebrating.custom",
            expires_at=time.monotonic() + 30.0,
        )
    )
    assert controller.snapshot().state == "celebrating.custom"
    adapter.publish(
        BuddyLifecycleEvent(source="explicit", owner=owner, state="idle", terminal=True)
    )
    assert controller.snapshot().state == "idle"

    with pytest.raises(ValueError):
        BuddyLifecycleEvent(source="console-run", owner=owner, state="model.directive")
    with pytest.raises(ValueError):
        BuddyLifecycleEvent(source="authored", owner=owner, state="api_key.exposed")
    assert (
        adapter.publish(
            BuddyLifecycleEvent(
                source="explicit",
                owner=owner,
                state="thinking",
                expires_at=time.monotonic() - 1.0,
            )
        )
        is False
    )


@pytest.mark.unit
def test_explicit_expiry_uses_the_adapter_clock_domain() -> None:
    now = [100.0]
    controller = PersonaBuddyController(clock=lambda: now[0])
    adapter = PersonaBuddyConsoleAdapter(controller)
    adapter._clock = lambda: now[0]

    event = BuddyLifecycleEvent(
        source="explicit",
        owner="trusted:custom-clock",
        state="thinking",
        expires_at=110.0,
    )

    assert adapter.publish(event) is True
    assert controller.snapshot().state == "thinking"

    now[0] = 111.0
    assert adapter.active_owner_count() == 0
    assert controller.snapshot().state == "idle"


@pytest.mark.asyncio
async def test_terminal_dispose_fences_every_console_producer_and_bind() -> None:
    """Disposal latches before cleanup and every later producer fails closed."""
    controller = PersonaBuddyController()
    adapter = PersonaBuddyConsoleAdapter(controller)
    adapter.run_state("session-a", "validating")
    adapter.approval_round("session-a", "round-1", pending=True)
    adapter.tool_step("run-a", 1, "tool_call")
    adapter.wake("conversation-a", "run-a", active=True)
    voice_generation = adapter.next_voice_generation("session-a")
    adapter.voice_state("session-a", voice_generation, "speaking")
    adapter.publish(
        BuddyLifecycleEvent(
            source="explicit",
            owner="trusted:operation",
            state="thinking",
            expires_at=time.monotonic() + 30.0,
        )
    )

    adapter.dispose()
    adapter.dispose()
    adapter.bind_controller(PersonaBuddyController())
    await controller.shutdown()

    assert (
        adapter.publish(
            BuddyLifecycleEvent(
                source="authored", owner="trusted:late", state="thinking"
            )
        )
        is False
    )
    assert adapter.run_state("session-a", "validating") is None
    assert adapter.approval_round("session-a", "round-2", pending=True) is None
    assert adapter.tool_step("run-b", 2, "tool_call") is None
    adapter.release_run("run-a")
    assert adapter.wake("conversation-a", "run-b", active=True) is None
    adapter.clear_wakes()
    assert adapter.next_voice_generation("session-a") is None
    assert adapter.voice_state("session-a", voice_generation + 1, "listening") is None
    adapter.release_voice("session-a", voice_generation)
    adapter.release_session("session-a")
    adapter.release_all()

    assert adapter.active_owner_count() == 0
    assert controller.snapshot().state == "idle"
    for name, value in vars(adapter).items():
        if name.endswith(("_owners", "_generation")) or name == "_tokens":
            assert not value, (name, value)


@pytest.mark.unit
def test_release_all_remains_reusable_but_dispose_is_terminal() -> None:
    controller = PersonaBuddyController()
    adapter = PersonaBuddyConsoleAdapter(controller)

    adapter.run_state("session-a", "validating")
    adapter.release_all()
    assert adapter.run_state("session-a", "validating") is not None
    adapter.dispose()
    assert adapter.run_state("session-a", "validating") is None


@pytest.mark.unit
def test_expired_timed_owner_is_pruned_from_adapter_bookkeeping() -> None:
    now = [time.monotonic()]
    controller = PersonaBuddyController(clock=lambda: now[0])
    adapter = PersonaBuddyConsoleAdapter(controller)
    adapter._clock = lambda: now[0]
    adapter.publish(
        BuddyLifecycleEvent(
            source="explicit",
            owner="trusted:expiring",
            state="thinking",
            expires_at=now[0] + 10.0,
        )
    )
    assert adapter.active_owner_count() == 1

    now[0] += 11.0

    assert adapter.active_owner_count() == 0
    assert adapter._tokens == {}
