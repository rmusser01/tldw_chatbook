"""Mounted Task 15 recovery projection and action handling."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Static

from Tests.Chat.test_console_dispatch_queue_recovery import (
    _authority,
    _destination,
    _truth,
)
from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunStatus,
    ConsoleSubmissionOrigin,
)
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchCheckpointState,
)
from tldw_chatbook.UI.Console_Modules.dispatch_recovery import (
    ConsoleDispatchRecoveryRegion,
)
from tldw_chatbook.UI.Console_Modules.prompt_queue import ConsolePromptQueueRegion
from tldw_chatbook.Chat.console_prompt_queue import PromptQueueMode
from tldw_chatbook.Chat.console_prompt_queue_coordinator import _PromptChain
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


@pytest.mark.asyncio
async def test_mounted_recovery_is_literal_actionable_and_owns_send_with_empty_queue():
    _app, host = _ready_host()
    async with host.run_test(size=(100, 34)) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session = store.create_session(
            session_id="recovery-session",
            title="Recovery",
            ephemeral=True,
        )
        user = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="private prompt",
            persist=False,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=False,
        )
        recovery = store.register_ephemeral_dispatch_recovery(
            session.id,
            user_message_id=user.id,
            assistant_message_id=assistant.id,
            preparation_id="preparation-mounted",
            attempt_id="attempt-mounted",
            checkpoint_state=ConsoleDispatchCheckpointState.ACCEPTED,
            origin="manual",
            queue_entry_id=None,
            frozen_authority=_authority(),
            resolved_destination=_destination(),
            reconstructability=_truth(),
            runtime_active=True,
        )
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("must remain blocked")

        # A healthy provider-owned turn blocks a second send but is not recovery UI.
        store.begin_ephemeral_dispatch(
            session.id,
            assistant_message_id=assistant.id,
            new_attempt_id="attempt-live",
        )
        console._sync_console_composer_action_state(can_save_chatbook=False)
        await pilot.pause()
        region = console.query_one(
            "#console-dispatch-recovery", ConsoleDispatchRecoveryRegion
        )
        send = composer.query_one("#console-send-message", Button)
        assert region.display is False
        assert send.disabled is True

        # A delivery-unknown owner is visible even though the queue is empty.
        store.mark_dispatch_recovery_needed(session.id, assistant.id)
        recovery = store.dispatch_recovery_for_session(session.id)
        assert recovery is not None
        untrusted_copy = "[red]<private>[/red] " + ("x" * 1200)
        untrusted_warning = "[bold]Retry can duplicate delivery.[/bold]"
        store._dispatch_recoveries_by_session[session.id] = replace(
            recovery,
            visible_copy=untrusted_copy,
            warning=untrusted_warning,
        )
        console._sync_console_composer_action_state(can_save_chatbook=False)
        await pilot.pause()

        assert region.display is True
        copy = region.query_one("#console-dispatch-recovery-copy", Static)
        warning = region.query_one("#console-dispatch-recovery-warning", Static)
        assert copy._render_markup is False
        assert warning._render_markup is False
        assert str(copy.render()) == untrusted_copy
        assert str(warning.render()) == untrusted_warning
        assert send.disabled is True
        assert controller.prompt_queue_registry.snapshot(session.id).total_count == 0

        # Exercise the mounted production callback, not the isolated widget seam.
        region.query_one("#console-dispatch-recovery-discard", Button).press()
        await pilot.pause()
        for _ in range(20):
            if store.dispatch_recovery_for_session(session.id) is None:
                break
            await pilot.pause(0.01)

        assert store.dispatch_recovery_for_session(session.id) is None
        assert store.get_message(assistant.id).status == "discarded"


@pytest.mark.asyncio
async def test_mounted_queued_recovery_has_one_action_surface_and_drains_exact_owner():
    _app, host = _ready_host()
    async with host.run_test(size=(100, 34)) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session = store.create_session(
            session_id="queued-recovery-session",
            title="Queued recovery",
            ephemeral=True,
        )
        user = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="accepted queued prompt",
            persist=False,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=False,
        )
        registry = controller.prompt_queue_registry
        begun = registry.begin_chain(session.id, context_epoch=0, expected_revision=0)
        first = registry.admit(
            session.id,
            text="accepted queued prompt",
            expected_revision=begun.snapshot.revision,
        )
        second = registry.admit(
            session.id,
            text="later queued prompt",
            expected_revision=first.snapshot.revision,
        )
        assert first.entry_id is not None
        assert second.entry_id is not None
        claimed = registry.claim_next(
            session.id,
            expected_revision=second.snapshot.revision,
        )
        assert claimed.applied
        bound = registry.bind_claimed_preparation(
            session.id,
            entry_id=first.entry_id,
            preparation_id="preparation-queued",
        )
        assert bound.applied
        settled = registry.settle_durable_acceptance(
            session.id,
            entry_id=first.entry_id,
            preparation_id="preparation-queued",
        )
        assert settled.applied
        coordinator = controller.prompt_queue_coordinator
        coordinator._chains[session.id] = _PromptChain()
        assert coordinator.hydrate_dispatch_recovery(
            session.id,
            queue_entry_id=first.entry_id,
            preparation_id="preparation-queued",
            checkpoint_state=ConsoleDispatchCheckpointState.ACCEPTED,
        )
        store.register_ephemeral_dispatch_recovery(
            session.id,
            user_message_id=user.id,
            assistant_message_id=assistant.id,
            preparation_id="preparation-queued",
            attempt_id="attempt-queued",
            checkpoint_state=ConsoleDispatchCheckpointState.ACCEPTED,
            origin="queued",
            queue_entry_id=first.entry_id,
            frozen_authority=_authority(),
            resolved_destination=_destination(),
            reconstructability=_truth(),
        )
        drained: list[str] = []

        async def submit_queued(_text: str, **kwargs: object):
            entry_id = kwargs["entry_id"]
            assert isinstance(entry_id, str)
            drained.append(entry_id)
            coordinator.turn_accepted(
                session.id,
                origin=ConsoleSubmissionOrigin.QUEUED,
                context_epoch=0,
                entry_id=entry_id,
            )
            return SimpleNamespace(
                accepted=True,
                terminal_status=ConsoleRunStatus.COMPLETED,
            )

        coordinator._submit_queued = submit_queued
        console._sync_console_composer_action_state(can_save_chatbook=False)
        await pilot.pause()

        recovery_region = console.query_one(
            "#console-dispatch-recovery", ConsoleDispatchRecoveryRegion
        )
        queue_region = console.query_one(
            "#console-prompt-queue", ConsolePromptQueueRegion
        )
        assert recovery_region.display is True
        assert queue_region.display is True
        assert [str(button.label) for button in recovery_region.query(Button)] == [
            "Retry response",
            "Discard",
        ]
        assert [str(button.label) for button in queue_region.query(Button)] == [
            "Manage",
            "Resume",
        ]
        assert (
            queue_region.query_one("#console-prompt-queue-manage", Button).disabled
            is False
        )
        assert (
            queue_region.query_one("#console-prompt-queue-pause", Button).disabled
            is True
        )
        assert "Paused for response recovery" in str(
            queue_region.query_one("#console-prompt-queue-summary", Static).render()
        )
        snapshot = registry.snapshot(session.id)
        assert snapshot.total_count == 1
        assert snapshot.mode is PromptQueueMode.PAUSED

        recovery_region.query_one("#console-dispatch-recovery-discard", Button).press()
        for _ in range(30):
            if drained:
                break
            await pilot.pause(0.01)

        assert drained == [second.entry_id]
        assert store.dispatch_recovery_for_session(session.id) is None
        assert registry.snapshot(session.id).total_count == 0


@pytest.mark.asyncio
async def test_mounted_recovery_action_is_pinned_to_the_displayed_session():
    _app, host = _ready_host()
    async with host.run_test(size=(100, 34)) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        store = controller.store
        owners: dict[str, str] = {}
        for index in (1, 2):
            session = store.create_session(
                session_id=f"stale-recovery-{index}",
                title=f"Recovery {index}",
                ephemeral=True,
            )
            user = store.append_message(
                session.id,
                role=ConsoleMessageRole.USER,
                content=f"private {index}",
                persist=False,
            )
            assistant = store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="",
                persist=False,
            )
            store.register_ephemeral_dispatch_recovery(
                session.id,
                user_message_id=user.id,
                assistant_message_id=assistant.id,
                preparation_id=f"preparation-stale-{index}",
                attempt_id=f"attempt-stale-{index}",
                checkpoint_state=ConsoleDispatchCheckpointState.ACCEPTED,
                origin="manual",
                queue_entry_id=None,
                frozen_authority=_authority(),
                resolved_destination=_destination(),
                reconstructability=_truth(),
            )
            owners[session.id] = assistant.id

        first_session = "stale-recovery-1"
        second_session = "stale-recovery-2"
        store._activate_session(first_session)
        console._sync_console_composer_action_state(can_save_chatbook=False)
        await pilot.pause()
        region = console.query_one(
            "#console-dispatch-recovery", ConsoleDispatchRecoveryRegion
        )

        # Simulate navigation winning after paint but before Button.Pressed runs.
        store._activate_session(second_session)
        region.query_one("#console-dispatch-recovery-discard", Button).press()
        for _ in range(30):
            if store.dispatch_recovery_for_session(first_session) is None:
                break
            await pilot.pause(0.01)

        assert store.dispatch_recovery_for_session(first_session) is None
        assert store.get_message(owners[first_session]).status == "discarded"
        assert store.dispatch_recovery_for_session(second_session) is not None
        assert store.get_message(owners[second_session]).status != "discarded"
