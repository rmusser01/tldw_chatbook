from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from textual.app import ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button
from textual.widgets import TextArea

from Tests.UI.test_console_dictation import _mounted_console, _ready_host

from tldw_chatbook.Chat.console_chat_models import ConsoleControllerActivity
from tldw_chatbook.Chat.console_prompt_queue import (
    ConsolePromptQueueRegistry,
    MAX_CONSOLE_QUEUE_ENTRIES,
    PromptQueuePauseReason,
    QueueMutationStatus,
)
from tldw_chatbook.UI.Console_Modules.prompt_queue import (
    ConsolePromptDispatchStatus,
    ConsolePromptQueueRegion,
    ConsolePromptQueueUIController,
    derive_prompt_queue_presentation,
)
from tldw_chatbook.Widgets.Console.console_session_surface import (
    ConsoleSessionSurface,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_prompt_queue_modal import (
    ConsolePromptQueueModal,
)


def _activity(
    session_id: str = "session-a",
    *,
    preparing: bool = False,
    accepted: bool = False,
    occupies: bool | None = None,
    count: int = 0,
    paused: bool = False,
) -> ConsoleControllerActivity:
    return ConsoleControllerActivity(
        session_id=session_id,
        occupies_slot=preparing or accepted if occupies is None else occupies,
        preparing_before_acceptance=preparing,
        accepted_live_turn=accepted,
        needs_approval=False,
        queued_count=count,
        queue_paused=paused,
        terminal_notification_eligible=False,
    )


def _registry_with_chain() -> ConsolePromptQueueRegistry:
    registry = ConsolePromptQueueRegistry()
    snapshot = registry.snapshot("session-a")
    registry.begin_chain(
        "session-a", context_epoch=1, expected_revision=snapshot.revision
    )
    return registry


def test_presentation_uses_exact_send_queue_boundaries() -> None:
    registry = ConsolePromptQueueRegistry()
    empty = registry.snapshot("session-a")

    preparing = derive_prompt_queue_presentation(
        empty, _activity(preparing=True)
    )
    assert preparing.send_label == "Preparing..."
    assert preparing.send_enabled is False

    handoff = derive_prompt_queue_presentation(
        empty, _activity(occupies=True)
    )
    assert handoff.send_label == "Preparing..."
    assert handoff.send_enabled is False

    chain = _registry_with_chain()
    chained = chain.snapshot("session-a")
    queue = derive_prompt_queue_presentation(
        chained, _activity(accepted=True)
    )
    assert queue.send_label == "Queue"
    assert queue.send_enabled is True

    for index in range(MAX_CONSOLE_QUEUE_ENTRIES):
        chained = chain.admit(
            "session-a",
            text=f"prompt {index}",
            expected_revision=chained.revision,
        ).snapshot
    full = derive_prompt_queue_presentation(
        chained,
        _activity(accepted=True, count=MAX_CONSOLE_QUEUE_ENTRIES),
    )
    assert full.send_label == "Queue full"
    assert full.send_enabled is False


def test_background_session_label_exposes_count_only() -> None:
    label = ConsoleSessionSurface._tab_label("Session", queued_count=3)

    assert label == "Q3 Session"


@pytest.mark.parametrize(
    ("reason", "state", "label", "action"),
    [
        (PromptQueuePauseReason.FAILED, "Turn failed", "Retry", "retry-failed"),
        (PromptQueuePauseReason.STOPPED, "Turn stopped", "Resume next", "resume-next"),
        (
            PromptQueuePauseReason.CONTEXT_CHANGED,
            "Context changed",
            "Review",
            "review",
        ),
        (
            PromptQueuePauseReason.DISPATCH_REFUSED,
            "Start refused",
            "Try again",
            "toggle-pause",
        ),
    ],
)
def test_paused_shelf_exposes_state_specific_primary_action(
    reason: PromptQueuePauseReason,
    state: str,
    label: str,
    action: str,
) -> None:
    registry = _registry_with_chain()
    snapshot = registry.admit(
        "session-a",
        text="waiting",
        expected_revision=registry.snapshot("session-a").revision,
    ).snapshot
    snapshot = registry.pause(
        "session-a", reason=reason, expected_revision=snapshot.revision
    ).snapshot

    presentation = derive_prompt_queue_presentation(
        snapshot, _activity(count=1, paused=True)
    )

    assert presentation.state_label == state
    assert presentation.pause_label == label
    assert presentation.primary_action == action


class _RegionApp(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ConsolePromptQueueRegion(id="queue")


@pytest.mark.asyncio
async def test_region_is_revision_guarded_and_hides_preview_when_collapsed() -> None:
    registry = _registry_with_chain()
    snapshot = registry.admit(
        "session-a",
        text="safe preview",
        expected_revision=registry.snapshot("session-a").revision,
    ).snapshot
    presentation = derive_prompt_queue_presentation(
        snapshot, _activity(accepted=True, count=1)
    )

    app = _RegionApp()
    async with app.run_test(size=(80, 24)) as pilot:
        region = app.query_one("#queue", ConsolePromptQueueRegion)
        assert region.sync_presentation("session-a", presentation) is True
        await pilot.pause()
        assert region.sync_presentation("session-a", presentation) is False
        assert region.has_class("-visible")
        assert region.query_one("#console-prompt-queue-summary").renderable == (
            "Queue 1/10 · Draining"
        )

        collapsed = derive_prompt_queue_presentation(
            snapshot,
            _activity(accepted=True, count=1),
            composer_collapsed=True,
        )
        region.sync_presentation("session-a", collapsed)
        assert not region.has_class("-visible")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (100, 30), (160, 40)])
async def test_mounted_shelf_and_neighboring_composer_fit_terminal(size) -> None:
    _app, host = _ready_host()
    async with host.run_test(size=size) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        session_id = controller.store.active_session_id
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        snapshot = controller.prompt_queue_registry.begin_chain(
            session_id,
            context_epoch=controller.store.conversation_context_epoch(session_id),
            expected_revision=snapshot.revision,
        ).snapshot
        controller.prompt_queue_registry.admit(
            session_id,
            text="geometry-safe queued prompt",
            expected_revision=snapshot.revision,
        )
        controller.prompt_queue_coordinator.publish_registry_change(session_id)
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        region = console.query_one(
            "#console-prompt-queue", ConsolePromptQueueRegion
        )
        composer = console.query_one(
            "#console-native-composer", ConsoleComposerBar
        )
        manage = region.query_one("#console-prompt-queue-manage", Button)
        pause = region.query_one("#console-prompt-queue-pause", Button)
        send = composer.query_one("#console-send-message", Button)

        assert region.display
        assert region.region.height == 1
        # task-17661: ALL transient strips sit at the top of the control
        # deck, above the status line — the shelf's nearest lower neighbor
        # in the default (chips-above) placement is the status row, and the
        # composer keeps its quiet gap below the chips. DOM-order geometry,
        # exact in any harness.
        chips = console.query_one("#console-status-chips")
        assert region.region.y + region.region.height == chips.region.y
        assert (
            chips.region.y + chips.region.height
            <= composer.region.y - composer.styles.margin.top
        )
        assert manage.region.right <= region.region.right
        assert pause.region.right <= region.region.right
        assert manage.region.right <= pause.region.x
        assert send.label.plain == "Queue"
        assert send.region.right <= composer.region.right


@pytest.mark.asyncio
async def test_confirm_navigation_is_a_pure_allow_with_manager_open() -> None:
    """TASK-31520 rewrite: the lifecycle dialog is retired with the loss.

    Pre-reuse, a busy/paused queue raised a Stay/Leave dialog from
    ``confirm_navigation`` because leaving destroyed the screen. The chat
    route is reusable now -- the queue registry, the paused chain, and the
    session all survive navigation -- so the gate answers True. What still
    matters: the CALL must be a pure gate with no side effects -- it must
    not dismiss the open manager, discard the unsaved edit, or move focus
    (the real navigation seam owns overlay dismissal separately).
    """
    _app, host = _ready_host()
    async with host.run_test(size=(100, 30)) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        session_id = controller.store.active_session_id
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        snapshot = controller.prompt_queue_registry.begin_chain(
            session_id,
            context_epoch=controller.store.conversation_context_epoch(session_id),
            expected_revision=snapshot.revision,
        ).snapshot
        snapshot = controller.prompt_queue_registry.admit(
            session_id,
            text="keep this private edit",
            expected_revision=snapshot.revision,
        ).snapshot
        controller.prompt_queue_registry.pause(
            session_id,
            reason=PromptQueuePauseReason.MANUAL,
            expected_revision=snapshot.revision,
        )
        controller.prompt_queue_coordinator.publish_registry_change(session_id)
        await console._sync_native_console_chat_ui()
        await pilot.pause()
        await pilot.click("#console-prompt-queue-manage")
        await pilot.pause()
        assert isinstance(host.screen_stack[-1], ConsolePromptQueueModal)
        await pilot.click("#console-prompt-queue-edit")
        edit = host.screen_stack[-1].query_one(
            "#console-prompt-queue-edit-input", TextArea
        )
        edit.text = "unsaved manager edit"
        edit.focus()

        assert await console.confirm_navigation() is True, (
            "nothing is lost by navigating under reuse; the gate must allow"
        )
        await pilot.pause()
        assert isinstance(host.screen_stack[-1], ConsolePromptQueueModal), (
            "the gate must not dismiss the open manager"
        )
        assert edit.text == "unsaved manager edit"
        assert edit.has_focus


@pytest.mark.asyncio
async def test_full_console_manager_mounts_entry_children_before_live_list_insert() -> None:
    """Opening Manage must not race child mounts against an unattached row."""

    _app, host = _ready_host()
    async with host.run_test(size=(100, 30)) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        session_id = controller.store.active_session_id
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        snapshot = controller.prompt_queue_registry.begin_chain(
            session_id,
            context_epoch=controller.store.conversation_context_epoch(session_id),
            expected_revision=snapshot.revision,
        ).snapshot
        controller.prompt_queue_registry.admit(
            session_id,
            text="mounted before children",
            expected_revision=snapshot.revision,
        )
        controller.prompt_queue_coordinator.publish_registry_change(session_id)
        await console._sync_native_console_chat_ui()

        console.query_one("#console-prompt-queue-manage", Button).press()
        await pilot.pause()

        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsolePromptQueueModal)
        entry_buttons = list(modal.query(".console-prompt-queue-entry-select"))
        assert len(entry_buttons) == 1
        assert entry_buttons[0].is_mounted
        assert entry_buttons[0].parent is not None
        assert entry_buttons[0].parent.is_mounted


class _FakeChatController:
    def __init__(self, *, accepted: bool, preparing: bool = False) -> None:
        self.prompt_queue_registry = _registry_with_chain() if accepted else ConsolePromptQueueRegistry()
        self.store = SimpleNamespace(
            active_session_id="session-a",
            conversation_context_epoch=lambda _session_id: 8,
        )
        self._accepted = accepted
        self._preparing = preparing

    def activity_for(self, session_id: str) -> ConsoleControllerActivity:
        snapshot = self.prompt_queue_registry.snapshot(session_id)
        return _activity(
            session_id,
            preparing=self._preparing,
            accepted=self._accepted,
            count=snapshot.total_count,
        )

    def queue_prompt(self, session_id: str, *, text: str, expected_revision: int):
        return self.prompt_queue_registry.admit(
            session_id, text=text, expected_revision=expected_revision
        )

    def edit_queued_prompt(
        self,
        session_id: str,
        *,
        entry_id: str,
        text: str,
        expected_revision: int,
    ):
        return self.prompt_queue_registry.edit(
            session_id,
            entry_id=entry_id,
            text=text,
            expected_revision=expected_revision,
        )

    def send_refusal_copy(self, _session_id: str) -> str:
        return ""


def _ui_controller(
    fake: _FakeChatController,
    calls: dict[str, Any],
    *,
    edit_refusal=lambda _text: "",
) -> ConsolePromptQueueUIController:
    async def append_system(text: str) -> None:
        calls["system"].append(text)

    async def sync_ui() -> None:
        calls["sync"].append(True)

    return ConsolePromptQueueUIController(
        chat_controller_accessor=lambda: fake,
        ensure_active_session=lambda: None,
        blocked_reason_accessor=lambda: "",
        setup_blocked_reason_accessor=lambda: "",
        restore_stash=lambda stash: calls["restored"].append(stash),
        append_system_message=append_system,
        notify=lambda text, severity: calls["notified"].append((text, severity)),
        focus_composer=lambda: calls["focused"].append(True),
        inflight_stashes_accessor=lambda: calls["inflight"],
        note_follow_intent=lambda: calls["follow"].append(True),
        launch_chain=lambda draft, session_id: calls["staged"].append(
            (draft, session_id)
        ),
        commit_queued_draft=lambda session_id, stash: calls["queued"].append(
            (session_id, stash)
        ),
        edit_refusal=edit_refusal,
        sync_ui=sync_ui,
    )


def _calls() -> dict[str, Any]:
    return {
        "system": [],
        "sync": [],
        "restored": [],
        "notified": [],
        "focused": [],
        "staged": [],
        "queued": [],
        "inflight": {},
        "follow": [],
    }


@pytest.mark.asyncio
async def test_dispatch_admits_exact_text_behind_accepted_turn() -> None:
    fake = _FakeChatController(accepted=True)
    calls = _calls()
    controller = _ui_controller(fake, calls)

    outcome = await controller.dispatch("  exact text\n", stash=None)

    assert outcome.status is ConsolePromptDispatchStatus.QUEUED
    snapshot = fake.prompt_queue_registry.snapshot("session-a")
    entry = snapshot.entries[0]
    body = fake.prompt_queue_registry.read_waiting_text(
        "session-a", entry_id=entry.entry_id, expected_revision=snapshot.revision
    )
    assert body.text == "  exact text\n"
    assert calls["staged"] == []
    assert calls["queued"] == [("session-a", None)]


@pytest.mark.asyncio
async def test_dispatch_stages_one_manual_chain_when_queue_does_not_own_work() -> None:
    fake = _FakeChatController(accepted=False)
    calls = _calls()
    controller = _ui_controller(fake, calls)

    outcome = await controller.dispatch("send now", stash=None)

    assert outcome.status is ConsolePromptDispatchStatus.SENT
    assert calls["staged"] == [("send now", "session-a")]
    assert fake.prompt_queue_registry.snapshot("session-a").total_count == 0


@pytest.mark.asyncio
async def test_dispatch_restores_exact_stash_when_queue_is_full() -> None:
    fake = _FakeChatController(accepted=True)
    snapshot = fake.prompt_queue_registry.snapshot("session-a")
    for index in range(MAX_CONSOLE_QUEUE_ENTRIES):
        snapshot = fake.prompt_queue_registry.admit(
            "session-a",
            text=f"queued {index}",
            expected_revision=snapshot.revision,
        ).snapshot
    calls = _calls()
    controller = _ui_controller(fake, calls)
    stash = object()

    outcome = await controller.dispatch("must survive", stash=stash)

    assert outcome.status is ConsolePromptDispatchStatus.REFUSED
    assert calls["restored"] == [stash]
    assert calls["queued"] == []
    assert calls["staged"] == []
    assert outcome.detail == "Queue full (10/10). Manage or remove an item."


@pytest.mark.asyncio
async def test_pre_acceptance_race_restores_stash_instead_of_launching() -> None:
    fake = _FakeChatController(accepted=False)
    calls = _calls()
    controller = _ui_controller(fake, calls)
    stash = object()
    activity_calls = 0

    def activity_for(session_id: str) -> ConsoleControllerActivity:
        nonlocal activity_calls
        activity_calls += 1
        return _activity(session_id, preparing=activity_calls > 1)

    fake.activity_for = activity_for  # type: ignore[method-assign]
    outcome = await controller.dispatch("race-safe", stash=stash)

    assert outcome.status is ConsolePromptDispatchStatus.REFUSED
    assert calls["restored"] == [stash]
    assert calls["staged"] == []
    assert "Preparing" in outcome.detail


@pytest.mark.asyncio
async def test_finished_chain_boundary_reroutes_to_one_normal_send() -> None:
    fake = _FakeChatController(accepted=False)
    calls = _calls()
    controller = _ui_controller(fake, calls)
    activity_calls = 0

    def activity_for(session_id: str) -> ConsoleControllerActivity:
        nonlocal activity_calls
        activity_calls += 1
        return _activity(session_id, accepted=activity_calls == 1)

    fake.activity_for = activity_for  # type: ignore[method-assign]
    outcome = await controller.dispatch("boundary", stash=None)

    assert outcome.status is ConsolePromptDispatchStatus.SENT
    assert calls["staged"] == [("boundary", "session-a")]
    assert fake.prompt_queue_registry.snapshot("session-a").total_count == 0


def test_manager_edit_refuses_recognized_slash_command_without_mutation() -> None:
    fake = _FakeChatController(accepted=True)
    snapshot = fake.prompt_queue_registry.snapshot("session-a")
    snapshot = fake.prompt_queue_registry.admit(
        "session-a", text="ordinary", expected_revision=snapshot.revision
    ).snapshot
    entry_id = snapshot.entries[0].entry_id
    calls = _calls()
    controller = _ui_controller(
        fake,
        calls,
        edit_refusal=lambda text: (
            "Slash commands cannot be queued." if text == "/help" else ""
        ),
    )

    result = controller.edit_waiting(
        "session-a",
        entry_id,
        text="/help",
        expected_revision=snapshot.revision,
    )

    assert result.status is QueueMutationStatus.INVALID
    after = fake.prompt_queue_registry.read_waiting_text(
        "session-a",
        entry_id=entry_id,
        expected_revision=snapshot.revision,
    )
    assert after.text == "ordinary"


@pytest.mark.asyncio
async def test_use_current_context_rejects_an_epoch_that_changed_after_review() -> None:
    fake = _FakeChatController(accepted=True)
    calls = _calls()
    controller = _ui_controller(fake, calls)
    snapshot = fake.prompt_queue_registry.snapshot("session-a")

    result = await controller.recover(
        "session-a",
        action="use-current-context",
        expected_revision=snapshot.revision,
        reviewed_context_epoch=7,
    )

    assert result.status is QueueMutationStatus.INVALID
    assert "changed since review" in result.detail


@pytest.mark.asyncio
async def test_dirty_queue_edit_vetoes_navigation_and_preserves_text() -> None:
    """TASK-31701: the one lossy Console navigation is guarded again.

    A queue-manager edit whose text diverges from the queued entry vetoes
    `flush_pending_work` (the seam app.py consults before dismissing
    overlays), preserving the modal, the typed text, and focus. Saving
    the edit clears the veto -- navigation is lossless again.
    """
    _app, host = _ready_host()
    async with host.run_test(size=(100, 30)) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        session_id = controller.store.active_session_id
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        snapshot = controller.prompt_queue_registry.begin_chain(
            session_id,
            context_epoch=controller.store.conversation_context_epoch(session_id),
            expected_revision=snapshot.revision,
        ).snapshot
        controller.prompt_queue_registry.admit(
            session_id,
            text="original queued text",
            expected_revision=snapshot.revision,
        )
        controller.prompt_queue_coordinator.publish_registry_change(session_id)
        await console._sync_native_console_chat_ui()
        await pilot.pause()
        await pilot.click("#console-prompt-queue-manage")
        await pilot.pause()
        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsolePromptQueueModal)

        # No edit open: nothing to protect.
        assert console.flush_pending_work() is True

        await pilot.click("#console-prompt-queue-edit")
        edit = modal.query_one("#console-prompt-queue-edit-input", TextArea)
        # Edit open but text unchanged: still lossless, still allowed.
        assert console.flush_pending_work() is True

        edit.text = "edited but not yet saved"
        assert console.flush_pending_work() is False, (
            "a dirty edit must veto -- the navigation seam would dismiss "
            "the modal and discard the typed text"
        )
        assert isinstance(host.screen_stack[-1], ConsolePromptQueueModal), (
            "the veto must not disturb the open manager"
        )
        assert edit.text == "edited but not yet saved"

        # Saving resolves the veto.
        await pilot.click("#console-prompt-queue-save")
        await pilot.pause()
        assert console.flush_pending_work() is True, (
            "a saved edit loses nothing; navigation must be allowed again"
        )
