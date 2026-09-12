from __future__ import annotations

from dataclasses import replace
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

from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_chat_models import ConsoleControllerActivity
from tldw_chatbook.Chat.console_prompt_queue import (
    ConsolePromptQueueRegistry,
    MAX_CONSOLE_QUEUE_ENTRIES,
    PromptQueuePauseReason,
    QueueMutationStatus,
)
from tldw_chatbook.Chat.console_runtime import (
    ConsoleRuntime,
    ConsoleTurnRecoveryEntry,
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


class _RecoveryRegionApp(ConsolidatedCSSApp):
    def __init__(self, actions: list[tuple[str, int, str]]) -> None:
        super().__init__()
        self.actions = actions

    def compose(self) -> ComposeResult:
        yield ConsolePromptQueueRegion(
            id="queue",
            on_primary_requested=lambda session_id, revision, action: (
                self.actions.append((session_id, revision, action))
            ),
        )


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
async def test_recovery_shelf_buttons_pin_the_displayed_turn_id() -> None:
    registry = ConsolePromptQueueRegistry()
    presentation = derive_prompt_queue_presentation(
        registry.snapshot("session-a"),
        _activity(),
        turn_recovery_id="turn-a",
    )
    actions: list[tuple[str, int, str]] = []
    app = _RecoveryRegionApp(actions)

    async with app.run_test(size=(100, 24)) as pilot:
        region = app.query_one("#queue", ConsolePromptQueueRegion)
        region.sync_presentation("session-a", presentation)
        await pilot.pause()
        await pilot.click("#console-prompt-queue-manage")
        await pilot.click("#console-prompt-queue-pause")

    assert actions == [
        ("session-a", 0, "turn-recovery:restore:turn-a"),
        ("session-a", 0, "turn-recovery:discard:turn-a"),
    ]


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
async def test_navigation_confirmation_is_pure_and_preserves_manager_edit() -> None:
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

        assert await console.confirm_navigation() is True
        await pilot.pause()

        assert isinstance(host.screen_stack[-1], ConsolePromptQueueModal)
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
        self.prompt_queue_coordinator = SimpleNamespace(
            dispatch_recovery_blocks_queue=lambda _session_id: False
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

    def queue_prompt(
        self, session_id: str, *, text: str, expected_revision: int, configuration=None
    ):
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
        configuration=None,
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
    turn_recovery_ids=None,
    restore_turn_recovery=None,
    discard_turn_recovery=None,
    load_recovered_turn=None,
) -> ConsolePromptQueueUIController:
    async def append_system(text: str) -> None:
        calls["system"].append(text)

    async def sync_ui() -> None:
        calls["sync"].append(True)

    kwargs = {}
    kwargs.update(
        turn_recovery_ids=turn_recovery_ids or (lambda _session_id: ()),
        restore_turn_recovery=restore_turn_recovery or (lambda _turn_id: None),
        discard_turn_recovery=discard_turn_recovery or (lambda _turn_id: False),
        load_recovered_turn=load_recovered_turn or (lambda _session_id: None),
    )
    return ConsolePromptQueueUIController(
        chat_controller_accessor=lambda: fake,
        capture_configuration=lambda _: None,
        ensure_active_session=lambda: None,
        blocked_reason_accessor=lambda: "",
        setup_blocked_reason_accessor=lambda: "",
        append_system_message=append_system,
        notify=lambda text, severity: calls["notified"].append((text, severity)),
        focus_composer=lambda: calls["focused"].append(True),
        note_follow_intent=lambda: calls["follow"].append(True),
        launch_chain=lambda draft, session_id: (
            calls["staged"].append((draft, session_id)) or "turn-a"
        ),
        commit_captured_draft=lambda session_id, stash: calls["committed"].append(
            (session_id, stash)
        ),
        commit_queued_draft=lambda session_id, stash: calls["queued"].append(
            (session_id, stash)
        ),
        edit_refusal=edit_refusal,
        sync_ui=sync_ui,
        **kwargs,
    )


def _calls() -> dict[str, Any]:
    return {
        "system": [],
        "sync": [],
        "notified": [],
        "focused": [],
        "staged": [],
        "queued": [],
        "committed": [],
        "inflight": {},
        "follow": [],
    }


def _runtime_with_two_recoveries():
    store = ConsoleChatStore()
    store.create_session(
        session_id="session-a", title="Recovery", workspace_id="global"
    )
    first_attachment = PendingAttachment(
        "/private/first-secret.png",
        "first-secret.png",
        "image",
        "attachment",
        data=b"first-secret-bytes",
    )
    second_attachment = PendingAttachment(
        "/private/second-secret.png",
        "second-secret.png",
        "image",
        "attachment",
        data=b"second-secret-bytes",
    )
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(store)
    entries = (
        ConsoleTurnRecoveryEntry(
            turn_id="turn-a",
            session_id="session-a",
            draft="first secret draft",
            attachments=(first_attachment,),
            insertion_order=1,
        ),
        ConsoleTurnRecoveryEntry(
            turn_id="turn-b",
            session_id="session-a",
            draft="second secret draft",
            attachments=(second_attachment,),
            insertion_order=2,
        ),
    )
    runtime._turn_recoveries.update((entry.turn_id, entry) for entry in entries)
    runtime._recovery_turns_by_session["session-a"] = [
        entry.turn_id for entry in entries
    ]
    return runtime, store, entries


def _recovery_ui_controller(runtime, fake, calls, loaded):
    return _ui_controller(
        fake,
        calls,
        turn_recovery_ids=lambda session_id: tuple(
            entry.turn_id for entry in runtime.recoveries_for_session(session_id)
        ),
        restore_turn_recovery=runtime.restore_turn_recovery,
        discard_turn_recovery=runtime.discard_turn_recovery,
        load_recovered_turn=lambda session_id: loaded.append(session_id),
    )


def test_fresh_controller_projects_oldest_recovery_without_secret_body() -> None:
    runtime, store, _entries = _runtime_with_two_recoveries()
    fake = _FakeChatController(accepted=False)
    fake.store = store
    calls = _calls()

    fresh = _recovery_ui_controller(runtime, fake, calls, [])
    presentation = fresh.presentation_for("session-a")
    rendered = repr(presentation)

    assert presentation.count == 0
    assert presentation.shelf_visible
    assert presentation.state_label == "Unsent turn needs attention"
    assert presentation.primary_action == "turn-recovery"
    assert presentation.turn_recovery_id == "turn-a"
    for secret in (
        "turn-a",
        "first secret draft",
        "first-secret.png",
        "/private/first-secret.png",
        "first-secret-bytes",
    ):
        assert secret not in rendered


@pytest.mark.asyncio
async def test_restore_restages_exact_attachments_then_reveals_next_recovery() -> None:
    runtime, store, entries = _runtime_with_two_recoveries()
    suffix = PendingAttachment(
        "/later.png", "later.png", "image", "attachment", data=b"later"
    )
    store.add_pending_attachment("session-a", suffix)
    fake = _FakeChatController(accepted=False)
    fake.store = store
    calls = _calls()
    loaded: list[str] = []
    controller = _recovery_ui_controller(runtime, fake, calls, loaded)

    await controller.handle_primary_intent(
        "session-a",
        action="turn-recovery:restore:turn-a",
        expected_revision=0,
    )

    assert store.session_draft("session-a") == "first secret draft"
    assert store.pending_attachments("session-a") == [
        entries[0].attachments[0],
        suffix,
    ]
    assert store.pending_attachments("session-a")[0] is entries[0].attachments[0]
    assert loaded == ["session-a"]
    assert calls["focused"] == [True]
    assert controller.presentation_for("session-a").turn_recovery_id == "turn-b"


@pytest.mark.asyncio
async def test_discard_releases_exact_oldest_and_reveals_next_recovery() -> None:
    runtime, store, _entries = _runtime_with_two_recoveries()
    fake = _FakeChatController(accepted=False)
    fake.store = store
    calls = _calls()
    controller = _recovery_ui_controller(runtime, fake, calls, [])

    await controller.handle_primary_intent(
        "session-a",
        action="turn-recovery:discard:turn-a",
        expected_revision=0,
    )

    assert [entry.turn_id for entry in runtime.recoveries_for_session("session-a")] == [
        "turn-b"
    ]
    assert controller.presentation_for("session-a").turn_recovery_id == "turn-b"
    assert store.session_draft("session-a") == ""


@pytest.mark.asyncio
async def test_stale_recovery_action_is_a_warning_and_does_not_touch_next() -> None:
    runtime, store, entries = _runtime_with_two_recoveries()
    fake = _FakeChatController(accepted=False)
    fake.store = store
    calls = _calls()
    controller = _recovery_ui_controller(runtime, fake, calls, [])
    assert runtime.discard_turn_recovery("turn-a")

    await controller.handle_primary_intent(
        "session-a",
        action="turn-recovery:restore:turn-a",
        expected_revision=0,
    )

    assert runtime.recoveries_for_session("session-a") == (entries[1],)
    assert store.session_draft("session-a") == ""
    assert store.pending_attachments("session-a") == []
    assert calls["notified"] == [
        ("That unsent turn is no longer available.", "warning")
    ]


@pytest.mark.asyncio
async def test_ambiguous_restore_warns_without_changing_recovery_or_store() -> None:
    runtime, store, entries = _runtime_with_two_recoveries()
    existing = PendingAttachment(
        "/existing.png",
        "existing.png",
        "image",
        "attachment",
        data=b"existing",
    )
    store.set_session_draft("session-a", "new live draft")
    store.add_pending_attachment("session-a", existing)
    fake = _FakeChatController(accepted=False)
    fake.store = store
    calls = _calls()
    controller = _recovery_ui_controller(runtime, fake, calls, [])

    await controller.handle_primary_intent(
        "session-a",
        action="turn-recovery:restore:turn-a",
        expected_revision=0,
    )

    assert runtime.recoveries_for_session("session-a") == entries
    assert store.session_draft("session-a") == "new live draft"
    assert store.pending_attachments("session-a") == [existing]
    assert calls["focused"] == []
    assert calls["notified"] == [
        ("That unsent turn could not be restored safely.", "warning")
    ]


@pytest.mark.asyncio
async def test_restore_for_closed_session_warns_and_keeps_exact_recovery() -> None:
    runtime, store, entries = _runtime_with_two_recoveries()
    store.close_session("session-a")
    fake = _FakeChatController(accepted=False)
    fake.store = store
    calls = _calls()
    controller = _recovery_ui_controller(runtime, fake, calls, [])

    await controller.handle_primary_intent(
        "session-a",
        action="turn-recovery:restore:turn-a",
        expected_revision=0,
    )

    assert runtime.recoveries_for_session("session-a") == entries
    assert calls["focused"] == []
    assert calls["notified"] == [
        ("That unsent turn could not be restored safely.", "warning")
    ]


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
@pytest.mark.parametrize("admission", ("busy", "race", "edit"))
async def test_wired_queue_admission_freezes_view_source_filter(
    monkeypatch, admission
) -> None:
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnExecutionContext
    from Tests.Chat.test_console_turn_execution_context import _authority, _destination
    from Tests.Chat.test_console_turn_preparation import _preparation_values
    from tldw_chatbook.Chat.console_library_policy import ConsoleAutoRetrieve
    from tldw_chatbook.Chat.console_turn_preparation import (
        ConsoleTurnPreparation,
        ConsoleTurnPreparationState,
    )

    _app, host = _ready_host()
    async with host.run_test(size=(100, 30)) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session_id = store.active_session_id
        selected = ["notes"]
        monkeypatch.setattr(
            console._session, "_rag_source_types_accessor", lambda: tuple(selected)
        )
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        armed = controller.prompt_queue_registry.begin_chain(
            session_id,
            context_epoch=store.conversation_context_epoch(session_id),
            expected_revision=snapshot.revision,
        )
        assert armed.applied
        activity_calls = 0

        def activity(owner):
            nonlocal activity_calls
            activity_calls += 1
            return _activity(owner, accepted=admission != "race" or activity_calls > 1)

        monkeypatch.setattr(controller, "activity_for", activity)
        if admission == "race":
            # The initial UI activity snapshot is stale while admission becomes
            # accepted between that snapshot and the exact launch boundary.
            monkeypatch.setattr(controller, "send_refusal_copy", lambda _: "")
        # Use the production wiring and builder, not an injected capture double.
        if admission == "edit":
            queued = controller.queue_prompt(
                session_id, text="original", expected_revision=armed.snapshot.revision
            )
            outcome = console._prompt_queue.edit_waiting(
                session_id,
                queued.snapshot.entries[0].entry_id,
                text="edited",
                expected_revision=queued.snapshot.revision,
            )
            assert outcome.applied
        else:
            outcome = await console._prompt_queue.dispatch(
                "queued", session_id=session_id
            )
            assert outcome.status is ConsolePromptDispatchStatus.QUEUED
        request = (
            controller.prompt_queue_registry._states[session_id]
            .waiting[0]
            .custody_request
        )
        assert request.configuration.rag_defaults["source_types"] == ("notes",)
        selected[:] = ["media", "conversations"]
        runtime = console._console_runtime()
        assert runtime.detach_view(console, runtime._attached_generation)
        assert runtime.view is None
        authority = _authority()
        authority = replace(
            authority,
            policy=replace(
                authority.policy, auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC
            ),
        )
        context = ConsoleTurnExecutionContext(
            request.configuration, authority, _destination()
        )
        assert controller._frozen_rag_source_types(context) == ("notes",)
        values = _preparation_values(session_id=session_id, execution_context=context)
        values.update(
            executed_draft=request.draft,
            transient_user_message_id=None,
            attachment_ids=(),
            evidence_ids=(),
            prefill_id=None,
        )
        preparation = ConsoleTurnPreparation(**values)
        assert store.begin_preparation(preparation) is preparation
        requests = []

        async def search(query, source_types, mode, **kwargs):
            requests.append((query, source_types, mode, kwargs))
            return {"results": []}

        monkeypatch.setattr(
            controller.app, "library_rag_search_service", SimpleNamespace(search=search)
        )
        result = await controller.prepare_library_for_turn(preparation.preparation_id)
        assert result.state is ConsoleTurnPreparationState.READY
        assert requests[0][0] == request.draft
        assert requests[0][1] == ("notes",)


@pytest.mark.asyncio
@pytest.mark.parametrize("admission", ("busy", "race", "edit"))
async def test_wired_queue_rejects_wrong_owner_before_draft_or_queue_mutation(
    monkeypatch, admission
) -> None:
    _app, host = _ready_host()
    async with host.run_test(size=(100, 30)) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session_id = store.active_session_id
        other = store.create_session(ephemeral=True)
        wrong = controller.resolve_runtime_turn_configuration_snapshot(other.id)
        store.switch_session(session_id)
        monkeypatch.setattr(
            console._session, "_build_console_turn_execution_context", lambda _: wrong
        )
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        armed = controller.prompt_queue_registry.begin_chain(
            session_id,
            context_epoch=store.conversation_context_epoch(session_id),
            expected_revision=snapshot.revision,
        )
        activity_calls = 0

        def activity(owner):
            nonlocal activity_calls
            activity_calls += 1
            return _activity(owner, accepted=admission != "race" or activity_calls > 1)

        monkeypatch.setattr(controller, "activity_for", activity)
        if admission == "race":
            monkeypatch.setattr(controller, "send_refusal_copy", lambda _: "")
        if admission == "edit":
            queued = controller.queue_prompt(
                session_id, text="original", expected_revision=armed.snapshot.revision
            )
        before = controller.prompt_queue_registry.snapshot(session_id)
        committed = []
        monkeypatch.setattr(
            console._prompt_queue,
            "_commit_queued_draft",
            lambda *args: committed.append(args),
        )
        monkeypatch.setattr(
            console._prompt_queue,
            "_commit_captured_draft",
            lambda *args: committed.append(args),
        )
        if admission == "edit":
            result = console._prompt_queue.edit_waiting(
                session_id,
                queued.snapshot.entries[0].entry_id,
                text="replacement",
                expected_revision=before.revision,
            )
            assert result.status is QueueMutationStatus.INVALID
        else:
            result = await console._prompt_queue.dispatch(
                "keep draft", session_id=session_id
            )
            assert result.status is ConsolePromptDispatchStatus.REFUSED
        assert "different session" in result.detail
        assert controller.prompt_queue_registry.snapshot(session_id) == before
        assert committed == []


@pytest.mark.asyncio
async def test_dispatch_stages_one_manual_chain_when_queue_does_not_own_work() -> None:
    fake = _FakeChatController(accepted=False)
    calls = _calls()
    controller = _ui_controller(fake, calls)

    outcome = await controller.dispatch("send now", stash=None)

    assert outcome.status is ConsolePromptDispatchStatus.SENT
    assert calls["staged"] == [("send now", "session-a")]
    assert calls["committed"] == [("session-a", None)]
    assert fake.prompt_queue_registry.snapshot("session-a").total_count == 0


@pytest.mark.asyncio
async def test_runtime_custody_succeeds_before_composer_revision_is_committed() -> None:
    fake = _FakeChatController(accepted=False)
    calls = _calls()
    stash = object()
    events: list[str] = []

    controller = _ui_controller(fake, calls)
    controller._launch_chain = lambda draft, session_id: (
        events.append("custody") or "turn-a"
    )
    controller._commit_captured_draft = lambda session_id, captured: events.append(
        f"composer-commit:{session_id}"
    )

    outcome = await controller.dispatch("send now", stash=stash)

    assert outcome.status is ConsolePromptDispatchStatus.SENT
    assert events == ["custody", "composer-commit:session-a"]


@pytest.mark.asyncio
async def test_dispatch_uses_explicit_owning_session_instead_of_active_session() -> None:
    fake = _FakeChatController(accepted=False)
    calls = _calls()
    controller = _ui_controller(fake, calls)

    outcome = await controller.dispatch(
        "belongs to b", session_id="session-b", stash=None
    )

    assert outcome.status is ConsolePromptDispatchStatus.SENT
    assert calls["staged"] == [("belongs to b", "session-b")]
    assert calls["committed"] == [("session-b", None)]


@pytest.mark.asyncio
async def test_synchronous_custody_refusal_leaves_composer_revision_untouched() -> None:
    fake = _FakeChatController(accepted=False)
    calls = _calls()
    stash = object()
    staged_attachments = [object()]
    controller = _ui_controller(fake, calls)

    def refuse(_draft: str, _session_id: str) -> str:
        raise RuntimeError("custody refused")

    controller._launch_chain = refuse

    outcome = await controller.dispatch("keep me", stash=stash)

    assert outcome.status is ConsolePromptDispatchStatus.REFUSED
    assert calls["committed"] == []
    assert staged_attachments == staged_attachments


@pytest.mark.asyncio
async def test_dispatch_keeps_captured_stash_when_queue_is_full() -> None:
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
    assert calls["queued"] == []
    assert calls["staged"] == []
    assert outcome.detail == "Queue full (10/10). Manage or remove an item."


@pytest.mark.asyncio
async def test_pre_acceptance_race_keeps_captured_stash_instead_of_launching() -> None:
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


def _bare_modal_for_dirty_check(
    *, editing: str | None, edit_text: str, baseline: str | None, read_result
):
    """A ConsolePromptQueueModal with only the dirty-check's seams stubbed.

    Args:
        editing: Value for ``_editing_entry_id``.
        edit_text: Text the stubbed edit TextArea reports.
        baseline: Value for ``_editing_baseline_text``.
        read_result: What the stubbed controller's ``read_waiting_text``
            returns.

    Returns:
        The bare modal instance.
    """
    modal = ConsolePromptQueueModal.__new__(ConsolePromptQueueModal)
    modal._editing_entry_id = editing
    modal._editing_baseline_text = baseline
    modal._revision = 7
    modal.session_id = "session-1"
    modal._queue_controller = SimpleNamespace(
        read_waiting_text=lambda *args, **kwargs: read_result
    )
    modal.query_one = lambda selector, widget_type=None: SimpleNamespace(
        text=edit_text
    )
    return modal


def test_has_unsaved_edit_unit_states() -> None:
    """Isolated decision table for the dirty check (Qodo #2425).

    The race states (STALE_REVISION / LOCKED / NOT_FOUND) must judge
    against the edit-time baseline: the save path refuses them too, so
    the veto is the user's only copy of the modified text.
    """
    from tldw_chatbook.Chat.console_prompt_queue import QueueMutationStatus

    applied = SimpleNamespace(status=QueueMutationStatus.APPLIED, text="queued")

    # No edit open: never dirty.
    assert (
        _bare_modal_for_dirty_check(
            editing=None, edit_text="anything", baseline=None, read_result=applied
        ).has_unsaved_edit()
        is False
    )
    # Edit open, text matches the current queue: clean.
    assert (
        _bare_modal_for_dirty_check(
            editing="e1", edit_text="queued", baseline="queued", read_result=applied
        ).has_unsaved_edit()
        is False
    )
    # Edit open, text diverges from the current queue: dirty.
    assert (
        _bare_modal_for_dirty_check(
            editing="e1", edit_text="typed", baseline="queued", read_result=applied
        ).has_unsaved_edit()
        is True
    )
    # Race states: current unreadable -> judge against the baseline.
    for status in (
        QueueMutationStatus.STALE_REVISION,
        QueueMutationStatus.LOCKED,
        QueueMutationStatus.NOT_FOUND,
    ):
        raced = SimpleNamespace(status=status, text=None)
        assert (
            _bare_modal_for_dirty_check(
                editing="e1", edit_text="typed", baseline="queued", read_result=raced
            ).has_unsaved_edit()
            is True
        ), f"{status}: modified text must stay protected through the race"
        assert (
            _bare_modal_for_dirty_check(
                editing="e1", edit_text="queued", baseline="queued", read_result=raced
            ).has_unsaved_edit()
            is False
        ), f"{status}: an unmodified edit protects nothing"
    # Defensive: no baseline at all -> any typed text is protected.
    raced = SimpleNamespace(status=QueueMutationStatus.NOT_FOUND, text=None)
    assert (
        _bare_modal_for_dirty_check(
            editing="e1", edit_text="typed", baseline=None, read_result=raced
        ).has_unsaved_edit()
        is True
    )
    assert (
        _bare_modal_for_dirty_check(
            editing="e1", edit_text="", baseline=None, read_result=raced
        ).has_unsaved_edit()
        is False
    )


def test_flush_pending_work_unit_stack_walk() -> None:
    """Isolated contract for the screen hook (Qodo #2425).

    Vetoes (with one warning notification) when any stacked screen
    reports an unsaved edit; allows when none does or none provides the
    probe.
    """
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    class _App:
        def __init__(self, stack):
            self.screen_stack = stack

    class _Host(ChatScreen):
        # Textual's `app` is a read-only property; the bare unit needs a
        # stubbed stack, so the subclass overrides the lookup.
        _stub_app = None

        @property
        def app(self):
            return self._stub_app

    screen = _Host.__new__(_Host)
    notes: list[str] = []
    screen.notify = lambda message, **kwargs: notes.append(str(message))

    plain = SimpleNamespace()  # no has_unsaved_edit at all
    clean = SimpleNamespace(has_unsaved_edit=lambda: False)
    dirty = SimpleNamespace(has_unsaved_edit=lambda: True)

    screen._stub_app = _App([plain, clean])
    assert ChatScreen.flush_pending_work(screen) is True
    assert notes == []

    screen._stub_app = _App([plain, clean, dirty])
    assert ChatScreen.flush_pending_work(screen) is False
    assert len(notes) == 1 and "Unsaved queue edit" in notes[0]
