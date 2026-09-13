"""Paint and acknowledgement contracts for the Console outcome notice."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from Tests.UI.test_console_native_chat_flow import (
    _build_test_app,
    _configure_native_ready_console,
    _wait_for_selector,
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_switcher_state import CapturedReceipt
from tldw_chatbook.Chat.console_switcher_state import (
    ActivityGroup,
    ConsoleSwitcherEntry,
    ConsoleSwitcherTarget,
    SwitcherTargetKind,
    UnavailableSessionNotice,
)
from tldw_chatbook.Widgets.Console.console_activity_outcome_notice import (
    ConsoleActivityOutcomeNotice,
    ConsoleActivityOutcomePresentation,
)
from tldw_chatbook.Widgets.Console.console_session_surface import (
    ConsoleSessionSurface,
)
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSwitcherChoice,
)


def _presentation(*statuses: str) -> ConsoleActivityOutcomePresentation:
    return ConsoleActivityOutcomePresentation(
        title="Agent [release] result",
        profile_authority="profile-a",
        authority_token="runtime-a",
        session_id="session-a",
        conversation_id="conversation-a",
        receipts=tuple(
            CapturedReceipt(activity_id=f"activity-{index}", status=status)
            for index, status in enumerate(statuses, start=1)
        ),
    )


class _NoticeApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[tuple[ConsoleActivityOutcomePresentation, int]] = []

    def compose(self) -> ComposeResult:
        yield ConsoleActivityOutcomeNotice(
            mark_seen=self._mark_seen,
            id="notice",
        )

    def _mark_seen(
        self,
        presentation: ConsoleActivityOutcomePresentation,
        generation: int,
    ) -> bool:
        self.requests.append((presentation, generation))
        return True


@pytest.mark.asyncio
async def test_notice_is_hidden_until_literal_safe_presentation_is_shown():
    app = _NoticeApp()
    async with app.run_test(size=(80, 12)) as pilot:
        notice = app.query_one("#notice", ConsoleActivityOutcomeNotice)
        assert notice.display is False

        generation = notice.show(_presentation("failed"))
        await pilot.pause()

        assert notice.display is True
        assert notice.is_current(generation, _presentation("failed"))
        content = notice.query_one("#console-activity-outcome-copy", Static)
        assert isinstance(content.renderable, Text)
        assert "Agent [release] result" in content.renderable.plain
        assert "FAILED" in content.renderable.plain
        assert notice.query_one("#console-activity-outcome-mark-seen", Button).display


@pytest.mark.asyncio
async def test_mark_seen_uses_current_frozen_presentation_then_hides_notice():
    app = _NoticeApp()
    presentation = _presentation("failed", "done")
    async with app.run_test(size=(80, 12)) as pilot:
        notice = app.query_one("#notice", ConsoleActivityOutcomeNotice)
        generation = notice.show(presentation)
        await pilot.pause()
        await pilot.click("#console-activity-outcome-mark-seen")
        await pilot.pause()

        assert app.requests == [(presentation, generation)]
        assert notice.display is False
        assert notice.presentation_generation > generation


@pytest.mark.asyncio
async def test_dismiss_and_replacement_invalidate_without_marking_seen():
    app = _NoticeApp()
    async with app.run_test(size=(80, 12)) as pilot:
        notice = app.query_one("#notice", ConsoleActivityOutcomeNotice)
        first = _presentation("failed")
        first_generation = notice.show(first)
        second = _presentation("stopped")
        second_generation = notice.show(second)

        assert second_generation > first_generation
        assert not notice.is_current(first_generation, first)
        await pilot.pause()
        await pilot.click("#console-activity-outcome-dismiss")
        await pilot.pause()

        assert app.requests == []
        assert notice.display is False
        assert notice.presentation_generation > second_generation


@pytest.mark.asyncio
async def test_session_surface_always_mounts_hidden_notice_between_work_and_transcript():
    app = App()
    app.compose = lambda: iter(
        (ConsoleSessionSurface(type("Host", (), {"notify": lambda *a, **k: None})()),)
    )
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        surface = app.query_one(ConsoleSessionSurface)
        notice = surface.query_one(ConsoleActivityOutcomeNotice)
        child_ids = [child.id for child in surface.children]
        assert notice.display is False
        assert (
            child_ids.index("console-task-surface")
            < child_ids.index("console-activity-outcome-notice")
            < child_ids.index("console-transcript-surface")
        )


class _RecordingReceiptService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.threads: list[str] = []
        self.degraded = False
        self.projection_generation = 1

    def acknowledge(self, activity_ids) -> int:  # type: ignore[no-untyped-def]
        captured = tuple(activity_ids)
        self.calls.append(captured)
        self.threads.append(threading.current_thread().name)
        return len(captured)


class _DegradingReceiptService(_RecordingReceiptService):
    def acknowledge(self, activity_ids) -> int:  # type: ignore[no-untyped-def]
        captured = tuple(activity_ids)
        self.calls.append(captured)
        self.threads.append(threading.current_thread().name)
        return 0 if self.degraded else len(captured)


def _native_choice(
    *,
    session_id: str,
    profile: str,
    token: str,
    receipts: tuple[CapturedReceipt, ...],
) -> ConsoleSwitcherChoice:
    target = ConsoleSwitcherTarget(
        kind=SwitcherTargetKind.NATIVE_SESSION,
        profile_authority=profile,
        authority_token=token,
        session_id=session_id,
        conversation_id=None,
        scope_type="global",
        workspace_id=None,
        receipts=receipts,
    )
    return ConsoleSwitcherChoice(
        "activate",
        ConsoleSwitcherEntry(
            row_key=f"session:{profile}:{session_id}",
            title="Background agent",
            subtitle="FINISHED · UNSEEN",
            native_session_id=session_id,
            conversation_id=None,
            scope_type="global",
            workspace_id=None,
            is_active=False,
            target=target,
            group=ActivityGroup.NEW_RESULTS,
        ),
    )


@pytest.mark.asyncio
async def test_success_receipt_acknowledges_only_after_destination_notice_paints():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.click("#console-new-chat-tab")
        await pilot.pause()
        store = console._console_chat_store
        target_id = store.sessions()[0].id
        service = _RecordingReceiptService()
        runtime = console._console_runtime()
        runtime._activity_receipts = service
        profile, token = console._workspace._console_switcher_authority()

        await console._session._apply_console_switcher_choice(
            _native_choice(
                session_id=target_id,
                profile=profile,
                token=token,
                receipts=(CapturedReceipt(activity_id="done-1", status="done"),),
            )
        )
        assert service.calls == []
        await pilot.pause()

        notice = console.query_one(ConsoleActivityOutcomeNotice)
        assert notice.display
        assert service.calls == [("done-1",)]
        assert service.threads[0] != threading.current_thread().name


@pytest.mark.asyncio
async def test_failed_receipt_waits_for_explicit_mark_seen():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.click("#console-new-chat-tab")
        await pilot.pause()
        store = console._console_chat_store
        target_id = store.sessions()[0].id
        service = _RecordingReceiptService()
        runtime = console._console_runtime()
        runtime._activity_receipts = service
        profile, token = console._workspace._console_switcher_authority()

        await console._session._apply_console_switcher_choice(
            _native_choice(
                session_id=target_id,
                profile=profile,
                token=token,
                receipts=(CapturedReceipt(activity_id="failed-1", status="failed"),),
            )
        )
        await pilot.pause()
        assert service.calls == []
        await pilot.click("#console-activity-outcome-mark-seen")
        await pilot.pause()

        assert service.calls == [("failed-1",)]


@pytest.mark.asyncio
async def test_mixed_receipts_auto_ack_only_done_then_mark_only_failure():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        service = _RecordingReceiptService()
        console._console_runtime()._activity_receipts = service
        store = console._console_chat_store
        profile, token = console._workspace._console_switcher_authority()

        await console._session._apply_console_switcher_choice(
            _native_choice(
                session_id=store.active_session_id,
                profile=profile,
                token=token,
                receipts=(
                    CapturedReceipt(activity_id="done-captured", status="done"),
                    CapturedReceipt(activity_id="failed-captured", status="failed"),
                ),
            )
        )
        await pilot.pause()
        assert service.calls == [("done-captured",)]

        await pilot.click("#console-activity-outcome-mark-seen")
        await pilot.pause()
        assert service.calls == [
            ("done-captured",),
            ("failed-captured",),
        ]


@pytest.mark.asyncio
async def test_new_receipt_during_navigation_is_not_inferred_or_acknowledged():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        service = _RecordingReceiptService()
        service.new_arrival = CapturedReceipt(activity_id="done-newer", status="done")
        console._console_runtime()._activity_receipts = service
        store = console._console_chat_store
        profile, token = console._workspace._console_switcher_authority()

        await console._session._apply_console_switcher_choice(
            _native_choice(
                session_id=store.active_session_id,
                profile=profile,
                token=token,
                receipts=(CapturedReceipt(activity_id="done-selected", status="done"),),
            )
        )
        await pilot.pause()

        assert service.calls == [("done-selected",)]


@pytest.mark.asyncio
async def test_missing_native_target_never_falls_back_or_acknowledges():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        service = _RecordingReceiptService()
        console._console_runtime()._activity_receipts = service
        store = console._console_chat_store
        active_id = store.active_session_id
        profile, token = console._workspace._console_switcher_authority()

        await console._session._apply_console_switcher_choice(
            _native_choice(
                session_id="vanished-session",
                profile=profile,
                token=token,
                receipts=(CapturedReceipt(activity_id="done-vanished", status="done"),),
            )
        )
        await pilot.pause()

        assert store.active_session_id == active_id
        assert service.calls == []
        assert not console.query_one(ConsoleActivityOutcomeNotice).display


@pytest.mark.asyncio
async def test_authority_mismatch_never_navigates_or_acknowledges():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        service = _RecordingReceiptService()
        console._console_runtime()._activity_receipts = service
        store = console._console_chat_store
        active_id = store.active_session_id
        profile, token = console._workspace._console_switcher_authority()

        await console._session._apply_console_switcher_choice(
            _native_choice(
                session_id=active_id,
                profile=profile,
                token="replaced-runtime",
                receipts=(CapturedReceipt(activity_id="done-stale", status="done"),),
            )
        )
        await pilot.pause()

        assert store.active_session_id == active_id
        assert service.calls == []
        assert not console.query_one(ConsoleActivityOutcomeNotice).display


@pytest.mark.asyncio
async def test_switch_away_before_deferred_paint_invalidates_acknowledgement(
    monkeypatch,
):
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.click("#console-new-chat-tab")
        await pilot.pause()
        store = console._console_chat_store
        sessions = store.sessions()
        target_id = sessions[0].id
        return_id = store.active_session_id
        service = _RecordingReceiptService()
        console._console_runtime()._activity_receipts = service
        profile, token = console._workspace._console_switcher_authority()
        notice = console.query_one(ConsoleActivityOutcomeNotice)
        deferred: list[tuple[object, tuple[object, ...]]] = []

        def hold(callback, *args, **_kwargs):  # type: ignore[no-untyped-def]
            deferred.append((callback, args))

        monkeypatch.setattr(notice, "call_after_refresh", hold)
        await console._session._apply_console_switcher_choice(
            _native_choice(
                session_id=target_id,
                profile=profile,
                token=token,
                receipts=(CapturedReceipt(activity_id="done-deferred", status="done"),),
            )
        )
        assert service.calls == []
        assert deferred

        await console._session._activate_native_console_session(return_id)
        callback, args = deferred.pop()
        callback(*args)  # type: ignore[operator]
        await pilot.pause()

        assert store.active_session_id == return_id
        assert service.calls == []


@pytest.mark.asyncio
async def test_failed_ack_completion_revalidates_notice_after_switch_away():
    class _BlockingFailureService(_DegradingReceiptService):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def acknowledge(self, activity_ids) -> int:  # type: ignore[no-untyped-def]
            self.calls.append(tuple(activity_ids))
            self.threads.append(threading.current_thread().name)
            self.entered.set()
            assert self.release.wait(5)
            return 0

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.click("#console-new-chat-tab")
        await pilot.pause()
        store = console._console_chat_store
        target_id = store.sessions()[0].id
        return_id = store.active_session_id
        service = _BlockingFailureService()
        service.degraded = True
        console._console_runtime()._activity_receipts = service
        profile, token = console._workspace._console_switcher_authority()

        await console._session._apply_console_switcher_choice(
            _native_choice(
                session_id=target_id,
                profile=profile,
                token=token,
                receipts=(CapturedReceipt(activity_id="done-blocked", status="done"),),
            )
        )
        await pilot.pause()
        assert await asyncio.to_thread(service.entered.wait, 2)

        await console._session._activate_native_console_session(return_id)
        service.release.set()
        await pilot.pause(0.1)

        notice = console.query_one(ConsoleActivityOutcomeNotice)
        assert store.active_session_id == return_id
        assert not notice.display
        assert service.threads[0] != threading.current_thread().name


@pytest.mark.asyncio
async def test_failed_auto_ack_exposes_exact_retry_and_recovers():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        service = _DegradingReceiptService()
        service.degraded = True
        console._console_runtime()._activity_receipts = service
        store = console._console_chat_store
        profile, token = console._workspace._console_switcher_authority()

        await console._session._apply_console_switcher_choice(
            _native_choice(
                session_id=store.active_session_id,
                profile=profile,
                token=token,
                receipts=(CapturedReceipt(activity_id="done-retry", status="done"),),
            )
        )
        await pilot.pause()
        button = console.query_one("#console-activity-outcome-mark-seen", Button)
        assert service.calls == [("done-retry",)]
        assert button.display

        service.degraded = False
        await pilot.click("#console-activity-outcome-mark-seen")
        await pilot.pause()

        assert service.calls == [("done-retry",), ("done-retry",)]
        assert not console.query_one(ConsoleActivityOutcomeNotice).display


@pytest.mark.asyncio
async def test_persisted_target_resumes_exact_conversation_before_acknowledgement():
    app = _build_test_app()
    _configure_native_ready_console(app)
    conversation_id = "saved-switcher-target"
    workspace_id = app.workspace_registry_service.get_active_workspace().workspace_id
    app.chat_conversation_scope_service = SimpleNamespace(
        get_conversation_tree=lambda requested, **_kwargs: (
            {
                "conversation": {
                    "id": requested,
                    "title": "Saved switcher target",
                    "scope_type": "workspace",
                    "workspace_id": workspace_id,
                },
                "root_threads": [],
            }
        )
    )
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        service = _RecordingReceiptService()
        console._console_runtime()._activity_receipts = service
        profile, token = console._workspace._console_switcher_authority()
        target = ConsoleSwitcherTarget(
            kind=SwitcherTargetKind.PERSISTED_CONVERSATION,
            profile_authority=profile,
            authority_token=token,
            session_id=None,
            conversation_id=conversation_id,
            scope_type="workspace",
            workspace_id=workspace_id,
            receipts=(CapturedReceipt(activity_id="done-saved", status="done"),),
        )
        choice = ConsoleSwitcherChoice(
            "activate",
            ConsoleSwitcherEntry(
                row_key=f"conversation:{profile}:{conversation_id}",
                title="Saved switcher target",
                subtitle="FINISHED · UNSEEN",
                native_session_id=None,
                conversation_id=conversation_id,
                scope_type="workspace",
                workspace_id=workspace_id,
                is_active=False,
                target=target,
                group=ActivityGroup.NEW_RESULTS,
            ),
        )

        await console._session._apply_console_switcher_choice(choice)
        assert service.calls == []
        await pilot.pause()

        active = console._session._active_native_console_session()
        assert active.persisted_conversation_id == conversation_id
        assert console.query_one(ConsoleActivityOutcomeNotice).display
        assert service.calls == [("done-saved",)]


@pytest.mark.asyncio
async def test_unavailable_result_marks_only_its_frozen_receipts_without_switching():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        service = _RecordingReceiptService()
        console._console_runtime()._activity_receipts = service
        store = console._console_chat_store
        active_id = store.active_session_id
        profile, token = console._workspace._console_switcher_authority()
        unavailable = UnavailableSessionNotice(
            stable_result_key=(f"unavailable-session:{profile}:gone-native-session"),
            profile_authority=profile,
            authority_token=token,
            session_id="gone-native-session",
            group=ActivityGroup.WAITING_FOR_YOU,
            latest_at=None,
            receipts=(
                CapturedReceipt(activity_id="failed-gone", status="failed"),
                CapturedReceipt(activity_id="done-gone", status="done"),
            ),
            primary_status="failed",
            all_statuses=("failed", "done"),
        )

        await console._session._apply_console_switcher_choice(
            ConsoleSwitcherChoice("mark_seen", unavailable)
        )
        await pilot.pause()

        assert store.active_session_id == active_id
        assert service.calls == [("failed-gone", "done-gone")]


@pytest.mark.asyncio
async def test_stale_unavailable_notice_token_cannot_acknowledge_receipts():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        service = _RecordingReceiptService()
        console._console_runtime()._activity_receipts = service
        profile, _token = console._workspace._console_switcher_authority()
        stale_notice = SimpleNamespace(
            profile_authority=profile,
            authority_token="replaced-runtime-token",
            receipts=(CapturedReceipt(activity_id="stale-receipt", status="failed"),),
        )

        await console._session._mark_unavailable_switcher_notice_seen(stale_notice)
        await pilot.pause()

        assert service.calls == []


@pytest.mark.asyncio
async def test_notice_remount_invalidates_deferred_callback(monkeypatch):
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        service = _RecordingReceiptService()
        console._console_runtime()._activity_receipts = service
        store = console._console_chat_store
        profile, token = console._workspace._console_switcher_authority()
        old_notice = console.query_one(ConsoleActivityOutcomeNotice)
        deferred: list[tuple[object, tuple[object, ...]]] = []

        def hold(callback, *args, **_kwargs):  # type: ignore[no-untyped-def]
            deferred.append((callback, args))

        monkeypatch.setattr(old_notice, "call_after_refresh", hold)
        await console._session._apply_console_switcher_choice(
            _native_choice(
                session_id=store.active_session_id,
                profile=profile,
                token=token,
                receipts=(
                    CapturedReceipt(activity_id="done-before-remount", status="done"),
                ),
            )
        )
        assert deferred

        surface = console.query_one(ConsoleSessionSurface)
        await old_notice.remove()
        await surface.mount(
            ConsoleActivityOutcomeNotice(id="console-activity-outcome-notice")
        )
        callback, args = deferred.pop()
        callback(*args)  # type: ignore[operator]
        await pilot.pause()

        assert service.calls == []
        assert not surface.query_one(ConsoleActivityOutcomeNotice).display
