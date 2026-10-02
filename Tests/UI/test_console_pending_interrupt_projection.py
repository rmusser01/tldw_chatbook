"""Session-owned interrupt copy and counts through the real Console view."""

import threading
import time

import pytest
from textual.widgets import Button, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_headless_approval import (
    _arm,
    _arm_install,
    _risk_row,
    _toast_text,
    _wait_for_round,
)
from Tests.UI.test_console_store_continuity import _navigate
from Tests.UI.test_console_turn_activity_line import _rendered_row_text
from Tests.UI.test_console_turn_navigation_continuity import _build_navigation_app
from tldw_chatbook.Agents.agent_models import AGENT_KIND_PRIMARY, STEP_TOOL_CALL
from tldw_chatbook.Chat.console_agent_bridge import AgentLiveSnapshot, AgentLiveStep
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_display_state import (
    CONSOLE_INSPECTOR_NO_APPROVAL_REASON,
)
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.Utils.token_counter import resolve_context_window
from tldw_chatbook.Widgets.Console.console_run_inspector import ConsoleRunInspector
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

QUESTIONS = [
    {
        "question": "Which database?",
        "header": "Database",
        "multiSelect": False,
        "options": [{"label": "SQLite", "description": "Keep it local"}],
    }
]


def _build_app(tmp_path):
    assert save_setting_to_cli_config("splash_screen", "enabled", False)
    app, gateway = _build_navigation_app(tmp_path)
    gateway.cached_context_window = lambda settings: resolve_context_window(
        settings.provider, settings.model or ""
    )
    return app


async def _wait(pilot, predicate):
    for _ in range(80):
        if predicate():
            return
        await pilot.pause(0.05)
    assert predicate(), "Console interrupt projection did not settle"


async def _seed_console(app, pilot):
    await _wait(pilot, lambda: app._initial_screen_pushed)
    console = await _navigate(app, pilot, "chat", expect="ChatScreen")
    await _wait(pilot, lambda: bool(list(console.query("#console-native-composer"))))
    controller = console._ensure_console_chat_controller()
    store = console._console_chat_store
    session_id = store.active_session_id
    assert session_id
    outcome = await controller.submit_draft(
        "Seed this conversation.", session_id=session_id
    )
    assert outcome.accepted
    assert next(
        row for row in store.sessions() if row.id == session_id
    ).persisted_conversation_id
    return console, controller, store, session_id


def _arm_question(controller, session_id):
    worker = threading.Thread(
        target=controller.request_user_questions,
        args=(QUESTIONS,),
        kwargs={"session_id": session_id},
        daemon=True,
    )
    worker.start()
    return worker


def _start_live_turn(console, controller, store, session_id):
    assistant = store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "Agent running."),
        session_id=session_id,
    )
    bridge = console._agent._ensure_console_agent_bridge()
    assert bridge is not None
    conversation_id = console._character._current_console_rail_conversation_id()
    assert conversation_id
    bridge._publish_live(
        conversation_id,
        "interrupt-projection",
        AgentLiveSnapshot(
            status="running",
            steps=(
                AgentLiveStep(
                    STEP_TOOL_CALL, "ask_user", AGENT_KIND_PRIMARY, time.monotonic()
                ),
            ),
        ),
        primary=True,
    )
    return assistant.id


async def _assert_projection(console, pilot, assistant_id, copy, approval_count):
    await console._sync_native_console_transcript()
    console._sync_console_mode_bar()
    console._sync_console_rail_and_controls()
    await pilot.pause()
    transcript = console.query_one(ConsoleTranscript)
    text = _rendered_row_text(transcript, assistant_id)
    assert copy in text, text
    assert str(console.query_one("#console-run-chip").render()) == f"Run: {copy}."
    inspector = console.query_one("#console-run-inspector-state", ConsoleRunInspector)
    rendered = "\n".join(str(row.render()) for row in inspector.query(Static))
    assert f"Live work: {copy}" in rendered, rendered
    assert f"Approvals: {approval_count} pending" in rendered, rendered
    assert inspector.state.pending_approval_count == approval_count


async def _finish_worker(pilot, worker):
    await _wait(pilot, lambda: not worker.is_alive())
    worker.join()


async def _stop_workers(controller, workers, pilot):
    controller.begin_shutdown()
    for worker in workers:
        await _finish_worker(pilot, worker)


@pytest.mark.asyncio
@private_profile_test
async def test_question_and_approval_copy_survives_sibling_view_and_remount(
    request, tmp_path
):
    """Keep question and approval copy bound to its session across remounts.

    Args:
        request: Pytest request selecting the isolated private-profile child.
        tmp_path: Temporary directory for the navigation app fixture.
    """
    app = _build_app(tmp_path)
    workers = []
    async with app.run_test(size=(160, 48)) as pilot:
        console, controller, store, session_id = await _seed_console(app, pilot)
        assistant_id = _start_live_turn(console, controller, store, session_id)
        try:
            question_worker = _arm_question(controller, session_id)
            workers.append(question_worker)
            await _wait(
                pilot,
                lambda: (
                    bool(list(console.query("#chat-question-card")))
                    and console.query_one("#chat-question-card").display
                    and bool(console.query_one("#chat-question-card")._request_id)
                ),
            )
            await _assert_projection(
                console, pilot, assistant_id, "Waiting for your answer", 0
            )

            approval_worker, _ = _arm(controller, session_id, call=_risk_row())
            workers.append(approval_worker)
            assert await _wait_for_round(controller, session_id)
            await _wait(
                pilot,
                lambda: (
                    console.query_one("#chat-approval-card").display
                    and console.query_one("#chat-approval-card")._batch_round_id
                    in controller._pending_approval_rounds
                ),
            )
            assert console.query_one("#chat-question-card").display
            await _assert_projection(
                console, pilot, assistant_id, "Waiting for your approval", 1
            )
            console.query_one("#console-native-composer").focus()
            await pilot.press("alt+a")
            assert app.focused in tuple(
                console.query_one("#chat-approval-card").walk_children()
            )

            sibling = controller.new_session(title="Unrelated session")
            await pilot.pause()
            assert console._console_pending_approval_count() == 0
            assert not console._agent.console_turn_activity()
            controller.switch_session(session_id)
            await _wait(
                pilot,
                lambda: (
                    bool(list(console.query("#chat-question-card")))
                    and console.query_one("#chat-question-card").display
                    and bool(console.query_one("#chat-question-card")._request_id)
                ),
            )

            runtime = console._console_runtime()
            await _navigate(
                app, pilot, "library", expect="LibraryScreen", allow_confirmation=False
            )
            assert not runtime.has_answerable_view()
            # Console routes suspend their installed view. Evict that suspended
            # view to exercise Textual teardown and a fresh mount as well.
            app._reusable_screen_instances.pop("chat")
            app.uninstall_screen(console)
            await console.remove()
            await _wait(pilot, lambda: runtime.view is None)
            assert controller.pending_round_count(session_id) == 1
            assert controller.pending_round_kinds(session_id) == frozenset(
                {"question", "approval"}
            )
            assert controller.pending_round_count(sibling.id) == 0
            assert question_worker.is_alive() and approval_worker.is_alive()
            reopened = await _navigate(app, pilot, "chat", expect="ChatScreen")
            assert reopened is not console
            await _wait(
                pilot,
                lambda: (
                    bool(list(reopened.query("#chat-approval-card")))
                    and reopened.query_one("#chat-approval-card").display
                    and reopened.query_one("#chat-approval-card")._batch_round_id
                    in controller._pending_approval_rounds
                ),
            )
            await _assert_projection(
                reopened, pilot, assistant_id, "Waiting for your approval", 1
            )

            round_id = reopened.query_one("#chat-approval-card")._batch_round_id
            controller.resolve_pending_approval(
                {"builtin__write_file": "deny"}, round_id=round_id
            )
            await _finish_worker(pilot, approval_worker)
            await _assert_projection(
                reopened, pilot, assistant_id, "Waiting for your answer", 0
            )
            question_id = reopened.query_one("#chat-question-card")._request_id
            controller.resolve_pending_question(
                [
                    {
                        "question": "Which database?",
                        "selected": [],
                        "other_text": None,
                        "unanswered": True,
                    }
                ],
                request_id=question_id,
            )
            await _finish_worker(pilot, question_worker)
            assert not controller.pending_round_kinds(session_id)
        finally:
            await _stop_workers(controller, workers, pilot)


@pytest.mark.asyncio
@private_profile_test
async def test_inspector_counts_queued_approval_rounds_for_its_own_session(
    request, tmp_path
):
    """Count queued approvals only for the session shown in Inspector.

    Args:
        request: Pytest request selecting the isolated private-profile child.
        tmp_path: Temporary directory for the navigation app fixture.
    """
    app = _build_app(tmp_path)
    workers = []
    async with app.run_test(size=(160, 48)) as pilot:
        console, controller, store, session_id = await _seed_console(app, pilot)
        assistant_id = _start_live_turn(console, controller, store, session_id)
        try:
            for _ in range(2):
                worker, _ = _arm(controller, session_id, call=_risk_row())
                workers.append(worker)
            await _wait(
                pilot,
                lambda: len(controller._pending_approvals.get(session_id, ())) == 2,
            )
            await _assert_projection(
                console, pilot, assistant_id, "Waiting for your approval", 2
            )
            assert controller.pending_round_count(session_id) == 2
            await _wait(
                pilot,
                lambda: (
                    console.query_one("#chat-approval-card").display
                    and console.query_one("#chat-approval-card")._batch_round_id
                    in controller._pending_approval_rounds
                ),
            )
            round_id = console.query_one("#chat-approval-card")._batch_round_id
            controller.resolve_pending_approval(
                {"builtin__write_file": "deny"}, round_id=round_id
            )
            await _wait(pilot, lambda: controller.pending_round_count(session_id) == 1)
            await _assert_projection(
                console, pilot, assistant_id, "Waiting for your approval", 1
            )
        finally:
            await _stop_workers(controller, workers, pilot)


@pytest.mark.asyncio
@private_profile_test
async def test_review_routes_reach_visible_skill_confirm_before_queued_approval(
    request, tmp_path
):
    """Route each review action to the displayed decision before queued approval.

    Args:
        request: Pytest request selecting the isolated private-profile child.
        tmp_path: Temporary directory for the navigation app fixture.
    """
    app = _build_app(tmp_path)
    workers = []
    async with app.run_test(size=(160, 48), notifications=True) as pilot:
        console, controller, store, session_id = await _seed_console(app, pilot)
        assistant_id = _start_live_turn(console, controller, store, session_id)
        try:
            install_worker, _ = _arm_install(controller, session_id)
            workers.append(install_worker)
            await _wait(
                pilot,
                lambda: (
                    console.query_one("#chat-skill-install-card").display
                    and bool(console.query_one("#chat-skill-install-card")._request_id)
                ),
            )
            approval_worker, _ = _arm(controller, session_id, call=_risk_row())
            workers.append(approval_worker)
            assert await _wait_for_round(controller, session_id)
            assert not console.query_one("#chat-approval-card").display
            assert console._task_resume_state.pending_skill_install is not None
            await _assert_projection(
                console, pilot, assistant_id, "Waiting for your approval", 1
            )

            install_card = console.query_one("#chat-skill-install-card")
            allow = install_card.query_one("#skill-install-allow", Button)
            for entry_point in ("shortcut", "inspector", "session_tab"):
                console.query_one("#console-native-composer").focus()
                if entry_point == "shortcut":
                    await pilot.press("alt+a")
                elif entry_point == "inspector":
                    console.query_one(
                        "#console-inspector-review-approval", Button
                    ).press()
                else:
                    console.query_one(
                        f"#console-session-tab-{session_id}", Button
                    ).press()
                await pilot.pause()
                assert app.focused is allow, (entry_point, _toast_text(app))
                assert app.screen is console
                assert CONSOLE_INSPECTOR_NO_APPROVAL_REASON not in _toast_text(app)

            controller.resolve_pending_skill_install(
                False, request_id=install_card._request_id
            )
            await _finish_worker(pilot, install_worker)
            await _wait(
                pilot,
                lambda: (
                    console.query_one("#chat-approval-card").display
                    and console.query_one("#chat-approval-card")._batch_round_id
                    in controller._pending_approval_rounds
                ),
            )
            console.query_one("#console-native-composer").focus()
            await pilot.press("alt+a")
            approval_card = console.query_one("#chat-approval-card")
            assert app.focused in tuple(approval_card.walk_children())
            assert controller.pending_round_count(session_id) == 1
        finally:
            await _stop_workers(controller, workers, pilot)


@pytest.mark.asyncio
@private_profile_test
async def test_chat_create_confirmation_has_its_own_kind_and_review_route(
    request, tmp_path
):
    """Keep real chat creation separate from tool approvals and reviewable.

    Args:
        request: Pytest request selecting the isolated private-profile child.
        tmp_path: Temporary directory for the navigation app fixture.
    """
    app = _build_app(tmp_path)
    workers = []
    async with app.run_test(size=(160, 48), notifications=True) as pilot:
        console, controller, store, session_id = await _seed_console(app, pilot)
        assistant_id = _start_live_turn(console, controller, store, session_id)
        result = {}
        worker = threading.Thread(
            target=lambda: result.update(
                controller.request_chat_create_confirm(
                    {"tool": "new_chat", "title": "Proposed chat"},
                    session_id=session_id,
                )
            ),
            daemon=True,
        )
        workers.append(worker)
        worker.start()
        try:
            await _wait(
                pilot,
                lambda: (
                    bool(list(console.query("#chat-create-card")))
                    and console.query_one("#chat-create-card").display
                    and bool(console.query_one("#chat-create-card")._request_id)
                ),
            )
            assert controller.pending_round_kinds(session_id) == {"chat_create"}
            await _assert_projection(
                console, pilot, assistant_id, "Waiting for your confirmation", 0
            )
            card = console.query_one("#chat-create-card")
            allow = card.query_one("#chat-create-allow", Button)
            for entry_point in ("shortcut", "session_tab"):
                console.query_one("#console-native-composer").focus()
                if entry_point == "shortcut":
                    await pilot.press("alt+a")
                else:
                    console.query_one(
                        f"#console-session-tab-{session_id}", Button
                    ).press()
                await pilot.pause()
                assert app.focused is allow, (entry_point, _toast_text(app))
                assert CONSOLE_INSPECTOR_NO_APPROVAL_REASON not in _toast_text(app)

            approval_worker, _ = _arm(controller, session_id, call=_risk_row())
            workers.append(approval_worker)
            assert await _wait_for_round(controller, session_id)
            await _wait(pilot, lambda: console.query_one("#chat-approval-card").display)
            await _assert_projection(
                console, pilot, assistant_id, "Waiting for your approval", 1
            )
            console.query_one("#console-native-composer").focus()
            await pilot.press("alt+a")
            assert app.focused in tuple(
                console.query_one("#chat-approval-card").walk_children()
            )
            sibling = controller.new_session(title="Unrelated session")
            await pilot.pause()
            assert console._console_pending_approval_count() == 0
            assert not controller.pending_round_kinds(sibling.id)
            controller.switch_session(session_id)
            await _wait(pilot, lambda: console.query_one("#chat-create-card").display)
            controller.resolve_pending_chat_create(False, False, request_id=card._request_id)
            await _finish_worker(pilot, worker)
            assert result == {"allow": False, "remember": False}
            assert controller.pending_round_kinds(session_id) == {"approval"}
        finally:
            await _stop_workers(controller, workers, pilot)
