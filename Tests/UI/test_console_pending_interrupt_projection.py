"""Session-owned interrupt copy and counts through the real Console view."""

import gc
import threading
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import drain_active_service_patches, drain_created_dirs
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
    await _wait(
        pilot,
        lambda: all(
            any(row.is_mounted for row in inspector.query(f"#{row_id}"))
            for row_id in (
                "console-inspector-live-work",
                "console-inspector-approvals",
            )
        )
        and f"Live work: {copy}"
        in "\n".join(str(row.render()) for row in inspector.query(Static))
        and f"Approvals: {approval_count} pending"
        in "\n".join(str(row.render()) for row in inspector.query(Static))
        and inspector.state.pending_approval_count == approval_count,
    )
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


async def _verify_question_and_approval_copy_survives_sibling_view_and_remount(
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


async def _verify_inspector_counts_queued_approval_rounds_for_its_own_session(
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


async def _verify_review_routes_reach_visible_skill_confirm_before_queued_approval(
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


async def _verify_chat_create_confirmation_has_its_own_kind_and_review_route(
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
        from Tests.Chat.test_console_chat_create_integration import (
            _prepare_close_new_chat,
        )
        from tldw_chatbook.Agents.run_context import use_run_id

        controller._active_assistant_message_ids[session_id] = assistant_id
        prepared, run = _prepare_close_new_chat(controller, session_id, "Proposed chat")
        result = {}

        def request_confirmation():
            with use_run_id(run):
                result.update(
                    controller.request_chat_create_confirm(
                        prepared, session_id=session_id
                    )
                )

        worker = threading.Thread(target=request_confirmation, daemon=True)
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
            controller.resolve_pending_chat_create(
                False, False, request_id=card._request_id
            )
            await _finish_worker(pilot, worker)
            assert result == {"allow": False, "remember": False}
            assert controller.pending_round_kinds(session_id) == {"approval"}
        finally:
            await _stop_workers(controller, workers, pilot)


_CHAT_CREATE_PRESENTATION_SYNC_TIMEOUT_SECONDS = 5


async def _verify_late_chat_create_projection_spares_the_active_sibling(
    request, tmp_path
):
    """Keep a sibling confirmation through late source projection and clear.

    Args:
        request: Pytest request selecting the isolated private-profile child.
        tmp_path: Parent directory for both source-transition app fixtures.
    """
    for closing in (True, False):
        await _verify_late_chat_create_projection_transition(
            request, tmp_path, closing=closing
        )


async def _verify_late_chat_create_projection_transition(
    request, tmp_path, *, closing: bool
):
    """A delayed source marshal cannot replace a live sibling's confirmation.

    Args:
        request: Pytest request selecting the isolated private-profile child.
        tmp_path: Separate app and database directory for each source transition.
        closing: Complete source Close or retain it across navigation and a clear.
    """
    case_path = tmp_path / ("close" if closing else "navigate")
    case_path.mkdir()
    app = _build_app(case_path)
    workers = []
    async with app.run_test(size=(160, 48)) as pilot:
        console, controller, store, source_id = await _seed_console(app, pilot)
        entered = threading.Event()
        release = threading.Event()
        results = {}
        errors = []
        marshalled = threading.Event()
        clear_requested = threading.Event()
        clear_entered = threading.Event()
        clear_release = threading.Event()
        cleared = threading.Event()
        original_marshal = controller._marshal_pending_chat_create

        def delayed_marshal(payload):
            """Pause source projection or its queued clear before real UI dispatch.

            Args:
                payload: Original controller confirmation or teardown payload.
            """
            is_source_clear = (
                payload is None
                and threading.current_thread() is source_worker
                and clear_requested.is_set()
            )
            if is_source_clear:
                clear_entered.set()
                assert clear_release.wait(
                    _CHAT_CREATE_PRESENTATION_SYNC_TIMEOUT_SECONDS
                )
            if payload and payload.get("session_id") == source_id:
                entered.set()
                assert release.wait(_CHAT_CREATE_PRESENTATION_SYNC_TIMEOUT_SECONDS)
            original_marshal(payload)
            if payload and payload.get("session_id") == source_id:
                marshalled.set()
            if is_source_clear:
                cleared.set()

        def request_confirmation(session_id, key):
            """Exercise the real worker request and report failures to its owner.

            Args:
                session_id: Existing source or live sibling session.
                key: Result identity for this worker.
            """
            from Tests.Chat.test_console_chat_create_integration import (
                _prepare_close_new_chat,
            )
            from tldw_chatbook.Agents.run_context import use_run_id

            try:
                prepared, run = _prepare_close_new_chat(
                    controller, session_id, f"Proposed {key}"
                )
                with use_run_id(run):
                    results[key] = controller.request_chat_create_confirm(
                        prepared,
                        session_id=session_id,
                    )
            except Exception as exc:  # noqa: BLE001 - expose worker failure
                errors.append((key, type(exc).__name__))

        controller._marshal_pending_chat_create = delayed_marshal
        source_worker = threading.Thread(
            target=request_confirmation, args=(source_id, "source"), daemon=True
        )
        workers.append(source_worker)
        source_worker.start()
        try:
            await _wait(pilot, entered.is_set)
            source_round = controller.pending_chat_create_ids()[0]
            sibling = controller.new_session(title="Live sibling")
            sibling_worker = threading.Thread(
                target=request_confirmation,
                args=(sibling.id, "sibling"),
                daemon=True,
            )
            workers.append(sibling_worker)
            sibling_worker.start()
            await _wait(
                pilot,
                lambda: (
                    bool(list(console.query("#chat-create-card")))
                    and console.query_one("#chat-create-card").display
                    and console.query_one("#chat-create-card")._request_id
                    != source_round
                ),
            )
            sibling_round = console.query_one("#chat-create-card")._request_id
            assert sibling_round
            if closing:
                ticket = controller.begin_session_close(
                    source_id,
                    expected_revision=controller.lifecycle_impact(
                        session_id=source_id
                    ).revision,
                )
                controller.finalize_session_close(ticket)
                assert not any(s.id == source_id for s in store.sessions())
            release.set()
            await _wait(pilot, marshalled.is_set)
            if closing:
                await _finish_worker(pilot, source_worker)
                assert results["source"] == {"allow": False, "remember": False}
            else:
                await _wait(
                    pilot,
                    lambda: (
                        controller.pending_chat_create_ids()
                        == [source_round, sibling_round]
                    ),
                )
                await pilot.pause()
            assert not errors, errors
            assert store.active_session_id == sibling.id
            state = console._task_resume_state
            assert state.pending_chat_create["request_id"] == sibling_round
            card = console.query_one("#chat-create-card")
            assert card.display and card._request_id == sibling_round
            if not closing:
                # Capture a clear while its source is viewed, then restore
                # the sibling before that real UI dispatch completes.
                controller.switch_session(source_id)
                clear_requested.set()
                controller.resolve_pending_chat_create(
                    False, False, request_id=source_round
                )
                await _wait(pilot, clear_entered.is_set)
                controller.switch_session(sibling.id)
                await _wait(
                    pilot,
                    lambda: (
                        console.query_one("#chat-create-card")._request_id
                        == sibling_round
                    ),
                )
                clear_release.set()
                await _wait(pilot, cleared.is_set)
                await _finish_worker(pilot, source_worker)
                current = console._task_resume_state.pending_chat_create
                assert current is not None, (
                    "Delayed source clear erased the live sibling"
                )
                assert current["request_id"] == sibling_round
                assert (
                    console.query_one("#chat-create-card")._request_id == sibling_round
                )
            controller.resolve_pending_chat_create(
                True, False, request_id=sibling_round
            )
            await _finish_worker(pilot, sibling_worker)
            assert results["sibling"] == {"allow": True, "remember": False}
            assert controller.pending_chat_create_ids() == []
        finally:
            release.set()
            clear_release.set()
            controller._marshal_pending_chat_create = original_marshal
            await _stop_workers(controller, workers, pilot)


@pytest.mark.parametrize(
    ("registered", "task_approval", "expected"),
    [
        (0, None, 0),
        (0, {"round_id": "legacy"}, 1),
        (0, {"round_id": "finishing", "phase": "finishing"}, 0),
        (0, {"round_id": "owned", "session_id": "viewed"}, 1),
        (0, {"round_id": "other", "session_id": "other"}, 0),
        (0, {"round_id": "unscoped", "session_id": None}, 1),
        (0, {"round_id": "unscoped", "session_id": ""}, 1),
        (0, {"round_id": "other", "session_id": "other", "phase": "finishing"}, 0),
        (2, {"round_id": "other", "session_id": "other"}, 2),
        (2, None, 2),
        (2, {"round_id": "legacy"}, 2),
        (2, {"round_id": "finishing", "phase": "finishing"}, 2),
    ],
)
def test_registry_count_keeps_tool_card_compatibility_kind_specific(
    registered: int, task_approval: dict[str, object] | None, expected: int
) -> None:
    """Use the typed tool card without importing unrelated global attention.

    Args:
        registered: Active-session approval count from the real registry seam.
        task_approval: Tool-specific card state, including a finishing control.
        expected: Honest count preserving queued rounds and legacy compatibility.
    """
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = SimpleNamespace(
        _console_chat_controller=SimpleNamespace(
            pending_round_count=lambda session_id: registered
        ),
        _console_chat_store=SimpleNamespace(active_session_id="viewed"),
        _task_resume_state=SimpleNamespace(pending_approval=task_approval),
        app_instance=SimpleNamespace(
            console_pending_approval_count=99,
            pending_console_approval={"kind": "question"},
        ),
    )
    assert ChatScreen._console_pending_approval_count(screen) == expected


async def _verify_legacy_tool_approval_counts_in_the_mounted_view(request, tmp_path):
    """Count a real unscoped tool card on every generic approval-count surface.

    Args:
        request: Existing isolated private-profile collection owner.
        tmp_path: Separate database directory for this fresh mounted Console.
    """
    app = _build_app(tmp_path)
    workers = []
    async with app.run_test(size=(160, 48)) as pilot:
        console, controller, store, session_id = await _seed_console(app, pilot)
        runtime = console._console_runtime()
        attachment_generation = runtime._attached_generation
        # This journey produces an approval for an already mounted view.
        await _wait(
            pilot,
            lambda: (
                console._console_attach_reconciled
                and not console._console_attach_reconcile_running
                and runtime.view is console
                and runtime._attached_generation == attachment_generation
                and runtime._reconciled_view is console
                and runtime.has_answerable_view()
            ),
        )
        assert runtime.chat_controller is controller
        assert runtime.chat_store is store
        assert store.active_session_id == session_id
        _start_live_turn(console, controller, store, session_id)
        try:
            worker, result = _arm(controller, None, call=_risk_row())
            workers.append(worker)
            await _wait(
                pilot,
                lambda: (
                    bool(list(console.query("#chat-approval-card")))
                    and console.query_one("#chat-approval-card").display
                    and bool(console.query_one("#chat-approval-card")._batch_round_id)
                ),
            )
            round_id = console.query_one("#chat-approval-card")._batch_round_id
            assert round_id in controller._pending_approval_rounds
            assert worker.is_alive()
            # Legacy calls have a host round, but deliberately no kind/badge entry.
            assert controller.pending_round_count(session_id) == 0
            assert console._task_resume_state.pending_approval["round_id"] == round_id
            assert console._console_pending_approval_count() == 1
            console._sync_console_rail_and_controls()
            inspector = console.query_one(
                "#console-run-inspector-state", ConsoleRunInspector
            )
            # A busy config lock defers refresh; observe its actual publication.
            await _wait(
                pilot,
                lambda: (
                    inspector.state.pending_approval_count == 1
                    and "Approvals: 1 pending"
                    in "\n".join(str(row.render()) for row in inspector.query(Static))
                ),
            )
            assert inspector.state.pending_approval_count == 1
            rendered = "\n".join(str(row.render()) for row in inspector.query(Static))
            assert "Approvals: 1 pending" in rendered, rendered
            attention = console._workspace._workspace_files_attention_snapshot()
            assert attention.pending_approval_count == 1
            assert "1 approval waiting" in attention.status_copy
            controller.resolve_pending_approval(
                {"builtin__write_file": "deny"}, round_id=round_id
            )
            await _finish_worker(pilot, worker)
            assert result["decisions"] == {"builtin__write_file": "deny"}
            assert console._console_pending_approval_count() == 0
            console._sync_console_rail_and_controls()
            await _wait(pilot, lambda: inspector.state.pending_approval_count == 0)
            assert inspector.state.pending_approval_count == 0
            assert (
                console._workspace._workspace_files_attention_snapshot().pending_approval_count
                == 0
            )
        finally:
            await _stop_workers(controller, workers, pilot)


@pytest.mark.asyncio
@private_profile_test
async def test_session_owned_pending_projection_journeys(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    """Run every mounted projection journey with fresh app and worker ownership.

    Args:
        request: Pytest request selecting the one isolated private-profile child.
        tmp_path: Parent directory for each journey's separate database fixture.
    """
    journeys = (
        _verify_question_and_approval_copy_survives_sibling_view_and_remount,
        _verify_inspector_counts_queued_approval_rounds_for_its_own_session,
        _verify_review_routes_reach_visible_skill_confirm_before_queued_approval,
        _verify_chat_create_confirmation_has_its_own_kind_and_review_route,
        _verify_late_chat_create_projection_spares_the_active_sibling,
        _verify_legacy_tool_approval_counts_in_the_mounted_view,
    )
    for index, journey in enumerate(journeys):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        try:
            await journey(request, case_path)
        finally:
            drain_active_service_patches()
            drain_created_dirs()
            gc.unfreeze()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                gc.collect()
