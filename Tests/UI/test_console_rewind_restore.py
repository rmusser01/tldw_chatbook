"""Screen wiring for the Console `/rewind` command's restore path (SP2 Task 1).

Mirrors ``Tests/UI/test_console_edit_resend_wiring.py``'s harness style:
``ConsoleHarness`` mounts a real ``ChatScreen`` over a real
``ConsoleChatStore``/``ConsoleChatController`` pair. Legacy wiring tests drive
the screen's restore machinery directly (``_console_command_rewind`` /
``_apply_console_rewind_choice``); the TASK-2705 regressions use the mounted
keyboard/Send product paths and real modal interactions. Focused modal behavior
is also covered by ``Tests/Chat/test_console_rewind_modal.py``.

Restore is pure tree navigation (SP1 primitives): the selected USER prompt's
PARENT (found by an id lookup in ``active_path_message_ids``, never
positional) becomes the new active leaf, and the prompt's own text is written
back into the composer via the same ``_insert_prompt_text_into_composer``
seam ``/prompt`` uses.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger
from textual.events import Key
from textual.widgets import Button, Static

from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
    derive_console_session_title,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult
from tldw_chatbook.Chat.console_command_grammar import CommandParse
from tldw_chatbook.Chat.console_context_compaction import (
    EffectiveMemoryKind,
    EffectiveMemoryResult,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar, ConsoleTranscript
from tldw_chatbook.Widgets.Console.console_rewind_modal import (
    ConsoleRewindChoice,
    ConsoleRewindModal,
)
from Tests.UI.app_factory import attach_chachanotes_db


CONSOLE_RUN_ALREADY_RUNNING_COPY = "A run is already running in this tab."
KIND_SUMMARIZE_UP_TO = "summarize-up-to"
KIND_SUMMARIZE_FROM = "summarize-from"


def _press_enter_synchronously(console) -> None:
    """Deliver Enter to the screen key handler without pumping messages."""
    console.on_key(Key(key="enter", character="\r"))


async def _seed_u1_a1_u2_a2(console):
    """Build a linear U1->A1->U2->A2 session and return (session, ids-by-label)."""
    store = console._ensure_console_chat_store()
    session = store.ensure_session()
    u1 = store.append_message(session.id, role=ConsoleMessageRole.USER, content="U1")
    a1 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="A1"
    )
    u2 = store.append_message(session.id, role=ConsoleMessageRole.USER, content="U2")
    a2 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="A2"
    )
    await console._sync_native_console_chat_ui()
    return session, {"u1": u1, "a1": a1, "u2": u2, "a2": a2}


def _switch_to_sibling_descendant(store, session_id: str, anchor_id: str):
    """Replace the active tail after ``anchor_id`` with a sibling branch."""
    store.set_active_leaf(session_id, anchor_id)
    user = store.append_message(
        session_id,
        role=ConsoleMessageRole.USER,
        content="Sibling branch user",
    )
    assistant = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Sibling branch assistant",
    )
    return user, assistant


@pytest.mark.asyncio
async def test_restore_mid_path_truncates_active_path_and_refills_composer():
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)

        spy_insert = MagicMock(return_value=True)
        console._commands._insert_prompt_text_into_composer = spy_insert

        await console._commands._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(kind="restore", message_id=ids["u2"].id, prompt_text="U2"),
        )
        await pilot.pause()

    assert store.active_path_message_ids(session.id) == [ids["u1"].id, ids["a1"].id]
    spy_insert.assert_called_once_with("U2", replace=True)


@pytest.mark.asyncio
async def test_restore_to_first_prompt_clears_active_leaf_to_empty_path(monkeypatch):
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)

        spy_before = MagicMock(wraps=store.set_active_path_before)
        spy_leaf = MagicMock(wraps=store.set_active_leaf)
        monkeypatch.setattr(store, "set_active_path_before", spy_before)
        monkeypatch.setattr(store, "set_active_leaf", spy_leaf)
        spy_insert = MagicMock(return_value=True)
        console._commands._insert_prompt_text_into_composer = spy_insert

        await console._commands._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(kind="restore", message_id=ids["u1"].id, prompt_text="U1"),
        )
        await pilot.pause()

    assert store.active_leaf(session.id) is None
    assert store.active_path_message_ids(session.id) == []
    spy_before.assert_called_once_with(session.id, ids["u1"].id)
    spy_leaf.assert_not_called()
    spy_insert.assert_called_once_with("U1", replace=True)


@pytest.mark.asyncio
async def test_first_prompt_warns_if_restart_cursor_is_unsaved(monkeypatch):
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)

        spy_insert = MagicMock(return_value=True)
        console._commands._insert_prompt_text_into_composer = spy_insert
        original = store.set_active_path_before

        def apply_but_report_unsaved(session_id: str, message_id: str) -> bool:
            assert original(session_id, message_id) is True
            return False

        monkeypatch.setattr(
            store, "set_active_path_before", apply_but_report_unsaved
        )
        notices: list[tuple[str, str]] = []
        app.notify = lambda text, **kwargs: notices.append(
            (str(text), kwargs.get("severity", ""))
        )

        await console._commands._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(
                kind="restore",
                message_id=ids["u1"].id,
                prompt_text="truncated preview",
            ),
        )
        await pilot.pause()

    assert store.active_path_message_ids(session.id) == []
    spy_insert.assert_called_once_with("U1", replace=True)
    assert (
        "Rewound for this session, but the restart position could not be saved.",
        "warning",
    ) in notices


@pytest.mark.asyncio
async def test_restore_blocked_while_a_run_is_streaming_makes_no_mutation():
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)
        original_path = store.active_path_message_ids(session.id)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        controller = console._ensure_console_chat_controller()
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response.")
        )

        spy_insert = MagicMock(return_value=True)
        console._commands._insert_prompt_text_into_composer = spy_insert

        notices: list[tuple[str, str]] = []
        app.notify = lambda message_text, **kwargs: notices.append(
            (str(message_text), kwargs.get("severity", ""))
        )

        await console._commands._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(kind="restore", message_id=ids["u2"].id, prompt_text="U2"),
        )
        await pilot.pause()
        assert console._is_descendant_or_self(host.focused, composer)

    # No mutation at all: active path unchanged, composer untouched.
    assert store.active_path_message_ids(session.id) == original_path
    spy_insert.assert_not_called()
    assert (CONSOLE_RUN_ALREADY_RUNNING_COPY, "warning") in notices


@pytest.mark.asyncio
async def test_none_choice_just_refocuses_composer_without_mutation():
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)
        original_path = store.active_path_message_ids(session.id)

        spy_focus = MagicMock()
        console._focus_console_composer_if_needed = spy_focus

        await console._commands._apply_console_rewind_choice(session.id, None)
        await pilot.pause()

    assert store.active_path_message_ids(session.id) == original_path
    spy_focus.assert_called_once_with(force=True)


@pytest.mark.parametrize(
    ("kind", "worker_name"),
    [
        (KIND_SUMMARIZE_UP_TO, "_summarize_console_up_to"),
        (KIND_SUMMARIZE_FROM, "_summarize_console_from"),
    ],
)
@pytest.mark.asyncio
async def test_summary_choice_dispatches_symmetric_exclusive_worker_without_mutation(
    kind, worker_name, monkeypatch
):
    """Either direction losing the shared exclusive group permits overlap."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)
        original_path = store.active_path_message_ids(session.id)
        captured_path = tuple(original_path)
        controller = console._ensure_console_chat_controller()

        async def marker_worker(*_args):
            return None

        routed = MagicMock(side_effect=lambda *_args: marker_worker())
        monkeypatch.setattr(console, worker_name, routed)
        spy_worker = MagicMock(side_effect=lambda coro, **kwargs: coro.close())
        console.run_worker = spy_worker

        await console._commands._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(
                kind=kind, message_id=ids["u2"].id, prompt_text="U2"
            ),
        )
        await pilot.pause()

    run_calls = [
        call
        for call in spy_worker.call_args_list
        if call.kwargs.get("group") == f"console-run-{session.id}"
    ]
    assert len(run_calls) == 1
    assert run_calls[0].kwargs["exclusive"] is True
    group = run_calls[0].kwargs.get("group")
    assert isinstance(group, str) and group.startswith("console-run-"), group
    assert group == f"console-run-{session.id}", group
    routed.assert_called_once_with(
        controller,
        session.id,
        ids["u2"].id,
        captured_path,
    )
    # Summarize never mutates the transcript tree, and nothing is stored until
    # the (unrun) worker succeeds.
    assert store.active_path_message_ids(session.id) == original_path
    assert store.session_context_summary(session.id) == (None, None)


@pytest.mark.parametrize(
    ("kind", "status", "run_copy"),
    [
        (KIND_SUMMARIZE_UP_TO, ConsoleRunStatus.VALIDATING, "Sending request."),
        (KIND_SUMMARIZE_FROM, ConsoleRunStatus.STREAMING, "Streaming response."),
        (KIND_SUMMARIZE_FROM, ConsoleRunStatus.VALIDATING, "Compacting context."),
    ],
)
@pytest.mark.asyncio
async def test_summary_choices_refuse_sending_streaming_and_compacting(
    kind, status, run_copy
):
    """No summary worker may cancel an already-active Console operation."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)
        original_path = store.active_path_message_ids(session.id)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        controller = console._ensure_console_chat_controller()
        controller._set_run_state(ConsoleRunState(status, run_copy))
        spy_worker = MagicMock(side_effect=lambda coro, **kwargs: coro.close())
        console.run_worker = spy_worker

        notices: list[tuple[str, str]] = []
        app.notify = lambda message_text, **kwargs: notices.append(
            (str(message_text), kwargs.get("severity", ""))
        )

        await console._commands._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(
                kind=kind, message_id=ids["u2"].id, prompt_text="U2"
            ),
        )
        await pilot.pause()
        assert console._is_descendant_or_self(host.focused, composer)

    assert not any(
        call.kwargs.get("group") == f"console-run-{session.id}"
        for call in spy_worker.call_args_list
    )
    assert (CONSOLE_RUN_ALREADY_RUNNING_COPY, "warning") in notices
    assert store.active_path_message_ids(session.id) == original_path
    assert store.session_context_summary(session.id) == (None, None)


@pytest.mark.asyncio
async def test_both_summary_workers_share_one_non_overlapping_exclusive_group(
    monkeypatch,
):
    """Wrong worker groups let up-to and from-here execute concurrently."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        session, ids = await _seed_u1_a1_u2_a2(console)
        release = asyncio.Event()
        started = [asyncio.Event(), asyncio.Event()]
        active = 0
        max_active = 0
        starts = 0

        async def held_worker(*_args):
            nonlocal active, max_active, starts
            slot = starts
            starts += 1
            active += 1
            max_active = max(max_active, active)
            started[slot].set()
            try:
                await release.wait()
            finally:
                active -= 1

        monkeypatch.setattr(console, "_summarize_console_up_to", held_worker)
        monkeypatch.setattr(console, "_summarize_console_from", held_worker)
        real_run_worker = console.run_worker
        workers = []

        def capture_worker(coro, **kwargs):
            worker = real_run_worker(coro, **kwargs)
            workers.append(worker)
            return worker

        monkeypatch.setattr(console, "run_worker", capture_worker)

        await console._commands._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(
                kind=KIND_SUMMARIZE_UP_TO,
                message_id=ids["u2"].id,
                prompt_text="U2",
            ),
        )
        await asyncio.wait_for(started[0].wait(), timeout=1)
        await console._commands._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(
                kind=KIND_SUMMARIZE_FROM,
                message_id=ids["u2"].id,
                prompt_text="U2",
            ),
        )
        await asyncio.wait_for(started[1].wait(), timeout=1)
        assert max_active == 1

        release.set()
        await workers[-1].wait()
        await pilot.pause()
        assert active == 0


@pytest.mark.asyncio
async def test_queued_summary_refuses_sibling_descendant_path_change(
    monkeypatch,
):
    """Membership is insufficient when the selected prompt stays on a new branch."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)
        original_path = tuple(store.active_path_message_ids(session.id))
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep sibling draft")
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(ids["a1"].id)
        await pilot.pause()
        assert transcript.selected_message_id == ids["a1"].id

        controller = console._ensure_console_chat_controller()
        summarize = AsyncMock(
            return_value=ConsoleSubmitResult(True, False, "controller success")
        )
        monkeypatch.setattr(controller, "summarize_from", summarize)
        queued: list = []
        monkeypatch.setattr(
            console,
            "run_worker",
            lambda coroutine, **_kwargs: queued.append(coroutine),
        )
        notices: list[tuple[str, str]] = []
        app.notify = lambda text, **kwargs: notices.append(
            (str(text), kwargs.get("severity", ""))
        )

        await console._commands._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(
                kind=KIND_SUMMARIZE_FROM,
                message_id=ids["u1"].id,
                prompt_text="U1",
            ),
        )
        assert len(queued) == 1

        _switch_to_sibling_descendant(store, session.id, ids["a1"].id)
        changed_path = tuple(store.active_path_message_ids(session.id))
        assert changed_path != original_path
        assert ids["u1"].id in changed_path
        changed_messages = tuple(
            (message.id, message.role, message.content)
            for message in store.messages_for_session(session.id)
        )

        await queued[0]
        await pilot.pause()

        summarize.assert_not_awaited()
        assert tuple(store.active_path_message_ids(session.id)) == changed_path
        assert tuple(
            (message.id, message.role, message.content)
            for message in store.messages_for_session(session.id)
        ) == changed_messages
        assert composer.draft_text() == "keep sibling draft"
        assert transcript.selected_message_id == ids["a1"].id
        assert console._is_descendant_or_self(host.focused, composer)
        assert (
            "Conversation changed before summarization could start.",
            "warning",
        ) in notices


@pytest.mark.asyncio
async def test_summary_worker_refuses_if_captured_selection_changes_before_start(
    monkeypatch,
):
    """A queued worker must not apply its stale selected prompt to a new path."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep this draft")
        controller = console._ensure_console_chat_controller()
        summarize = AsyncMock(
            return_value=ConsoleSubmitResult(True, False, "controller success")
        )
        monkeypatch.setattr(controller, "summarize_from", summarize)
        notices: list[tuple[str, str]] = []
        app.notify = lambda text, **kwargs: notices.append(
            (str(text), kwargs.get("severity", ""))
        )

        store.set_active_leaf(session.id, ids["a1"].id)
        changed_path = store.active_path_message_ids(session.id)
        changed_messages = tuple(
            (message.id, message.role, message.content)
            for message in store.messages_for_session(session.id)
        )
        await console._summarize_console_from(
            controller, session.id, ids["u2"].id
        )
        await pilot.pause()

        assert store.active_path_message_ids(session.id) == changed_path
        assert tuple(
            (message.id, message.role, message.content)
            for message in store.messages_for_session(session.id)
        ) == changed_messages
        assert composer.draft_text() == "keep this draft"
        assert console._is_descendant_or_self(host.focused, composer)
        summarize.assert_not_awaited()
        assert (
            "Conversation changed before summarization could start.",
            "warning",
        ) in notices


@pytest.mark.asyncio
async def test_summary_worker_refuses_if_captured_session_changes_before_start(
    monkeypatch,
):
    """A queued worker must never apply an old session's selection elsewhere."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)
        old_messages = tuple(
            (message.id, message.role, message.content)
            for message in store.messages_for_session(session.id)
        )
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep this draft")
        controller = console._ensure_console_chat_controller()
        summarize = AsyncMock(
            return_value=ConsoleSubmitResult(True, False, "controller success")
        )
        monkeypatch.setattr(controller, "summarize_from", summarize)
        notices: list[tuple[str, str]] = []
        app.notify = lambda text, **kwargs: notices.append(
            (str(text), kwargs.get("severity", ""))
        )

        other_session = store.create_session(title="other")
        composer.load_draft("keep this draft")
        other_path = store.active_path_message_ids(other_session.id)
        await console._summarize_console_from(
            controller, session.id, ids["u2"].id
        )
        await pilot.pause()

        assert store.active_session_id == other_session.id
        assert store.active_path_message_ids(other_session.id) == other_path
        assert tuple(
            (message.id, message.role, message.content)
            for message in store.messages_for_session(session.id)
        ) == old_messages
        assert composer.draft_text() == "keep this draft"
        assert console._is_descendant_or_self(host.focused, composer)
        summarize.assert_not_awaited()
        assert (
            "Conversation changed before summarization could start.",
            "warning",
        ) in notices


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "accepted", "controller_copy", "expected_copy", "severity"),
    [
        (
            KIND_SUMMARIZE_FROM,
            True,
            "Summarized 2 turns from that message.",
            "Conversation memory updated.",
            "information",
        ),
        (
            KIND_SUMMARIZE_UP_TO,
            False,
            "That span is too large to summarize in one call. Choose a later start.",
            "That span is too large to summarize in one call. Choose a later start.",
            "warning",
        ),
        (
            KIND_SUMMARIZE_FROM,
            False,
            "Conversation changed while summarizing. No memory was saved.",
            "Conversation changed while summarizing. No memory was saved.",
            "warning",
        ),
        (
            KIND_SUMMARIZE_FROM,
            False,
            "The active provider is not ready. Review Console setup.",
            "The active provider is not ready. Review Console setup.",
            "warning",
        ),
        (
            KIND_SUMMARIZE_FROM,
            False,
            "Summarization was cancelled.",
            "Summarization was cancelled.",
            "warning",
        ),
        (
            KIND_SUMMARIZE_FROM,
            False,
            "Couldn't summarize the conversation. Try again.",
            "Couldn't summarize the conversation. Try again.",
            "warning",
        ),
    ],
)
async def test_summary_worker_uses_bounded_terminal_copy_and_preserves_ui_state(
    kind,
    accepted,
    controller_copy,
    expected_copy,
    severity,
    monkeypatch,
):
    """Terminal handling must never echo transcript content or edit the UI."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)
        original_path = store.active_path_message_ids(session.id)
        original_messages = tuple(
            (message.id, message.role, message.content)
            for message in store.messages_for_session(session.id)
        )
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("private draft must stay")
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(ids["a1"].id)
        await pilot.pause()
        assert transcript.selected_message_id == ids["a1"].id
        controller = console._ensure_console_chat_controller()
        # TASK-26017: the preview gate runs the REAL planning before the
        # stubbed summarize; a None preview takes the documented fail-open
        # path (no modal) so these pins keep exercising the terminal flow.
        monkeypatch.setattr(
            controller, "preview_summarize", AsyncMock(return_value=None)
        )
        result = ConsoleSubmitResult(accepted, False, controller_copy)
        method_name = (
            "summarize_from"
            if kind == KIND_SUMMARIZE_FROM
            else "summarize_up_to"
        )
        summarize = AsyncMock(return_value=result)
        monkeypatch.setattr(controller, method_name, summarize)
        notices: list[tuple[str, str]] = []
        app.notify = lambda text, **kwargs: notices.append(
            (str(text), kwargs.get("severity", ""))
        )

        if kind == KIND_SUMMARIZE_FROM:
            await console._summarize_console_from(
                controller, session.id, ids["u2"].id
            )
        else:
            await console._summarize_console_up_to(
                controller, session.id, ids["u2"].id
            )
        await pilot.pause()

        assert notices[0] == ("Summarizing selected range...", "information")
        assert notices[-1] == (expected_copy, severity)
        assert store.active_path_message_ids(session.id) == original_path
        assert tuple(
            (message.id, message.role, message.content)
            for message in store.messages_for_session(session.id)
        ) == original_messages
        assert composer.draft_text() == "private draft must stay"
        assert transcript.selected_message_id == ids["a1"].id
        assert console._is_descendant_or_self(host.focused, composer)


@pytest.mark.asyncio
async def test_summary_worker_refocuses_after_unexpected_error_without_leaking_it(
    monkeypatch,
):
    """An unexpected exception must not leak its content into recovery copy."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        session, ids = await _seed_u1_a1_u2_a2(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep")
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(ids["a1"].id)
        await pilot.pause()
        assert transcript.selected_message_id == ids["a1"].id
        controller = console._ensure_console_chat_controller()
        # TASK-26017: same preview-gate bypass as the terminal-copy pins.
        monkeypatch.setattr(
            controller, "preview_summarize", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            controller,
            "summarize_from",
            AsyncMock(side_effect=RuntimeError("PRIVATE TRANSCRIPT CONTENT")),
        )
        notices: list[tuple[str, str]] = []
        app.notify = lambda text, **kwargs: notices.append(
            (str(text), kwargs.get("severity", ""))
        )

        await console._summarize_console_from(
            controller, session.id, ids["u2"].id
        )
        await pilot.pause()

        assert notices[-1] == (
            "Couldn't summarize the conversation. Try again.",
            "warning",
        )
        assert all("PRIVATE" not in text for text, _severity in notices)
        assert composer.draft_text() == "keep"
        assert transcript.selected_message_id == ids["a1"].id
        assert console._is_descendant_or_self(host.focused, composer)


@pytest.mark.asyncio
async def test_console_command_rewind_notifies_when_no_prompts_yet():
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        console._ensure_console_chat_store().ensure_session()

        notices: list[str] = []
        app.notify = lambda message_text, **kwargs: notices.append(str(message_text))

        screens_before = len(host.screen_stack)
        await console._commands._console_command_rewind(CommandParse("command", "rewind", ""))
        await pilot.pause()

        assert len(host.screen_stack) == screens_before
    assert "Nothing to rewind." in notices


@pytest.mark.asyncio
async def test_restore_refills_composer_with_full_prompt_not_truncated_preview():
    """A restore target longer than the modal's ~60-char preview must refill
    the composer with the message's FULL text, not the truncated preview
    that `RewindPromptRow.preview` / `ConsoleRewindChoice.prompt_text` carry
    for display purposes only. Regression for the bug where restoring to any
    prompt over the preview's `max_length` silently clipped the re-edit.
    """
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    long_prompt = "A" * 120

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        u1 = store.append_message(
            session.id, role=ConsoleMessageRole.USER, content=long_prompt
        )
        await console._sync_native_console_chat_ui()

        # Mirrors how `_console_rewind_prompt_rows` builds the modal row's
        # `preview`, which is what a real modal selection would put in
        # `ConsoleRewindChoice.prompt_text`.
        preview = derive_console_session_title(long_prompt, max_length=60)
        assert preview != long_prompt  # sanity: the preview really is truncated

        spy_insert = MagicMock(return_value=True)
        console._commands._insert_prompt_text_into_composer = spy_insert

        await console._commands._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(kind="restore", message_id=u1.id, prompt_text=preview),
        )
        await pilot.pause()

    spy_insert.assert_called_once_with(long_prompt, replace=True)


@pytest.mark.asyncio
async def test_restore_to_a_stale_message_id_makes_no_mutation_and_notifies():
    """A restore choice targeting an id no longer on the active path (e.g. the
    modal was opened, the tree changed underneath it, then a stale row was
    picked) must not touch the store or the composer -- just notify. The
    preceding `path.index()` lookup is what guards the full-text fetch this
    task adds, so this documents that guard still holds.
    """
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)
        original_path = store.active_path_message_ids(session.id)
        original_leaf = store.active_leaf(session.id)

        spy_insert = MagicMock(return_value=True)
        console._commands._insert_prompt_text_into_composer = spy_insert

        notices: list[tuple[str, str]] = []
        app.notify = lambda message_text, **kwargs: notices.append(
            (str(message_text), kwargs.get("severity", ""))
        )

        await console._commands._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(
                kind="restore", message_id="does-not-exist", prompt_text="x"
            ),
        )
        await pilot.pause()

    assert store.active_path_message_ids(session.id) == original_path
    assert store.active_leaf(session.id) == original_leaf
    spy_insert.assert_not_called()
    assert any(
        "no longer exists" in text and severity == "error"
        for text, severity in notices
    )


@pytest.mark.asyncio
async def test_restore_choice_guards_against_changed_active_session():
    """TASK-549: if the active session changed while the modal was up (a
    ``ModalScreen`` blocks this today, but the guard is future-proofing), a
    restore choice captured for the OLD session must no-op with a notify
    instead of mutating the now-different active session's tree.
    """
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)
        original_path = store.active_path_message_ids(session.id)

        # Simulate the active session changing out from under the modal
        # between "opened" and "choice applied".
        other_session = store.create_session(title="other")
        store.switch_session(other_session.id)
        other_path = store.active_path_message_ids(other_session.id)

        spy_insert = MagicMock(return_value=True)
        console._commands._insert_prompt_text_into_composer = spy_insert

        notices: list[tuple[str, str]] = []
        app.notify = lambda message_text, **kwargs: notices.append(
            (str(message_text), kwargs.get("severity", ""))
        )

        await console._commands._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(kind="restore", message_id=ids["u2"].id, prompt_text="U2"),
        )
        await pilot.pause()

    # No mutation anywhere: neither session's tree nor the composer changed.
    assert store.active_path_message_ids(session.id) == original_path
    assert store.active_path_message_ids(other_session.id) == other_path
    spy_insert.assert_not_called()
    assert any(
        "session changed" in text.lower() and severity == "warning"
        for text, severity in notices
    )


@pytest.mark.asyncio
async def test_summarize_choice_guards_against_changed_active_session():
    """Same session-changed guard covers the summarize-up-to branch: no
    worker is dispatched and nothing is stored."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)
        original_path = store.active_path_message_ids(session.id)

        other_session = store.create_session(title="other")
        store.switch_session(other_session.id)

        spy_worker = MagicMock(side_effect=lambda coro, **kwargs: coro.close())
        real_run_worker = console.run_worker

        def capture_summary_worker(work, **kwargs):
            # Workspace alias refreshes may still be queued after the session
            # switch. Only the guarded summary dispatch is under test here.
            if kwargs.get("group", "").startswith("console-run-"):
                return spy_worker(work, **kwargs)
            return real_run_worker(work, **kwargs)

        console.run_worker = capture_summary_worker

        notices: list[tuple[str, str]] = []
        app.notify = lambda message_text, **kwargs: notices.append(
            (str(message_text), kwargs.get("severity", ""))
        )

        await console._commands._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(
                kind="summarize-up-to", message_id=ids["u2"].id, prompt_text="U2"
            ),
        )
        await pilot.pause()

    assert spy_worker.call_count == 0
    assert store.active_path_message_ids(session.id) == original_path
    assert store.session_context_summary(session.id) == (None, None)
    assert any(
        "session changed" in text.lower() and severity == "warning"
        for text, severity in notices
    )


@pytest.mark.asyncio
async def test_console_command_rewind_pushes_modal_with_newest_first_rows():
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        session, ids = await _seed_u1_a1_u2_a2(console)

        await console._commands._console_command_rewind(CommandParse("command", "rewind", ""))
        await pilot.pause()

        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsoleRewindModal)
        assert [row.message_id for row in modal._prompts] == [
            ids["u2"].id,
            ids["u1"].id,
        ]
        assert [row.index_label for row in modal._prompts] == ["#2", "#1"]
        await pilot.click("#console-rewind-row-0")
        await pilot.pause()
        assert not modal.query_one(
            "#console-rewind-action-summarize", Button
        ).disabled
        assert not modal.query_one(
            "#console-rewind-action-summarize-from", Button
        ).disabled


@pytest.mark.asyncio
async def test_rewind_callback_refuses_sibling_descendant_path_change(monkeypatch):
    """A modal-open branch switch must fail even when its prompt stays on-path."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)
        original_path = tuple(store.active_path_message_ids(session.id))
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep callback draft")
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(ids["a1"].id)
        await pilot.pause()

        await console._commands._console_command_rewind(CommandParse("command", "rewind", ""))
        await pilot.pause()
        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsoleRewindModal)

        _switch_to_sibling_descendant(store, session.id, ids["a1"].id)
        changed_path = tuple(store.active_path_message_ids(session.id))
        assert changed_path != original_path
        assert ids["u1"].id in changed_path
        changed_messages = tuple(
            (message.id, message.role, message.content)
            for message in store.messages_for_session(session.id)
        )
        controller = console._ensure_console_chat_controller()
        summarize = AsyncMock(
            return_value=ConsoleSubmitResult(True, False, "controller success")
        )
        monkeypatch.setattr(controller, "summarize_from", summarize)
        notices: list[tuple[str, str]] = []
        app.notify = lambda text, **kwargs: notices.append(
            (str(text), kwargs.get("severity", ""))
        )

        await pilot.click("#console-rewind-row-1")
        await pilot.pause()
        await pilot.click("#console-rewind-action-summarize-from")
        await pilot.pause()

        assert host.screen_stack[-1] is console
        summarize.assert_not_awaited()
        assert tuple(store.active_path_message_ids(session.id)) == changed_path
        assert tuple(
            (message.id, message.role, message.content)
            for message in store.messages_for_session(session.id)
        ) == changed_messages
        assert composer.draft_text() == "keep callback draft"
        assert transcript.selected_message_id == ids["a1"].id
        assert console._is_descendant_or_self(host.focused, composer)
        assert (
            "Conversation changed before summarization could start.",
            "warning",
        ) in notices


@pytest.mark.asyncio
async def test_console_rewind_disables_summaries_for_incomplete_tip_only():
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, _ids = await _seed_u1_a1_u2_a2(console)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="unfinished prompt",
        )

        await console._commands._console_command_rewind(CommandParse("command", "rewind", ""))
        await pilot.pause()
        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsoleRewindModal)
        await pilot.click("#console-rewind-row-0")
        await pilot.pause()

        assert not modal.query_one(
            "#console-rewind-action-restore", Button
        ).disabled
        assert modal.query_one(
            "#console-rewind-action-summarize", Button
        ).disabled
        assert modal.query_one(
            "#console-rewind-action-summarize-from", Button
        ).disabled
        assert not modal.query_one(
            "#console-rewind-action-cancel", Button
        ).disabled
        assert modal._summary_disabled_reason == (
            "Finish the current exchange before summarizing."
        )


@pytest.mark.asyncio
async def test_console_rewind_disables_summaries_while_run_is_active():
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        _session, _ids = await _seed_u1_a1_u2_a2(console)
        controller = console._ensure_console_chat_controller()
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response.")
        )

        await console._commands._console_command_rewind(CommandParse("command", "rewind", ""))
        await pilot.pause()
        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsoleRewindModal)
        await pilot.click("#console-rewind-row-0")
        await pilot.pause()

        assert modal.query_one(
            "#console-rewind-action-summarize", Button
        ).disabled
        assert modal.query_one(
            "#console-rewind-action-summarize-from", Button
        ).disabled
        assert modal._summary_disabled_reason == CONSOLE_RUN_ALREADY_RUNNING_COPY


@pytest.mark.asyncio
async def test_console_rewind_marks_both_actions_as_replacing_typed_memory(
    monkeypatch,
):
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        session, _ids = await _seed_u1_a1_u2_a2(console)
        controller = console._ensure_console_chat_controller()
        monkeypatch.setattr(
            controller,
            "context_control_inputs",
            MagicMock(
                return_value=(
                    MagicMock(),
                    None,
                    EffectiveMemoryResult(EffectiveMemoryKind.LEGACY_PREFIX),
                )
            ),
        )

        await console._commands._console_command_rewind(CommandParse("command", "rewind", ""))
        await pilot.pause()
        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsoleRewindModal)
        assert modal._has_effective_memory is True


@pytest.mark.asyncio
async def test_console_rewind_memory_lookup_error_warns_conservatively_without_leak(
    monkeypatch,
):
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await _seed_u1_a1_u2_a2(console)
        controller = console._ensure_console_chat_controller()
        private_error = "PRIVATE MEMORY LOOKUP BODY"
        monkeypatch.setattr(
            controller,
            "context_control_inputs",
            MagicMock(side_effect=RuntimeError(private_error)),
        )
        notices: list[str] = []
        app.notify = lambda text, **_kwargs: notices.append(str(text))
        diagnostics: list[str] = []
        sink_id = logger.add(
            diagnostics.append,
            level="WARNING",
            format="{message}",
        )
        try:
            await console._commands._console_command_rewind(
                CommandParse("command", "rewind", "")
            )
            await pilot.pause()
        finally:
            logger.remove(sink_id)

        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsoleRewindModal)
        assert modal._has_effective_memory is True
        await pilot.click("#console-rewind-row-0")
        await pilot.pause()
        rendered_copy = [
            str(modal.query_one(selector, Static).render())
            for selector in (
                "#console-rewind-action-summarize-copy",
                "#console-rewind-action-summarize-from-copy",
            )
        ]
        assert all(
            "Replaces current conversation memory" in copy
            for copy in rendered_copy
        )
        assert any(
            "Console rewind effective-memory lookup failed" in message
            for message in diagnostics
        )
        assert all(
            private_error not in text
            for text in [*notices, *diagnostics, *rendered_copy]
        )


@pytest.mark.asyncio
async def test_keyboard_rewind_cancel_consumes_command_and_preserves_late_draft():
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _seed_u1_a1_u2_a2(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.load_draft("/rewind")
        console._sync_console_workbench_actions_from_draft()
        console._sync_console_command_popup()
        assert console._dismiss_console_command_popup()
        _press_enter_synchronously(console)
        composer.insert_text("next draft")
        assert composer.draft_text() == "next draft"

        await pilot.pause()
        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsoleRewindModal)

        await pilot.press("escape")
        await pilot.pause()

        assert host.screen_stack[-1] is console
        assert composer.draft_text() == "next draft"
        assert composer.has_focus


@pytest.mark.asyncio
async def test_visible_send_rewind_cancel_clears_command_and_refocuses_empty_composer():
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _seed_u1_a1_u2_a2(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/rewind")
        console._sync_console_workbench_actions_from_draft()
        await pilot.pause()

        send = console.query_one("#console-send-message", Button)
        assert not send.disabled
        await pilot.click("#console-send-message")
        await pilot.pause()

        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsoleRewindModal)
        assert composer.draft_text() == ""

        await pilot.click("#console-rewind-row-0")
        await pilot.pause()
        await pilot.click("#console-rewind-action-cancel")
        await pilot.pause()
        await pilot.pause()

        assert host.screen_stack[-1] is console
        assert send.disabled
        assert console._is_descendant_or_self(host.focused, composer)


@pytest.mark.asyncio
async def test_rewind_restore_replaces_late_keyboard_text_with_full_prompt():
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)
    full_prompt = "selected full prompt " + ("x" * 100)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content=full_prompt,
        )
        await console._sync_native_console_chat_ui()
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.load_draft("/rewind")
        console._sync_console_workbench_actions_from_draft()
        console._sync_console_command_popup()
        assert console._dismiss_console_command_popup()
        _press_enter_synchronously(console)
        composer.insert_text("late text")
        await pilot.pause()
        assert isinstance(host.screen_stack[-1], ConsoleRewindModal)

        await pilot.click("#console-rewind-row-0")
        await pilot.pause()
        await pilot.click("#console-rewind-action-restore")
        await pilot.pause()

        assert host.screen_stack[-1] is console
        assert composer.draft_text() == full_prompt


@pytest.mark.asyncio
async def test_rewind_no_prompts_restores_keyboard_stash_ahead_of_late_text():
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)
    notices: list[tuple[str, str]] = []

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Existing response without a USER prompt.",
        )
        await console._sync_native_console_chat_ui()
        console.app_instance.notify = lambda message_text, **kwargs: notices.append(
            (str(message_text), kwargs.get("severity", ""))
        )
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.load_draft("/rewind")
        console._sync_console_workbench_actions_from_draft()
        console._sync_console_command_popup()
        assert console._dismiss_console_command_popup()

        _press_enter_synchronously(console)
        composer.insert_text("next")
        for _ in range(40):
            if notices:
                break
            await pilot.pause(0.01)

        assert host.screen_stack[-1] is console
        assert composer.draft_text() == "/rewindnext"
        assert ("Nothing to rewind.", "warning") in notices


@pytest.mark.asyncio
async def test_rewind_with_args_keeps_restore_before_dispatch_behavior():
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _seed_u1_a1_u2_a2(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.load_draft("/rewind anything")
        console._sync_console_workbench_actions_from_draft()
        console._sync_console_command_popup()

        _press_enter_synchronously(console)
        composer.insert_text("next")
        await pilot.pause()

        assert isinstance(host.screen_stack[-1], ConsoleRewindModal)
        assert composer.draft_text() == "/rewind anythingnext"


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["keyboard", "visible-send"])
async def test_rewind_modal_launch_failure_preserves_draft(source, monkeypatch):
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Existing prompt.",
        )
        await console._sync_native_console_chat_ui()
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/rewind")

        if source == "keyboard":
            stash = composer.stash_draft_for_send()
            assert stash is not None and stash.text == "/rewind"
            console._console_pending_send_stash = stash
            composer.insert_text("late")

        real_push_screen = console.app.push_screen

        def fail_rewind_modal(screen, *args, **kwargs):
            if isinstance(screen, ConsoleRewindModal):
                raise RuntimeError("rewind modal launch failed")
            return real_push_screen(screen, *args, **kwargs)

        monkeypatch.setattr(console.app, "push_screen", fail_rewind_modal)

        with pytest.raises(RuntimeError, match="rewind modal launch failed"):
            await console._submission._send_console_message_from_visible_action()

        expected = "/rewindlate" if source == "keyboard" else "/rewind"
        assert composer.draft_text() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["identity", "edit-retype", "generation"])
async def test_visible_rewind_cleanup_preserves_a_changed_composer(
    mutation, monkeypatch
):
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/rewind")
        opening_snapshot = composer.capture_draft_snapshot()
        replacement = MagicMock(spec=ConsoleComposerBar)

        async def succeed_after_mutation(_parse):
            if mutation == "identity":
                monkeypatch.setattr(
                    console, "_console_composer_or_none", lambda: replacement
                )
            elif mutation == "edit-retype":
                composer.insert_text("x")
                composer.delete_left()
            else:
                composer.load_draft("/rewind")
            return True

        clear_draft = MagicMock()
        monkeypatch.setattr(console._commands, "_console_command_rewind", succeed_after_mutation)
        monkeypatch.setattr(console._commands, "_clear_console_composer_draft", clear_draft)

        assert not await console._submission._send_console_message_from_visible_action()

        current_snapshot = composer.capture_draft_snapshot()
        assert composer.draft_text() == "/rewind"
        clear_draft.assert_not_called()
        if mutation == "identity":
            assert console._console_composer_or_none() is replacement
            assert replacement is not composer
            assert current_snapshot.edit_serial == opening_snapshot.edit_serial
            assert current_snapshot.generation == opening_snapshot.generation
        elif mutation == "edit-retype":
            assert console._console_composer_or_none() is composer
            assert current_snapshot.edit_serial > opening_snapshot.edit_serial
            assert current_snapshot.generation == opening_snapshot.generation
        else:
            assert console._console_composer_or_none() is composer
            assert current_snapshot.edit_serial == opening_snapshot.edit_serial
            assert current_snapshot.generation > opening_snapshot.generation
