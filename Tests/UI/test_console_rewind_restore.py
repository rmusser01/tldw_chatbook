"""Screen wiring for the Console `/rewind` command's restore path (SP2 Task 1).

Mirrors ``Tests/UI/test_console_edit_resend_wiring.py``'s harness style:
``ConsoleHarness`` mounts a real ``ChatScreen`` over a real
``ConsoleChatStore``/``ConsoleChatController`` pair, and each test drives the
screen's own restore machinery directly (``_console_command_rewind`` /
``_apply_console_rewind_choice``) rather than clicking through the modal --
the modal's own click/dismiss behavior is covered by
``Tests/Chat/test_console_rewind_modal.py``.

Restore is pure tree navigation (SP1 primitives): the selected USER prompt's
PARENT (found by an id lookup in ``active_path_message_ids``, never
positional) becomes the new active leaf, and the prompt's own text is written
back into the composer via the same ``_insert_prompt_text_into_composer``
seam ``/prompt`` uses.
"""

from unittest.mock import MagicMock

import pytest

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
from tldw_chatbook.Chat.console_command_grammar import CommandParse
from tldw_chatbook.Widgets.Console.console_rewind_modal import (
    ConsoleRewindChoice,
    ConsoleRewindModal,
)


CONSOLE_RUN_ALREADY_RUNNING_COPY = "A Console run is already running."


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


@pytest.mark.asyncio
async def test_restore_mid_path_truncates_active_path_and_refills_composer():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)

        spy_insert = MagicMock(return_value=True)
        console._insert_prompt_text_into_composer = spy_insert

        await console._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(kind="restore", message_id=ids["u2"].id, prompt_text="U2"),
        )
        await pilot.pause()

    assert store.active_path_message_ids(session.id) == [ids["u1"].id, ids["a1"].id]
    spy_insert.assert_called_once_with("U2", replace=True)


@pytest.mark.asyncio
async def test_restore_to_first_prompt_clears_active_leaf_to_empty_path():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)

        spy_insert = MagicMock(return_value=True)
        console._insert_prompt_text_into_composer = spy_insert

        await console._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(kind="restore", message_id=ids["u1"].id, prompt_text="U1"),
        )
        await pilot.pause()

    assert store.active_leaf(session.id) is None
    assert store.active_path_message_ids(session.id) == []
    spy_insert.assert_called_once_with("U1", replace=True)


@pytest.mark.asyncio
async def test_restore_blocked_while_a_run_is_streaming_makes_no_mutation():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)
        original_path = store.active_path_message_ids(session.id)

        controller = console._ensure_console_chat_controller()
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response.")
        )

        spy_insert = MagicMock(return_value=True)
        console._insert_prompt_text_into_composer = spy_insert

        notices: list[tuple[str, str]] = []
        app.notify = lambda message_text, **kwargs: notices.append(
            (str(message_text), kwargs.get("severity", ""))
        )

        await console._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(kind="restore", message_id=ids["u2"].id, prompt_text="U2"),
        )
        await pilot.pause()

    # No mutation at all: active path unchanged, composer untouched.
    assert store.active_path_message_ids(session.id) == original_path
    spy_insert.assert_not_called()
    assert (CONSOLE_RUN_ALREADY_RUNNING_COPY, "warning") in notices


@pytest.mark.asyncio
async def test_none_choice_just_refocuses_composer_without_mutation():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)
        original_path = store.active_path_message_ids(session.id)

        spy_focus = MagicMock()
        console._focus_console_composer_if_needed = spy_focus

        await console._apply_console_rewind_choice(session.id, None)
        await pilot.pause()

    assert store.active_path_message_ids(session.id) == original_path
    spy_focus.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_summarize_up_to_choice_dispatches_console_run_worker_without_mutation():
    """SP2 Task 3: the summarize-up-to choice runs the boundary-summary flow on
    the exclusive ``console-run`` worker group and never does tree surgery."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)
        original_path = store.active_path_message_ids(session.id)

        # Capture the dispatched worker without running it -- the provider
        # outcome is irrelevant to the screen's dispatch contract.
        spy_worker = MagicMock(side_effect=lambda coro, **kwargs: coro.close())
        console.run_worker = spy_worker

        await console._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(
                kind="summarize-up-to", message_id=ids["u2"].id, prompt_text="U2"
            ),
        )
        await pilot.pause()

    assert spy_worker.call_count == 1
    assert spy_worker.call_args.kwargs.get("group") == "console-run"
    # Summarize never mutates the transcript tree, and nothing is stored until
    # the (unrun) worker succeeds.
    assert store.active_path_message_ids(session.id) == original_path
    assert store.session_context_summary(session.id) == (None, None)


@pytest.mark.asyncio
async def test_summarize_up_to_choice_blocked_while_a_run_is_streaming():
    """A summarize refuses (no worker) while a run streams, like restore does."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)
        original_path = store.active_path_message_ids(session.id)

        controller = console._ensure_console_chat_controller()
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response.")
        )
        spy_worker = MagicMock(side_effect=lambda coro, **kwargs: coro.close())
        console.run_worker = spy_worker

        notices: list[tuple[str, str]] = []
        app.notify = lambda message_text, **kwargs: notices.append(
            (str(message_text), kwargs.get("severity", ""))
        )

        await console._apply_console_rewind_choice(
            session.id,
            ConsoleRewindChoice(
                kind="summarize-up-to", message_id=ids["u2"].id, prompt_text="U2"
            ),
        )
        await pilot.pause()

    assert spy_worker.call_count == 0
    assert (CONSOLE_RUN_ALREADY_RUNNING_COPY, "warning") in notices
    assert store.active_path_message_ids(session.id) == original_path
    assert store.session_context_summary(session.id) == (None, None)


@pytest.mark.asyncio
async def test_console_command_rewind_notifies_when_no_prompts_yet():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        console._ensure_console_chat_store().ensure_session()

        notices: list[str] = []
        app.notify = lambda message_text, **kwargs: notices.append(str(message_text))

        screens_before = len(host.screen_stack)
        await console._console_command_rewind(CommandParse("command", "rewind", ""))
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
        console._insert_prompt_text_into_composer = spy_insert

        await console._apply_console_rewind_choice(
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
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session, ids = await _seed_u1_a1_u2_a2(console)
        original_path = store.active_path_message_ids(session.id)
        original_leaf = store.active_leaf(session.id)

        spy_insert = MagicMock(return_value=True)
        console._insert_prompt_text_into_composer = spy_insert

        notices: list[tuple[str, str]] = []
        app.notify = lambda message_text, **kwargs: notices.append(
            (str(message_text), kwargs.get("severity", ""))
        )

        await console._apply_console_rewind_choice(
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
async def test_console_command_rewind_pushes_modal_with_newest_first_rows():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        session, ids = await _seed_u1_a1_u2_a2(console)

        await console._console_command_rewind(CommandParse("command", "rewind", ""))
        await pilot.pause()

        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsoleRewindModal)
        assert [row.message_id for row in modal._prompts] == [
            ids["u2"].id,
            ids["u1"].id,
        ]
        assert [row.index_label for row in modal._prompts] == ["#2", "#1"]
