"""Screen wiring for Console branching Phase B Task 3.

After Task 1 widened the edit modal to return ``ConsoleEditResult`` instead
of a bare string, ``_apply_edit`` still called ``store.update_message_content``
with the whole dataclass -- breaking in-place Save (regression covered by
``Tests/UI/test_console_native_chat_flow.py::
test_console_selected_message_edit_action_opens_modal_and_saves_content``).
This file pins the branch this task adds: ``resend=False`` keeps routing to
``store.update_message_content`` (restored), ``resend=True`` routes to the
NEW ``controller.edit_and_resend_message`` path via a ``console-run`` worker,
gated by the same ``run_state.is_send_allowed`` check as retry/regenerate/
continue -- and the modal is only offered ``can_resend=True`` for a USER
message (the only role ``edit_and_resend_message`` accepts).
"""

from unittest.mock import AsyncMock

import pytest
from textual.widgets import TextArea

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)

from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Widgets.Console import ConsoleTranscript


CONSOLE_RUN_ALREADY_RUNNING_COPY = "A Console run is already running."


async def _open_edit_modal(console, pilot, message_id: str):
    """Select ``message_id`` and click its Edit action; return the pushed modal."""
    transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
    transcript.select_message(message_id)
    await console._sync_native_console_chat_ui()
    await _wait_for_selector(
        console, pilot, f"#console-message-action-edit-{message_id}"
    )
    await pilot.click(f"#console-message-action-edit-{message_id}")
    host = console.app
    await _wait_for_selector(
        host.screen_stack[-1], pilot, "#console-edit-message-modal"
    )
    return host.screen_stack[-1]


async def _wait_until(predicate, pilot, *, attempts: int = 80) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError("Condition never became true.")


@pytest.mark.asyncio
async def test_edit_modal_offers_resend_only_for_user_message():
    """A USER message's edit modal shows the resend button; an ASSISTANT
    message's does not -- ``edit_and_resend_message`` only accepts USER rows."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        user_message = store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="hi"
        )
        assistant_message = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="hello"
        )
        await console._sync_native_console_chat_ui()

        user_modal = await _open_edit_modal(console, pilot, user_message.id)
        assert user_modal.query("#console-edit-message-resend")
        await pilot.click("#console-edit-message-cancel")
        await pilot.pause()

        assistant_modal = await _open_edit_modal(console, pilot, assistant_message.id)
        assert not assistant_modal.query("#console-edit-message-resend")
        await pilot.click("#console-edit-message-cancel")
        await pilot.pause()


@pytest.mark.asyncio
async def test_edit_save_still_routes_to_update_message_content_in_place():
    """``resend=False`` (Save) must still land through the ORIGINAL in-place
    path, not the new resend worker -- this is the transitional-break fix."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="original"
        )
        await console._sync_native_console_chat_ui()

        controller = console._ensure_console_chat_controller()
        spy_resend = AsyncMock()
        controller.edit_and_resend_message = spy_resend

        edit_modal = await _open_edit_modal(console, pilot, message.id)
        editor = edit_modal.query_one("#console-edit-message-body", TextArea)
        editor.text = "edited in place"
        await pilot.click("#console-edit-message-save")
        await pilot.pause()

    assert store.get_message(message.id).content == "edited in place"
    assert console._last_console_action.action_id == "edit"
    spy_resend.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_and_resend_dispatches_controller_edit_and_resend_message():
    """``resend=True`` (Edit & resend) routes to
    ``controller.edit_and_resend_message`` via a ``console-run`` worker,
    leaving the original message untouched (the controller forks a branch)."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="original"
        )
        await console._sync_native_console_chat_ui()

        controller = console._ensure_console_chat_controller()
        spy_resend = AsyncMock(
            return_value=ConsoleSubmitResult(
                accepted=True, should_clear_draft=True, visible_copy=""
            )
        )
        controller.edit_and_resend_message = spy_resend

        edit_modal = await _open_edit_modal(console, pilot, message.id)
        editor = edit_modal.query_one("#console-edit-message-body", TextArea)
        editor.text = "edited and resent"
        await pilot.click("#console-edit-message-resend")
        await pilot.pause()

        await _wait_until(lambda: spy_resend.await_count > 0, pilot)

    spy_resend.assert_awaited_once_with(message.id, "edited and resent")
    # The in-place path must NOT have fired -- the original node is untouched.
    assert store.get_message(message.id).content == "original"


@pytest.mark.asyncio
async def test_edit_and_resend_blocked_while_a_run_is_already_active():
    """Mirrors retry/regenerate/continue's mid-run gate (TASK-232): a
    ``console-run`` worker must never be spawned while one is already
    streaming -- it would cancel the in-flight run at creation time, before
    the controller's own rejection can run."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        message = store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="original"
        )
        await console._sync_native_console_chat_ui()

        controller = console._ensure_console_chat_controller()
        spy_resend = AsyncMock()
        controller.edit_and_resend_message = spy_resend
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response.")
        )

        notices: list[str] = []
        app.notify = lambda message_text, **kwargs: notices.append(str(message_text))

        edit_modal = await _open_edit_modal(console, pilot, message.id)
        editor = edit_modal.query_one("#console-edit-message-body", TextArea)
        editor.text = "edited and resent"
        await pilot.click("#console-edit-message-resend")
        await pilot.pause()

    spy_resend.assert_not_awaited()
    assert any(CONSOLE_RUN_ALREADY_RUNNING_COPY in note for note in notices)
