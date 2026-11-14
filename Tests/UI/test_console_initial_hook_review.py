"""The stock waiting Send owns hook review beyond a disposable modal."""

import pytest
from textual.screen import ModalScreen

from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from Tests.UI.test_console_hook_review_send_freeze import (
    DRAFT,
    NavigatedConsoleHarness,
    _open_review_from,
    _record_dispatch,
    _release,
    _task_factory,
    _until,
)
from Tests.UI.test_console_workbench_contract import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_hooks_review_modal import (
    ConsoleHooksReviewModal,
)

pytestmark = pytest.mark.bootstrap_profile
hook_file = _hook_file


async def test_waiting_send_review_is_resident_after_modal_unmount(hook_file):
    """Use the proven navigated Send route and its existing projection API."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = NavigatedConsoleHarness(app)
    runtime = None
    try:
        with _task_factory("eager"):
            async with host.run_test(size=(120, 40)) as pilot:
                assert await _until(
                    lambda: isinstance(host.screen, ChatScreen)
                    and host.screen.is_mounted,
                    10,
                ), "the navigation worker never mounted the Console"
                console = host.screen
                await pilot.pause()
                calls = _record_dispatch(console)
                runtime = console._console_runtime()
                controller = console._ensure_console_chat_controller()
                console, composer, modal, requester = await _open_review_from(
                    "send-button", host, pilot
                )
                try:
                    session_id = console._console_visible_send_session_id()
                    pending = controller.pending_decision_projection(session_id)
                    assert (
                        pending is not None
                    ), "the actual waiting Send opened a modal without a resident decision"
                    assert pending.decision_type == "hook_review"
                    review_id = pending.decision_id
                    assert review_id and console._hooks._busy

                    # Navigation removes presentation without an explicit Cancel.
                    await host.pop_screen()
                    assert await _until(
                        lambda: modal not in host.screen_stack
                        and modal.parent is None
                        and not modal.is_running,
                        5,
                    )
                    retained = controller.pending_decision_projection(session_id)
                    assert retained is not None
                    assert retained.decision_type == "hook_review"
                    assert retained.decision_id == review_id
                    assert console._hooks._busy
                    assert calls == [] and composer.draft_text() == DRAFT

                    controller.project_pending_decision_for_active_session()
                    assert await _until(
                        lambda: isinstance(host.screen, ConsoleHooksReviewModal)
                        and host.screen is not modal
                        and host.screen.is_mounted,
                        5,
                    ), "the resident review did not remount on the current Console"
                    remounted = host.screen
                    current = controller.pending_decision_projection(session_id)
                    assert current is not None and current.decision_id == review_id
                    await remounted.request_safe_cancel(source="test_explicit_cancel")
                    assert await _until(lambda: not console._hooks._busy, 5)
                    assert controller.pending_decision_projection(session_id) is None
                    assert calls == [] and composer.draft_text() == DRAFT
                finally:
                    # Old behavior must still tear down after the meaningful RED.
                    if isinstance(host.screen, ConsoleHooksReviewModal):
                        await host.screen.request_safe_cancel(source="test_cleanup")
                    await _release(host, modal, requester)
    finally:
        if runtime is not None:
            await runtime.dispose()


async def test_stopped_review_retires_after_covering_modal_is_removed(hook_file):
    """Stop settles the resident request while another real modal covers it."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = NavigatedConsoleHarness(app)
    runtime = None
    try:
        with _task_factory("eager"):
            async with host.run_test(size=(120, 40)) as pilot:
                assert await _until(
                    lambda: isinstance(host.screen, ChatScreen)
                    and host.screen.is_mounted,
                    10,
                ), "the navigation worker never mounted the Console"
                console = host.screen
                await pilot.pause()
                calls = _record_dispatch(console)
                runtime = console._console_runtime()
                controller = console._ensure_console_chat_controller()
                console, composer, modal, requester = await _open_review_from(
                    "send-button", host, pilot
                )
                cover = ModalScreen()
                try:
                    session_id = console._console_visible_send_session_id()
                    pending = controller.pending_decision_projection(session_id)
                    assert (
                        pending is not None and pending.decision_type == "hook_review"
                    )
                    await host.push_screen(cover)
                    assert await _until(lambda: cover.is_mounted, 5)
                    assert host.screen is cover and modal in host.screen_stack

                    assert controller.stop_active_run()
                    assert await _until(
                        lambda: controller.pending_decision_projection(session_id)
                        is None
                        and not console._hooks._busy,
                        5,
                    ), "Stop left the covered initial review pending"
                    assert (
                        host.screen is cover
                    ), "review cleanup dismissed its covering modal"
                    assert calls == [] and composer.draft_text() == DRAFT

                    await host.pop_screen()
                    assert await _until(
                        lambda: host.screen is console
                        and modal not in host.screen_stack
                        and modal.parent is None
                        and not modal.is_running,
                        5,
                    ), "the terminal hook review remained after its cover was removed"
                    controller.project_pending_decision_for_active_session()
                    await pilot.pause()
                    assert not any(
                        isinstance(screen, ConsoleHooksReviewModal)
                        for screen in host.screen_stack
                    )
                    assert not console._hooks._busy
                    assert calls == [] and composer.draft_text() == DRAFT
                finally:
                    if host.screen is cover:
                        await host.pop_screen()
                    if isinstance(host.screen, ConsoleHooksReviewModal):
                        await host.screen.request_safe_cancel(source="test_cleanup")
                    await _release(host, modal, requester)
    finally:
        if runtime is not None:
            await runtime.dispose()
