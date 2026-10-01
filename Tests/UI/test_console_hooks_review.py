"""Native Console hook review routing, draft retention, and literal details."""

import asyncio

import pytest

from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from Tests.UI.test_console_workbench_contract import (
    ConsoleHarness,
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from tldw_chatbook.Widgets.Console.console_hooks_review_modal import (
    ConsoleHooksReviewModal,
)

pytestmark = pytest.mark.bootstrap_profile

hook_file = _hook_file


def _send_in_worker(console, draft):
    """Send the way a worker caller does (spoken "Console, send."), which
    waits for the review and returns the settled outcome. A caller outside a
    worker gets AWAITING_REVIEW at once and the review runs in a worker of its
    own (TASK-33621.28), so it cannot observe these outcomes."""
    return console.run_worker(
        console._dispatch_console_draft_send(draft), group="test-hook-send"
    )


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_hooks_action_is_reachable_and_returns_focus(size, hook_file):
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=size) as pilot:
        console = host.screen
        await _wait_for_selector(console, pilot, "#console-control-hooks")
        button = console.query_one("#console-control-hooks")
        assert button.region.width and button.region.right <= console.region.right
        await console._refresh_console_hooks()
        assert str(button.label).endswith(" 1")
        assert button.tooltip == "Hook permissions: 1 need review"
        button.focus()
        await pilot.press("enter")
        async with asyncio.timeout(5):
            while not isinstance(host.screen, ConsoleHooksReviewModal):
                await pilot.pause(0.01)
        modal = host.screen
        await _wait_for_selector(modal, pilot, "#console-hooks-review")
        await pilot.pause()
        assert not modal._selected
        assert (
            modal.query_one("#console-hooks-cancel").region.bottom
            <= modal.region.bottom
        )
        await pilot.press("escape")
        await pilot.pause()
        assert host.screen is console and console.focused is button


async def test_next_send_cancel_keeps_draft_then_allow_resumes_once(hook_file):
    from tldw_chatbook.UI.Console_Modules.prompt_queue import (
        ConsolePromptDispatchResult,
        ConsolePromptDispatchStatus,
    )

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen
        composer = console._console_composer_or_none()
        composer.load_draft("retained draft")
        await pilot.pause()
        session_id = console._console_visible_send_session_id()
        calls = []

        async def dispatch(draft, **kwargs):
            calls.append(draft)
            return ConsolePromptDispatchResult(
                ConsolePromptDispatchStatus.SENT, session_id, ""
            )

        console._prompt_queue.dispatch = dispatch
        first = _send_in_worker(console, "retained draft")
        async with asyncio.timeout(5):
            while not isinstance(host.screen, ConsoleHooksReviewModal):
                await pilot.pause(0.01)
        await pilot.press("escape")
        assert not await first.wait()
        assert composer.draft_text() == "retained draft" and calls == []
        second = _send_in_worker(console, "retained draft")
        async with asyncio.timeout(5):
            while not isinstance(host.screen, ConsoleHooksReviewModal):
                await pilot.pause(0.01)
        modal = host.screen
        await _wait_for_selector(modal, pilot, "#hook-review-details-0", timeout=5)
        await pilot.click("#hook-review-details-0")
        from textual.widgets import Static

        detail = str(modal.query_one(".hook-review-detail", Static).render())
        assert '"-c"' in detail and '"pass"' in detail
        await pilot.click("#console-hooks-allow-all")
        assert await second.wait()
        assert calls == ["retained draft"]
        third = await console._dispatch_console_draft_send("retained draft")
        assert third and calls == ["retained draft", "retained draft"]
        assert host.screen is console


async def test_escape_during_approval_keeps_send_cancelled(hook_file):
    from tldw_chatbook.UI.Console_Modules.prompt_queue import (
        ConsolePromptDispatchResult,
        ConsolePromptDispatchStatus,
    )

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen
        composer = console._console_composer_or_none()
        composer.load_draft("keep while saving")
        await pilot.pause()
        calls = []

        async def dispatch(draft, **kwargs):
            calls.append(draft)
            return ConsolePromptDispatchResult(
                ConsolePromptDispatchStatus.SENT,
                console._console_visible_send_session_id(),
                "",
            )

        console._prompt_queue.dispatch = dispatch
        send = _send_in_worker(console, "keep while saving")
        async with asyncio.timeout(5):
            while not isinstance(host.screen, ConsoleHooksReviewModal):
                await pilot.pause(0.01)
        modal = host.screen
        await _wait_for_selector(modal, pilot, "#console-hooks-allow-all", timeout=5)
        saved = asyncio.Event()
        release = asyncio.Event()
        original = modal._approve

        async def paused_write(expected, keys):
            current = await original(expected, keys)
            saved.set()
            await release.wait()
            return current

        modal._approve = paused_write
        await pilot.click("#console-hooks-allow-all")
        await asyncio.wait_for(saved.wait(), 5)
        await pilot.press("escape")
        assert not await send.wait()
        release.set()
        await pilot.pause()
        assert host.screen is console and calls == []
        assert composer.draft_text() == "keep while saving"
        assert console._console_runtime().ensure_hook_permissions().snapshot().ready


async def test_partial_approval_keeps_review_open_and_settings_cancels_send(hook_file):
    import toml

    raw = toml.loads(hook_file.read_text())
    raw["hooks"]["hook"].append({**raw["hooks"]["hook"][0], "id": "two"})
    hook_file.write_text(toml.dumps(raw))
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(80, 24)) as pilot:
        console = host.screen
        composer = console._console_composer_or_none()
        composer.load_draft("keep for settings")
        await pilot.pause()
        posted = []
        original_post = console.post_message

        def post(message):
            from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

            if isinstance(message, NavigateToScreen):
                posted.append(message)
            return original_post(message)

        console.post_message = post
        send = _send_in_worker(console, "keep for settings")
        async with asyncio.timeout(5):
            while not isinstance(host.screen, ConsoleHooksReviewModal):
                await pilot.pause(0.01)
        modal = host.screen
        await _wait_for_selector(modal, pilot, "#hook-review-select-0", timeout=5)
        await pilot.click("#hook-review-select-0")
        await pilot.click("#console-hooks-allow-selected")
        async with asyncio.timeout(5):
            while modal.snapshot.pending_count != 1 or modal._busy:
                await pilot.pause(0.01)
        assert host.screen is modal and not send.is_finished
        assert await pilot.click("#console-hooks-settings")
        assert not await asyncio.wait_for(send.wait(), 5)
        await pilot.pause()
        assert composer.draft_text() == "keep for settings"
        assert posted


async def test_all_hooks_exposes_current_revoke_and_disable_actions(hook_file):
    import toml

    from tldw_chatbook.Agents.hook_permissions import HookPermissions

    owner = HookPermissions()
    pending = owner.snapshot()
    owner.approve(pending, [pending.rows[0].entry.key])
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(80, 24)) as pilot:
        console = host.screen
        await _wait_for_selector(console, pilot, "#console-control-hooks")
        console.query_one("#console-control-hooks").focus()
        await pilot.press("enter")
        async with asyncio.timeout(5):
            while not isinstance(host.screen, ConsoleHooksReviewModal):
                await pilot.pause(0.01)
        modal = host.screen
        await _wait_for_selector(modal, pilot, "#console-hooks-all")
        modal.query_one("#console-hooks-all").focus()
        await pilot.press("enter")
        await _wait_for_selector(modal, pilot, "#hook-review-revoke-0")
        modal.query_one("#hook-review-revoke-0").focus()
        await pilot.press("enter")
        async with asyncio.timeout(5):
            while modal.snapshot.ready or modal._busy:
                await pilot.pause(0.01)
        assert owner.snapshot().rows[0].state == "pending"
        modal.query_one("#hook-review-disable-0").focus()
        await pilot.press("enter")
        async with asyncio.timeout(5):
            while not modal.snapshot.ready or modal._busy:
                await pilot.pause(0.01)
        assert toml.loads(hook_file.read_text())["hooks"]["hook"][0]["enabled"] is False
        await pilot.press("escape")
        await pilot.pause()
        assert host.screen is console


async def test_dismissal_during_row_refresh_does_not_query_removed_modal(hook_file):
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(80, 24)) as pilot:
        console = host.screen
        await _wait_for_selector(console, pilot, "#console-control-hooks")
        console.query_one("#console-control-hooks").focus()
        await pilot.press("enter")
        async with asyncio.timeout(5):
            while not isinstance(host.screen, ConsoleHooksReviewModal):
                await pilot.pause(0.01)
        modal = host.screen
        await _wait_for_selector(modal, pilot, "#console-hooks-list")
        region = modal.query_one("#console-hooks-list")
        entered, release = asyncio.Event(), asyncio.Event()

        async def paused_recompose():
            entered.set()
            await release.wait()

        region.recompose = paused_recompose
        refresh = asyncio.create_task(modal._render_rows())
        await asyncio.wait_for(entered.wait(), 5)
        await pilot.press("escape")
        await pilot.pause()
        assert host.screen is console
        synced = []
        modal._sync_actions = lambda: synced.append(True)
        release.set()
        await asyncio.wait_for(refresh, 5)
        assert synced == []


@pytest.mark.parametrize("caller", ["worker", "plain-task"])
async def test_a_review_awaited_off_a_worker_task_is_logged(caller, hook_file):
    """TASK-33621.28 review: W003 cannot see ``request_hook_review``'s await
    (the modal settles its own answer), and the freeze it caused on the app
    pump logged nothing. A caller off a worker task is now an ERROR in the
    app log; a worker caller -- every caller today -- is not. Run under
    ``asyncio.eager_task_factory``, as Textual's ``run_async`` runs the real
    app (``run_test`` does not): there a worker's first step runs before
    ``Worker._task`` is set, and the live hfrf1 run logged a false ERROR."""
    from loguru import logger

    errors: list[str] = []
    sink = logger.add(
        lambda message: errors.append(message.record["message"]),
        level="ERROR",
        filter=lambda record: "TASK-33621.28" in record["message"],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    try:
        async with host.run_test(size=(120, 40)) as pilot:
            loop = asyncio.get_running_loop()
            previous = loop.get_task_factory()
            loop.set_task_factory(asyncio.eager_task_factory)
            try:
                console = host.screen
                owner = console._console_runtime().ensure_hook_permissions()
                snapshot = await asyncio.to_thread(owner.snapshot)
                review = console._request_console_hooks_review(
                    snapshot, False, lambda: None
                )
                if caller == "worker":
                    worker = console.run_worker(review, group="test-hook-review")
                    pending = worker.wait()
                else:
                    pending = asyncio.ensure_future(review)
                async with asyncio.timeout(5):
                    while not isinstance(host.screen, ConsoleHooksReviewModal):
                        await pilot.pause(0.01)
                await _wait_for_selector(host.screen, pilot, "#console-hooks-review")
                await pilot.press("escape")
                result = await asyncio.wait_for(pending, 5)
                assert result.kind == "cancel" and host.screen is console
            finally:
                loop.set_task_factory(previous)
    finally:
        logger.remove(sink)
    assert len(errors) == (1 if caller == "plain-task" else 0), errors
    if errors:
        # Attributable without a traceback: which screen's caller blocked,
        # and whether a Send was waiting on the review.
        assert "screen=ChatScreen" in errors[0], errors
        assert "waiting_for_send=False" in errors[0], errors
