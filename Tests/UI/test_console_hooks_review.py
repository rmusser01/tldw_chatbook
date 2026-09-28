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
        first = asyncio.create_task(
            console._dispatch_console_draft_send("retained draft")
        )
        async with asyncio.timeout(5):
            while not isinstance(host.screen, ConsoleHooksReviewModal):
                await pilot.pause(0.01)
        await pilot.press("escape")
        assert not await first
        assert composer.draft_text() == "retained draft" and calls == []
        second = asyncio.create_task(
            console._dispatch_console_draft_send("retained draft")
        )
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
        assert await second
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
        send = asyncio.create_task(
            console._dispatch_console_draft_send("keep while saving")
        )
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
        assert not await send
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
        send = asyncio.create_task(
            console._dispatch_console_draft_send("keep for settings")
        )
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
        assert host.screen is modal and not send.done()
        assert await pilot.click("#console-hooks-settings")
        assert not await asyncio.wait_for(send, 5)
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
