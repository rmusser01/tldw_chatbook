"""A selected mount assertion depends on its actual surface, not every pump."""

import asyncio

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from Tests.Chat.console_execution_mount_wait import wait_for_selected_composer


class MountWaitApp(App):
    def compose(self) -> ComposeResult:
        yield Static("Unrelated", id="unrelated")


@pytest.mark.asyncio
@pytest.mark.parametrize("unrelated_state", ["idle", "detached", "blocked"])
async def test_mount_wait_requires_real_target_without_waiting_for_unrelated_pump(
    unrelated_state,
):
    app = MountWaitApp()
    entered, release = asyncio.Event(), asyncio.Event()

    async def unrelated_callback():
        entered.set()
        await release.wait()

    async with app.run_test() as pilot:
        unrelated = app.screen.query_one("#unrelated")
        if unrelated_state == "detached":
            await unrelated.remove()
            assert not unrelated.is_attached
        elif unrelated_state == "blocked":
            unrelated.call_later(unrelated_callback)
            await entered.wait()
        pending = asyncio.create_task(wait_for_selected_composer(app))
        try:
            await asyncio.sleep(0)
            assert not pending.done()
            composer = Static("Composer", id="console-native-composer")
            await app.screen.mount(composer)
            assert composer.is_mounted and composer.is_attached
            assert not pending.done()  # Mounted alone is not reconciled.
            app.screen._console_process_surface_reconciled = True
            async with asyncio.timeout(10):
                await pending
            assert app.screen.query_one("#console-native-composer") is composer
            if unrelated_state == "blocked":
                assert entered.is_set() and not release.is_set()
        finally:
            release.set()
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("target_state", ["absent", "unreconciled"])
async def test_mount_wait_preserves_caller_deadline_for_missing_condition(target_state):
    app = MountWaitApp()
    async with app.run_test():
        app.screen._console_process_surface_reconciled = target_state == "absent"
        if target_state == "unreconciled":
            await app.screen.mount(Static("Composer", id="console-native-composer"))
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await wait_for_selected_composer(app)


@pytest.mark.asyncio
async def test_mount_wait_propagates_original_application_failure():
    app = MountWaitApp()
    async with app.run_test():
        error = RuntimeError("Original app failure")
        app._exception = error
        try:
            with pytest.raises(RuntimeError) as failure:
                await wait_for_selected_composer(app)
            assert failure.value is error
        finally:
            app._exception = None
