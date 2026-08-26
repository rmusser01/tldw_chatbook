"""task-16300: navigation must never leave a content screen resident.

`tldw_chatbook/app.py`'s `_create_navigation_screen` documents the
invariant this suite defends: *"Screens must never be cached and
re-mounted: ``switch_screen`` unmounts the outgoing screen"* -- the
premise under which `ScreenStateStore.save_state`/`restore_state` is the
ONLY continuity mechanism, root-caused from the 2026-07-11 rapid-switch
freeze.

Textual breaks that premise from below. `App.switch_screen`
(`.venv/lib/python3.12/site-packages/textual/app.py:3001-3032`) pops only
``self._screen_stack[-1]`` and appends the new screen; ``_replace_screen``
then unmounts only that popped screen. So when a pushed screen (nav
overflow menu, command palette, picker, confirm dialog) sits above the
content screen, navigating replaces THE MODAL and leaves the content
screen resident in the stack -- mounted, pump running, ``on_unmount``
never fired, timers beating behind whatever the user is now looking at.
The wake-integrity arc (tasks 15970/15971) traced two live Console bugs
to exactly that state.

Two further consequences of the same pop-the-top mechanism are pinned
here: the outgoing screen's pre-navigation hooks were asked of the MODAL
rather than of the content screen (so Console's busy-fleet confirm never
ran and no snapshot was taken), and ``switch_screen`` calls
``top_screen._pop_result_callback()`` WITHOUT invoking it (textual
app.py:3020), stranding the result future of any modal opened through
``push_screen_wait`` -- its awaiting worker never resumes.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.screen import ModalScreen, Screen
from textual.widgets import Static

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen


class _ProbeContentScreen(Screen):
    """Stand-in for a routed content screen (constructed as ``cls(app)``)."""

    def __init__(self, app_instance, screen_name: str = "chat") -> None:
        super().__init__()
        self.app_instance = app_instance
        self.screen_name = screen_name
        self.nav_bar_active = screen_name
        self.unmount_calls = 0
        self.hook_calls: list[str] = []

    def compose(self):
        yield Static(f"content:{self.screen_name}")

    def on_unmount(self) -> None:
        self.unmount_calls += 1

    def flush_pending_work(self) -> bool:
        self.hook_calls.append("flush")
        return True

    def confirm_navigation(self) -> bool:
        self.hook_calls.append("confirm")
        return True

    def save_state(self) -> dict:
        self.hook_calls.append("save_state")
        return {"probe": self.screen_name}


class _ProbeOverlay(ModalScreen[object]):
    """A pushed screen with the hooks a content screen would answer.

    The hooks exist so the suite can prove they are asked of the CONTENT
    screen and never of the overlay that happens to sit on top of it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.hook_calls: list[str] = []
        self.unmount_calls = 0

    def compose(self):
        yield Static("overlay")

    def on_unmount(self) -> None:
        self.unmount_calls += 1

    def flush_pending_work(self) -> bool:
        self.hook_calls.append("flush")
        return True

    def confirm_navigation(self) -> bool:
        self.hook_calls.append("confirm")
        return True

    def save_state(self) -> dict:
        self.hook_calls.append("save_state")
        return {"overlay": True}


class _StubbornOverlay(ModalScreen[object]):
    """An overlay whose ``dismiss`` refuses to leave the stack."""

    def __init__(self) -> None:
        super().__init__()
        self.dismiss_calls = 0

    def compose(self):
        yield Static("stubborn")

    def dismiss(self, result=None):  # noqa: D102 - deliberate no-op
        self.dismiss_calls += 1
        return None


def _install_probe_routes(app, monkeypatch, screens: dict) -> None:
    """Route ``chat``/``library`` to probe screens, recording each build."""

    def fake_resolve(target: str):
        name = "chat" if target == "chat" else "library"

        def _factory(app_instance, _name=name):
            screen = _ProbeContentScreen(app_instance, _name)
            screens.setdefault(_name, []).append(screen)
            return screen

        return name, name, _factory

    monkeypatch.setattr(app, "_resolve_screen_navigation_target", fake_resolve)


async def _mount_initial_probe(app, pilot, screens: dict) -> _ProbeContentScreen:
    """Push the starting content screen the way startup's push does."""
    screen = _ProbeContentScreen(app, "chat")
    screens.setdefault("chat", []).append(screen)
    await app.push_screen(screen)
    app._initial_screen_pushed = True
    app.current_tab = "chat"
    await pilot.pause()
    assert app.screen is screen
    return screen


def _content_screens(app) -> list:
    return [s for s in app.screen_stack if isinstance(s, _ProbeContentScreen)]


@pytest.mark.asyncio
async def test_navigation_under_a_pushed_screen_leaves_no_resident_screen(
    monkeypatch,
):
    """AC#1/#2: the outgoing content screen is replaced, not the overlay.

    RED before the fix: ``switch_screen`` popped the overlay and the
    stack ended as ``[Screen, outgoing, incoming]`` with the outgoing
    screen still running.
    """
    app = _build_test_app()
    screens: dict[str, list] = {}
    _install_probe_routes(app, monkeypatch, screens)

    async with app.run_test(size=(120, 40)) as pilot:
        outgoing = await _mount_initial_probe(app, pilot, screens)
        app.push_screen(_ProbeOverlay())
        await pilot.pause()
        assert isinstance(app.screen, _ProbeOverlay)

        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()

        incoming = app.screen
        assert isinstance(incoming, _ProbeContentScreen)
        assert incoming.screen_name == "library"
        assert outgoing not in app.screen_stack, (
            "navigating under a pushed screen must replace the CONTENT "
            "screen, not the overlay -- a resident hidden screen keeps its "
            "pump, timers and controller alive behind the new screen"
        )
        assert _content_screens(app) == [incoming], (
            f"exactly one content screen may remain; stack was "
            f"{[type(s).__name__ for s in app.screen_stack]}"
        )
        assert outgoing.unmount_calls == 1, (
            "the outgoing screen's teardown must run, exactly as it does "
            "on a navigation with no pushed screen"
        )
        assert not outgoing.is_running, "the outgoing screen's pump must stop"


@pytest.mark.asyncio
async def test_plain_navigation_still_replaces_the_content_screen(monkeypatch):
    """Control: with no overlay the behaviour is unchanged."""
    app = _build_test_app()
    screens: dict[str, list] = {}
    _install_probe_routes(app, monkeypatch, screens)

    async with app.run_test(size=(120, 40)) as pilot:
        outgoing = await _mount_initial_probe(app, pilot, screens)
        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()

        assert outgoing not in app.screen_stack
        assert outgoing.unmount_calls == 1
        assert _content_screens(app) == [app.screen]


@pytest.mark.asyncio
async def test_navigation_hooks_are_asked_of_the_content_screen(monkeypatch):
    """AC#4: flush/confirm/save_state belong to the outgoing CONTENT screen.

    RED before the fix: ``current_screen = self.screen`` resolved to the
    overlay, so Console's busy-fleet ``confirm_navigation`` never ran and
    no ``ScreenStateStore`` snapshot was taken for the tab being left.
    """
    app = _build_test_app()
    screens: dict[str, list] = {}
    _install_probe_routes(app, monkeypatch, screens)

    async with app.run_test(size=(120, 40)) as pilot:
        outgoing = await _mount_initial_probe(app, pilot, screens)
        overlay = _ProbeOverlay()
        app.push_screen(overlay)
        await pilot.pause()

        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()

        assert outgoing.hook_calls == ["flush", "confirm", "save_state"], (
            "the outgoing content screen must be flushed, asked to confirm, "
            "and snapshotted -- the overlay is not the screen being left"
        )
        assert overlay.hook_calls == [], (
            "the overlay must never be mistaken for the outgoing screen"
        )
        assert app.screen_state_store.restore(
            "chat", app._current_runtime_identity()
        ) == {"probe": "chat"}


@pytest.mark.asyncio
async def test_content_screen_veto_keeps_the_overlay_and_the_screen(monkeypatch):
    """A vetoed navigation costs the user nothing: overlay and screen stay."""
    app = _build_test_app()
    screens: dict[str, list] = {}
    _install_probe_routes(app, monkeypatch, screens)

    async with app.run_test(size=(120, 40)) as pilot:
        outgoing = await _mount_initial_probe(app, pilot, screens)
        monkeypatch.setattr(outgoing, "confirm_navigation", lambda: False)
        overlay = _ProbeOverlay()
        app.push_screen(overlay)
        await pilot.pause()

        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()

        assert outgoing in app.screen_stack, "a veto must keep the screen"
        assert overlay in app.screen_stack, (
            "a veto must not have cost the user the dialog they had open -- "
            "overlays are only dismissed once the switch is committed"
        )
        assert app.screen is overlay


@pytest.mark.asyncio
async def test_awaited_modal_resumes_its_caller_with_a_cancel_result(monkeypatch):
    """AC#3: an in-flight ``push_screen_wait`` awaiter is never stranded.

    RED before the fix: ``switch_screen`` pops the top screen's result
    callback WITHOUT calling it (textual app.py:3020), so the future
    behind ``push_screen_wait`` was never resolved and the awaiting worker
    hung for the rest of the session.
    """
    app = _build_test_app()
    screens: dict[str, list] = {}
    _install_probe_routes(app, monkeypatch, screens)

    async with app.run_test(size=(120, 40)) as pilot:
        await _mount_initial_probe(app, pilot, screens)

        resumed = asyncio.Event()
        box: dict[str, object] = {}

        async def _await_modal() -> None:
            box["result"] = await app.push_screen_wait(_ProbeOverlay())
            resumed.set()

        app.run_worker(_await_modal(), name="probe-modal-awaiter")
        for _ in range(50):
            if isinstance(app.screen, _ProbeOverlay):
                break
            await pilot.pause()
        assert isinstance(app.screen, _ProbeOverlay)
        assert not resumed.is_set()

        await app.handle_screen_navigation(NavigateToScreen("library"))
        for _ in range(50):
            if resumed.is_set():
                break
            await pilot.pause()

        assert resumed.is_set(), (
            "navigating out from under an awaited modal must resume its "
            "caller -- dropping the result future strands that worker "
            "forever (it has no timeout and nothing else resolves it)"
        )
        assert box["result"] is None, (
            "the awaiter must observe the same no-result value a user "
            "dismissal produces, so existing callers need no new branch"
        )


@pytest.mark.asyncio
async def test_console_is_not_left_resident_under_the_nav_overflow_menu(tmp_path):
    """The reported live shape, with the real Console screen and menu.

    This is the exact construction the wake-integrity arc used to
    *produce* the leak (`_leak_resident_chat`): a real ChatScreen, the
    real nav overflow menu pushed above it, a real navigation to Library.
    It now asserts the inverse -- the Console screen leaves the stack and
    its controller is told to shut down -- so the resident Console that
    ate the unseen mark (15971) and read the wrong composer (15970)
    cannot come back unnoticed.
    """
    from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
    from Tests.UI.test_console_native_chat_flow import (
        _configure_native_ready_console,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from tldw_chatbook.UI.Navigation.nav_overflow_menu import NavOverflowMenu
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)

    async with app.run_test(size=(160, 48)) as pilot:
        chat = ChatScreen(app)
        await app.push_screen(chat)
        app._initial_screen_pushed = True
        await pilot.pause()
        await _wait_for_selector(chat, pilot, "#console-native-composer")
        controller = chat._ensure_console_chat_controller()
        assert not controller._shutdown_requested.is_set()

        app.push_screen(NavOverflowMenu())
        await pilot.pause()
        assert isinstance(app.screen, NavOverflowMenu)

        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()

        assert type(app.screen).__name__ == "LibraryScreen"
        assert chat not in app.screen_stack, (
            "the Console screen must not stay resident behind Library -- "
            "that resident screen is what delivered wakes off-screen and "
            "view-cleared the unseen mark the user never saw (15970/15971)"
        )
        assert not chat.is_running
        assert controller._shutdown_requested.is_set(), (
            "leaving Console must shut its controller down, exactly as a "
            "navigation with no menu open already did"
        )


@pytest.mark.asyncio
async def test_undismissable_overlay_aborts_navigation_instead_of_leaking(
    monkeypatch,
):
    """Fail closed: if the stack cannot be reduced, do not switch.

    Switching anyway is what produces the resident screen, so an overlay
    that will not leave has to stop the navigation loudly rather than
    silently reintroduce the leak.
    """
    app = _build_test_app()
    screens: dict[str, list] = {}
    _install_probe_routes(app, monkeypatch, screens)
    notifications: list[str] = []
    monkeypatch.setattr(
        app, "notify", lambda message, **kwargs: notifications.append(message)
    )

    async with app.run_test(size=(120, 40)) as pilot:
        outgoing = await _mount_initial_probe(app, pilot, screens)
        stubborn = _StubbornOverlay()
        app.push_screen(stubborn)
        await pilot.pause()

        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()

        assert _content_screens(app) == [outgoing], (
            "aborting must leave the one content screen it started with, "
            "not add a second live one"
        )
        assert outgoing.unmount_calls == 0
        assert notifications, "the abort must be reported to the user"
        assert stubborn.dismiss_calls == 1, (
            "a screen that ignored one dismissal must not be dismissed "
            "again -- ``dismiss`` is not side-effect free (the file pickers "
            "invoke their caller's callback from inside it), so retrying "
            "until a loop bound runs out would fire those callbacks once "
            "per attempt"
        )
