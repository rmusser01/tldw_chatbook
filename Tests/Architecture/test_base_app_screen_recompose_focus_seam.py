"""`BaseAppScreen`'s focus-restore seam must cover BOTH doors into a recompose.

Tier-2 review S19 [D3]. task-31946 built the capture/restore pair as "the ONE
seam" and put it in ``refresh(recompose=True)``. But ``refresh(recompose=True)``
only SCHEDULES ``Widget._check_recompose``, which awaits ``recompose()`` -- and
nine production call sites skip ``refresh`` entirely and ``await
self.recompose()`` directly (``UI/Screens/workflows_screen.py``, three
``UI/Library_Modules/`` modules, six sites in ``UI/Screens/library_screen.py``).
Every one of those tore the tree down with no restore queued, leaving
``screen.focused`` at ``None`` and the keyboard dead until the user clicked --
the app-wide soft-lock ``lessons-textual.md`` (TASK-22281) documents.

The fix moves the capture/restore into ``recompose()``, the coroutine both
doors converge on. These tests are the pin for "both doors", so a future edit
cannot quietly move it back up one level.

Gate-free on purpose: a bare ``textual.app.App`` harness, not the real app, so
`Backup_Recovery`'s ADR-126 recovery gate never runs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input

from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen


class _FocusProbeScreen(BaseAppScreen):
    """A minimal `BaseAppScreen` that composes one focusable widget.

    ``compose`` is overridden rather than ``compose_content`` so the real
    nav bar / footer (which want a live ``TldwCli``) are never built; the
    ``#screen-content`` id is kept because ``_first_focusable_in_content``
    scopes its fallback to it.
    """

    def __init__(self) -> None:
        super().__init__(SimpleNamespace(), "probe")

    def compose(self) -> ComposeResult:
        from textual.containers import Container

        with Container(id="screen-content"):
            yield Input(id="probe-input")


class _ProbeApp(App):
    def on_mount(self) -> None:
        self.push_screen(_FocusProbeScreen())


async def _focused_id(pilot) -> str | None:
    return getattr(pilot.app.screen.focused, "id", None)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_direct_recompose_restores_focus() -> None:
    """The door nine production call sites actually use.

    ``await self.recompose()`` bypasses ``refresh()`` entirely, so before
    this fix nothing captured or restored focus on this path.
    """
    async with _ProbeApp().run_test() as pilot:
        screen = pilot.app.screen
        screen.set_focus(screen.query_one("#probe-input", Input))
        await pilot.pause()
        assert await _focused_id(pilot) == "probe-input"

        await screen.recompose()
        for _ in range(6):
            await pilot.pause()

        assert await _focused_id(pilot) == "probe-input", (
            "a direct `await self.recompose()` must restore keyboard focus "
            "-- it is the same teardown `refresh(recompose=True)` performs"
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_refresh_recompose_still_restores_focus() -> None:
    """The door task-31946 already covered must keep working.

    ``refresh(recompose=True)`` routes through ``_check_recompose`` ->
    ``recompose()``, so moving the seam down covers it too -- and exactly
    once, not twice.
    """
    async with _ProbeApp().run_test() as pilot:
        screen = pilot.app.screen
        screen.set_focus(screen.query_one("#probe-input", Input))
        await pilot.pause()
        assert await _focused_id(pilot) == "probe-input"

        screen.refresh(recompose=True)
        for _ in range(6):
            await pilot.pause()

        assert await _focused_id(pilot) == "probe-input"


class _OrderingScreen(_FocusProbeScreen):
    """Queues a post-recompose pass of its own, the way subclasses do."""

    def __init__(self) -> None:
        super().__init__()
        self.order: list[str] = []

    def restore_focus_after_recompose(self, previous: str | None) -> None:
        self.order.append("restore")
        super().restore_focus_after_recompose(previous)

    def _own_pass(self) -> None:
        self.order.append("own_pass")


class _OrderingApp(App):
    def on_mount(self) -> None:
        self.push_screen(_OrderingScreen())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_callers_own_post_recompose_pass_still_runs_before_the_restore() -> None:
    """The two-hop queue order is load-bearing and must survive the move.

    PR L review item 3: the Library's stage-visibility pass HIDES containers,
    and a widget that becomes hidden blurs itself (`Widget._on_hide` ->
    `blur()`), so a restore landing before that pass would put focus straight
    back to `None`. `Widget.focusable` reads `visible`, which cannot see a
    hide that has not happened yet -- ordering is the only fix.
    """
    async with _OrderingApp().run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, _OrderingScreen)
        screen.set_focus(screen.query_one("#probe-input", Input))
        await pilot.pause()

        screen.refresh(recompose=True)
        screen.call_after_refresh(screen._own_pass)
        for _ in range(8):
            await pilot.pause()

        assert screen.order == ["own_pass", "restore"], (
            f"the caller's own pass must run first; got {screen.order}"
        )
        assert await _focused_id(pilot) == "probe-input"
