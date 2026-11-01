"""``BaseAppScreen.TAB_REGION`` and ``arrival_focus_target()`` (Roleplay frame B0).

A bare ``textual.app.App`` hosts a real ``BaseAppScreen`` subclass whose
``compose`` is overridden (the ``test_base_app_screen_recompose_focus_seam``
pattern), so no ``TldwCli`` is mounted. The module still imports
``tldw_chatbook.app`` at collection time: ``Tests/UI/conftest.py``'s autouse
fixture would otherwise import it for the first time inside the per-test
sandbox and every test would error at setup with ``RecoveryRequired``
(lessons-testing-evidence).

The behaviour-neutral claim is checked against an IN-TEST "before" arm: a
subclass that re-declares Textual's own ``("tab", "app.focus_next")`` pair,
which is exactly what every route ran before B0. Never against recorded
sequences: first-mount content timing varies (B0 research E3).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Input, TextArea

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
from tldw_chatbook.app_command_providers import _bindings_to_shortcuts
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Navigation.screen_registry import registered_screen_routes

_REGION = "#screen-content, #screen-content *"


class _ProbeScreen(BaseAppScreen):
    """Nav chrome, a content region with four controls, and a footer button."""

    def __init__(self) -> None:
        super().__init__(SimpleNamespace(), "probe")

    def compose(self) -> ComposeResult:
        with Horizontal(id="probe-nav"):
            yield Button("Home", id="probe-nav-home")
            yield Button("Console", id="probe-nav-console")
        with Container(id="screen-content"):
            yield Input(id="probe-a")
            yield Button("B", id="probe-b")
            yield TextArea(id="probe-c")
            yield Button("D", id="probe-d")
        yield Button("Footer", id="probe-footer")


class _StockProbeScreen(_ProbeScreen):
    """The pre-B0 arm: Textual's own app-namespaced Tab bindings."""

    BINDINGS = [
        Binding("tab", "app.focus_next", "Focus Next", show=False),
        Binding("shift+tab", "app.focus_previous", "Focus Previous", show=False),
    ]


class _OptInProbeScreen(_ProbeScreen):
    TAB_REGION = _REGION

    def arrival_focus_target(self):
        return self.query_one("#probe-b")


class _EmptyOptInProbeScreen(_ProbeScreen):
    """Opted in, but the content region holds nothing focusable yet."""

    TAB_REGION = _REGION

    def compose(self) -> ComposeResult:
        with Horizontal(id="probe-nav"):
            yield Button("Home", id="probe-nav-home")
            yield Button("Console", id="probe-nav-console")
        yield Container(id="screen-content")
        yield Button("Footer", id="probe-footer")


class _OwnTabProbeScreen(_ProbeScreen):
    """The ChatScreen/LibraryScreen shape: a screen that re-declares tab."""

    BINDINGS = [Binding("tab", "probe_tab", "Probe tab", show=False)]

    def __init__(self) -> None:
        super().__init__()
        self.probe_tabs = 0

    def action_probe_tab(self) -> None:
        self.probe_tabs += 1


class _Host(App):
    def __init__(self, screen_type: type[_ProbeScreen]) -> None:
        super().__init__()
        self._screen_type = screen_type

    def on_mount(self) -> None:
        self.push_screen(self._screen_type())


async def _walk(screen_type, key: str, start: str | None, presses: int = 10) -> list:
    app = _Host(screen_type)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = app.screen
        if start is None:
            screen.set_focus(None)
        else:
            screen.query_one(f"#{start}").focus()
        await pilot.pause()
        landed = []
        for _ in range(presses):
            await pilot.press(key)
            landed.append(getattr(screen.focused, "id", None))
        return landed


_STARTS = [None, "probe-nav-home", "probe-a", "probe-c", "probe-d", "probe-footer"]


@pytest.mark.parametrize("key", ["tab", "shift+tab"])
@pytest.mark.parametrize("start", _STARTS)
async def test_tab_region_none_walks_exactly_like_textuals_stock_binding(start, key) -> None:
    assert await _walk(_ProbeScreen, key, start) == await _walk(_StockProbeScreen, key, start)


def test_generic_f1_help_lists_the_same_rows_as_textuals_screen() -> None:
    """``App._show_generic_screen_help`` renders ``getattr(screen, "BINDINGS")``."""
    assert _bindings_to_shortcuts(BaseAppScreen.BINDINGS) == _bindings_to_shortcuts(
        Screen.BINDINGS
    )


def test_merged_tab_bindings_are_screen_scoped_and_not_priority() -> None:
    """A priority Tab would preempt every screen's own on_key Tab trap."""
    merged = BaseAppScreen._merged_bindings.key_to_bindings
    assert [(b.action, b.priority) for b in merged["tab"]] == [("region_focus_next", False)]
    assert [(b.action, b.priority) for b in merged["shift+tab"]] == [
        ("region_focus_previous", False)
    ]
    assert set(merged) == set(Screen._merged_bindings.key_to_bindings)
    # The merge would still inherit Screen's copy binding without the re-spread;
    # F1's generic help reads BaseAppScreen.BINDINGS itself, so pin the list.
    assert [b.action for b in BaseAppScreen.BINDINGS if b.key == "ctrl+c,super+c"] == [
        "screen.copy_text"
    ]


async def test_region_actions_are_allowed_by_default() -> None:
    app = _Host(_ProbeScreen)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert app.screen.TAB_REGION is None
        assert app.screen.check_action("region_focus_next", ()) is True
        assert app.screen.check_action("region_focus_previous", ()) is True


async def test_default_arrival_focus_target_is_the_first_content_control() -> None:
    app = _Host(_ProbeScreen)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert app.screen.arrival_focus_target() is app.screen.query_one("#probe-a")


async def test_opt_in_region_wraps_inside_content_and_never_reaches_chrome() -> None:
    landed = await _walk(_OptInProbeScreen, "tab", "probe-d", presses=8)
    assert landed[0] == "probe-a"
    assert not any(
        name and name.startswith(("probe-nav", "probe-footer")) for name in landed
    ), landed


async def test_opt_in_from_chrome_keeps_the_app_wide_walk() -> None:
    assert await _walk(_OptInProbeScreen, "tab", "probe-nav-home", presses=2) == [
        "probe-nav-console",
        "probe-a",
    ]


async def test_opt_in_with_nothing_focused_lands_on_arrival_focus_target() -> None:
    assert await _walk(_OptInProbeScreen, "tab", None, presses=1) == ["probe-b"]


async def test_opt_in_with_an_empty_content_region_never_lands_tab_on_chrome() -> None:
    """The default hook falls back to the screen's first focusable widget (the
    nav bar) when the content holds nothing focusable; that target is outside
    the region and must be ignored, not focused."""
    assert await _walk(_EmptyOptInProbeScreen, "tab", None, presses=1) == [None]


async def test_a_screen_that_redeclares_tab_keeps_its_own_binding() -> None:
    app = _Host(_OwnTabProbeScreen)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.screen.query_one("#probe-a").focus()
        await pilot.pause()
        await pilot.press("tab")
        assert app.screen.probe_tabs == 1
        assert app.screen.focused is app.screen.query_one("#probe-a")


#: Screens that keep their own Tab binding by design, with the action name.
_OWN_TAB = {
    # TASK-2154.11: Console's region-scoped Tab.
    "tldw_chatbook.UI.Screens.chat_screen.ChatScreen": ("focus_next", "focus_previous"),
    # task-32052: Library's Tab confinement; TAB_REGION adoption is FU-1.
    "tldw_chatbook.UI.Screens.library_screen.LibraryScreen": ("focus_next", "focus_previous"),
}


def _production_screen_classes() -> list[type[BaseAppScreen]]:
    for route in registered_screen_routes():
        # Returns None on ImportError; a blanket except here would hide real
        # crashes in production screens.
        route.load_screen_class()
    seen: set[type] = set()
    stack = list(BaseAppScreen.__subclasses__())
    while stack:
        screen_class = stack.pop()
        if screen_class in seen:
            continue
        seen.add(screen_class)
        stack.extend(screen_class.__subclasses__())
    return sorted(
        (cls for cls in seen if cls.__module__.startswith("tldw_chatbook.")),
        key=lambda cls: f"{cls.__module__}.{cls.__qualname__}",
    )


def test_every_production_screen_inherits_the_region_binding_or_is_allowlisted() -> None:
    classes = _production_screen_classes()
    names = {f"{cls.__module__}.{cls.__qualname__}" for cls in classes}
    assert len(classes) >= 15, sorted(names)
    assert set(_OWN_TAB) <= names
    assert "tldw_chatbook.UI.Screens.personas_screen.PersonasScreen" in names
    wrong: dict[str, object] = {}
    for cls in classes:
        name = f"{cls.__module__}.{cls.__qualname__}"
        merged = cls._merged_bindings.key_to_bindings
        actions = (
            tuple(binding.action for binding in merged.get("tab", [])),
            tuple(binding.action for binding in merged.get("shift+tab", [])),
        )
        expected = _OWN_TAB.get(name, ("region_focus_next", "region_focus_previous"))
        if actions != ((expected[0],), (expected[1],)):
            wrong[name] = actions
        if cls.TAB_REGION is not None:
            wrong[name] = ("TAB_REGION", cls.TAB_REGION)
    assert wrong == {}
