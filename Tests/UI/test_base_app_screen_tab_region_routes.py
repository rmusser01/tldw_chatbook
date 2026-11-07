"""Every route with ``TAB_REGION is None`` keeps Textual's stock Tab walk (Roleplay frame B0).

Mounted real screens: slow (about 3 s per screen class), so NOT in the PR-gate
census. The test is ARM-AGNOSTIC: the B0 gate run executes it on both paired
arms (Task 11 copies this file into the base arm), so a red on head only is a
B0 change, and a red on both arms is a limit of the probe, not of B0.

The oracle is Textual's own stock walk taken in the SAME mount: from a saved
focus S, ``app.action_focus_next()`` -- exactly what Screen's
``("tab", "app.focus_next")`` binding ran before B0, and what
``region_focus_next`` calls while ``TAB_REGION is None`` -- records its
target; focus goes back to S; then the real key press must land on the same
widget. Never a recorded sequence and never a model of the chain: first-mount
content timing varies (B0 research E3: one cold Meetings mount lacked a
late-mounted button).

Parametrised over ``module.Class`` strings read from the registry, so
collecting this file imports no screen module; each test loads its own class.
"""

from __future__ import annotations

import time

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Navigation.screen_registry import registered_screen_routes

#: Re-declare Tab themselves (allowlisted in test_base_app_screen_tab_region.py);
#: their own suites pin their Tab.
_OWN_TAB_CLASS_NAMES = frozenset({"ChatScreen", "LibraryScreen"})


def _routes_by_class_path() -> dict[str, object]:
    """``"module.Class"`` -> its first registered route, importing nothing."""
    routes: dict[str, object] = {}
    for route in registered_screen_routes():
        if route.class_name in _OWN_TAB_CLASS_NAMES:
            continue
        routes.setdefault(f"{route.module_path}.{route.class_name}", route)
    return routes


_ROUTES = _routes_by_class_path()


class _Host(ConsolidatedCSSApp):
    def __init__(self, app_instance, screen_class) -> None:
        super().__init__()
        self.app_instance = app_instance
        self._screen_class = screen_class

    async def on_mount(self) -> None:
        await self.push_screen(self._screen_class(self.app_instance))

    def on_navigate_to_screen(self, message) -> None:
        """Swallow navigation requests: this host mounts one screen only."""


def _inside_content(widget) -> bool:
    """Whether ``widget`` sits in ``#screen-content``.

    Steps that START in the chrome are not compared: the nav bar's overflow
    re-lays itself out as focus moves through it, so the oracle's walk and
    the key press that follows can see different chains (measured red,
    intermittently, on BOTH arms while planning: Shift+Tab from
    ``nav-meetings``). The first step of each direction is always compared,
    so Shift+Tab still crosses from the content into the chrome once.
    """
    return any(ancestor.id == "screen-content" for ancestor in widget.ancestors)


def _ident(widget) -> str | None:
    if widget is None:
        return None
    return widget.id or f"<{type(widget).__name__}>"


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize(
    "class_path", sorted(_ROUTES), ids=lambda path: path.rsplit(".", 1)[-1]
)
async def test_tab_from_the_first_content_control_follows_the_stock_walk(class_path) -> None:
    screen_class = _ROUTES[class_path].load_screen_class()
    if not (isinstance(screen_class, type) and issubclass(screen_class, BaseAppScreen)):
        pytest.skip(f"{class_path}: not a loadable BaseAppScreen in this environment")
    host = _Host(_build_test_app(), screen_class)
    async with host.run_test(size=(140, 42)) as pilot:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if isinstance(host.screen, screen_class) and host.screen.region.width > 0:
                break
            await pilot.pause(0.02)
        screen = host.screen
        assert isinstance(screen, screen_class), type(screen)
        await pilot.pause(0.8)
        # Arm-agnostic: the base arm has neither the attribute nor the actions.
        assert getattr(screen, "TAB_REGION", None) is None
        if hasattr(screen, "action_region_focus_next"):
            assert screen.check_action("region_focus_next", ()) is True
            assert screen.check_action("region_focus_previous", ()) is True
        mismatches = []
        compared = 0
        for key, stock_walk in (
            ("tab", host.action_focus_next),
            ("shift+tab", host.action_focus_previous),
        ):
            screen.set_focus(screen._first_focusable_in_content())
            await pilot.pause(0.1)
            for step in range(3):
                start = screen.focused
                if start is None or (step and not _inside_content(start)):
                    break
                compared += 1
                stock_walk()
                expected = screen.focused
                screen.set_focus(start)
                await pilot.pause(0.05)
                await pilot.press(key)
                await pilot.pause(0.05)
                if screen.focused is not expected:
                    mismatches.append(
                        (key, _ident(start), _ident(expected), _ident(screen.focused))
                    )
        assert compared >= 2, compared
        assert mismatches == []
