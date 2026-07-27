"""Lab mode strip: presence, active state, and cross-screen navigation.

The Lab destination seats three screens -- Models (llm), Speech (stts),
Evals (evals). Each mounts a LabModeStrip under its DestinationHeader whose
chips post NavigateToScreen for the other modes' routes; the chip for the
owning screen is highlighted and inert. Before the strip existed, the Evals
inline workbench was unreachable from the rest of the shell.
"""

from __future__ import annotations

import time
from importlib import import_module
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest
from textual import on
from textual.app import App
from textual.widgets import Button

from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.lab_mode_strip import LAB_MODE_CHIPS, LabModeStrip
from tldw_chatbook.UI.Workbench.workbench_widgets import DestinationHeader
from Tests.UI.test_screen_navigation import _build_test_app


# (route, owning screen class name, module path, active chip id)
_LAB_SCREENS = (
    ("llm", "LLMScreen", "tldw_chatbook.UI.Screens.llm_screen", "lab-mode-models"),
    ("stts", "STTSScreen", "tldw_chatbook.UI.Screens.stts_screen", "lab-mode-speech"),
    ("evals", "EvalsScreen", "tldw_chatbook.UI.Screens.evals_screen", "lab-mode-evals"),
)

# Subset of _LAB_SCREENS whose compose_content() is still the flat pattern
# (DestinationHeader/LabModeStrip yielded directly, callable via
# list(screen.compose_content()) with no running app). "evals" is excluded
# here (PR3a Task 3): it now wraps its content in
# with Vertical(id="evals-shell"): ..., a three-pane workbench shell, and
# entering that context manager requires an active Textual app --
# test_lab_screen_composes_mode_strip_under_destination_header below would
# raise NoActiveAppError for it. See
# test_evals_composes_mode_strip_under_destination_header_via_real_app for
# the equivalent coverage through a real running app.
#
# "llm" is excluded for the same reason as of the Lab-frame PR2 adoption
# (Task 6): LLMScreen now extends LabScreen, whose compose_content() only
# enters `with Horizontal(id="lab-status-row"): ...` when lab_status_chips()
# is non-empty -- and Models' status chip (running-server count) always is.
# Entering that context manager needs an active Textual app for the same
# reason evals' does. See
# test_llm_composes_mode_strip_under_destination_header_via_real_app.
#
# The other three async tests below (_StripHarness-based, or exercising the
# real shell) still parametrize/exercise all of _LAB_SCREENS including evals
# and llm -- they never call compose_content() directly.
#
# "stts" joins them with the Speech adoption: STTSScreen now extends
# LabScreen too, so its compose_content() is the frame's -- which always
# enters `with Horizontal(id="lab-header-row"): ...`, and Speech's own
# capability chip is composed inside it. Same NoActiveAppError, same
# remedy. That leaves this subset EMPTY, which is the honest outcome: all
# three Lab modes are on the frame now, and no Lab screen composes flat any
# more. Kept rather than deleted so the next flat-composing mode (if one is
# ever added) lands here instead of silently skipping the check.
_LAB_SCREENS_FLAT_COMPOSE = tuple(
    entry for entry in _LAB_SCREENS if entry[0] not in ("evals", "llm", "stts")
)


class _StripHarness(App[None]):
    """Bare harness mounting only the strip; records navigation requests."""

    def __init__(self, active_route: str):
        super().__init__()
        self._active_route = active_route
        self.navigated: list[str] = []

    def compose(self):
        yield LabModeStrip(active_route=self._active_route, id="lab-mode-strip")

    @on(NavigateToScreen)
    def _record_navigation(self, message: NavigateToScreen) -> None:
        self.navigated.append(message.screen_name)


@pytest.mark.parametrize(
    ("route", "class_name", "module", "active_chip"), _LAB_SCREENS_FLAT_COMPOSE
)
def test_lab_screen_composes_mode_strip_under_destination_header(
    route, class_name, module, active_chip
):
    app = _build_test_app()
    screen = getattr(import_module(module), class_name)(app)

    widgets = list(screen.compose_content())

    assert isinstance(widgets[0], DestinationHeader), route
    strip = widgets[1]
    assert isinstance(strip, LabModeStrip), route
    assert strip.id == "lab-mode-strip"
    assert strip.active_route == route


@pytest.mark.asyncio
async def test_evals_composes_mode_strip_under_destination_header_via_real_app():
    """Equivalent of test_lab_screen_composes_mode_strip_under_destination_header
    for Evals, whose compose_content() now wraps content in
    with Vertical(id="evals-shell"): ... (see _LAB_SCREENS_FLAT_COMPOSE's
    comment) and so cannot be driven via a bare list(compose_content())
    call outside a running app."""
    from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen

    class _EvalsHarness(App[None]):
        def __init__(self, app_instance):
            super().__init__()
            self._app_instance = app_instance

        async def on_mount(self) -> None:
            await self.push_screen(EvalsScreen(self._app_instance))

    app_instance = _build_test_app()
    app = _EvalsHarness(app_instance)

    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.1)
        screen = app.screen_stack[-1]

        header = screen.query_one("#evals-destination-header", DestinationHeader)
        strip = screen.query_one("#lab-mode-strip", LabModeStrip)
        assert strip.active_route == "evals"
        # DestinationHeader precedes the mode strip in document order --
        # the same structural contract the flat-pattern screens assert via
        # widgets[0]/widgets[1] above.
        header_index = list(screen.walk_children()).index(header)
        strip_index = list(screen.walk_children()).index(strip)
        assert header_index < strip_index


@pytest.mark.asyncio
async def test_llm_composes_mode_strip_under_destination_header_via_real_app():
    """Equivalent of test_lab_screen_composes_mode_strip_under_destination_header
    for Models (llm), whose compose_content() is now LabScreen's -- see
    _LAB_SCREENS_FLAT_COMPOSE's comment -- and so cannot be driven via a bare
    list(compose_content()) call outside a running app once a status row is
    present."""
    from tldw_chatbook.UI.Screens.llm_screen import LLMScreen

    class _LLMHarness(App[None]):
        def __init__(self, app_instance):
            super().__init__()
            self._app_instance = app_instance

        async def on_mount(self) -> None:
            await self.push_screen(LLMScreen(self._app_instance))

    app_instance = _build_test_app()
    app = _LLMHarness(app_instance)

    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.1)
        screen = app.screen_stack[-1]

        header = screen.query_one("#lab-destination-header", DestinationHeader)
        strip = screen.query_one("#lab-mode-strip", LabModeStrip)
        assert strip.active_route == "llm"
        # DestinationHeader precedes the mode strip in document order --
        # the same structural contract the flat-pattern screens assert via
        # widgets[0]/widgets[1] above.
        header_index = list(screen.walk_children()).index(header)
        strip_index = list(screen.walk_children()).index(strip)
        assert header_index < strip_index


@pytest.mark.asyncio
@pytest.mark.parametrize(("route", "class_name", "module", "active_chip"), _LAB_SCREENS)
async def test_active_chip_reflects_current_screen(
    route, class_name, module, active_chip
):
    app = _StripHarness(route)

    async with app.run_test() as pilot:
        await pilot.pause()
        for mode_id, _label, mode_route, _tooltip in LAB_MODE_CHIPS:
            chip = app.query_one(f"#lab-mode-{mode_id}", Button)
            assert chip.has_class("is-active") == (mode_route == route), mode_id
        # Exactly one chip is active, and it is the owning screen's.
        active = [
            button.id
            for button in app.query(".lab-mode-chip")
            if button.has_class("is-active")
        ]
        assert active == [active_chip]


@pytest.mark.asyncio
async def test_inactive_chips_post_navigation_to_their_routes():
    app = _StripHarness("llm")

    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#lab-mode-evals", Button).press()
        await pilot.pause()
        app.query_one("#lab-mode-speech", Button).press()
        await pilot.pause()

    assert app.navigated == ["evals", "stts"]


@pytest.mark.asyncio
async def test_active_chip_press_is_a_noop():
    app = _StripHarness("evals")

    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#lab-mode-evals", Button).press()
        await pilot.pause()

    assert app.navigated == []


def _prepare_clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for env_var, path_name in (
        ("HOME", "home"),
        ("XDG_CONFIG_HOME", "xdg-config"),
        ("XDG_DATA_HOME", "xdg-data"),
        ("XDG_CACHE_HOME", "xdg-cache"),
    ):
        path = tmp_path / path_name
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(env_var, str(path))


def _test_cli_setting(section: str, key: str, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


async def _wait_until(
    pilot,
    condition: Callable[[], bool],
    *,
    timeout_seconds: float = 10.0,
    interval_seconds: float = 0.05,
    context: str = "",
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return
        await pilot.pause(interval_seconds)
    if condition():
        return
    context_suffix = f" for {context}" if context else ""
    raise AssertionError(
        f"condition was not met within {timeout_seconds:.1f}s{context_suffix}"
    )


@pytest.mark.asyncio
async def test_lab_route_and_mode_strip_navigate_the_real_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """End to end: NavigateToScreen("lab") seats Models, and the strip moves
    between Lab screens while the Lab nav button stays boxed."""
    from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen
    from tldw_chatbook.UI.Screens.home_screen import HomeScreen
    from tldw_chatbook.UI.Screens.llm_screen import LLMScreen

    _prepare_clean_environment(monkeypatch, tmp_path)
    # Isolate the navigation test from the Models screen's HuggingFace
    # widgets: their init-fired reactives and scan/download workers schedule
    # deferred DOM updates (call_later/thread completion) that race child
    # mounting and screen switches under run_test -- a pre-existing family of
    # races in those widgets, unrelated to shell navigation.
    from tldw_chatbook.Widgets.HuggingFace.download_manager import DownloadManager
    from tldw_chatbook.Widgets.HuggingFace.local_models_widget import LocalModelsWidget
    from tldw_chatbook.Widgets.HuggingFace.model_search_widget import ModelSearchWidget

    async def _noop_async_update(self, *args):
        return None

    monkeypatch.setattr(ModelSearchWidget, "perform_search", lambda self: None)
    monkeypatch.setattr(ModelSearchWidget, "_update_results_list", _noop_async_update)
    monkeypatch.setattr(LocalModelsWidget, "scan_models", lambda self: None)
    monkeypatch.setattr(LocalModelsWidget, "_refresh_model_list", _noop_async_update)
    monkeypatch.setattr(LocalModelsWidget, "_update_summary", lambda self: None)
    monkeypatch.setattr(DownloadManager, "_refresh_downloads_list", _noop_async_update)
    monkeypatch.setattr(DownloadManager, "_update_summary", lambda self: None)
    app = _build_test_app()
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(160, 45)) as pilot:
            await _wait_until(
                pilot,
                lambda: isinstance(app.screen, HomeScreen),
                context="initial home",
            )

            # The critique repro: the "lab" destination id must seat Lab's
            # primary route (Models) instead of leaving the app on MCP.
            app.post_message(NavigateToScreen("lab"))
            await _wait_until(
                pilot, lambda: isinstance(app.screen, LLMScreen), context="lab -> llm"
            )
            assert app.screen.query_one("#lab-mode-models", Button).has_class(
                "is-active"
            )
            assert app.screen.query_one("#nav-lab", Button).has_class("is-active")

            app.screen.query_one("#lab-mode-evals", Button).press()
            await _wait_until(
                pilot,
                lambda: isinstance(app.screen, EvalsScreen),
                context="chip -> evals",
            )
            assert app.screen.query_one("#lab-mode-evals", Button).has_class(
                "is-active"
            )
            assert app.screen.query_one("#nav-lab", Button).has_class("is-active")

            app.screen.query_one("#lab-mode-models", Button).press()
            await _wait_until(
                pilot, lambda: isinstance(app.screen, LLMScreen), context="chip -> llm"
            )
            assert app.screen.query_one("#lab-mode-models", Button).has_class(
                "is-active"
            )
            assert app.screen.query_one("#nav-lab", Button).has_class("is-active")


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLED_STYLESHEET = _REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


class _BundledStripHarness(App[None]):
    """Mount the strip with the production stylesheet.

    The bundle is required: the bug under test lives in the bundle's global
    `.is-active` rule, which beats LabModeStrip.DEFAULT_CSS. A harness
    without CSS_PATH passes vacuously.
    """

    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def __init__(self, active_route: str) -> None:
        super().__init__()
        self._active_route = active_route

    def compose(self):
        yield LabModeStrip(active_route=self._active_route, id="lab-mode-strip")


def _has_border(widget) -> bool:
    """True when any edge declares a border style."""
    border = widget.styles.border
    return any(
        edge[0] for edge in (border.top, border.right, border.bottom, border.left)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("route", "active_chip"), [
    ("llm", "lab-mode-models"),
    ("stts", "lab-mode-speech"),
    ("evals", "lab-mode-evals"),
])
async def test_active_mode_chip_has_no_border_so_its_label_renders(route, active_chip):
    """The active chip must not gain the bundle's global `.is-active` border.

    The strip is one row tall. A bordered chip becomes a three-row box, so
    only its top border renders and the mode label disappears entirely --
    leaving no way to see which Lab mode is active.
    """
    app = _BundledStripHarness(route)
    async with app.run_test(size=(80, 6)) as pilot:
        await pilot.pause()
        chip = app.query_one(f"#{active_chip}")

        assert "is-active" in chip.classes
        assert not _has_border(chip), (
            f"{active_chip} has a border; its label is clipped by the 1-row strip"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(("route", "active_chip", "other_chip"), [
    ("llm", "lab-mode-models", "lab-mode-speech"),
    ("stts", "lab-mode-speech", "lab-mode-evals"),
    ("evals", "lab-mode-evals", "lab-mode-models"),
])
async def test_focused_non_active_chip_does_not_impersonate_the_active_chip(
    route, active_chip, other_chip
):
    """A focused, non-active chip must not read as the active one.

    The app's global `Button:focus` grants `bold underline` -- exactly the
    `.is-active` chip's signature (`$ds-focus-bg`/`$ds-focus-fg`/`bold
    underline`). Without a focus guard, tabbing to a non-active chip would
    make two chips look active at once, defeating the point of highlighting
    the active mode at all. Underline must stay exclusive to `.is-active`.
    """
    app = _BundledStripHarness(route)
    async with app.run_test(size=(80, 6)) as pilot:
        await pilot.pause()
        active_chip_widget = app.query_one(f"#{active_chip}")
        other_chip_widget = app.query_one(f"#{other_chip}")

        app.set_focus(other_chip_widget)
        await pilot.pause()

        active_text_style = active_chip_widget.styles.text_style
        other_text_style = other_chip_widget.styles.text_style

        assert active_text_style != other_text_style
        assert active_text_style.underline, (
            "the active chip lost its underline signature"
        )
        assert not other_text_style.underline, (
            f"focused non-active chip {other_chip} carries underline; it now "
            "reads as active too"
        )


def _rendered_text(app: App) -> str:
    """Join every compositor strip's segment text into one blob.

    Textual 8.2.7 has no `App.export_text()`; `screen._compositor.render_strips()`
    is the way to read what was actually rendered, as opposed to inferring it
    from styles.
    """
    strips = app.screen._compositor.render_strips()
    return "\n".join("".join(segment.text for segment in strip) for strip in strips)


@pytest.mark.asyncio
@pytest.mark.parametrize(("route", "active_label"), [
    ("llm", "Models"),
    ("stts", "Speech"),
    ("evals", "Evals"),
])
async def test_active_mode_chip_label_is_actually_rendered(route, active_label):
    """Assert the active chip's rendered label text, not a styles proxy.

    The spec requires the active chip's *rendered label* be present -- a
    test asserting only the `is-active` class or the absence of a border
    passes even when the label is clipped by the surrounding strip. This
    reads the real compositor output, so it fails if the label is ever
    clipped again for any reason (not just the specific border bug fixed
    here).
    """
    app = _BundledStripHarness(route)
    async with app.run_test(size=(80, 6)) as pilot:
        await pilot.pause()
        rendered = _rendered_text(app)

        assert active_label in rendered, (
            f"active label {active_label!r} not found in rendered strip output:\n{rendered}"
        )
