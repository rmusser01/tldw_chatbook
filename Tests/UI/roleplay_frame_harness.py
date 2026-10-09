"""Roleplay frame test harness: two styled tiers, the size matrix, painted geometry.

Spec section 5.7.1 (Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md):

- ``RoleplayMockApp`` is the delegating ``PersonasTestApp`` that used to live in
  ``test_personas_workbench.py`` (moved verbatim; both old modules re-export
  it under the old names). It is the UNSTYLED tier: ``ConsolidatedCSSApp``
  loads no app bundle, so unstyled tests never assert geometry.
- ``StyledRoleplayMockApp`` is styled tier 1: the boot bundle plus every lazy
  split sheet (``APP_STYLESHEETS``, derived from the build's own
  ``SCREEN_OWNED_SPLITS``), so it carries ``screen_feature_roleplay.tcss``.
- ``painted_rows``, ``click_meta_cells``, ``settle`` and ``wait_until`` (moved
  here from ``test_roleplay_hostile_names.py``, TASK-34400) and
  ``seed_mock_characters`` are the one copy every Roleplay test imports.
- ``roleplay_full_app()`` is styled tier 2: a real ``TldwCli`` that reaches
  Roleplay through a real route (the initial tab, or Ctrl+4 from Home), so
  ``TldwCli._ensure_screen_owned_css`` loads the Roleplay sheet exactly as it
  does for a user. It is seeded through the same ``ccp_character_handler``
  seams as the mock tier (``seed_mock_characters``), not through a temporary
  ChaChaNotes as spec 5.7.1 describes: a recorded B1 deviation, so this tier
  proves styling and routing, not persistence. Frame slice B2a converts it to
  DB seeding before its volume tests (TASK-33910.3 carries the note).

A real ``TldwCli`` that pushes ``PersonasScreen`` itself skips
``_ensure_screen_owned_css`` and paints the inline header without its rules
(TASK-32187's Watchlists trap; ``Tests/UI/full_app_destination_context.py``).
Reach Roleplay by navigation, as ``roleplay_full_app`` does, or call
``app._ensure_screen_owned_css("personas")`` before the push;
``Tests/UI/test_roleplay_stylesheet.py`` scans the suite for the bare push.

Geometry is asserted relative to the MEASURED nav bar and header, never as
absolute rows (spec 5.7.2 item 1; ADR-210's compact nav moves rows below 35).
Not collected by pytest (the file name has no ``test_`` prefix).
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from textual.css.stylesheet import CssSource
from textual.geometry import Region
from textual.pilot import Pilot
from textual.screen import Screen
from textual.widget import Widget

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
from Tests.app_module_patches import patch_app_global
from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import APP_STYLESHEETS, CSS_DIR, ConsolidatedCSSApp
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.glyph_fallback import set_ascii_glyph_mode

#: The lazily loaded Roleplay sheet (build_css.SCREEN_OWNED_SPLITS).
ROLEPLAY_SHEET = CSS_DIR / "screen_feature_roleplay.tcss"

#: Spec 5.7.1 geometry matrix for B1: 80x24 is the degrade check, the other
#: three are the design centre. 60x24 joins with B7: B1's worst-case header
#: (every chip, a server, the longest kind) fits from 65 columns, and the
#: spec's below-64 layout is B7's and TASK-33910.24's.
ROLEPLAY_SIZES: tuple[tuple[int, int], ...] = (
    (80, 24),
    (120, 36),
    (160, 45),
    (220, 55),
)

#: The size matrix as a parametrize mark: ``@size_matrix`` runs a test once
#: per B1 size with a ``roleplay_size`` argument (ids ``80x24`` ...). A mark,
#: not a fixture, so test modules need no fixture import.
size_matrix = pytest.mark.parametrize(
    "roleplay_size", ROLEPLAY_SIZES, ids=lambda size: f"{size[0]}x{size[1]}"
)


class RoleplayMockApp(ConsolidatedCSSApp):
    """The unstyled tier: today's delegating ``PersonasTestApp``, moved verbatim."""

    def __init__(self, mock_app_instance):
        super().__init__()
        self._mock = mock_app_instance
        self.character_persona_scope_service = (
            mock_app_instance.character_persona_scope_service
        )

    # Delegating these to a MagicMock would make Textual see phantom dynamic
    # hooks (``compute_*``/``watch_*``/...) on the App and crash at mount.
    _NON_DELEGATED_PREFIXES = (
        "_",
        "watch_",
        "compute_",
        "validate_",
        "action_",
        "key_",
        "on_",
    )

    def __getattr__(self, name):
        if name.startswith(self._NON_DELEGATED_PREFIXES):
            raise AttributeError(name)
        return getattr(self.__dict__["_mock"], name)

    def compose(self):
        # Mirrors the real app: an `AppFooterStatus` composed directly on
        # the app's own default screen (see app.py's `compose()`).
        # Task-264: `PersonasScreen` (via `BaseAppScreen.compose()`) now
        # mounts its OWN `AppFooterStatus` too, and
        # `PersonasScreen._register_footer_shortcuts()` resolves that
        # screen-owned instance via ``self.query_one("AppFooterStatus")`` --
        # so this default-screen widget is only kept around as a foil (the
        # tests below assert the registration does NOT land here).
        yield AppFooterStatus(id="app-footer-status")

    async def _ensure_tts_profile_service(self):
        """Delegate the real app's private lazy loader when a test provides it."""

        loader = self.__dict__["_mock"].__dict__.get("_ensure_tts_profile_service")
        if not callable(loader):
            return None
        result = loader()
        if inspect.isawaitable(result):
            result = await result
        return result

    def on_mount(self) -> None:
        self.push_screen(PersonasScreen(self))


class StyledRoleplayMockApp(RoleplayMockApp):
    """Styled tier 1: the boot bundle plus every lazy split sheet (spec 5.7.1)."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]


#: The names the Roleplay test modules have always used (re-exported there).
PersonasTestApp = RoleplayMockApp
StyledPersonasTestApp = StyledRoleplayMockApp


def seed_mock_characters(
    monkeypatch: pytest.MonkeyPatch, records: list[dict[str, Any]]
) -> None:
    """Route the screen's character seams over ``records`` (both tiers).

    The same seams ``test_personas_workbench.stub_characters`` patches:
    ``fetch_all_characters``/``fetch_character_by_id`` plus the paged loader.

    Args:
        monkeypatch: The test's monkeypatch.
        records: Character dicts with at least ``id`` and ``name``.
    """
    import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as handler
    from Tests.UI.test_personas_dictionaries import patch_character_paging

    monkeypatch.setattr(
        handler, "fetch_all_characters", lambda: [dict(r) for r in records]
    )
    monkeypatch.setattr(
        handler,
        "fetch_character_by_id",
        lambda character_id: next(
            (dict(r) for r in records if str(r["id"]) == str(character_id)), None
        ),
    )
    patch_character_paging(monkeypatch)


def _settings_for_full_app(ascii_glyphs: bool) -> Callable[..., Any]:
    """``get_cli_setting`` for the full-app tier: no splash, chosen glyph mode.

    The app reads ``splash_screen.enabled`` at compose and resets the glyph
    mode from ``appearance.ascii_glyphs`` at compose (app.py), so both must
    come from this patch, live for the whole ``run_test``. Measured while
    planning: arrival takes 7.7 s with the splash and 0.7 s without.
    """

    def settings(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        if section == "appearance" and key == "ascii_glyphs":
            return ascii_glyphs
        return default

    return settings


# wait_until, painted_rows, click_meta_cells and settle moved here unchanged
# from test_roleplay_hostile_names.py (TASK-34400) in B1: this harness is the
# one home of the Roleplay paint helpers.
async def wait_until(
    pilot: Pilot,
    predicate: Callable[[], bool],
    *,
    timeout: float = 20.0,
    what: str = "",
) -> None:
    """Poll ``predicate`` with a monotonic deadline.

    Args:
        pilot: The running test pilot.
        predicate: Returns True once the awaited state holds.
        timeout: Seconds to wait before failing.
        what: Names the awaited state in the failure message.

    Raises:
        AssertionError: ``predicate`` stayed False for ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError(f"timed out after {timeout}s waiting for {what or predicate}")


def _first_list_item_present(app) -> bool:
    screen = app.screen
    return isinstance(screen, PersonasScreen) and bool(
        screen.query("#personas-library-rows > ListItem")
    )


@asynccontextmanager
async def roleplay_full_app(
    *,
    size: tuple[int, int],
    entry: str = "initial_tab",
    ascii_glyphs: bool = False,
    notifications: bool = False,
    on_home: Callable[[Any], None] | None = None,
) -> AsyncIterator[Any]:
    """Styled tier 2: a real ``TldwCli`` arriving at Roleplay by a real route.

    Seed data through the same seams first (``seed_mock_characters``; no
    ChaChaNotes DB yet, see the module docstring). Yields the pilot once the
    Roleplay screen shows its first list item.

    Args:
        size: Terminal ``(columns, rows)``.
        entry: ``"initial_tab"`` boots straight into Roleplay (the app's
            ``_push_initial_screen`` path); ``"ctrl+4"`` boots Home and presses
            Ctrl+4 (the in-app navigation path).
        ascii_glyphs: Run with ``appearance.ascii_glyphs`` on.
        notifications: Mount toasts (``run_test`` defaults to off).
        on_home: With ``entry="ctrl+4"``, called with the app while Home is
            showing, just before Ctrl+4 (to observe the pre-visit state).
    """
    if entry not in ("initial_tab", "ctrl+4"):
        raise ValueError(f"unknown entry {entry!r}")
    app = _build_test_app(
        configured_default="personas" if entry == "initial_tab" else "home"
    )
    settings = _settings_for_full_app(ascii_glyphs)
    try:
        with patch_app_global("get_cli_setting", side_effect=settings):
            async with app.run_test(size=size, notifications=notifications) as pilot:
                await wait_until(
                    pilot,
                    lambda: getattr(app, "_initial_screen_pushed", False),
                    what="the initial screen",
                )
                if entry == "ctrl+4":
                    await wait_until(
                        pilot,
                        lambda: type(app.screen).__name__ == "HomeScreen",
                        what="Home",
                    )
                    if on_home is not None:
                        on_home(app)
                    await pilot.press("ctrl+4")
                await wait_until(
                    pilot,
                    lambda: _first_list_item_present(app),
                    what="Roleplay's list",
                )
                await settle(pilot)
                yield pilot
    finally:
        # The app set the process-wide glyph mode at compose; never leak it.
        set_ascii_glyph_mode(False)


#: The two styled tiers (spec 5.7.1): ``"mock"`` (StyledRoleplayMockApp) and
#: ``"full"`` (roleplay_full_app). ``@styled_tiers`` gives a test a
#: ``styled_tier`` argument per tier.
STYLED_TIERS: tuple[str, ...] = ("mock", "full")
styled_tiers = pytest.mark.parametrize("styled_tier", STYLED_TIERS)


@asynccontextmanager
async def open_styled_roleplay(
    tier: str,
    mock_app_instance: Any,
    *,
    size: tuple[int, int],
    notifications: bool = False,
    ascii_glyphs: bool = False,
    app_class: type[StyledRoleplayMockApp] = StyledRoleplayMockApp,
) -> AsyncIterator[Any]:
    """Mount Roleplay under one styled tier and wait for its first list item.

    Args:
        tier: ``"mock"`` or ``"full"``.
        mock_app_instance: The ``Tests/UI/conftest.py`` fixture (mock tier only).
        size: Terminal ``(columns, rows)``.
        notifications: Mount toasts.
        ascii_glyphs: Run with the ASCII glyph mode on (restored afterwards).
        app_class: A ``StyledRoleplayMockApp`` subclass (mock tier only).
    """
    if tier == "full":
        async with roleplay_full_app(
            size=size, notifications=notifications, ascii_glyphs=ascii_glyphs
        ) as pilot:
            yield pilot
        return
    if tier != "mock":
        raise ValueError(f"unknown tier {tier!r}")
    app = app_class(mock_app_instance)
    # The mock tier has no app compose that resets the process-wide mode.
    set_ascii_glyph_mode(ascii_glyphs)
    try:
        async with app.run_test(size=size, notifications=notifications) as pilot:
            await wait_until(
                pilot, lambda: _first_list_item_present(app), what="Roleplay's list"
            )
            await settle(pilot)
            yield pilot
    finally:
        set_ascii_glyph_mode(False)


def chrome_bottoms(screen) -> tuple[int, int]:
    """``(nav bottom, header bottom)`` as measured, in screen rows (0-based)."""
    nav = screen.query_one(MainNavigationBar).region
    header = screen.query_one("#personas-header").region
    return nav.bottom, header.bottom


def first_list_item(screen) -> Widget:
    """The first row of the items list (B2a keeps the id, replaces the type)."""
    return screen.query("#personas-library-rows > ListItem").first()


def painted_rows(screen: Screen) -> list[str]:
    """Every compositor row as plain text: what the terminal shows.

    Args:
        screen: The mounted screen whose compositor output is read.

    Returns:
        One string per terminal row, top to bottom, as currently painted.
    """
    return [strip.text for strip in screen._compositor.render_strips()]


def painted_text(screen, region: Region) -> str:
    """The plain text the compositor paints inside ``region`` (one line per row)."""
    rows = painted_rows(screen)
    return "\n".join(
        rows[y][region.x : region.right] for y in range(region.y, region.bottom)
    )


def assert_painted_inside(widget: Widget, pane: Widget) -> None:
    """Every cell of ``widget`` lies in ``pane``'s painted window and is not covered.

    The painted window is ``scrollable_content_region``, not the pane's outer
    region (lessons-testing-evidence, "painted window, not pane rectangle");
    and containment is not visibility, so the compositor's hit test must
    find ``widget`` (or a descendant) on its middle row (a docked sibling
    can cover a contained widget).
    """
    window = pane.scrollable_content_region
    region = widget.region
    assert region.width > 0 and region.height > 0, (
        f"{widget!r} paints nothing: {region}"
    )
    assert region.intersection(window) == region, (
        f"{widget!r} {region} escapes {pane!r}'s painted window {window}"
    )
    middle = region.y + (region.height - 1) // 2
    hit, _ = widget.screen.get_widget_at(region.x, middle)
    assert hit is widget or widget in hit.ancestors, (
        f"{widget!r} is covered by {hit!r} at ({region.x}, {middle})"
    )


def click_meta_cells(screen: Screen) -> list[tuple[int, int, str, str]]:
    """Every painted cell run that carries an ``@click`` action.

    Args:
        screen: The mounted screen whose compositor output is scanned.

    Returns:
        ``(x, y, text, action)`` for each painted segment whose style meta
        holds ``@click``; empty when no painted text is clickable markup.
    """
    hits = []
    for y, strip in enumerate(screen._compositor.render_strips()):
        x = 0
        for segment in strip:
            meta = segment.style.meta if segment.style is not None else {}
            if meta and "@click" in meta:
                hits.append((x, y, segment.text, meta["@click"]))
            x += segment.cell_length
    return hits


def drop_rule_from_loaded_sheet(app, sheet: Path, selector: str) -> None:
    """Delete one rule block from an already-loaded sheet and restyle the app.

    The executable form of the "delete one ``_roleplay.tcss`` header rule"
    discrimination check (spec B1): it edits the PARSED source the app holds,
    so it works the same whether the sheet arrived through a harness
    ``CSS_PATH`` or through the app's route loader.

    Args:
        app: A running app.
        sheet: The stylesheet file the app loaded.
        selector: The exact selector text heading the block to delete.

    Raises:
        KeyError: The app never loaded ``sheet`` (itself a finding).
    """
    key = (str(sheet), "")
    source = app.stylesheet.source[key]
    head = f"\n{selector} {{"
    start = source.content.index(head)
    end = source.content.index("}", start) + 1
    mutated = source.content[:start] + source.content[end:]
    app.stylesheet.source[key] = CssSource(
        mutated, source.is_defaults, source.tie_breaker, source.scope
    )
    app.stylesheet.reparse()
    app.stylesheet.update(app)


async def settle(pilot: Pilot) -> None:
    """Let the screen's own workers finish, then repaint twice.

    Only workers owned by the current screen: the full app runs app-wide
    workers that never finish, so ``app.workers.wait_for_complete()`` would
    wait forever there.

    Args:
        pilot: The running test pilot.
    """
    await pilot.pause()
    screen = pilot.app.screen
    unfinished = [
        worker
        for worker in pilot.app.workers
        if screen in worker.node.ancestors_with_self and not worker.is_finished
    ]
    if unfinished:
        await pilot.app.workers.wait_for_complete(unfinished)
    await pilot.pause()
    await asyncio.sleep(0)
    await pilot.pause()
