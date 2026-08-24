"""Mounted tests for master-shell navigation."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual import events
from textual.app import App

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.containers import Horizontal
from textual.widgets import Button

import tldw_chatbook
from tldw_chatbook.UI.Navigation.main_navigation import (
    MainNavigationBar,
    _straddles_viewport,
)

#: task-3801: the real, generated app stylesheet -- mirrors
#: `Tests/UI/test_mcp_inspector.py`'s `_BUNDLED_CSS_PATH` /
#: `InspectorAppWithBundledCSS`. A bare `App()` with no `CSS_PATH` only
#: exercises `MainNavigationBar.BUNDLED_CSS`, never the separately
#: maintained `.nav-button.nav-button-clip-ghost:disabled` override in
#: `css/components/_navigation.tcss` -- which is the tier that actually wins
#: live, since `App.CSS_PATH` stylesheets outrank widget `DEFAULT_CSS`
#: regardless of specificity or `!important` (see the rule's docstring in
#: `main_navigation.py`).
_BUNDLED_CSS_PATH = str(Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss")

#: The selector `test_ghost_rule_is_width_neutral_under_the_bundled_stylesheet`
#: pins, and the box-model properties that made it geometry-non-neutral once
#: before (task-3225 review round 4's `border: solid $background` regression,
#: see that test's docstring).
_GHOST_RULE_SELECTOR = ".nav-button.nav-button-clip-ghost:disabled"
_BOX_MODEL_PROPERTIES = ("border", "padding", "margin", "width", "height")


def _bundled_css_rule_body(css_path: str, selector: str) -> str:
    """Return the declaration block for `selector` in a generated CSS bundle.

    Fails loudly (via assertion) rather than returning an empty string when
    the bundle is missing or the selector can't be found, so a moved/renamed
    bundle file or a renamed selector breaks LOUD instead of letting a
    geometry-only test go quietly non-proving (task-3801 review finding: the
    original harness never checked either of those things, so it could pass
    for the wrong reason).
    """
    bundle = Path(css_path)
    assert bundle.is_file(), f"bundled stylesheet missing: {css_path!r}"
    css = bundle.read_text(encoding="utf-8")
    # Strip comments first so a selector mentioned only in a docstring/comment
    # (this module has several) can't be mistaken for the live rule.
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert match, f"{selector!r} not found in bundled stylesheet {css_path!r}"
    return match.group(1)


def _declared_properties(rule_body: str) -> list[str]:
    """Return the property names declared in a `prop: value;` rule body.

    Matches on `name :` rather than doing a bare substring search so a value
    that happens to contain a box-model word (e.g. `background: $background`
    contains no such word today, but a future `color: $border-muted` could)
    is never mistaken for a property declaration.
    """
    return re.findall(r"([a-zA-Z-]+)\s*:", rule_body)


def test_compact_navigation_labels_preserve_full_meaning():
    from tldw_chatbook.UI.Navigation.shell_destinations import get_shell_destination

    wc = get_shell_destination("watchlists_collections")

    assert wc.label == "Watchlists"
    assert wc.full_label == "Watchlists"
    assert "Collections" not in wc.tooltip
    assert (
        wc.navigation_priority < get_shell_destination("settings").navigation_priority
    )


def test_master_shell_navigation_uses_compact_spacing_for_full_destination_rail():
    css = MainNavigationBar.BUNDLED_CSS

    assert "margin: 0;" in css
    assert "padding: 0;" in css
    assert ".nav-overflow-hint" in css


@pytest.mark.asyncio
async def test_master_shell_navigation_order_and_labels():
    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="home")

    app = TestApp()

    async with app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)

        actual = [
            (button.id, str(button.label).strip())
            for button in app.query(".nav-button")
        ]

    assert actual == [
        ("nav-home", "\u23031 Home"),
        ("nav-console", "\u23032 Console"),
        ("nav-library", "\u23033 Library"),
        ("nav-research", "F10 Research"),
        ("nav-artifacts", "\u23034 Artifacts"),
        ("nav-personas", "\u23035 Roleplay"),
        ("nav-watchlists_collections", "\u23036 Watchlists"),
        ("nav-schedules", "\u23037 Schedules"),
        ("nav-workflows", "\u23038 Workflows"),
        ("nav-mcp", "\u23039 MCP"),
        ("nav-acp", "\u23030 ACP"),
        ("nav-lab", "F7 Lab"),
        ("nav-logs", "F8 Logs"),
        ("nav-settings", "F9 Settings"),
    ]


def test_nav_button_label_numbering_scheme():
    from tldw_chatbook.UI.Navigation.main_navigation import nav_button_label

    # F-002: the labels used to read "1 Home" -- implying a bare-digit key
    # -- while the actual binding is ctrl+digit. The label now carries the
    # control glyph so the affordance matches the keybinding at zero extra
    # width per tab. ctrl+1..ctrl+9 cover the first nine destinations,
    # ctrl+0 the tenth; Lab, Logs, Settings carry their F7/F8/F9 routes.
    digits = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")
    for index, digit in enumerate(digits):
        assert nav_button_label(index, "Label") == f"\u2303{digit} Label"
    assert nav_button_label(10, "Lab") == "F7 Lab"
    assert nav_button_label(11, "Logs") == "F8 Logs"
    assert nav_button_label(12, "Settings") == "F9 Settings"


@pytest.mark.asyncio
async def test_master_shell_navigation_keeps_active_destination_visible_on_mount():
    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="settings")

    app = TestApp()

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause(0.6)

        nav = app.query_one(MainNavigationBar)
        strip = nav.query_one("#nav-destination-strip")
        active_button = app.query_one("#nav-settings", Button)

        # Overflow hides (never clips) the destinations that don't fit; the
        # active one always renders in full.
        assert active_button.display
        assert active_button.region.width > 0
        assert active_button.region.x >= strip.region.x
        assert active_button.region.right <= strip.region.right


@pytest.mark.asyncio
async def test_master_shell_navigation_reveals_active_destination_when_it_changes():
    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="home")

        def on_navigate_to_screen(self, message):
            pass

    app = TestApp()

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause(0.6)

        nav = app.query_one(MainNavigationBar)
        strip = nav.query_one("#nav-destination-strip")
        settings_button = app.query_one("#nav-settings", Button)
        # Settings overflows at 60 cols: the strip scrolls to reveal the
        # active destination instead of hiding or clipping buttons.
        assert strip.max_scroll_x > 0

        nav._activate_navigation_button(settings_button)
        await pilot.pause(0.6)

        assert settings_button.has_class("is-active")
        assert settings_button.region.width > 0
        assert settings_button.region.right <= strip.region.right


@pytest.mark.asyncio
async def test_master_shell_navigation_docks_overflow_hint_outside_destination_strip():
    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="settings")

    app = TestApp()

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause(0.6)

        nav = app.query_one(MainNavigationBar)
        strip = nav.query_one("#nav-destination-strip")
        hint = app.query_one("#nav-overflow-hint")

        assert hint.parent is nav
        assert hint not in strip.children
        # Even with most destinations hidden by overflow, the hint stays
        # visible at the bar's right edge.
        assert hint.display
        assert hint.region.width > 0
        assert hint.region.right == nav.region.right


@pytest.mark.asyncio
async def test_master_shell_navigation_uses_terminal_tab_rail():
    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="console")

    app = TestApp()

    async with app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)

        nav_buttons = list(app.query(".nav-button"))
        separators = list(app.query(".nav-separator"))
        active_button = app.query_one("#nav-console", Button)

    assert nav_buttons
    assert all(button.has_class("ascii-nav-tab") for button in nav_buttons)
    assert separators == []
    assert active_button.has_class("is-active")
    assert ".nav-separator" not in MainNavigationBar.BUNDLED_CSS
    assert "background: $primary-darken-1;" in MainNavigationBar.BUNDLED_CSS


@pytest.mark.asyncio
async def test_home_and_console_remain_first_primary_destinations():
    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="home")

    app = TestApp()

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause(0.1)
        buttons = list(app.query(".nav-button"))

    assert [(button.id, str(button.label).strip()) for button in buttons[:2]] == [
        ("nav-home", "⌃1 Home"),
        ("nav-console", "⌃2 Console"),
    ]


@pytest.mark.asyncio
async def test_master_shell_navigation_routes_to_primary_route():
    events = []

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="home")

        def on_mount(self):
            self.query_one("#nav-console", Button).press()

        def on_navigate_to_screen(self, message):
            events.append(message.screen_name)

    app = TestApp()

    async with app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)

    assert events == ["chat"]


@pytest.mark.asyncio
async def test_active_destination_subroute_can_return_to_primary_route():
    events = []

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="library", active_route="study")

        def on_navigate_to_screen(self, message):
            events.append(message.screen_name)

    app = TestApp()

    async with app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)
        app.query_one("#nav-library", Button).press()
        await pilot.pause(0.1)

    assert events == ["library"]


@pytest.mark.asyncio
async def test_active_destination_primary_route_still_noops():
    events = []

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="library", active_route="library")

        def on_navigate_to_screen(self, message):
            events.append(message.screen_name)

    app = TestApp()

    async with app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)
        app.query_one("#nav-library", Button).press()
        await pilot.pause(0.1)

    assert events == []


@pytest.mark.asyncio
async def test_media_folded_route_returns_to_library_primary_route():
    """task-2851 AC#2: the Library nav button must never become a permanent
    no-op after a Library-folded legacy route occupies the slot.

    The reported bug's "sticky" half was that, after the command palette's
    "Media & Content: Open Media Library" entry hijacked the Library tab
    (nav bar highlighted "library", but the mounted screen was the legacy
    ``MediaScreen``), re-selecting Library allegedly did not restore the
    canonical ``LibraryScreen``. Re-verified live against this branch's HEAD
    (dev has moved since the 6ffa56516 finding): the "already active" guard
    in ``_activate_navigation_button`` already distinguishes a folded
    subroute (``active_route="media"``) from Library's own primary route
    (``active_route="library"``) correctly -- pressing the Library button
    posts a real ``NavigateToScreen("library")`` rather than short-
    circuiting as a no-op. This pins that behavior explicitly for the exact
    route the bug named, mirroring
    ``test_active_destination_subroute_can_return_to_primary_route`` (which
    covers the same guard for "study").
    """
    events = []

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="library", active_route="media")

        def on_navigate_to_screen(self, message):
            events.append(message.screen_name)

    app = TestApp()

    async with app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)
        app.query_one("#nav-library", Button).press()
        await pilot.pause(0.1)

    assert events == ["library"]


@pytest.mark.asyncio
async def test_every_visible_master_shell_nav_destination_resolves():
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    app = _build_test_app()

    # Research Workspace is registered lazily before its screen-owning task
    # creates the module, so route metadata (not importability) is Task 1's
    # contract for this one destination.
    for destination in SHELL_DESTINATION_ORDER:
        if destination.destination_id == "research":
            continue
        _screen_name, _tab_id, screen_class = app._resolve_screen_navigation_target(
            destination.primary_route
        )
        assert screen_class is not None, destination.primary_route


def test_folded_routes_highlight_owning_destination():
    folded = {
        "search": ("library", "search"),
        "media": ("library", "media"),
        "study": ("library", "study"),
        "writing": ("library", "writing"),
        "research": ("research", "research"),
        "ingest": ("library", "ingest"),
        "llm": ("lab", "llm"),
        "stts": ("lab", "stts"),
        "evals": ("lab", "evals"),
        "stats": ("settings", "stats"),
        # The retired Coding screen folds into Console.
        "coding": ("console", "chat"),
    }

    for route, (destination_id, canonical_route) in folded.items():
        nav = MainNavigationBar(active=route)
        assert nav.active_destination_id == destination_id, route
        assert nav.active_route == canonical_route, route


@pytest.mark.asyncio
async def test_folded_screen_boxes_owning_destination_button():
    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="search", active_route="search")

    app = TestApp()

    async with app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)
        assert app.query_one("#nav-library", Button).has_class("is-active")
        assert not app.query_one("#nav-lab", Button).has_class("is-active")

    class LabApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="llm", active_route="llm")

    lab_app = LabApp()

    async with lab_app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)
        assert lab_app.query_one("#nav-lab", Button).has_class("is-active")
        assert not lab_app.query_one("#nav-library", Button).has_class("is-active")


def test_shell_destination_hotkeys_keep_existing_destination_owners():
    """Inserting Research cannot move any existing destination's shortcut."""
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Navigation.shell_destinations import (
        SHELL_DESTINATION_ORDER,
        SHELL_DESTINATION_SHORTCUTS,
    )

    hotkey_bindings = [
        binding
        for binding in TldwCli.BINDINGS
        if binding.action.startswith("shell_destination(")
    ]

    assert dict(SHELL_DESTINATION_SHORTCUTS) == {
        "home": "ctrl+1",
        "console": "ctrl+2",
        "library": "ctrl+3",
        "artifacts": "ctrl+4",
        "personas": "ctrl+5",
        "watchlists_collections": "ctrl+6",
        "schedules": "ctrl+7",
        "workflows": "ctrl+8",
        "mcp": "ctrl+9",
        "acp": "ctrl+0",
        "lab": "f7",
        "logs": "f8",
        "settings": "f9",
        "research": "f10",
    }
    assert len(hotkey_bindings) == len(SHELL_DESTINATION_ORDER)
    for binding in hotkey_bindings:
        destination_id = binding.action.removeprefix("shell_destination(").removesuffix(")")
        destination = next(
            candidate
            for candidate in SHELL_DESTINATION_ORDER
            if candidate.destination_id == destination_id
        )
        assert binding.key == SHELL_DESTINATION_SHORTCUTS[destination_id]
        assert binding.action == f"shell_destination({destination_id})"
        assert destination.accessible_label in binding.description


def test_action_shell_destination_posts_primary_route():
    """The single hotkey action navigates to each destination's primary route."""
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    posted = []
    fake_app = SimpleNamespace(post_message=posted.append)

    for destination in SHELL_DESTINATION_ORDER:
        TldwCli.action_shell_destination(fake_app, destination.destination_id)
        message = posted[-1]
        assert isinstance(message, NavigateToScreen)
        assert message.screen_name == destination.primary_route, (
            destination.destination_id
        )

    # Textual binding actions pass the argument as a string.
    posted.clear()
    TldwCli.action_shell_destination(fake_app, "research")
    assert isinstance(posted[-1], NavigateToScreen)
    assert posted[-1].screen_name == "research_workspace"

    # Unknown destination IDs are a safe no-op.
    posted.clear()
    TldwCli.action_shell_destination(fake_app, "not-a-destination")
    assert posted == []


@pytest.mark.asyncio
async def test_nav_overflow_menu_reaches_hidden_destinations_at_100_cols():
    """F-001: at 100 columns the strip clips mid-button (the review's
    "8 Workflows" -> "8" artifact) and later destinations have no click
    path in the strip itself. The "More ▾" affordance opens the overflow
    menu, where every clipped destination is listed and pressable --
    TASK-2154.21 replaced the old paging hint with this menu so a press
    navigates directly instead of scrolling page by page."""

    class TestApp(ConsolidatedCSSApp):
        def __init__(self):
            super().__init__()
            self.nav_requests: list[str] = []

        def compose(self):
            yield MainNavigationBar(active="home")

        def on_navigate_to_screen(self, message) -> None:
            self.nav_requests.append(message.screen_name)

    app = TestApp()

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.5)

        strip = app.query_one("#nav-destination-strip")
        hint = app.query_one("#nav-overflow-hint")

        # Overflowing: the affordance is on duty.
        assert strip.max_scroll_x > 0
        assert hint.display is True

        def visible(button):
            return (
                button.region.x >= strip.region.x
                and button.region.right <= strip.region.right
                and button.region.width > 0
            )

        settings = app.query_one("#nav-settings", Button)
        assert not visible(settings), "test premise: Settings starts off-screen"

        # One press opens the menu; the clipped destination is a row there.
        hint.press()
        await pilot.pause(0.5)
        menu = app.screen_stack[-1]
        assert menu.__class__.__name__ == "NavOverflowMenu"
        menu_settings = menu.query_one("#nav-overflow-settings", Button)
        assert "Settings" in str(menu_settings.label)

        # Pressing the row navigates exactly like the strip button would.
        menu_settings.press()
        await pilot.pause(0.5)
        assert "settings" in app.nav_requests
        assert app.screen_stack[-1].__class__.__name__ != "NavOverflowMenu"


@pytest.mark.asyncio
async def test_more_hint_never_scrolls_the_strip_so_it_cannot_overscroll():
    """PR #1322 review: the old paging hint could land past the end of the
    strip when its increment exceeded the remaining scroll. TASK-2154.21's
    overflow menu removed the pager entirely -- a press opens the menu and
    leaves the strip's scroll position untouched, so overscroll is
    structurally impossible."""

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="home")

    app = TestApp()

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.5)

        strip = app.query_one("#nav-destination-strip")
        hint = app.query_one("#nav-overflow-hint", Button)

        assert strip.max_scroll_x > 0
        assert strip.scroll_offset.x == 0

        hint.press()
        await pilot.pause(0.5)

        assert app.screen_stack[-1].__class__.__name__ == "NavOverflowMenu"
        assert strip.scroll_offset.x == 0


# --- task-3200: nav-bar mid-word tab cut at narrow widths -----------------
#
# UAT (task-2858 P2 batch, LIB-18) found the destination strip's
# `overflow-x: auto` scroll clipping a partially-visible trailing button
# mid-word at 80 columns -- e.g. "Watchlists" rendered as "Watc" right
# before the "More ›" hint. `_ghost_clipped_buttons` blanks (rather than
# hides) any button whose CURRENT render straddles either edge of the
# strip's scroll viewport by coloring every surface to match the bar's
# background (`.nav-button-clip-ghost` in `DEFAULT_CSS`), so the same
# geometric straddle that used to leak a partial word now paints nothing
# readable. These tests pin BOTH the geometry (region no longer straddles
# in a way that shows any label glyphs -- a ghosted straddle is exempted)
# AND the actual POST-CLIP rendered text (via the compositor, not the
# un-clipped `Button.label` source), at two active-tab positions --  an
# early one (no scrolling needed) and a late one (the strip must scroll,
# which is what exposes a straddling neighbor) -- since scroll position is
# what determines which button (if any) lands on the edge.


def _readable_nav_text(app: App) -> str:
    """Every compositor segment's text that is actually READABLE, joined --
    what a person looking at the terminal would see, as opposed to any
    widget's un-clipped `.label` source string.

    `render_strips()` (Textual 8.2.7 has no `App.export_text()`, per the
    same-shaped helper in `test_console_session_tab_strip.py`) returns the
    POST-CLIP characters, but ghosting (`.nav-button-clip-ghost`) makes a
    straddling button invisible by setting its foreground color EQUAL to
    its background, not by removing the characters -- so a segment whose
    `style.color == style.bgcolor` renders nothing a person can read even
    though its `.text` still contains the real glyphs. Filtering those out
    before joining is what makes this an honest "what did the screen show"
    check instead of a "what characters exist in the buffer" one.
    """
    strips = app.screen._compositor.render_strips()
    lines = []
    for strip in strips:
        chars = []
        for segment in strip:
            style = segment.style
            if style is not None and style.color == style.bgcolor:
                continue  # foreground matches background: invisible
            chars.append(segment.text)
        lines.append("".join(chars))
    return "\n".join(lines)


def _straddling_buttons(app: App, strip) -> list[str]:
    """Nav buttons whose region partially overlaps the strip's visible
    edge WITHOUT being clip-ghosted -- i.e. buttons that would leak a
    partial label onto the screen."""
    offenders = []
    for button in app.query(".nav-button"):
        if not button.display or button.has_class("nav-button-clip-ghost"):
            continue
        region = button.region
        if region.width <= 0:
            continue
        fully_in = region.x >= strip.region.x and region.right <= strip.region.right
        fully_out = region.right <= strip.region.x or region.x >= strip.region.right
        if not fully_in and not fully_out:
            offenders.append(button.id)
    return offenders


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "width,active",
    [
        (80, "home"),  # early destination: no scrolling needed
        (80, "settings"),  # late destination: forces a scroll
        (100, "home"),
        (100, "watchlists_collections"),
    ],
)
async def test_nav_strip_never_renders_a_partial_destination_label(width, active):
    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active=active)

    app = TestApp()

    async with app.run_test(size=(width, 24)) as pilot:
        await pilot.pause(0.6)

        strip = app.query_one("#nav-destination-strip", Horizontal)

        # Geometry: nothing straddles the viewport edge with real content.
        assert _straddling_buttons(app, strip) == []

        # The active destination is always fully, genuinely visible (never
        # ghosted) -- the one invariant this fix must not regress.
        active_button = app.query_one(f"#nav-{active}", Button)
        assert active_button.display
        assert not active_button.has_class("nav-button-clip-ghost")
        assert active_button.region.width > 0
        assert active_button.region.x >= strip.region.x
        assert active_button.region.right <= strip.region.right

        # Rendered-text pin: every ghosted button's label contributes NO
        # READABLE fragment to the actual painted screen (foreground ==
        # background, not merely absent characters). Compare against the
        # first several characters (past the hotkey prefix) of each
        # ghosted button's label -- long enough that a coincidental
        # substring match elsewhere on screen is not a concern here.
        painted = _readable_nav_text(app)
        ghosted = [
            button
            for button in app.query(".nav-button")
            if button.has_class("nav-button-clip-ghost")
        ]
        for button in ghosted:
            label_text = str(button.label).strip()
            # Strip the "⌃N " / "F7 " hotkey prefix to get the destination
            # word itself (e.g. "Watchlists"), then check a chunk of it.
            word = label_text.split(" ", 1)[-1]
            fragment = word[:4]
            assert fragment not in painted, (
                f"{button.id} is ghosted but '{fragment}' still readable"
            )
            # Review finding: color alone left a ghosted button fully
            # interactive (Tab-reachable with no focus ring, clickable
            # while invisible). `disabled` must accompany the ghost class.
            assert button.disabled, f"{button.id} is ghosted but not disabled"
            assert not button.focusable, f"{button.id} is ghosted but still focusable"
        if width == 100 and active == "home":
            # Pin the specific finding this task fixed (task-2858 P2 /
            # LIB-18): at a narrow width with the strip scrolled to its
            # default position, some destination WILL straddle the "More
            # ›" hint's edge -- if nothing were ever ghosted here, the
            # geometry/rendered-text assertions above would be vacuous.
            assert ghosted, "test premise: expected a straddling destination at 100 cols"


# --- task-4020: re-critique RC-02 -- ghosting effectiveness under the real,
# BUNDLED stylesheet (the tier that actually governs what a user sees), not
# just `MainNavigationBar.BUNDLED_CSS` ------------------------------------
#
# Root cause: the 2026-08-09 re-critique's mechanical probe captured the nav
# bar with a COLORLESS text dump (`tmux capture-pane -p`, no `-e`/ANSI). That
# cannot distinguish a genuinely-rendered mid-word-clipped label (foreground
# != background, actually legible) from a correctly ghosted one (`.nav-
# button-clip-ghost:disabled` pins foreground EXACTLY equal to background --
# see the rule's own extensive docstring above): the underlying characters
# are still present in the compositor's cell buffer either way, so a tool
# that only reads characters (not their color) reports the SAME "Watc"/"M"
# fragment for both cases. Direct ANSI decoding of the real running app
# (`capture-pane -p -e`) at 80 and 120 cols, both for the exact fragments the
# re-critique quoted and for a left-edge scroll straddle, showed foreground
# RGB == background RGB for every one of them -- i.e. pixel-invisible in any
# color-aware terminal. The re-critique's OWN corroborating check (a click on
# the "blank" cell did not navigate) is further evidence FOR ghosting: a
# genuinely un-ghosted button is never `disabled`, so that click would have
# navigated. `_straddles_viewport`/`_ghost_clipped_buttons` are unaffected by
# the `NavOverflowMenu` rework; nothing here needed a behavior fix.
#
# The real, fixable gap (AC#4): every geometry/readable-text test above runs
# under a bare `App()` with ONLY `MainNavigationBar.BUNDLED_CSS` loaded --
# never `css/components/_navigation.tcss`'s separately-maintained override,
# which is the copy that actually wins in the shipped app (`App.CSS_PATH`
# always outranks widget `DEFAULT_CSS`, `!important` or not -- see the ghost
# rule's own docstring, and `test_ghost_rule_is_width_neutral_under_the_
# bundled_stylesheet` above, which already made exactly this point for
# geometry but never for the color/legibility invariant). A regression that
# broke `_navigation.tcss`'s color override (e.g. reverting it, or a bundle
# rebuild dropping it) would leave every test above GREEN while the real app
# regressed to genuinely-legible ghosted text -- the "test passes against
# broken code" failure mode AC#4 warns about, just one CSS tier removed from
# where task-3200 originally looked. The tests below close that gap.


def _plain_nav_text(app: App) -> str:
    """Every character the compositor painted, WITHOUT filtering by color --
    i.e. exactly what a colorless capture (`tmux capture-pane -p`, no `-e`)
    would show. Deliberately the naive counterpart to `_readable_nav_text`
    above: contrasting the two on the same render is what proves the
    re-critique's "mid-word cut" finding was a capture-tool artifact, not a
    rendering defect -- ghosted text is still IN the buffer (color-matched,
    not removed), so a colorless reader cannot tell it apart from a real
    legible label.
    """
    strips = app.screen._compositor.render_strips()
    return "\n".join(
        "".join(segment.text for segment in strip) for strip in strips
    )


@pytest.mark.asyncio
async def test_naive_colorless_capture_false_positives_on_ghosted_labels():
    """RED-documentation for task-4020 AC#2: reproduces the re-critique's
    exact observed effect (a colorless capture "sees" a mid-word-clipped
    label) under the REAL bundled stylesheet, at the REAL width, with the
    REAL destination the re-critique quoted -- then contrasts it with the
    color-aware check that proves the label is not actually legible.

    At 100 cols, active="home", one non-active destination straddles the
    "More ▾" edge. The concrete edge destination changes as destinations are
    added, so this test pins the user-visible ghosting behavior rather than
    its incidental position in the strip.
    """

    class TestApp(ConsolidatedCSSApp):
        CSS_PATH = _BUNDLED_CSS_PATH

        def compose(self):
            yield MainNavigationBar(active="home")

    app = TestApp()

    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause(0.6)

        ghosted = [
            button
            for button in app.query(".nav-button")
            if button.has_class("nav-button-clip-ghost")
        ]
        assert ghosted, (
            "test premise: a destination straddles and is ghosted at 100 cols "
            "under the bundled stylesheet"
        )
        destination = ghosted[0]
        fragment = str(destination.label).strip().split(" ", 1)[-1][:4]

        # A colorless capture sees the clipped fragment even though the
        # button is ghosted (fg == bg) and disabled.
        assert fragment in _plain_nav_text(app), (
            "expected the colorless capture to include the ghosted destination "
            "fragment"
        )

        # The color-aware check (already established by task-3200, reused
        # here against the SAME render) shows the fragment is not actually
        # readable: this is the correction to the re-critique's conclusion.
        readable_text = _readable_nav_text(app)
        assert fragment not in readable_text, (
            f"{destination.id}'s ghosted fragment is color-distinguishable "
            "from its background -- a real regression, not a measurement artifact"
        )
        assert destination.disabled, (
            "a ghosted button must stay disabled -- consistent with the "
            "re-critique's own click-test finding that the blank cell did "
            "not navigate"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "width,active",
    [
        (80, "home"),  # early destination: right-edge straddle, no scroll
        (80, "settings"),  # late destination: forces a scroll (left-edge too)
        (100, "home"),
        (100, "settings"),
        (120, "home"),
        (120, "mcp"),  # the re-critique's other quoted fragment ("⌃9 M")
        (120, "settings"),
    ],
)
async def test_nav_strip_never_renders_a_partial_destination_label_under_bundled_css(
    width, active
):
    """task-4020 AC#1/#4: the SAME invariant as
    `test_nav_strip_never_renders_a_partial_destination_label` above, now
    proven under the REAL bundled stylesheet (`App.CSS_PATH`) instead of
    only `MainNavigationBar.BUNDLED_CSS` -- see the section docstring above
    for why that tier distinction is exactly what the re-critique's
    "no ghosting observed" finding turned out to hinge on. Widths and
    active destinations mirror the re-critique's own 80/100/120 sweep with
    an early (home) and late (settings, and 120's mcp -- the re-critique's
    own second example) active tab, so this also stands as this task's
    live-verification-equivalent, deterministic regression coverage.
    """

    class TestApp(ConsolidatedCSSApp):
        CSS_PATH = _BUNDLED_CSS_PATH

        def compose(self):
            yield MainNavigationBar(active=active)

    app = TestApp()

    async with app.run_test(size=(width, 24)) as pilot:
        strip = app.query_one("#nav-destination-strip", Horizontal)
        active_button = app.query_one(f"#nav-{active}", Button)
        for _ in range(500):
            if (
                _straddling_buttons(app, strip) == []
                and active_button.region.width > 0
                and active_button.region.x >= strip.region.x
                and active_button.region.right <= strip.region.right
            ):
                break
            await pilot.pause(0.02)

        # Geometry: nothing straddles the viewport edge with real content.
        assert _straddling_buttons(app, strip) == []

        # The active destination is always fully, genuinely visible.
        assert active_button.display
        assert not active_button.has_class("nav-button-clip-ghost")
        assert active_button.region.width > 0
        assert active_button.region.x >= strip.region.x
        assert active_button.region.right <= strip.region.right

        # Color-aware rendered-text pin, under the tier that actually wins
        # live: every ghosted button's label contributes NO readable
        # fragment to the painted screen.
        painted = _readable_nav_text(app)
        ghosted = [
            button
            for button in app.query(".nav-button")
            if button.has_class("nav-button-clip-ghost")
        ]
        for button in ghosted:
            label_text = str(button.label).strip()
            word = label_text.split(" ", 1)[-1]
            fragment = word[:4]
            assert fragment not in painted, (
                f"{button.id} is ghosted but '{fragment}' is still readable "
                "under the bundled stylesheet"
            )
            assert button.disabled, f"{button.id} is ghosted but not disabled"
            assert not button.focusable, f"{button.id} is ghosted but still focusable"

        # AC#3: rule out the OTHER shape the re-critique's fragments could
        # have meant -- an ellipsis-truncation artifact (`…`) distinct from
        # ghosting (e.g. Rich's own text-overflow indicator, if some future
        # change ever let a button's box shrink below its label's natural
        # width). Live verification never reproduced a literal `…` glyph;
        # this pins that absence as a regression guard rather than leaving
        # it as an unverified assumption.
        assert "…" not in painted, (
            "an ellipsis-truncation artifact appeared in the nav strip -- "
            "this is a DIFFERENT mechanism than clip-ghosting (Rich's own "
            "text-overflow indicator on an under-sized button box) and "
            "would need its own fix, not a ghosting one"
        )


@pytest.mark.asyncio
async def test_tab_cycling_never_focuses_a_ghosted_nav_button():
    """Review finding: a ghosted (invisible) button was still Tab-reachable
    with no visible focus ring -- a keyboard user could land on it blind
    and Enter-navigate to a destination they never saw highlighted.
    `disabled` (paired with the ghost class in `_ghost_clipped_buttons`)
    removes it from the focus chain entirely (`Widget.focusable` excludes
    disabled widgets) -- cycle Tab all the way around the bar and confirm
    focus never lands on a ghosted button, only on genuinely visible ones.

    The per-press check reads each button's ghost/disabled state AT THE
    MOMENT it receives focus, not a snapshot taken before the loop starts:
    `on_descendant_focus` re-scrolls the strip on every Tab landing (to
    keep the newly-focused button fully visible), which legitimately
    changes WHICH buttons straddle an edge as the strip's scroll position
    moves -- a button ghosted at t=0 can be genuinely un-ghosted (fully
    visible, re-enabled) by the time Tab reaches it several presses later.
    An earlier version of this test compared against a single pre-loop
    snapshot and produced a false failure for exactly that reason (caught
    live: Tab correctly reached `nav-schedules` only after it had been
    scrolled fully into view and un-ghosted, but the stale snapshot still
    listed its id as ghosted).
    """

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="settings")

    app = TestApp()

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.6)

        assert any(
            button.has_class("nav-button-clip-ghost")
            for button in app.query(".nav-button")
        ), "test premise: expected a straddling destination at 80 cols"

        # More than enough presses to cycle all the way around the bar
        # (13 destinations + the overflow hint) at least twice.
        visited_while_ghosted = []
        for _ in range(30):
            await pilot.press("tab")
            focused = app.focused
            if focused is not None and focused.has_class("nav-button-clip-ghost"):
                visited_while_ghosted.append(focused.id)
            if focused is not None and focused.has_class("nav-button"):
                assert not focused.disabled, (
                    f"{focused.id} received focus while disabled"
                )

        assert visited_while_ghosted == []


@pytest.mark.asyncio
async def test_press_on_a_ghosted_nav_button_is_a_no_op():
    """Review finding: nothing rejected a ghosted button as a `.press()`
    target -- a mouse click into dead-looking space (or a programmatic
    `.press()`) would silently navigate. `Button.press()` already no-ops
    when `self.disabled` is set; `_ghost_clipped_buttons` sets it whenever
    the ghost class is applied, so calling `.press()` directly on a
    ghosted button must produce neither a `Button.Pressed` message nor a
    `NavigateToScreen` navigation, and the active destination must stay
    unchanged.
    """
    events = []

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="settings")

        def on_navigate_to_screen(self, message):
            events.append(message.screen_name)

    app = TestApp()

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.6)

        nav = app.query_one(MainNavigationBar)
        ghosted = [
            button
            for button in app.query(".nav-button")
            if button.has_class("nav-button-clip-ghost")
        ]
        assert ghosted, "test premise: expected a straddling destination at 80 cols"
        target = ghosted[0]
        assert target.disabled

        target.press()
        await pilot.pause(0.1)

        assert events == []
        assert nav.active_destination_id == "settings"
        assert target.has_class("nav-button-clip-ghost")


@pytest.mark.asyncio
async def test_click_on_ghosted_nav_button_via_border_route_is_a_no_op():
    """Review finding (round 5): `on_click` routes a click that resolves
    to the bar/strip itself (not directly to a specific widget) to ANY
    `NavigationButton` whose `region.contains_point` matches the click,
    then calls `_activate_navigation_button` directly -- bypassing
    `Button.press()`'s built-in `disabled` no-op entirely (the guard the
    sibling `test_press_on_a_ghosted_nav_button_is_a_no_op` above
    exercises). Ghosting (task-3200/3225) is purely cosmetic and
    geometry-neutral -- a ghosted button keeps its real screen region and
    stays part of the widget tree -- so its region still intersects real
    click coordinates; only its paint is blanked.

    This drives the natural (not mocked) resolution path: at row y=2 --
    the bar's own `border-bottom` row, height=3 total (rows 0/1 are
    content, row 2 is the border) -- `get_widget_at` resolves to the
    `MainNavigationBar` itself rather than to any child button, because
    those border pixels are drawn as part of the bar's own box, not
    attributed to a child's compositor region (confirmed directly:
    `get_widget_at(x, 2)` returns the bar for every x across the ghosted
    button's width, while rows 0/1 at the same x correctly return the
    button). That is exactly the "clicked the border, not a widget" case
    `on_click`'s own guard clause exists to route into the loop below --
    i.e. this is the real bug path, not a synthetic bypass of it.
    `active="artifacts"` at 80 cols reliably straddles/ghosts
    `nav-watchlists_collections` (`Region(x=64, y=0, width=15,
    height=3)`), the same defect class as the review's own probe
    (`nav-watchlists_collections`, `x=62-71, y=2`, a different active
    destination).
    """
    events_seen = []

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="artifacts")

        def on_navigate_to_screen(self, message):
            events_seen.append(message.screen_name)

    app = TestApp()

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.6)

        nav = app.query_one(MainNavigationBar)
        ghosted = [
            button
            for button in app.query(".nav-button")
            if button.has_class("nav-button-clip-ghost")
        ]
        assert ghosted, "test premise: expected a straddling destination at 80 cols"
        target = next(
            (b for b in ghosted if b.id == "nav-watchlists_collections"), ghosted[0]
        )
        assert target.disabled
        region = target.region

        # A cell inside the ghosted button's own region, on the bar's
        # border-bottom row -- naturally resolves to the bar itself (see
        # docstring above), not the button, which is the exact case
        # `on_click`'s border router exists to handle.
        click_x = region.x
        click_y = 2
        assert app.get_widget_at(click_x, click_y)[0] is nav, (
            "test premise: this coordinate must resolve to the bar itself"
        )

        click_event = events.Click(
            nav,
            click_x,
            click_y,
            0,
            0,
            1,
            False,
            False,
            False,
            screen_x=click_x,
            screen_y=click_y,
        )
        nav.on_click(click_event)
        await pilot.pause(0.1)

        assert events_seen == []
        assert nav.active_destination_id == "artifacts"
        assert target.has_class("nav-button-clip-ghost")
        assert target.disabled


@pytest.mark.asyncio
async def test_periodic_interval_does_not_drag_the_focused_button_out_of_view():
    """Review round 2 finding: the periodic 0.5s interval
    (`_update_overflow_hints`, `set_interval` in `on_mount`) called
    `_scroll_active_destination_into_view` every tick unconditionally
    targeting the ACTIVE destination, indifferent to keyboard focus. When
    Tab had focused a DIFFERENT, far-away button (`on_descendant_focus`
    scrolls to reveal it), the interval's very next tick dragged the strip
    back toward the active destination -- leaving the FOCUSED button
    straddling the edge: visibly mid-word-cut, still `app.focused`,
    un-ghosted, and `disabled=False` (Enter-navigable). A static exemption
    keyed on "whichever button held focus when `_ghost_clipped_buttons`
    last ran" could not catch this, because nothing had re-scrolled to
    reveal that focused button on the interval's own tick -- the exemption
    only meant "don't hide it," not "keep it visible".

    This is the reviewer's exact deterministic reproduction: `active=
    "schedules"` (far from the end of the bar), Tab to `nav-settings`
    (forces a scroll to reveal it, since Settings does not fit in the
    initial 80-col viewport alongside Schedules), then `pilot.pause(0.9)`
    -- long enough for at least one 0.5s interval tick to fire after the
    Tab-driven scroll settled. Before the fix this reliably (3/3 in the
    reviewer's own repro) left `nav-settings` straddling
    (`Region(x=66, width=15)` against `strip.region.right == 70`) while
    still focused, un-ghosted, and enabled.
    """

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="schedules")

    app = TestApp()

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.3)

        for _ in range(20):
            await pilot.press("tab")
            if getattr(app.focused, "id", None) == "nav-settings":
                break
        assert app.focused is not None and app.focused.id == "nav-settings", (
            "test premise: expected to Tab-focus nav-settings"
        )

        # One interval tick (0.5s) plus margin -- the exact window the
        # review finding traced the drag-back to.
        await pilot.pause(0.9)

        strip = app.query_one("#nav-destination-strip", Horizontal)
        settings = app.query_one("#nav-settings", Button)
        assert app.focused is settings, (
            "nav-settings should still hold focus after the interval tick"
        )
        assert not _straddles_viewport(settings.region, strip.region), (
            f"nav-settings straddles after an interval tick while focused: "
            f"{settings.region} vs strip {strip.region}"
        )
        assert not settings.has_class("nav-button-clip-ghost"), (
            "the focused button must not be ghosted while it holds focus"
        )
        assert not settings.disabled, (
            "the focused button must not be disabled while it holds focus"
        )


class _IntervalSuppressibleNavBar(MainNavigationBar):
    """`MainNavigationBar` whose 0.5s settle interval can be switched off
    mid-test (review round 4, task-3225).

    `set_interval(0.5, self._update_overflow_hints)` captures a BOUND
    method at mount, so patching the instance attribute afterwards does
    nothing -- the suppression has to be a branch inside an override.

    Why suppress it at all: the interval is the deliberate *backstop* for
    every settle trigger, and it is focus-aware since round 2, so it heals
    an `on_resize` that stranded the focused button within <= 0.5s. That
    makes it impossible to tell a WORKING `on_resize` from a BROKEN one by
    looking at the strip a few hundred ms later -- which is exactly how
    the shipped version of this test came to be vacuous (round 3 takeover
    finding: reverting `on_resize`'s `_recenter_strip` wiring did not turn
    it red). Suppressing the backstop for the duration of the resize
    isolates the property actually under test: `on_resize`'s OWN settle
    pass must leave the deliberately-focused button fully visible, without
    waiting on the interval to clean up after it.
    """

    suppress_interval = False

    def _update_overflow_hints(self) -> None:
        if self.suppress_interval:
            return
        super()._update_overflow_hints()


@pytest.mark.asyncio
async def test_resize_does_not_strand_the_focused_button():
    """Review round 3 finding: `on_resize` used to route straight through
    the plain, active-only `_scroll_active_destination_into_view`,
    indifferent to keyboard focus -- the exact defect class round 2 fixed
    for the periodic interval, independently reproduced through this
    OTHER trigger.

    Round 4 rewrite (task-3225). The originally-shipped version of this
    test (`active="schedules"`, Tab to `nav-settings`, GROW 80 -> 90) was
    vacuous twice over and is replaced rather than tweaked:

    1. Growing the terminal never produced a genuine straddle in this
       harness in the first place -- `nav-settings` sat flush at the
       boundary either way, so the assertion could not fail.
    2. Even in a scenario that DOES strand, two other mechanisms heal it
       before any wall-clock assertion can see it: the periodic interval
       (focus-aware since round 2) and `_ghost_clipped_buttons`'s
       best-effort `scroll_to_widget(focused)` nudge, which fires whenever
       the focused button STRADDLES. Directly traced: with `on_resize`
       reverted, `scroll_x` went 86 -> 75 (dragged toward active) -> 96
       (nudged back) inside 40ms.

    So this version (a) makes the resize strand the focused button FULLY
    off-screen rather than straddling -- `active="home"` at the far left,
    focus on `nav-settings` at the far right -- which is the case the
    ghost-pass nudge cannot rescue (it only nudges a straddler), and
    (b) suppresses the interval backstop for the duration of the resize
    (see `_IntervalSuppressibleNavBar`). What is left is `on_resize`'s own
    pass, and the assertion is the honest user-facing invariant: the
    button you Tab-focused is still FULLY visible after the resize, not
    merely "not straddling" (a button dragged entirely off-screen is not
    straddling either, and is strictly worse -- invisible, yet still
    focused and Enter-navigable).

    Mutation-tested both ways: reverting `on_resize` to
    `_scroll_active_destination_into_view` fails this at every checkpoint;
    restoring `_recenter_strip` passes it at every checkpoint.
    """

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield _IntervalSuppressibleNavBar(active="home")

    app = TestApp()

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.3)

        for _ in range(20):
            await pilot.press("tab")
            if getattr(app.focused, "id", None) == "nav-settings":
                break
        assert app.focused is not None and app.focused.id == "nav-settings", (
            "test premise: expected to Tab-focus nav-settings"
        )

        nav = app.query_one(_IntervalSuppressibleNavBar)
        strip = app.query_one("#nav-destination-strip", Horizontal)
        settings = app.query_one("#nav-settings", Button)
        assert settings.region.x >= strip.region.x, (
            "test premise: the focused button starts fully visible"
        )

        nav.suppress_interval = True
        await pilot.resize_terminal(90, 24)

        # Several checkpoints, not one: the round-3 takeover's other
        # finding was that the (then-shipped) fix corrected the geometry
        # transiently and drifted back ~0.3s later, so a single
        # post-resize assertion could not have caught it. The drift-back's
        # root cause (a ghost pass that reflowed the strip) is fixed in
        # `MainNavigationBar.BUNDLED_CSS`; this sweep is what keeps it
        # fixed.
        for step in range(8):
            await pilot.pause(0.1)
            assert app.focused is settings, (
                f"nav-settings should still hold focus after the resize "
                f"(checkpoint {step})"
            )
            assert (
                settings.region.x >= strip.region.x
                and settings.region.right <= strip.region.right
            ), (
                f"the focused button is not fully visible after a resize "
                f"(checkpoint {step}): {settings.region} vs strip "
                f"{strip.region}"
            )
            assert not _straddles_viewport(settings.region, strip.region)
            assert not settings.has_class("nav-button-clip-ghost")
            assert not settings.disabled


@pytest.mark.asyncio
async def test_ghosting_a_button_never_reflows_the_strip():
    """Review round 4 (task-3225): ghosting must be geometry-neutral.

    The whole reason task-3200 ghosts a clipped button with CSS instead of
    `display: none` is that hiding it changes the strip's layout, which
    cascades into new straddlers and breaks the "More ›" pager's reach.
    The shipped ghost rule quietly broke that invariant anyway: it
    declared `border: solid $background !important`, replacing Textual's
    `Button.-style-default` border (`border-top`/`border-bottom` only --
    ZERO horizontal cells) with a four-edge border, so a ghosted button
    measured 2 cells WIDER than the same button un-ghosted and shoved
    every button after it 2 cells right. That reflow is what produced the
    "~0.3s drift-back" filed as task-3225: a settle pass would scroll the
    focused button into view, then its own trailing ghost pass reflowed
    the strip and put it back into a straddling position, with nothing
    scheduled to re-check.

    Deterministic geometry assertion, no timing: ghost one button by hand
    and require every other button's region -- and its own -- to be
    unchanged. Goes red if any box-model property (border, padding,
    width, visibility) is ever reintroduced into the ghost rule.
    """

    class _NoAutoGhostBar(MainNavigationBar):
        """Ghosting is applied by hand here, so the widget's own settle
        passes must not race the assertion by re-deciding it."""

        def _ghost_clipped_buttons(self) -> None:
            return

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield _NoAutoGhostBar(active="home")

    app = TestApp()

    async with app.run_test(size=(200, 24)) as pilot:
        await pilot.pause(0.4)

        strip = app.query_one("#nav-destination-strip", Horizontal)
        buttons = list(strip.query(Button))
        assert len(buttons) > 3, "test premise: several destinations present"
        before = {button.id: button.region for button in buttons}
        virtual_before = strip.virtual_size

        victim = app.query_one("#nav-workflows", Button)
        victim.add_class("nav-button-clip-ghost")
        victim.disabled = True
        await pilot.pause(0.3)

        after = {button.id: button.region for button in strip.query(Button)}
        assert after == before, (
            "ghosting a nav button reflowed the strip -- the ghost rule is "
            "not geometry-neutral. Changed: "
            + repr(
                {
                    button_id: (before[button_id], after[button_id])
                    for button_id in before
                    if before[button_id] != after.get(button_id)
                }
            )
        )
        assert strip.virtual_size == virtual_before, (
            "ghosting a nav button changed the strip's virtual size "
            f"({virtual_before} -> {strip.virtual_size}), which moves "
            "max_scroll_x and the 'More ›' pager's reach"
        )


class _NoAutoGhostBarWithBundledCSS(MainNavigationBar):
    """Ghosting is applied by hand in the test below, so the widget's own
    settle passes must not race the assertion by re-deciding it -- same
    reasoning as `_NoAutoGhostBar` in
    `test_ghosting_a_button_never_reflows_the_strip`, above (its own
    docstring names the race this avoids: without this override, `#nav-
    workflows` is not actually straddling the viewport at this test's
    width, so the widget's own `_ghost_clipped_buttons` pass silently
    un-ghosts it again before the assertion runs)."""

    def _ghost_clipped_buttons(self) -> None:
        return


class _NavAppWithBundledCSS(ConsolidatedCSSApp):
    """Loads the REAL generated bundle (`App.CSS_PATH`), not just
    `MainNavigationBar.BUNDLED_CSS`. Mirrors `InspectorAppWithBundledCSS`
    in `Tests/UI/test_mcp_inspector.py`."""

    CSS_PATH = _BUNDLED_CSS_PATH

    def compose(self):
        yield _NoAutoGhostBarWithBundledCSS(active="home")


@pytest.mark.asyncio
async def test_ghost_rule_is_width_neutral_under_the_bundled_stylesheet():
    """task-3801: pin the bundle tier's ghost-rule override, not just DEFAULT_CSS.

    Specifically, this pins `.nav-button.nav-button-clip-ghost:disabled` in
    the generated `App.CSS_PATH` bundle, not just `MainNavigationBar.
    DEFAULT_CSS`. task-3225 review round 4 found and fixed a real regression: the
    DEFAULT_CSS-tier ghost rule once declared a four-edge `border: solid
    $background`, which is NOT geometry-neutral (Textual's own
    `Button.-style-default` has zero horizontal border cells) -- a ghosted
    button measured 2 cells wider than the same button un-ghosted and
    reflowed the strip (`test_ghosting_a_button_never_reflows_the_strip`,
    above, pins that fix). But `App.CSS_PATH` stylesheets outrank widget
    `DEFAULT_CSS` regardless of specificity, so in the REAL running app it
    is `css/components/_navigation.tcss`'s separately maintained copy of
    this rule that actually wins -- and nothing exercised THAT copy the
    same way: `test_ghosting_a_button_never_reflows_the_strip` runs under
    a bare `App()` with no `CSS_PATH`, so it cannot see a box-model
    property reintroduced into the bundle tier. This test closes that gap
    by loading the real bundle and comparing one button's region before
    and after ghosting, exactly as the DEFAULT_CSS-tier sibling does.

    Before any of that, it also guards its own premise: that the bundle
    file exists and still contains the rule under test. Without this, a
    renamed/moved bundle or a renamed selector would make the geometry
    assertions below vacuously pass (nothing to apply, so nothing to
    reflow) instead of failing loud -- silently non-proving rather than
    red. It also statically asserts the rule's declaration block carries
    no box-model property, which is the direct, source-level statement of
    the same task-3225 round-4 incident the runtime geometry check below
    proves dynamically.
    """
    rule_body = _bundled_css_rule_body(_BUNDLED_CSS_PATH, _GHOST_RULE_SELECTOR)
    box_model_hits = [
        prop
        for prop in _declared_properties(rule_body)
        if prop in _BOX_MODEL_PROPERTIES
        or any(prop.startswith(f"{box}-") for box in _BOX_MODEL_PROPERTIES)
    ]
    assert not box_model_hits, (
        f"{_GHOST_RULE_SELECTOR!r} in {_BUNDLED_CSS_PATH!r} declares box-model "
        f"property(ies) {box_model_hits} -- this is exactly the task-3225 "
        "round-4 regression (a border/padding/margin/width/height delta "
        "between ghosted and un-ghosted buttons reflows the nav strip)."
    )

    app = _NavAppWithBundledCSS()

    async with app.run_test(size=(200, 24)) as pilot:
        await pilot.pause(0.4)

        victim = app.query_one("#nav-workflows", Button)
        before = victim.region
        assert before.width > 0, "test premise: the button actually renders"

        victim.add_class("nav-button-clip-ghost")
        victim.disabled = True
        await pilot.pause(0.3)

        after = victim.region
        assert after == before, (
            "ghosting #nav-workflows under the REAL app stylesheet changed "
            f"its region ({before} -> {after}) -- the bundle-tier "
            ".nav-button.nav-button-clip-ghost:disabled rule in "
            "css/components/_navigation.tcss reintroduced a non-zero "
            "border/padding/margin delta between the ghosted and "
            "un-ghosted box model."
        )


@pytest.mark.asyncio
async def test_restore_active_does_not_strand_the_focused_button():
    """Review round 3 finding: `restore_active` used to route straight
    through the plain, active-only `_scroll_active_destination_into_view`
    too -- the same defect class, a third independent trigger.
    Live-reproduced: Tab to `nav-settings` (`active="schedules"`), an
    optimistic click-activate to `console` (mirroring `_activate_
    navigation_button`, before its navigation actually completes), then
    `restore_active("schedules")` left `nav-settings` genuinely
    straddling, un-ghosted, enabled, and still focused.
    """

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="schedules")

    app = TestApp()

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.3)

        for _ in range(20):
            await pilot.press("tab")
            if getattr(app.focused, "id", None) == "nav-settings":
                break
        assert app.focused is not None and app.focused.id == "nav-settings", (
            "test premise: expected to Tab-focus nav-settings"
        )

        nav = app.query_one(MainNavigationBar)
        console_button = app.query_one("#nav-console", Button)
        nav._activate_navigation_button(console_button)
        await pilot.pause(0.2)
        nav.restore_active("schedules")
        await pilot.pause(0.3)

        strip = app.query_one("#nav-destination-strip", Horizontal)
        settings = app.query_one("#nav-settings", Button)
        assert app.focused is settings, (
            "nav-settings should still hold focus after restore_active"
        )
        assert not _straddles_viewport(settings.region, strip.region), (
            f"nav-settings straddles after restore_active while focused: "
            f"{settings.region} vs strip {strip.region}"
        )
        assert not settings.has_class("nav-button-clip-ghost")
        assert not settings.disabled


# (rebase note) `test_pager_releases_focus_instead_of_stranding_it`
# (review round 3's own regression test for a focused nav button left
# straddling by a "More ›" pager press) was dropped here, not adapted:
# dev's parallel NV-01/TASK-2154.21 rework (merged independently of the
# whole task-3200 series) replaced in-strip paging with
# `handle_overflow_hint` opening a real `NavOverflowMenu` screen listing
# every destination -- pressing "#nav-overflow-hint" no longer scrolls
# the strip at all, so there is no more scroll-viewport position for a
# focused button to be left straddling against. The defect class this
# test pinned cannot recur under the current design; see
# `_refresh_overflow_hint_visibility`'s rebase note in
# `main_navigation.py` for the corresponding production-code note.


@pytest.mark.asyncio
async def test_recenter_strip_and_focused_strip_button_survive_a_detached_bar():
    """Review round 3 crash-guard: `self.screen` raises `NoScreen` once a
    widget is no longer attached to an active screen (`_focused_strip_
    button`'s round-2 fix). Confirm every focus-aware recenter entry point
    tolerates that -- constructing a `MainNavigationBar`, mounting it,
    then removing it, and calling the methods a stray deferred
    `call_after_refresh` callback could still reach afterward must never
    raise.
    """

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="home")

    app = TestApp()

    async with app.run_test(size=(80, 24)) as pilot:
        nav = app.query_one(MainNavigationBar)
        await nav.remove()
        await pilot.pause(0.1)

        assert nav._focused_strip_button() is None
        nav._recenter_strip()
        nav._ghost_clipped_buttons()
