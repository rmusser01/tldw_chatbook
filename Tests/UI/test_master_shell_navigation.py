"""Mounted tests for master-shell navigation."""

from types import SimpleNamespace

import pytest
from textual.app import App
from textual.containers import Horizontal
from textual.widgets import Button

from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar


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
    css = MainNavigationBar.DEFAULT_CSS

    assert "margin: 0;" in css
    assert "padding: 0;" in css
    assert ".nav-overflow-hint" in css


@pytest.mark.asyncio
async def test_master_shell_navigation_order_and_labels():
    class TestApp(App):
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
    class TestApp(App):
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
    class TestApp(App):
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
    class TestApp(App):
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
    class TestApp(App):
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
    assert ".nav-separator" not in MainNavigationBar.DEFAULT_CSS
    assert "background: $primary-darken-1;" in MainNavigationBar.DEFAULT_CSS


@pytest.mark.asyncio
async def test_home_and_console_remain_first_primary_destinations():
    class TestApp(App):
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

    class TestApp(App):
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

    class TestApp(App):
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

    class TestApp(App):
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

    class TestApp(App):
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

    for destination in SHELL_DESTINATION_ORDER:
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
        "research": ("library", "research"),
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
    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="search", active_route="search")

    app = TestApp()

    async with app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)
        assert app.query_one("#nav-library", Button).has_class("is-active")
        assert not app.query_one("#nav-lab", Button).has_class("is-active")

    class LabApp(App):
        def compose(self):
            yield MainNavigationBar(active="llm", active_route="llm")

    lab_app = LabApp()

    async with lab_app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)
        assert lab_app.query_one("#nav-lab", Button).has_class("is-active")
        assert not lab_app.query_one("#nav-library", Button).has_class("is-active")


def test_shell_destination_hotkeys_follow_destination_order():
    """Ctrl+1..9 then Ctrl+0 map onto SHELL_DESTINATION_ORDER, in order."""
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    hotkey_bindings = [
        binding
        for binding in TldwCli.BINDINGS
        if binding.action.startswith("shell_destination(")
    ]

    expected_keys = list(TldwCli.SHELL_DESTINATION_HOTKEYS) + list(
        TldwCli.SHELL_DESTINATION_FKEYS
    )
    assert expected_keys == [
        "ctrl+1",
        "ctrl+2",
        "ctrl+3",
        "ctrl+4",
        "ctrl+5",
        "ctrl+6",
        "ctrl+7",
        "ctrl+8",
        "ctrl+9",
        "ctrl+0",
        "f7",
        "f8",
        "f9",
    ]
    # One binding per hotkey, zipped against the destination order: ctrl+digits
    # cover the first ten, F7/F8/F9 the remaining three — every destination
    # has a keyboard route and none is skipped.
    assert len(hotkey_bindings) == min(len(expected_keys), len(SHELL_DESTINATION_ORDER))
    assert len(hotkey_bindings) == len(SHELL_DESTINATION_ORDER)
    for index, binding in enumerate(hotkey_bindings):
        destination = SHELL_DESTINATION_ORDER[index]
        assert binding.key == expected_keys[index]
        assert binding.action == f"shell_destination({index})"
        assert destination.accessible_label in binding.description
        # Index numbers belong to the key layer, not the nav labels.
        assert str(index + 1) not in destination.label


def test_action_shell_destination_posts_primary_route():
    """The single hotkey action navigates to each destination's primary route."""
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    posted = []
    fake_app = SimpleNamespace(post_message=posted.append)

    for index, destination in enumerate(SHELL_DESTINATION_ORDER):
        TldwCli.action_shell_destination(fake_app, index)
        message = posted[-1]
        assert isinstance(message, NavigateToScreen)
        assert message.screen_name == destination.primary_route, (
            destination.destination_id
        )

    # Textual binding actions pass the argument as a string.
    posted.clear()
    TldwCli.action_shell_destination(fake_app, "0")
    assert isinstance(posted[-1], NavigateToScreen)
    assert posted[-1].screen_name == SHELL_DESTINATION_ORDER[0].primary_route

    # Out-of-range indices are a safe no-op.
    posted.clear()
    TldwCli.action_shell_destination(fake_app, len(SHELL_DESTINATION_ORDER))
    TldwCli.action_shell_destination(fake_app, -1)
    TldwCli.action_shell_destination(fake_app, "not-a-number")
    assert posted == []


@pytest.mark.asyncio
async def test_nav_overflow_menu_reaches_hidden_destinations_at_100_cols():
    """F-001: at 100 columns the strip clips mid-button (the review's
    "8 Workflows" -> "8" artifact) and later destinations have no click
    path in the strip itself. The "More ▾" affordance opens the overflow
    menu, where every clipped destination is listed and pressable --
    TASK-2154.21 replaced the old paging hint with this menu so a press
    navigates directly instead of scrolling page by page."""

    class TestApp(App):
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

    class TestApp(App):
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
    class TestApp(App):
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
        if width == 100 and active == "home":
            # Pin the specific finding this task fixed (task-2858 P2 /
            # LIB-18): at a narrow width with the strip scrolled to its
            # default position, some destination WILL straddle the "More
            # ›" hint's edge -- if nothing were ever ghosted here, the
            # geometry/rendered-text assertions above would be vacuous.
            assert ghosted, "test premise: expected a straddling destination at 100 cols"

