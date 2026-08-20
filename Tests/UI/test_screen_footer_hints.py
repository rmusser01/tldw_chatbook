"""Task-264: every BaseAppScreen must carry its OWN AppFooterStatus instance.

`AppFooterStatus` used to be resolved through the App's default screen, which
is occluded whenever a `BaseAppScreen` is active. That made destination
registrations update invisible or missing chrome.

The fix: `BaseAppScreen.compose()` now yields its own `AppFooterStatus`, and
callers resolve it through the screen (`self.query_one(...)`) instead of the
app. These tests pin that contract directly against the real screens/
registration methods, not a hand-rolled fake.
"""

import ast
import logging
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.app import ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Static

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.UI.Screens.mcp_screen import MCPScreen
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus


def _test_cli_setting(section: str, key: str, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


async def _wait_for_screen(app, pilot, screen_type, tab: str):
    for _ in range(300):
        if (
            app.current_tab == tab
            and isinstance(app.screen, screen_type)
            and app.screen.query(AppFooterStatus)
        ):
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError(
        f"full TldwCli did not finish routing to {screen_type.__name__}."
    )


async def _close_production_app(app) -> None:
    try:
        if app._rich_log_handler:
            await app._rich_log_handler.stop_processor()
            logging.getLogger().removeHandler(app._rich_log_handler)
            app._rich_log_handler.close()
        await app.on_shutdown_request()
        await app.on_unmount()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_production_routes_own_and_preserve_contextual_footer_hints():
    """Exercise footer ownership through the full production application."""
    app = _build_test_app()
    app._initial_tab_value = "chat"

    try:
        with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
            async with app.run_test(size=(160, 48)) as pilot:
                screen = await _wait_for_screen(app, pilot, ChatScreen, "chat")
                screen_footer = screen.query_one(AppFooterStatus)
                assert "F6" in screen_footer.shortcut_text
                assert "Ctrl+K" in screen_footer.shortcut_text

                await app.handle_screen_navigation(NavigateToScreen("mcp"))
                screen = await _wait_for_screen(app, pilot, MCPScreen, "mcp")
                screen_footer = screen.query_one(AppFooterStatus)
                assert "mode" in screen_footer.shortcut_text
                assert "a add server" in screen_footer.shortcut_text
                # F-003: the Tokens chip is meaningful only in chat contexts --
                # on MCP it stays hidden, never a "Tokens: --" placeholder.
                token_chip = screen.query_one("#footer-token-count", Static)
                assert token_chip.display is False
                assert "Tokens: --" not in str(token_chip.renderable)
                # Past the 0.5s one-shot updater tick (which writes "" on
                # non-chat tabs), still hidden.
                await pilot.pause(0.7)
                assert token_chip.display is False
                assert "Tokens: --" not in str(token_chip.renderable)

                await app.handle_screen_navigation(NavigateToScreen("library"))
                screen = await _wait_for_screen(
                    app,
                    pilot,
                    LibraryScreen,
                    "library",
                )
                # task-2237 (R2): the landing footer advertises the full
                # keyboard story that works there (F-012 started with just
                # `/` focus search).
                # Merged footer (AppFooterStatus): task-2860 -- the screen's
                # own "F6 next pane" entry now renders verbatim (it used to
                # be silently dropped as a "reserved global key", leaving
                # only the global cluster's generic copy, then spelled
                # "F6 panes"); the global cluster in turn excludes its own
                # F6 entry ("F6 next pane" since task-4023 AC#5) because the
                # screen already covers that key, so F1/Ctrl+P/Ctrl+Q are
                # the only globals left.
                assert (
                    screen.query_one(AppFooterStatus).shortcut_text
                    == "/ focus search | i import content | n new note | "
                    "F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit"
                )
                for _ in range(300):
                    rows = list(screen.query("#library-row-browse-search"))
                    if rows:
                        rows[0].press()
                        break
                    await pilot.pause(0.01)
                else:
                    raise AssertionError("Library Search/RAG row did not mount.")
                for _ in range(300):
                    footers = list(screen.query(AppFooterStatus))
                    if (
                        footers
                        and "use Library context in Console"
                        in footers[0].shortcut_text
                    ):
                        screen_footer = footers[0]
                        break
                    await pilot.pause(0.01)
                else:
                    raise AssertionError("Library contextual footer did not update.")
                # Task 12/RAG-36 fix-review: LIBRARY_SHORTCUTS also advertises
                # the evidence-card `enter`/`o` keys (both `Binding(...,
                # show=False)`, so this footer registration is their ONLY
                # on-screen discoverability signal) -- pin the full four-hint
                # string so a future edit can't silently drop a key.
                # F-012: `/ focus search` closes the set -- it works on every
                # canvas, including this one.
                # task-4023 AC#4 (RC-10): the set now covers F6 itself, so
                # the screen entry renders "F6 next pane" verbatim and the
                # global cluster drops its own F6 entry (the task-2860
                # merge rule, same as the landing assertion above).
                assert screen_footer.shortcut_text == (
                    "u use Library context in Console | "
                    "enter select evidence | o open evidence | "
                    "/ focus search | F6 next pane | "
                    "F1 help · Ctrl+P palette · Ctrl+Q quit"
                )

                footer_before = screen_footer
                screen.refresh(recompose=True)
                for _ in range(300):
                    footers = list(screen.query(AppFooterStatus))
                    if footers:
                        footer_after = footers[0]
                        if (
                            footer_after is not footer_before
                            and "use Library context in Console"
                            in footer_after.shortcut_text
                        ):
                            break
                    await pilot.pause(0.01)
                else:
                    raise AssertionError(
                        "Library footer registration did not survive recompose."
                    )
                assert footer_after.shortcut_text == (
                    "u use Library context in Console | "
                    "enter select evidence | o open evidence | "
                    "/ focus search | F6 next pane | "
                    "F1 help · Ctrl+P palette · Ctrl+Q quit"
                )

                await app.handle_screen_navigation(NavigateToScreen("settings"))
                screen = await _wait_for_screen(
                    app,
                    pilot,
                    SettingsScreen,
                    "settings",
                )
                screen_footer = screen.query_one(AppFooterStatus)
                for token in ("save category", "revert category", "test category"):
                    assert token not in screen_footer.shortcut_text

                assert list(screen.query(AppFooterStatus)) == [screen_footer]
    finally:
        await _close_production_app(app)


def test_library_shortcuts_advertise_the_evidence_card_keys():
    """Task 12/RAG-36 fix-review: a cheap, direct pin on `LIBRARY_SHORTCUTS`
    itself (no pilot/app needed) -- both evidence-card `Binding`s
    (`enter`/`o`) are registered with `show=False`, so this tuple is the
    ONLY on-screen signal that they exist. A future edit to this tuple that
    silently drops one of the entries must fail here.

    F-012: `/ focus search` closes the set (it works on every canvas), and
    the landing/other-canvas set is pinned alongside it."""
    # task-4023 AC#4 (RC-10): F6 closes the set -- Search/RAG was the ONLY
    # per-mode Library set without it, so F1 (fed by this same tuple, the
    # task-2858 single-source rule) never listed the key even though the
    # footer's global hint cluster advertised "F6 panes" on that surface.
    assert LibraryScreen.LIBRARY_SHORTCUTS == (
        ("u", "use Library context in Console"),
        ("enter", "select evidence"),
        ("o", "open evidence"),
        ("/", "focus search"),
        ("F6", "next pane"),
    )
    # task-2237 (R2): the landing set advertises the full keyboard story --
    # the hub next-action accelerators and the pane-cycle key, not just `/`.
    assert LibraryScreen.LIBRARY_LANDING_SHORTCUTS == (
        ("/", "focus search"),
        ("i", "import content"),
        ("n", "new note"),
        ("F6", "next pane"),
    )


class _MinimalScreen(BaseAppScreen):
    """The lightest possible BaseAppScreen subclass -- just enough content
    to mount without pulling in a real destination's dependencies."""

    def compose_content(self) -> ComposeResult:
        yield Static("minimal screen content")


class _MinimalScreenHost(ConsolidatedCSSApp):
    """Hosts a bare BaseAppScreen subclass with no App-level footer of its
    own, so the only AppFooterStatus in the tree is the one the screen
    itself composes."""

    async def on_mount(self) -> None:
        await self.push_screen(_MinimalScreen(None, "minimal"))


@pytest.mark.asyncio
async def test_base_app_screen_composes_footer_status():
    """Every BaseAppScreen carries its own AppFooterStatus instance."""
    host = _MinimalScreenHost()

    async with host.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = host.screen_stack[-1]

        footer = screen.query_one(AppFooterStatus)

        assert footer.id == "screen-footer-status"
        assert footer.shortcut_text == AppFooterStatus.DEFAULT_SHORTCUT_TEXT


class _DefaultScreenFooterHost(ConsolidatedCSSApp):
    """Mirrors app.py's real shape: an `AppFooterStatus` composed directly
    on the App's own DEFAULT screen (id="app-footer-status", exactly like
    `TldwCli._create_main_ui_widgets`), with a real destination screen
    pushed on top of it. Before task-264 this was the ONLY footer in the
    tree, and it is what `self.app.query_one(AppFooterStatus)` used to
    (mis)resolve to from inside the pushed screen.
    """

    def __init__(self, app_instance, screen_factory):
        super().__init__()
        self.app_instance = app_instance
        self._screen_factory = screen_factory

    def compose(self) -> ComposeResult:
        yield AppFooterStatus(id="app-footer-status")

    async def on_mount(self) -> None:
        await self.push_screen(self._screen_factory(self.app_instance))


@pytest.mark.asyncio
async def test_console_registration_updates_the_screens_own_footer():
    """chat_screen's registration must land on ITS instance, not the app's
    default-screen one; text contains 'F6' and 'Ctrl+K'."""
    app_instance = _build_test_app()
    host = _DefaultScreenFooterHost(app_instance, ChatScreen)

    async with host.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        screen = host.screen_stack[-1]

        # Real registration path: ChatScreen.on_mount() already called
        # _register_console_footer_shortcuts() during push_screen above.
        screen_footer = screen.query_one(AppFooterStatus)
        assert screen_footer.id == "screen-footer-status"
        assert "F6" in screen_footer.shortcut_text
        assert "Ctrl+K" in screen_footer.shortcut_text

        # The app's default-screen footer (what `self.app.query_one(...)`
        # used to target) must be left untouched at its default text.
        default_screen_footer = host.query_one(AppFooterStatus)
        assert default_screen_footer is not screen_footer
        assert (
            default_screen_footer.shortcut_text == AppFooterStatus.DEFAULT_SHORTCUT_TEXT
        )


@pytest.mark.asyncio
async def test_mcp_registration_updates_the_screens_own_footer():
    """mcp_screen registration -> its footer text contains 'mode' and
    'a add server'."""
    app_instance = _build_test_app()
    host = _DefaultScreenFooterHost(app_instance, MCPScreen)

    async with host.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        screen = host.screen_stack[-1]

        # Real registration path: MCPScreen.on_mount() already called
        # _register_footer_shortcuts() during push_screen above.
        screen_footer = screen.query_one(AppFooterStatus)
        assert screen_footer.id == "screen-footer-status"
        assert "mode" in screen_footer.shortcut_text
        assert "a add server" in screen_footer.shortcut_text

        default_screen_footer = host.query_one(AppFooterStatus)
        assert default_screen_footer is not screen_footer
        assert (
            default_screen_footer.shortcut_text == AppFooterStatus.DEFAULT_SHORTCUT_TEXT
        )


@pytest.mark.asyncio
async def test_registration_survives_screen_recompose():
    """Screen-level recompose (settings' recompose=True reactives, library/
    chat `refresh(recompose=True)`) replaces the footer widget; the
    persisted registration must re-seed the fresh instance (task-264
    review)."""
    host = _MinimalScreenHost()

    async with host.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = host.screen_stack[-1]
        screen.register_footer_shortcuts(
            source="minimal", shortcuts=(("x", "do the thing"),)
        )
        footer_before = screen.query_one(AppFooterStatus)
        assert "do the thing" in footer_before.shortcut_text

        screen.refresh(recompose=True)
        await pilot.pause()

        footer_after = screen.query_one(AppFooterStatus)
        assert footer_after is not footer_before
        assert "do the thing" in footer_after.shortcut_text

        # And a source-guarded clear drops both the live text and the
        # persisted copy, so a later recompose stays at the default.
        screen.clear_footer_shortcuts(source="minimal")
        screen.refresh(recompose=True)
        await pilot.pause()
        footer_cleared = screen.query_one(AppFooterStatus)
        assert footer_cleared.shortcut_text == AppFooterStatus.DEFAULT_SHORTCUT_TEXT


@pytest.mark.asyncio
async def test_library_registration_updates_the_screens_own_footer():
    """task-264 review Important: Library's `u` hint (previously rendered by
    the retired Textual Footer's show=True binding) must reach its own
    footer via the registration API. task-2237: the landing canvas
    advertises LIBRARY_LANDING_SHORTCUTS; the `u` set registers once a
    browse row is entered."""
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app_instance = _build_test_app()
    host = _DefaultScreenFooterHost(app_instance, LibraryScreen)

    async with host.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        screen = host.screen_stack[-1]
        screen_footer = screen.query_one(AppFooterStatus)
        # Landing canvas: the task-2237 landing set, on the screen's OWN
        # footer (not the app default-screen one).
        assert "import content" in screen_footer.shortcut_text

        for _ in range(300):
            rows = list(screen.query("#library-row-browse-search"))
            if rows:
                rows[0].press()
                break
            await pilot.pause(0.01)
        else:
            raise AssertionError("Library Search/RAG row did not mount.")
        for _ in range(300):
            footers = list(screen.query(AppFooterStatus))
            if footers and "use Library context in Console" in footers[0].shortcut_text:
                screen_footer = footers[0]
                break
            await pilot.pause(0.01)
        else:
            raise AssertionError("Library contextual footer did not update.")
        assert "u" in screen_footer.shortcut_text


def test_library_note_footer_shortcut_contract_is_exact():
    """Keep screen-owned Notes copy stable without cross-test private imports."""
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    # task-4023 AC#5: the Notes workflow joins the shared footer grammar --
    # per-key (key, label) pairs, lowercase keys, the same action
    # vocabulary as the sibling canvases -- with a _COMPACT tier that only
    # compresses labels (same keys/destinations) for the ≤100-col
    # single-stage footer.
    assert LibraryScreen.LIBRARY_NOTES_NAVIGATOR_SHORTCUTS == (
        ("ctrl+n", "new note"),
        ("/", "find note"),
        ("esc", "focus rail"),
    )
    assert LibraryScreen.LIBRARY_NOTES_NAVIGATOR_SHORTCUTS_COMPACT == (
        ("ctrl+n", "new"),
        ("/", "find"),
        ("esc", "rail"),
    )
    assert LibraryScreen.LIBRARY_NOTES_EDITOR_SHORTCUTS == (
        ("ctrl+s", "save note"),
        ("esc", "back to notes"),
    )
    assert LibraryScreen.LIBRARY_NOTES_EDITOR_SHORTCUTS_COMPACT == (
        ("ctrl+s", "save"),
        ("esc", "notes"),
    )
    assert LibraryScreen.LIBRARY_NOTES_PREVIEW_SHORTCUTS == (
        ("pgup/pgdn", "scroll"),
        ("esc", "back to notes"),
    )
    assert LibraryScreen.LIBRARY_NOTES_CONTEXT_SHORTCUTS == (
        ("enter", "run action"),
        ("esc", "back to note"),
    )
    assert LibraryScreen.LIBRARY_NOTES_CONFLICT_SHORTCUTS == (
        ("enter", "choose version"),
    )


@pytest.mark.asyncio
async def test_settings_registration_updates_the_screens_own_footer():
    """task-264 review Important: Settings' footer hints must reach its own
    footer via registration. task-1340: the advertised set is per-category
    (ADR-031) -- the default Overview category supports no s/r/t action, so
    none are advertised; switching to Providers & Models (which implements
    save, revert, AND test) advertises all three."""
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    app_instance = _build_test_app()
    host = _DefaultScreenFooterHost(app_instance, SettingsScreen)

    async with host.run_test(size=(160, 48)) as pilot:
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        screen = host.screen_stack[-1]
        screen_footer = screen.query_one(AppFooterStatus)
        for token in ("save category", "revert category", "test category"):
            assert token not in screen_footer.shortcut_text

        screen._select_category("providers-models", restore_focus=True)
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        screen_footer = screen.query_one(AppFooterStatus)
        for token in ("save category", "revert category", "test provider"):
            assert token in screen_footer.shortcut_text


# ---------------------------------------------------------------------------
# task-289 drift guards: two invariants task-264 left as comments become
# red tests here.

_CSS_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"
# Full comment text, NOT the bare marker: splitting on a marker that sits
# INSIDE a comment leaves a dangling `*/` in the section, which corrupts the
# first parsed selector.
_FOOTER_SECTION_START = "/* --- Window Footer Widget --- */"
_FOOTER_SECTION_END = "/* --- End of Window Footer Widget --- */"


def _parse_css_blocks(css_text: str) -> dict[str, dict[str, str]]:
    """Parse flat (non-nested) tcss into {selector: {property: value}}.

    Comments are stripped first; selectors and values are whitespace-
    normalized. Good enough for the simple declaration blocks under test --
    NOT a general CSS parser. Known limit: a selector LIST (``#a, #b { }``)
    is kept as one compound key, so regrouping rules across the two files
    reports as "missing from footer section" (a loud false-fail, never a
    silent pass) -- split the list back out if you hit that.
    """
    text = re.sub(r"/\*.*?\*/", "", css_text, flags=re.DOTALL)
    blocks: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", text):
        selector = " ".join(match.group(1).split())
        declarations: dict[str, str] = {}
        for declaration in match.group(2).split(";"):
            if ":" not in declaration:
                continue
            prop, value = declaration.split(":", 1)
            declarations[prop.strip()] = " ".join(value.split())
        # CSS merges duplicate selector blocks in source order (later
        # properties win per-property); mirror that instead of overwriting
        # the whole block, so a legitimate split of one selector across two
        # blocks cannot false-fail the drift guard (Qodo #687-2).
        blocks.setdefault(selector, {}).update(declarations)
    return blocks


def _footer_section_blocks(css_path: Path) -> dict[str, dict[str, str]]:
    """The parsed footer-section blocks of a css file, keyed by selector."""
    text = css_path.read_text(encoding="utf-8")
    assert _FOOTER_SECTION_START in text and _FOOTER_SECTION_END in text, (
        f"{css_path.name} lost its '{_FOOTER_SECTION_START}' section markers -- "
        "the BUNDLED_CSS drift guard needs them to find the footer block."
    )
    section = text.split(_FOOTER_SECTION_START, 1)[1].split(_FOOTER_SECTION_END, 1)[0]
    return _parse_css_blocks(section)


def _default_css_divergences(bundle_blocks: dict[str, dict[str, str]]) -> list[str]:
    """Every BUNDLED_CSS declaration missing/different in the bundle blocks.

    BUNDLED_CSS scopes child selectors as ``AppFooterStatus #id``; the bundle
    declares the same ids unscoped, so the scope prefix is stripped before
    matching. BUNDLED_CSS is allowed to be a SUBSET (the bundle carries
    extras); a declaration present in BUNDLED_CSS but absent or different in
    the bundle is drift.
    """
    divergences = []
    for selector, declarations in _parse_css_blocks(
        AppFooterStatus.BUNDLED_CSS
    ).items():
        bundle_selector = selector.replace("AppFooterStatus #", "#")
        bundle_declarations = bundle_blocks.get(bundle_selector)
        if bundle_declarations is None:
            divergences.append(
                f"selector {bundle_selector!r} missing from footer section"
            )
            continue
        for prop, value in declarations.items():
            bundle_value = bundle_declarations.get(prop)
            if bundle_value != value:
                divergences.append(
                    f"{bundle_selector} {{ {prop}: {value} }} vs bundle "
                    f"{bundle_value!r}"
                )
    return divergences


def test_default_css_matches_the_live_bundle_source():
    """AppFooterStatus.BUNDLED_CSS must stay a faithful subset of the live
    bundle source (css/components/_widgets.tcss footer block) -- otherwise
    stylesheet-less harnesses silently diverge from production geometry
    (task-264's KEEP-IN-SYNC contract, previously comment-only)."""
    divergences = _default_css_divergences(
        _footer_section_blocks(_CSS_ROOT / "components" / "_widgets.tcss")
    )
    assert divergences == [], (
        "AppFooterStatus.BUNDLED_CSS diverged from _widgets.tcss's footer "
        f"block: {divergences}. Update BOTH sides (they are KEEP-IN-SYNC) "
        "and rebuild the bundle (python3 tldw_chatbook/css/build_css.py)."
    )


def test_built_bundle_carries_the_footer_rules():
    """The BUILT bundle (tldw_cli_modular.tcss, what production loads) must
    carry the same footer declarations -- catches an edited _widgets.tcss
    that was never rebuilt into the bundle."""
    divergences = _default_css_divergences(
        _footer_section_blocks(_CSS_ROOT / "tldw_cli_modular.tcss")
    )
    assert divergences == [], (
        "The built bundle's footer block diverged from AppFooterStatus."
        f"BUNDLED_CSS: {divergences}. If _widgets.tcss is already correct, "
        "the bundle is stale -- rerun python3 tldw_chatbook/css/build_css.py."
    )


def test_personas_screen_has_no_recompose_path_while_footer_hints_are_non_persisting():
    """PersonasScreen drives its footer through the NON-persisting
    ``set_shortcut_context`` path (its hint set is dynamic, re-registered on
    editing/selection transitions). That is only safe while the screen never
    recomposes -- a screen-level recompose replaces the footer widget and
    silently resets its hints (the task-264 fix-wave bug, solved for the
    static screens by BaseAppScreen's persisting registration).

    Guard, two rules (both literal-True only -- the house AST-guard style):
    - any ``recompose=True`` call inside a BaseAppScreen subclass body
      (recompose reactives, self.refresh(recompose=True)); non-screen child
      widget classes are excluded because a child-widget recompose only
      rebuilds that widget's children, never the screen's footer;
    - any ``.refresh(recompose=True)`` call OUTSIDE class bodies -- the
      library_screen.py precedent is module-level helpers that take the
      screen and call ``screen.refresh(recompose=True)``, invisible to a
      class-body-only walk (task-289 review).

    Escape hatch: an ACTUAL ``.register_footer_shortcuts(...)`` call in the
    file (the persisting API) disarms the guard -- checked as an AST call,
    not a substring, so a comment mentioning the API cannot disarm it.
    """
    import tldw_chatbook.UI.Screens.personas_screen as personas_module

    source = Path(personas_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    def _has_literal_true_recompose(call: ast.Call) -> bool:
        return any(
            keyword.arg == "recompose"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )

    recompose_sites: list[int] = []
    uses_persisting_api = False
    screen_classes_found = 0
    for top_level in tree.body:
        if isinstance(top_level, ast.ClassDef):
            base_names = {
                base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
                for base in top_level.bases
            }
            is_screen_class = "BaseAppScreen" in base_names
            screen_classes_found += is_screen_class
            for inner in ast.walk(top_level):
                if not isinstance(inner, ast.Call):
                    continue
                if (
                    isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "register_footer_shortcuts"
                ):
                    uses_persisting_api = True
                if is_screen_class and _has_literal_true_recompose(inner):
                    recompose_sites.append(inner.lineno)
        else:
            # Module-level statements/functions: only .refresh(recompose=True)
            # counts here (a bare reactive() at module scope isn't a screen
            # reactive), but the persisting-API call disarms from anywhere.
            for inner in ast.walk(top_level):
                if not isinstance(inner, ast.Call):
                    continue
                if (
                    isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "register_footer_shortcuts"
                ):
                    uses_persisting_api = True
                if (
                    isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "refresh"
                    and _has_literal_true_recompose(inner)
                ):
                    recompose_sites.append(inner.lineno)

    # Self-check: the walk identifies screen classes by the literal direct
    # base name "BaseAppScreen". If personas ever inherits via an alias or an
    # intermediate class, this guard goes blind -- fail loudly instead of
    # silently scanning nothing (Qodo #687-3a). Runtime truth check keeps the
    # assertion honest against renames.
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    assert issubclass(PersonasScreen, BaseAppScreen)
    assert screen_classes_found >= 1, (
        "The recompose guard found no class with a direct 'BaseAppScreen' "
        "base in personas_screen.py, but PersonasScreen IS a BaseAppScreen "
        "subclass at runtime -- the inheritance is no longer literal/direct "
        "and this guard's AST detection is blind. Update the guard's "
        "screen-class detection to match the new inheritance shape."
    )

    assert not (recompose_sites and not uses_persisting_api), (
        f"personas_screen.py gained recompose=True at line(s) {recompose_sites} "
        "while its footer hints still use the non-persisting "
        "set_shortcut_context path -- a recompose will silently reset them. "
        "Migrate the footer registration to BaseAppScreen."
        "register_footer_shortcuts (persisting) before adding a recompose "
        "path (see task-264 / task-289)."
    )
