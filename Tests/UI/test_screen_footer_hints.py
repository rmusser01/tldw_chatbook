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
        if app.current_tab == tab and isinstance(app.screen, screen_type):
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

                await app.handle_screen_navigation(NavigateToScreen("library"))
                screen = await _wait_for_screen(
                    app,
                    pilot,
                    LibraryScreen,
                    "library",
                )
                assert (
                    screen.query_one(AppFooterStatus).shortcut_text
                    == AppFooterStatus.DEFAULT_SHORTCUT_TEXT
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
                    screen_footer = screen.query_one(AppFooterStatus)
                    if "use Library context in Console" in screen_footer.shortcut_text:
                        break
                    await pilot.pause(0.01)
                else:
                    raise AssertionError("Library contextual footer did not update.")
                assert (
                    screen_footer.shortcut_text
                    == "u use Library context in Console"
                )

                footer_before = screen_footer
                screen.refresh(recompose=True)
                for _ in range(300):
                    footer_after = screen.query_one(AppFooterStatus)
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
                assert (
                    footer_after.shortcut_text
                    == "u use Library context in Console"
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
        "the DEFAULT_CSS drift guard needs them to find the footer block."
    )
    section = text.split(_FOOTER_SECTION_START, 1)[1].split(_FOOTER_SECTION_END, 1)[0]
    return _parse_css_blocks(section)


def _default_css_divergences(bundle_blocks: dict[str, dict[str, str]]) -> list[str]:
    """Every DEFAULT_CSS declaration missing/different in the bundle blocks.

    DEFAULT_CSS scopes child selectors as ``AppFooterStatus #id``; the bundle
    declares the same ids unscoped, so the scope prefix is stripped before
    matching. DEFAULT_CSS is allowed to be a SUBSET (the bundle carries
    extras); a declaration present in DEFAULT_CSS but absent or different in
    the bundle is drift.
    """
    divergences = []
    for selector, declarations in _parse_css_blocks(
        AppFooterStatus.DEFAULT_CSS
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
    """AppFooterStatus.DEFAULT_CSS must stay a faithful subset of the live
    bundle source (css/components/_widgets.tcss footer block) -- otherwise
    stylesheet-less harnesses silently diverge from production geometry
    (task-264's KEEP-IN-SYNC contract, previously comment-only)."""
    divergences = _default_css_divergences(
        _footer_section_blocks(_CSS_ROOT / "components" / "_widgets.tcss")
    )
    assert divergences == [], (
        "AppFooterStatus.DEFAULT_CSS diverged from _widgets.tcss's footer "
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
        f"DEFAULT_CSS: {divergences}. If _widgets.tcss is already correct, "
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
