"""Roleplay's lazy stylesheet: ownership, anchoring and parity (frame slice B1).

Spec 2.12 items 2-3, R18 and G18. ``features/_roleplay.tcss`` is split whole
into ``screen_feature_roleplay.tcss``, which the app parses on the first visit
to Roleplay. Static checks only (no app is mounted): the visit itself is pinned
by the harness self-tests (``test_roleplay_frame_harness.py``).
"""

from __future__ import annotations

import ast
import re
from dataclasses import astuple, fields
from pathlib import Path

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
from textual.css.model import SelectorType
from textual.css.parse import parse_selectors

from Tests.UI import test_consolidated_css_harness as harness_scan
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, CSS_DIR
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Constants import TAB_PERSONAS
from tldw_chatbook.css import build_css
from tldw_chatbook.UI.Persona_Modules.roleplay_frame_state import (
    ROLEPLAY_PANE_CLASS_NAMES,
)
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.adaptive_pane_shell import AdaptivePaneClasses
from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
    LIBRARY_ADAPTIVE_READER_CLASSES,
)

SHEET_NAME = "screen_feature_roleplay.tcss"
SHEET = CSS_DIR / SHEET_NAME
SOURCE = CSS_DIR / "features" / "_roleplay.tcss"
LIBRARY_SOURCE = CSS_DIR / "features" / "_library.tcss"
#: Shared tokens the Roleplay split claims so its copies leave boot. They are
#: never an anchor: a selector made only of them would restyle other screens.
SHARED_CLAIMS = frozenset(
    {
        "workbench-header-title",
        "workbench-header-subtitle",
        "workbench-header-status",
        "-active",
    }
)
_TOKEN = re.compile(r"[#.]([A-Za-z0-9_-]+)")
_PREAMBLE_END = ".tldw-agentic-split-preamble-end { }"
ROLEPLAY_PANE_CLASSES = AdaptivePaneClasses(*ROLEPLAY_PANE_CLASS_NAMES)


def _roleplay_prefixes() -> tuple[str, ...]:
    for split in build_css.SCREEN_OWNED_SPLITS:
        for owner, sheet in split.sheets.items():
            if sheet == SHEET_NAME:
                return tuple(split.prefixes[owner])
    raise AssertionError(f"no screen-owned split writes {SHEET_NAME}")


def _anchored(member: str) -> bool:
    """True when a selector carries at least one Roleplay-only token."""
    own = [p for p in _roleplay_prefixes() if p not in SHARED_CLAIMS]
    return any(
        token not in SHARED_CLAIMS
        and any(token == p or token.startswith(f"{p}-") for p in own)
        for token in _TOKEN.findall(member)
    )


def _moved_rules(text: str) -> list[tuple[str, str]]:
    """``(selector, body)`` per rule after the sheet's variable preamble."""
    body = text.split(_PREAMBLE_END, 1)[1] if _PREAMBLE_END in text else text
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    return [
        (" ".join(selector.split()), declarations)
        for selector, declarations in re.findall(r"([^{}]+)\{([^{}]*)\}", body)
    ]


def _members(selector: str) -> list[str]:
    return [" ".join(member.split()) for member in selector.split(",")]


def test_the_sheet_is_route_loaded_and_never_a_screen_css_path():
    """MF-02: Textual loads a screen's CSS_PATH under every app, so the sheet
    rides the route map, parsed by ``_ensure_screen_owned_css``."""
    assert TldwCli._SCREEN_OWNED_ROUTE_CSS[TAB_PERSONAS] == (SHEET_NAME,)
    assert not getattr(PersonasScreen, "CSS_PATH", None)


def test_every_roleplay_selector_carries_a_roleplay_token():
    """Claiming shared tokens is safe only while each selector is anchored."""
    rules = _moved_rules(SHEET.read_text(encoding="utf-8"))
    assert len(rules) >= 14
    for selector, _body in rules:
        for member in _members(selector):
            assert _anchored(member), f"{member!r} has no Roleplay token"
    # The guard itself is not vacuous:
    assert not _anchored(".workbench-header-title")
    assert not _anchored("Button.-active")
    assert _anchored("#personas-header .workbench-header-title")


def test_the_sheet_has_no_bare_type_subject():
    """The broad-selector census counts only boot sheets (spec 2.12 item 1),
    so this sheet pins itself: every subject is an id or a class."""
    for selector, _body in _moved_rules(SHEET.read_text(encoding="utf-8")):
        for selector_set in parse_selectors(selector):
            subject = selector_set.selectors[-1]
            assert subject.type in (SelectorType.ID, SelectorType.CLASS), selector
    bare = parse_selectors("#personas-header Static")[0].selectors[-1]
    assert bare.type is SelectorType.TYPE  # the check can fail


def test_the_boot_bundle_carries_no_roleplay_rule():
    """Every Roleplay rule moved: the bundle keeps only the module banner, and
    no header or shell token B1 adds is left in boot."""
    bundle = BUNDLED_STYLESHEET.read_text(encoding="utf-8")
    banner = "/* ===== MODULE: features/_roleplay.tcss ===== */"
    section = bundle.split(banner, 1)[1].split("/* ===== MODULE:", 1)[0]
    assert section.strip() == ""
    frame_tokens = {
        token
        for token in _TOKEN.findall(bundle)
        if token.startswith(("personas-header-", *ROLEPLAY_PANE_CLASS_NAMES))
    }
    assert frame_tokens == set()


def test_roleplay_pane_classes_carry_a_roleplay_split_prefix():
    """A shell class without the owner prefix would pin its rules to boot."""
    for css_class in astuple(ROLEPLAY_PANE_CLASSES):
        assert _anchored(f".{css_class}"), css_class


def _role_rules(text: str, classes: AdaptivePaneClasses) -> set[tuple[str, tuple]]:
    """The rules that use a shell class, with each class replaced by its role."""
    roles = {
        getattr(classes, field.name): "{" + field.name + "}"
        for field in fields(classes)
    }
    found = set()
    for selector, body in _moved_rules(text):
        if not any(token in roles for token in _TOKEN.findall(selector)):
            continue
        normalised = sorted(
            _TOKEN.sub(
                lambda m: m.group(0)[0] + roles.get(m.group(1), m.group(1)), member
            )
            for member in _members(selector)
        )
        declarations = tuple(
            sorted(" ".join(d.split()) for d in body.split(";") if d.strip())
        )
        found.add((", ".join(normalised), declarations))
    return found


def test_roleplay_shell_rules_match_the_library_block():
    """Spec 2.12 item 2: Roleplay's copy equals the Library's re-keyed block
    once class names are normalised to roles; six rules, nav/items have none."""
    library = _role_rules(
        LIBRARY_SOURCE.read_text(encoding="utf-8"), LIBRARY_ADAPTIVE_READER_CLASSES
    )
    roleplay = _role_rules(SOURCE.read_text(encoding="utf-8"), ROLEPLAY_PANE_CLASSES)
    assert len(library) == 6
    assert roleplay == library


def test_parity_sees_a_dropped_focus_reverse():
    """Negative control: the parity check is not vacuous."""
    mutated = SOURCE.read_text(encoding="utf-8").replace(
        "text-style: bold reverse;", "text-style: bold;"
    )
    library = _role_rules(
        LIBRARY_SOURCE.read_text(encoding="utf-8"), LIBRARY_ADAPTIVE_READER_CLASSES
    )
    assert _role_rules(mutated, ROLEPLAY_PANE_CLASSES) != library


def test_the_split_sheet_scan_reports_a_bundle_only_roleplay_harness(
    tmp_path, monkeypatch
):
    """``_SPLIT_SHEET_OWNERS`` lists no owner for the Roleplay sheet, so a
    harness pinned to the bundle alone that queries a Roleplay-only token is
    reported (spec B1: PersonasScreen must never exempt one)."""
    tests_root = tmp_path / "Tests"
    tests_root.mkdir()
    (tests_root / "test_synthetic_roleplay_harness.py").write_text(
        "from textual.app import App\n"
        "from Tests.UI.consolidated_css import BUNDLED_STYLESHEET\n"
        "from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen\n\n"
        "class BundleOnlyRoleplayHarness(App):\n"
        "    CSS_PATH = str(BUNDLED_STYLESHEET)\n\n"
        "    def on_mount(self):\n"
        "        self.push_screen(PersonasScreen(self))\n\n"
        "QUERY = '#personas-header-unsaved'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(harness_scan, "_TESTS_ROOT", tests_root)
    findings = harness_scan.scan_bundle_only_harnesses()
    assert [(cls, sheet) for _path, cls, sheet, _used in findings] == [
        ("BundleOnlyRoleplayHarness", SHEET_NAME)
    ]


#: Tests that build a real ``TldwCli`` and push ``PersonasScreen`` themselves
#: but assert no geometry (they check which center views are mounted), so the
#: sheet the push skips cannot mislead them.
_GEOMETRY_FREE_FULL_APP_PUSHES = frozenset(
    {"Tests/UI/test_personas_deferred_center_views.py"}
)


def _bare_roleplay_pushes(tests_root: Path) -> list[str]:
    """``path::function`` for every test function that builds the real app
    (``_build_test_app``) and constructs ``PersonasScreen`` itself without
    loading the route's sheet first (``_ensure_screen_owned_css``)."""
    found = []
    for path in sorted(tests_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "PersonasScreen" not in text or "_build_test_app" not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            called = {
                getattr(call.func, "id", None) or getattr(call.func, "attr", None)
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
            }
            if {"_build_test_app", "PersonasScreen"} <= called and (
                "_ensure_screen_owned_css" not in called
            ):
                relative = path.relative_to(tests_root.parent).as_posix()
                found.append(f"{relative}::{node.name}")
    return found


def test_no_full_app_test_pushes_roleplay_without_its_sheet(tmp_path):
    """A real app that pushes ``PersonasScreen`` itself skips the route loader,
    so the inline header paints without its rules (TASK-32187's Watchlists
    trap). Each such test loads the sheet first or is listed as geometry-free.
    Negative control: a synthetic offender is reported, a loaded one is not."""
    repo = Path(__file__).resolve().parents[2]
    offenders = [
        hit
        for hit in _bare_roleplay_pushes(repo / "Tests")
        if hit.split("::")[0] not in _GEOMETRY_FREE_FULL_APP_PUSHES
    ]
    assert offenders == []
    synthetic = tmp_path / "Tests"
    synthetic.mkdir()
    (synthetic / "test_synthetic_push.py").write_text(
        "async def test_bare():\n"
        "    app = _build_test_app()\n"
        "    await app.push_screen(PersonasScreen(app))\n\n\n"
        "async def test_loaded():\n"
        "    app = _build_test_app()\n"
        "    app._ensure_screen_owned_css('personas')\n"
        "    await app.push_screen(PersonasScreen(app))\n",
        encoding="utf-8",
    )
    assert _bare_roleplay_pushes(synthetic) == [
        "Tests/test_synthetic_push.py::test_bare"
    ]


def test_header_chip_words_clear_aa_on_every_theme():
    """Spec 4.12: the chips are words that must be READ, so their colour
    tokens resolve to AA (4.5:1) on the header's panel surface in every
    registered theme. Negative control: the decorative hue fails somewhere."""
    from textual.color import Color

    from Tests.UI.test_theme_contrast import (
        AA,
        _ratio,
        _resolve_color,
        _resolved_variables,
    )
    from tldw_chatbook.css.Themes.themes import ALL_THEMES

    variables = (CSS_DIR / "core" / "_variables.tcss").read_text(encoding="utf-8")
    alias = dict(re.findall(r"^\$(ds-[\w-]+):\s*\$([\w-]+);", variables, re.M))
    rules = dict(_moved_rules(SOURCE.read_text(encoding="utf-8")))

    def contrast(theme, textual_name: str) -> float:
        resolved = _resolved_variables(theme)
        panel = Color.parse(resolved["panel"])
        return _ratio(_resolve_color(resolved[textual_name], panel).hex, panel.hex)

    for selector in ("#personas-header-unsaved", "#personas-header-blocked"):
        token = re.search(r"color:\s*\$([\w-]+);", rules[selector])[1]
        for theme in ALL_THEMES:
            ratio = contrast(theme, alias[token])
            assert ratio >= AA, (selector, token, theme.name, round(ratio, 2))
    assert min(contrast(theme, "warning") for theme in ALL_THEMES) < AA


def test_the_roleplay_source_has_no_numeric_dimension_literal():
    """ADR-161's hard zero, pinned for this sheet: the repo-wide
    ``test_dimension_literal_ratchet`` is red on dev for other sheets, so a
    failure-set diff could not see a new literal here."""
    from Tests.UI.test_component_pattern_governance import _DIM

    text = re.sub(r"/\*.*?\*/", "", SOURCE.read_text(encoding="utf-8"), flags=re.S)
    assert _DIM.findall(text) == []
    assert _DIM.search("#probe { height: 1; }")  # the pattern can fire
