# test_consolidated_css_harness.py
# Description: Pins Tests/UI/consolidated_css.py's own CSS_PATH-merge behavior
# (TASK-15995).
#
# `ConsolidatedCSSApp.CSS_PATH` carries the two generated screen/modal sheets
# (TASK-15450) so a harness that pushes one of the seven `BUNDLED_SCREEN_CSS`
# modals gets its class-level CSS. But it is an ordinary class attribute, and
# ~27 real test harnesses declare their own `CSS_PATH` (most to also load the
# app bundle) -- which, absent a merge, shadows it wholesale via normal Python
# attribute lookup and drops the screen sheets. Textual's `App.__init__` also
# accepts a `css_path=` constructor kwarg that short-circuits the
# `self.CSS_PATH` class-attribute branch entirely (`css_path or
# self.CSS_PATH`), so the merge has to intercept both forms. This module pins
# both rather than relying on any of the 27 real harnesses, since none of them
# currently happens to push a `BUNDLED_SCREEN_CSS` modal (a vacuous-pass trap
# noted in the task).

from __future__ import annotations

import ast
import copy
import importlib
import re
from pathlib import Path

import pytest
from textual.color import Color

from Tests.UI.consolidated_css import (
    APP_STYLESHEETS,
    BUNDLED_STYLESHEET,
    ConsolidatedCSSApp,
)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Widgets.Note_Widgets.note_selection_dialog import (
    NoteSelectionDialog,
)


class _StyledHarness(ConsolidatedCSSApp):
    """Mirrors the real combiners: a subclass that declares its own CSS_PATH."""

    CSS_PATH = str(BUNDLED_STYLESHEET)


@pytest.mark.asyncio
async def test_css_path_class_attr_override_still_loads_screen_sheets():
    """A subclass ``CSS_PATH`` class attribute must not drop the screen sheets.

    `NoteSelectionDialog` (one of the seven `BUNDLED_SCREEN_CSS` modals) gives
    `#note-selection-container` a fixed `width: 80` only via its screen sheet
    entry; absent that sheet the container falls back to `Container`'s own
    `width: 1fr` default and fills the whole content area instead. This is a
    computed-geometry consequence of the CSS actually applying, not merely a
    check that pushing the modal raised no exception.
    """
    app = _StyledHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(NoteSelectionDialog([]))
        await pilot.pause()

        container = app.screen.query_one("#note-selection-container")
        assert container.region.width == 80, (
            "NoteSelectionDialog's BUNDLED_SCREEN_CSS width:80 rule for "
            "#note-selection-container did not apply (region.width="
            f"{container.region.width}) -- a subclass CSS_PATH class "
            "attribute override dropped ConsolidatedCSSApp's screen sheets"
        )


@pytest.mark.asyncio
async def test_css_path_kwarg_override_loads_screen_sheets_and_keeps_own_entry(
    tmp_path,
):
    """A ``css_path=`` constructor kwarg override must survive the merge too.

    Textual's ``App.__init__`` resolves ``css_path or self.CSS_PATH`` -- a
    kwarg short-circuits the class-attribute branch entirely, so the merge
    has to intercept it independently of a subclass's ``CSS_PATH`` attribute.
    None of the real harnesses use this form today, but the mechanism must
    still compose with it. Asserts both halves: the harness's own supplied
    stylesheet still applies, and the screen sheets are still merged in
    alongside it.
    """
    custom_css = tmp_path / "harness_only.tcss"
    custom_css.write_text("#note-search-input { background: red; }\n", encoding="utf-8")

    app = ConsolidatedCSSApp(css_path=str(custom_css))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(NoteSelectionDialog([]))
        await pilot.pause()

        search_input = app.screen.query_one("#note-search-input")
        assert search_input.styles.background == Color.parse("red"), (
            "the harness's own css_path= entry did not apply -- the merge "
            "must not replace it, only bracket it with the screen sheets"
        )

        container = app.screen.query_one("#note-selection-container")
        assert container.region.width == 80, (
            "a css_path= constructor kwarg override dropped the screen "
            "sheets (region.width=" + str(container.region.width) + ")"
        )


# --- TASK-21115: dynamic first-mount vs the stale-tie-breaker parse ----------


@pytest.mark.asyncio
async def test_dynamic_first_mount_of_a_consolidated_class_keeps_its_geometry():
    """A consolidated class first-mounted AFTER boot must still get its sheet.

    Measured failure shape (TASK-21115): a bare ``Vertical`` mounts at boot,
    registering Textual's ``Vertical { width: 1fr; height: 1fr }`` at
    tie-breaker 0 (it is that widget's OWN class). A consolidated
    Vertical-subclass then first-mounts dynamically: its registration lowers
    the stored ``Vertical`` tie-breaker to -1, but Textual's ``add_source``
    does not arm a reparse for a tie-breaker change, so the parsed rules
    still carry 0 -- exactly tying the consolidated sheet's
    ``ConsoleSelectionMenu { width: auto; ... }`` (specificity (0,0,1),
    tie-breaker 0) and beating it on source order. The menu mounted
    full-screen (measured: 80x40 instead of 24x6). Compose-time mounts never
    showed it, because the widget's registration and the first parse happen
    in the same mount batch.

    ``TieAwareStylesheet`` (used by both ``TldwCli`` and
    ``ConsolidatedCSSApp``) closes that window by treating a lowered
    tie-breaker as a CSS change. Born red against the plain ``Stylesheet``.
    """
    from textual.containers import Vertical

    from tldw_chatbook.Widgets.Console.console_selection_menu import (
        ConsoleSelectionMenu,
    )

    class _DynamicMountApp(ConsolidatedCSSApp):
        def compose(self):
            yield Vertical(id="boot-time-vertical")

    app = _DynamicMountApp()
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.pause()
        await app.mount(ConsoleSelectionMenu(screen_x=2, screen_y=10))
        await pilot.pause()
        menu = app.query_one(ConsoleSelectionMenu)
        assert str(menu.styles.width) == "auto" and str(menu.styles.height) == "auto", (
            f"menu resolved {menu.styles.width}x{menu.styles.height} -- the "
            "consolidated sheet lost to a stale base-class tie-breaker on a "
            "dynamic first mount (see css/tie_aware_stylesheet.py)"
        )
        assert menu.region.width < 40 and menu.region.height < 10, (
            f"menu mounted at {menu.region} -- full-container geometry means "
            "its BUNDLED_CSS did not apply"
        )


def test_tie_aware_stylesheet_arms_reparse_when_a_tie_breaker_lowers():
    """Unit pin for the mechanism itself, independent of any app boot.

    Upstream ``Stylesheet.add_source`` keeps the LOWEST tie-breaker ever
    offered for an existing source but leaves ``_require_parse`` unset when
    lowering it -- the exact staleness the test above measures end-to-end.
    """
    from textual.css.stylesheet import Stylesheet

    from tldw_chatbook.css.tie_aware_stylesheet import TieAwareStylesheet

    for cls, expect_armed in ((Stylesheet, False), (TieAwareStylesheet, True)):
        sheet = cls()
        sheet.add_source("X { height: 1; }", read_from=("probe", "X"), tie_breaker=0)
        sheet._require_parse = False  # simulate the post-parse steady state
        sheet.add_source("X { height: 1; }", read_from=("probe", "X"), tie_breaker=-1)
        assert sheet.source[("probe", "X")].tie_breaker == -1, (
            "both classes must keep upstream's lowest-offer-wins contract"
        )
        assert sheet._require_parse is expect_armed, (
            f"{cls.__name__}: _require_parse should be {expect_armed} after a "
            "tie-breaker lowering (upstream leaves the stale parse in place; "
            "the subclass exists to arm the reparse)"
        )
        # Review fix round (TASK-21115): arming _require_parse alone is a
        # half-fix. `Stylesheet.apply` reads the `rules_map` property BEFORE
        # `self.rules`; `rules_map` short-circuits on a non-None `_rules_map`
        # without honoring the armed reparse, and the reparse `self.rules`
        # then performs replaces the re-tied source's RuleSet OBJECTS (the
        # parse cache is keyed on tie_breaker) -- so `limit_rules`, built
        # from the STALE map, filters the fresh rules out entirely for that
        # apply. Upstream's own new-source path sets BOTH flags; so must the
        # lowering path.
        sheet._rules_map = {"seeded": []}
        sheet.add_source(
            "X { height: 1; }", read_from=("probe", "X"), tie_breaker=-2
        )
        if expect_armed:
            assert sheet._rules_map is None, (
                f"{cls.__name__}: a tie-breaker lowering must also null "
                "_rules_map -- an armed reparse behind a stale map makes "
                "apply() filter the freshly parsed rules out (see "
                "css/tie_aware_stylesheet.py)"
            )
        else:
            assert sheet._rules_map is not None, (
                "upstream Stylesheet is expected to leave the stale map in "
                "place -- if this changed, the subclass may be obsolete"
            )


@pytest.mark.asyncio
async def test_dynamic_first_mount_keeps_inherited_base_defaults():
    """Review fix round (TASK-21115): the shape the first fix missed.

    `test_dynamic_first_mount_of_a_consolidated_class_keeps_its_geometry`
    covers a class whose consolidated block RESTATES width/height
    (ConsoleSelectionMenu), so a stale ``_rules_map`` was masked there: the
    sheet's own rules were already in the map. A Vertical subclass that does
    NOT restate geometry -- live shape: ``ConsoleInspectorRail``, whose
    BUNDLED_CSS styles only descendant text -- relies on INHERITING Textual's
    ``Vertical { width: 1fr; height: 1fr }`` defaults. On its dynamic first
    mount the tie-breaker lowering arms the reparse, but with ``_rules_map``
    left non-None ``apply()`` builds ``limit_rules`` from the STALE map while
    ``self.rules`` reparses to NEW RuleSet objects (the parse cache is keyed
    on tie_breaker), so the fresh Vertical default rules are filtered out and
    the widget mounts with NO base geometry at all. Red on the arm-only fix;
    green once the lowering also nulls ``_rules_map``.
    """
    from textual.containers import Vertical

    class _InheritingRail(Vertical):
        """No own CSS anywhere -- geometry must come from Vertical's defaults."""

    class _BootApp(ConsolidatedCSSApp):
        def compose(self):
            yield Vertical(id="boot-time-vertical")

    app = _BootApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await app.mount(_InheritingRail())
        await pilot.pause()
        rail = app.query_one(_InheritingRail)
        assert str(rail.styles.width) == "1fr" and str(rail.styles.height) == "1fr", (
            f"rail resolved {rail.styles.width}x{rail.styles.height} -- the "
            "inherited Vertical defaults were filtered out of its first "
            "apply() by a stale _rules_map (see css/tie_aware_stylesheet.py)"
        )
        # Both flow children split the screen: the base defaults really applied.
        assert rail.region.height == 12, (
            f"rail region {rail.region} -- expected the 1fr split of a "
            "24-row screen shared with the boot-time Vertical"
        )


# --------------------------------------------------------------------------
# task-32204: the screen-owned-split sweep guard.
#
# TASK-25812/TASK-24459 moved the Library, Console, Settings, Evals,
# Scheduling and Watchlists rules out of the boot bundle into sheets the
# OWNING SCREEN loads (its own `CSS_PATH`, or `TldwCli._SCREEN_OWNED_ROUTE_CSS`
# for the three feature sheets). A harness that pins `CSS_PATH` to the bundle
# alone and composes one of those widgets DIRECTLY -- without pushing the
# screen that owns the sheet -- has measured unstyled geometry ever since:
# `Button` falls back to `width: auto`, `Vertical` to `height: 1fr`. The
# scan below is the sweep that found them, kept executable so the next one
# cannot reappear silently.
# --------------------------------------------------------------------------

_TESTS_ROOT = Path(__file__).resolve().parent.parent

#: Split sheet -> the screen classes whose own `CSS_PATH` loads it. A harness
#: that names one of these pushes the real screen, so Textual (or
#: `TldwCli._ensure_screen_owned_css`) loads the sheet exactly as production
#: does and the harness is correct as written.
# A harness is exempt from a sheet's check when ITS OWN class body (same-module
# bases included) names the screen that owns the sheet -- pushing that screen
# loads the sheet through the screen's ``CSS_PATH``. The imported harness bases
# that push a screen in ``on_mount`` are named too, because a subclass reaches
# them only through its base list: ``ConsoleHarness`` (three module-local
# production Console harnesses), ``LibraryHarness`` (``test_library_shell``) and
# ``DestinationHarness`` (``test_destination_shells``; pushes the route's screen).
_SPLIT_SHEET_OWNERS = {
    "screen_agentic_console.tcss": (
        "ChatScreen",
        "ConsoleHarness",
        "DestinationHarness",
    ),
    "screen_agentic_library.tcss": (
        "LibraryScreen",
        "LibraryHarness",
        "DestinationHarness",
    ),
    "screen_agentic_settings.tcss": ("SettingsScreen", "DestinationHarness"),
    "screen_feature_evals.tcss": ("EvalsScreen", "DestinationHarness"),
    "screen_feature_scheduling.tcss": (
        "SchedulesScreen",
        "SchedulingScreen",
        "DestinationHarness",
    ),
    "screen_feature_watchlists.tcss": (
        "WatchlistsCollectionsScreen",
        "DestinationHarness",
    ),
}

_SELECTOR_TOKEN_RE = re.compile(r"[#.]([A-Za-z][\w-]*)")


def _selector_tokens(path: Path) -> set[str]:
    """Every id/class token that heads a rule in ``path``."""
    tokens: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("/*")[0].strip()
        if "{" in line or line.endswith(","):
            tokens.update(_SELECTOR_TOKEN_RE.findall(line.split("{")[0]))
    return tokens


def _module_string_literals(tree: ast.AST) -> str:
    """Every string constant in ``tree`` except docstrings.

    Docstrings are excluded because this repo's tests discuss selectors in
    prose constantly; only a selector a test actually *queries* is evidence
    that it composes the widget.
    """
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return " ".join(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    )


_REAL_SHEET_SEQUENCES: dict[str, tuple[Path, ...]] = {
    "APP_STYLESHEETS": tuple(APP_STYLESHEETS),
    "TldwCli.CSS_PATH": tuple(Path(entry) for entry in TldwCli.CSS_PATH),
}


class _ResolveSheetSubscripts(ast.NodeTransformer):
    """Fold ``APP_STYLESHEETS[0]`` (and friends) to the ONE path it names.

    Task-8 review, finding 3: the scan used to exempt any pin whose source
    mentioned ``APP_STYLESHEETS`` -- so ``str(APP_STYLESHEETS[0])``, the
    bundle alone spelled differently, slipped through. Indexing a real
    sequence is resolved to its element before the name-inlining below, so
    that pin now expands to the bundle path only and is reported.
    """

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        sequence = _REAL_SHEET_SEQUENCES.get(ast.unparse(node.value))
        if (
            sequence is not None
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            try:
                return ast.Constant(str(sequence[node.slice.value]))
            except IndexError:
                return self.generic_visit(node)
        return self.generic_visit(node)


def _unparse_resolved(value: ast.AST) -> str:
    return ast.unparse(_ResolveSheetSubscripts().visit(copy.deepcopy(value)))


def _expanded_css_path_source(value: ast.AST, tree: ast.AST, text: str) -> str:
    """``value``'s source with the names it references inlined.

    Harnesses spell their pin a dozen ways (``str(BUNDLE)``,
    ``_REAL_CSS_PATH``, ``OtherHarness.CSS_PATH``, ``TldwCli.CSS_PATH``,
    ``[str(p) for p in APP_STYLESHEETS]``), so the literal source segment is
    not enough to tell which sheets a pin carries. Two substitution passes
    over module-level assignments (plus same-module ``Class.CSS_PATH``
    references and the real ``TldwCli.CSS_PATH`` / ``APP_STYLESHEETS``
    sequences) resolve every shape in this repo; anything still unresolved
    simply keeps its name and is treated as carrying nothing, which can only
    produce a report, never a silent pass.
    """
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = _unparse_resolved(node.value)
        elif isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "CSS_PATH"
                    for t in stmt.targets
                ):
                    bindings[f"{node.name}.CSS_PATH"] = _unparse_resolved(stmt.value)
    for name, sequence in _REAL_SHEET_SEQUENCES.items():
        bindings[name] = " ".join(str(entry) for entry in sequence)
    # ``OtherModuleHarness.CSS_PATH``: read the real attribute off the imported
    # class rather than guessing. Unresolvable stays unresolved (a report).
    imported = {
        alias.asname or alias.name: node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("Tests.")
        for alias in node.names
    }
    for name, module_name in imported.items():
        key = f"{name}.CSS_PATH"
        if key in bindings or not re.search(rf"\b{re.escape(key)}\b", text):
            continue
        try:
            pinned = getattr(importlib.import_module(module_name), name).CSS_PATH
        except Exception:
            continue
        entries = pinned if isinstance(pinned, (list, tuple)) else [pinned]
        bindings[key] = " ".join(str(entry) for entry in entries)

    source = _unparse_resolved(value)
    for _ in range(3):
        expanded = source
        for name, replacement in bindings.items():
            expanded = re.sub(
                rf"(?<![\w.]){re.escape(name)}(?![\w])",
                lambda _match, value=replacement: value,
                expanded,
            )
        if expanded == source:
            break
        source = expanded
    return source


def _harness_source_with_bases(node: ast.ClassDef, tree: ast.AST, text: str) -> str:
    """``node``'s source plus that of every same-module base, transitively."""
    classes = {
        cls.name: cls for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef)
    }
    seen: set[str] = set()
    pending = [node]
    parts: list[str] = []
    while pending:
        cls = pending.pop()
        if cls.name in seen:
            continue
        seen.add(cls.name)
        parts.append(ast.get_source_segment(text, cls) or "")
        for base in cls.bases:
            name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
            if name in classes:
                pending.append(classes[name])
    return "\n".join(parts)


def scan_bundle_only_harnesses() -> list[tuple[str, str, str, tuple[str, ...]]]:
    """Harnesses pinned without a split sheet whose widgets need it.

    Returns:
        ``(test file, harness class, sheet, sample selectors)`` per finding,
        sorted, where the file's own queries name selectors that ONLY the
        missing sheet styles.
    """
    css_dir = BUNDLED_STYLESHEET.parent
    bundle_tokens = _selector_tokens(BUNDLED_STYLESHEET)
    split_only = {
        sheet: _selector_tokens(css_dir / sheet) - bundle_tokens
        for sheet in _SPLIT_SHEET_OWNERS
    }

    findings: list[tuple[str, str, str, tuple[str, ...]]] = []
    for py in sorted(_TESTS_ROOT.rglob("*.py")):
        text = py.read_text(encoding="utf-8")
        if "CSS_PATH" not in text:
            continue
        tree = ast.parse(text)
        literals = _module_string_literals(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign) or not any(
                    isinstance(t, ast.Name) and t.id == "CSS_PATH"
                    for t in stmt.targets
                ):
                    continue
                pin = _expanded_css_path_source(stmt.value, tree, text)
                # Scoped to the HARNESS CLASS and its same-module bases, not
                # the file (review finding 3): a module that merely imports
                # ``LibraryScreen`` must not exempt every harness in it from
                # the library-sheet check, while a subclass of a harness that
                # pushes the screen in ITS ``on_mount`` is still exempt.
                harness_text = _harness_source_with_bases(node, tree, text)
                for sheet, owners in _SPLIT_SHEET_OWNERS.items():
                    if sheet in pin:
                        continue
                    if any(
                        re.search(rf"\b{owner}\b", harness_text) for owner in owners
                    ):
                        continue
                    used = sorted(
                        token
                        for token in split_only[sheet]
                        if re.search(rf"[#.]{re.escape(token)}\b", literals)
                    )
                    if used:
                        findings.append(
                            (
                                str(py.relative_to(_TESTS_ROOT.parent)),
                                node.name,
                                sheet,
                                tuple(used[:5]),
                            )
                        )
    return sorted(findings)


def test_no_harness_composes_split_sheet_widgets_with_the_bundle_alone():
    """task-32204 AC#2: a bundle-only pin over split-sheet widgets is a bug.

    The failure names the harness and the sheet it is missing so the fix is
    a one-line swap to ``APP_STYLESHEETS`` (``Tests/UI/consolidated_css.py``),
    which is what the running app ends up with once the owning screen has
    been visited.
    """
    findings = scan_bundle_only_harnesses()
    assert not findings, "\n".join(
        f"{path}::{cls} pins the bundle without {sheet} "
        f"but queries {', '.join(used)} -- "
        "use CSS_PATH = [str(p) for p in APP_STYLESHEETS]"
        for path, cls, sheet, used in findings
    )
