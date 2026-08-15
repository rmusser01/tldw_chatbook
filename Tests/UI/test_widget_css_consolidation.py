"""TASK-15450: the widget/screen CSS consolidation and the parse-cache cliff.

Textual keeps one stylesheet source per widget class that declares
``DEFAULT_CSS`` and caches parsed sources in an ``LRUCache(64)``. A full
13-destination tour of this app used to end at **94** sources; past that cliff a
sequential scan evicts 100% of the cache, so every ``Stylesheet.parse()`` ran
fully cold -- measured 127 / 378 / 134 ms back to back -- and Textual re-runs
that parse whenever a widget class not yet seen this session first mounts.

Those declarations now live in ``BUNDLED_CSS`` / ``BUNDLED_SCREEN_CSS`` and are
lifted into generated stylesheets by ``css/build_css.py``. These tests pin the
three things that make that safe:

* the selector rewrite is *exactly* what Textual's own scoping would produce,
* the generated sheets parse, and reproduce from their Python sources,
* the tour stays under the cliff, and a modal's first open no longer reparses.
"""

from __future__ import annotations

import ast
import asyncio
import os
from pathlib import Path

import pytest
from textual.app import App
from textual.css.parse import parse
from textual.css.stylesheet import Stylesheet, StylesheetParseError
from textual.css.tokenize import tokenize_values

from tldw_chatbook.css import build_css, widget_css

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPO_ROOT / "tldw_chatbook"
_CSS_ROOT = _PACKAGE_ROOT / "css"

#: Textual's parse cache size (``textual/css/stylesheet.py``). The whole point of
#: the consolidation is to keep the live source count comfortably below it.
_PARSE_CACHE_CAPACITY = 64

_GENERATED_SHEETS = (
    build_css.WIDGET_DEFAULTS_SELF_FILENAME,
    build_css.WIDGET_DEFAULTS_SCOPED_FILENAME,
    build_css.SCREEN_CSS_SELF_FILENAME,
    build_css.SCREEN_CSS_SCOPED_FILENAME,
)


def _css_variables() -> dict:
    """Textual theme variables, as the real app resolves them."""
    return tokenize_values(App().get_css_variables())


def _class_css_blocks() -> list[tuple[str, str, str]]:
    """Every class-level CSS string literal in the package.

    Returns:
        ``(module, class_name, css)`` triples, covering the consolidated
        attributes and any ``DEFAULT_CSS``/``CSS`` that has not moved.
    """
    wanted = {"DEFAULT_CSS", "CSS", widget_css.WIDGET_ATTR, widget_css.SCREEN_ATTR}
    blocks: list[tuple[str, str, str]] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(_PACKAGE_ROOT)
        if any(part in widget_css.EXCLUDED_DIRS for part in relative.parts):
            continue
        source = path.read_text(encoding="utf-8")
        if not any(name in source for name in wanted):
            continue
        for node in ast.walk(ast.parse(source, filename=str(path))):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                names = {t.id for t in stmt.targets if isinstance(t, ast.Name)}
                if not names & wanted:
                    continue
                value = stmt.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    blocks.append((relative.as_posix(), node.name, value.value))
    return blocks


def _selector_entries(css: str, scope: str, variables: dict):
    """Flatten a parse into ``(selector chain, specificity, declarations)``."""
    entries = []
    for rule in parse(
        scope, css, ("t", ""), variable_tokens=variables, is_default_rules=True
    ):
        for selector_set in rule.selector_set:
            chain = tuple(
                (
                    selector.name,
                    selector.type.name,
                    selector.combinator.name,
                    tuple(sorted(selector.pseudo_classes)),
                )
                for selector in selector_set.selectors
            )
            entries.append((chain, selector_set.specificity, rule.styles.css))
    return entries


def test_scope_rewrite_matches_textuals_own_scoping():
    """The rewrite reproduces Textual's ``SCOPED_CSS`` prefixing, rule for rule.

    Parsed two ways -- raw text with ``scope=ClassName``, versus the self/scoped
    streams parsed unscoped -- every selector set must survive with the same
    chain and the same declarations. Specificity is allowed to differ in exactly
    one way: a selector that gained a *written* scope prefix gains ``+1`` in the
    lowest-order component, because Textual's injected scope selector carries
    ``(0, 0, 0)`` where a written type selector carries ``(0, 0, 1)``. That
    shift is what the two streams (and their differing tie-breakers) exist to
    compensate for; see ``css/widget_css.py``.
    """
    variables = _css_variables()
    blocks = _class_css_blocks()
    assert blocks, "no class-level CSS found -- the AST scan is broken"

    problems: list[str] = []
    for module, class_name, css in blocks:
        original = _selector_entries(css, class_name, variables)
        own, scoped = widget_css.split_scoped_css(css, class_name)
        rebuilt = [(e, "self") for e in _selector_entries(own, "", variables)]
        rebuilt += [(e, "scoped") for e in _selector_entries(scoped, "", variables)]

        if len(original) != len(rebuilt):
            problems.append(
                f"{module}::{class_name}: {len(original)} selector sets became "
                f"{len(rebuilt)}"
            )
            continue

        pool: dict[tuple, list] = {}
        for (chain, specificity, declarations), stream in rebuilt:
            pool.setdefault((chain, declarations), []).append((specificity, stream))
        for chain, specificity, declarations in original:
            bucket = pool.get((chain, declarations))
            if not bucket:
                problems.append(f"{module}::{class_name}: lost selector {chain}")
                continue
            new_specificity, stream = bucket.pop()
            expected = (
                specificity
                if stream == "self"
                else (specificity[0], specificity[1], specificity[2] + 1)
            )
            if tuple(new_specificity) != tuple(expected):
                problems.append(
                    f"{module}::{class_name}: {chain} specificity {specificity} -> "
                    f"{new_specificity} in the {stream} stream (expected {expected})"
                )
    assert not problems, "scope rewrite diverged from Textual:\n" + "\n".join(problems)


def test_scope_rewrite_reproduces_textuals_comma_quirk():
    """Only the LAST selector of a comma list is scoped, as Textual does it.

    ``parse_rule_set`` flushes each earlier group when it meets the comma and
    scopes only the group left over, so ``A, .b {…}`` in scoped ``DEFAULT_CSS``
    leaves ``A`` matching app-wide. Widget CSS depends on that today, so the
    rewrite keeps it; ``scope_every_selector`` opts out, and the screen sheets
    use it because they are live from boot rather than from a modal's first open.
    """
    css = ".leaked, .scoped { color: red; }\n"
    own, scoped = widget_css.split_scoped_css(css, "MyWidget")
    assert ".leaked" in own and "MyWidget" not in own
    assert "MyWidget .scoped" in scoped

    own, scoped = widget_css.split_scoped_css(
        css, "MyWidget", scope_every_selector=True
    )
    assert own.strip() == "", "nothing should stay unscoped when scoping everything"
    assert "MyWidget .leaked" in scoped and "MyWidget .scoped" in scoped


def test_selector_already_naming_its_widget_is_not_prefixed():
    """A rule that already starts with the widget's type name is left alone."""
    own, scoped = widget_css.split_scoped_css(
        "MyWidget { height: 3; }\n.other { height: 1; }\n", "MyWidget"
    )
    assert "MyWidget MyWidget" not in own
    assert "MyWidget {" in own
    assert "MyWidget .other" in scoped


def test_top_level_variable_declarations_stay_out_of_selectors():
    """``$var: value;`` at the top level is trivia, not part of the next selector.

    ``EmojiPickerScreen`` carries exactly this ("local fallbacks so this CSS
    parses without the app bundle"); folding it into the following selector
    produced an unparseable sheet.
    """
    own, scoped = widget_css.split_scoped_css(
        "$fallback: $surface;\n.thing { background: $fallback; }\n", "MyWidget"
    )
    assert "$fallback: $surface;" in scoped
    assert "MyWidget $fallback" not in scoped
    assert "MyWidget .thing" in scoped


@pytest.mark.parametrize("filename", _GENERATED_SHEETS)
def test_generated_stylesheet_parses(filename: str):
    """Each generated sheet parses -- an invalid property fails the whole sheet.

    This is not hypothetical: three ``font-size:`` declarations (no such Textual
    property) meant that opening either selection dialog raised
    ``StylesheetParseError`` out of ``_load_screen_css``, and left the app unable
    to reparse for the rest of the session.
    """
    path = _CSS_ROOT / filename
    stylesheet = Stylesheet(variables=App().get_css_variables())
    stylesheet.add_source(
        path.read_text(encoding="utf-8"),
        read_from=(str(path), ""),
        is_default_css=filename.startswith("widget_defaults"),
    )
    stylesheet.parse()


def test_base_class_blocks_precede_their_subclasses():
    """Base-class CSS must be emitted before a subclass's, as Textual ordered it.

    Textual gave each class's own ``DEFAULT_CSS`` tie-breaker 0 and its bases
    ``-(depth)``, so a subclass won ties against its base. In one generated sheet
    that ordering has to come from source order instead.
    """
    sheets = "".join(
        (_CSS_ROOT / name).read_text(encoding="utf-8")
        for name in (
            build_css.WIDGET_DEFAULTS_SELF_FILENAME,
            build_css.WIDGET_DEFAULTS_SCOPED_FILENAME,
        )
    )
    order: dict[str, int] = {}
    for index, line in enumerate(sheets.splitlines()):
        if line.startswith("/* ===== WIDGET: "):
            order.setdefault(line.split()[3], index)

    problems = []
    for module, class_name, _css in _class_css_blocks():
        if class_name not in order:
            continue
        source = (_PACKAGE_ROOT / module).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef) or node.name != class_name:
                continue
            for base in node.bases:
                base_name = ast.unparse(base).split("[")[0].split(".")[-1]
                if base_name in order and order[base_name] > order[class_name]:
                    problems.append(
                        f"{base_name} is a base of {class_name} but is emitted "
                        "after it, inverting the tie-breaker Textual gave them"
                    )
    assert not problems, "\n".join(problems)


def test_consolidated_classes_declare_no_textual_css_attribute():
    """A consolidated class must not also keep ``DEFAULT_CSS``/``CSS``.

    Both would be live at once: the generated sheet *and* the per-class source
    the consolidation exists to remove, with duplicate rules in two tiers.
    """
    consolidated = {
        block.class_name
        for attr in (widget_css.WIDGET_ATTR, widget_css.SCREEN_ATTR)
        for block in widget_css.iter_blocks(_PACKAGE_ROOT, attr)
    }
    offenders = [
        f"{module}::{class_name}"
        for module, class_name, _css in _class_css_blocks()
        if class_name in consolidated
        and _declares(module, class_name, {"DEFAULT_CSS", "CSS"})
    ]
    assert not offenders, (
        f"consolidated classes still declaring Textual CSS: {offenders}"
    )


def _declares(module: str, class_name: str, names: set[str]) -> bool:
    """Whether ``class_name`` in ``module`` assigns any of ``names``."""
    source = (_PACKAGE_ROOT / module).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and {t.id for t in stmt.targets if isinstance(t, ast.Name)} & names
            ):
                return True
    return False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_destination_tour_stays_under_the_parse_cache_cliff():
    """A full 13-destination tour must leave the source count under the cliff.

    This is the measurement the whole task exists for: before consolidation the
    same tour ended at 94 sources with ``stylesheet.parse()`` costing 127-378 ms
    per call; it must now finish well under Textual's ``LRUCache(64)`` with the
    cache warm.
    """
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        await asyncio.sleep(2)
        for key in [f"ctrl+{digit}" for digit in "1234567890"] + ["f7", "f8", "f9"]:
            await pilot.press(key)
            await pilot.pause()
            await asyncio.sleep(0.75)
        sources = len(app.stylesheet.source)
        # Headroom, not just "under": a handful of unconsolidated widget classes
        # still register their own source when first mounted.
        assert sources < _PARSE_CACHE_CAPACITY - 8, (
            f"{sources} live stylesheet sources after the tour -- Textual's parse "
            f"cache holds {_PARSE_CACHE_CAPACITY}, past which every parse is cold"
        )


def test_every_class_level_css_block_parses_as_a_stylesheet():
    """Every class-level CSS block must survive ``Stylesheet.parse()``.

    An invalid property does not fail its own declaration -- it fails the
    **whole stylesheet**, and Textual then cannot reparse for the rest of the
    session (``reparse()`` builds a fresh ``Stylesheet`` that re-adds the bad
    source and raises again). On dev, ``font-size: 10`` in the Note and
    Conversation selection dialogs meant opening either one raised
    ``StylesheetParseError`` out of ``_load_screen_css``; ``VoiceProfileDialog``
    carried the same latent bug in its ``DEFAULT_CSS``.

    This pins the *class* of defect rather than its three instances, and it
    covers blocks that were never consolidated -- including every screen that
    still declares a plain ``CSS``. Note that
    ``test_scope_rewrite_matches_textuals_own_scoping`` would not catch this:
    it calls ``textual.css.parse.parse`` directly, which collects errors onto
    ``rule.errors`` instead of raising. Only ``Stylesheet.parse`` raises.
    """
    variables = App().get_css_variables()
    failures = []
    for module, class_name, css in _class_css_blocks():
        stylesheet = Stylesheet(variables=variables)
        stylesheet.add_source(
            css, read_from=(module, class_name), is_default_css=True, scope=class_name
        )
        try:
            stylesheet.parse()
        except Exception as exc:  # noqa: BLE001 - report every offender at once
            failures.append(f"{module}::{class_name}: {type(exc).__name__}")
    assert not failures, (
        "class-level CSS that Textual cannot parse -- this fails the whole "
        "stylesheet at runtime, not just the offending rule:\n" + "\n".join(failures)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module_path, class_name",
    [
        (
            "tldw_chatbook.Widgets.Note_Widgets.note_selection_dialog",
            "NoteSelectionDialog",
        ),
        (
            "tldw_chatbook.Widgets.conversation_selection_dialog",
            "ConversationSelectionDialog",
        ),
    ],
)
async def test_selection_dialog_opens_without_a_stylesheet_error(
    module_path: str, class_name: str
):
    """Pushing either selection dialog must not raise out of the CSS machinery.

    The static guard above pins the declarations; this pins the behaviour that
    was actually broken -- on dev, ``app.push_screen(NoteSelectionDialog([]))``
    raises ``StylesheetParseError`` from ``_load_screen_css``, and the app then
    cannot reparse for the rest of the session, so every later screen with CSS
    raises too. Mounted, not static: it is the push that used to blow up.

    Both dialogs also used to carry an *unrelated* pre-existing bug -- their
    ``on_mount`` called ``Vertical.clear()``, which does not exist -- so the
    push still ended in an ``AttributeError`` from the dialog's own code.
    TASK-15992 fixed that, so this now asserts a fully clean open: the
    StylesheetParseError assertion names the CSS failure mode specifically
    (the symptom that actually poisoned the session), and any other exception
    is re-raised unconditionally.
    """
    from importlib import import_module

    from Tests.UI.consolidated_css import ConsolidatedCSSApp

    dialog_class = getattr(import_module(module_path), class_name)

    app = ConsolidatedCSSApp()
    raised: Exception | None = None
    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(dialog_class([]))
            await pilot.pause()
            # The stylesheet must still parse: on dev it is poisoned by now, and
            # `reparse()` re-adds the bad source and raises on every later screen.
            app.stylesheet.parse()
    except Exception as exc:  # noqa: BLE001 - classified immediately below
        raised = exc

    assert not isinstance(raised, StylesheetParseError), (
        f"pushing {class_name} raised StylesheetParseError -- its class-level CSS "
        "is invalid again, which fails the whole stylesheet and stops the app "
        "reparsing for the rest of the session"
    )
    if raised is not None:
        raise raised


def test_staleness_check_counts_declarations_not_mentions():
    """A module that only *mentions* the marker is not a stylesheet input.

    The boot-time staleness check reads Python modules to decide whether the
    generated sheets need rebuilding. Testing for the bare marker anywhere in
    the file was wrong in a way that bit immediately: four package modules
    discuss ``BUNDLED_CSS`` while declaring none -- including ``app.py``, which
    tripped the check via the checking function's *own docstring*. Every edit to
    any of them then re-ran the build subprocess on the next source-tree boot and
    rewrote the committed bundle's ``Generated:`` timestamp -- committed-bundle
    churn, which is the exact class of problem the CSS guard exists to prevent.

    The check must therefore agree with the builder about what an input is: the
    same set of modules ``iter_blocks`` collects from, no more and no less.
    """
    from tldw_chatbook.app import _BUNDLED_CSS_DECLARATION_RE

    assert _BUNDLED_CSS_DECLARATION_RE.search('    BUNDLED_CSS = """\n')
    assert _BUNDLED_CSS_DECLARATION_RE.search('    BUNDLED_SCREEN_CSS = """\n')
    assert _BUNDLED_CSS_DECLARATION_RE.search("    BUNDLED_CSS: str = ''\n")
    assert not _BUNDLED_CSS_DECLARATION_RE.search(
        "# the class-level BUNDLED_CSS / BUNDLED_SCREEN_CSS literals\n"
        "markers = (widget_css.WIDGET_ATTR, widget_css.SCREEN_ATTR)\n"
    )

    declaring = {
        block.module
        for attr in (widget_css.WIDGET_ATTR, widget_css.SCREEN_ATTR)
        for block in widget_css.iter_blocks(_PACKAGE_ROOT, attr)
    }
    matched = set()
    skip = {"__pycache__", *widget_css.EXCLUDED_DIRS}
    for dirpath, dirnames, filenames in os.walk(_PACKAGE_ROOT):
        # Same exclusions as the staleness walk: a vendored module mentioning
        # the marker must not trigger a rebuild the builder then ignores.
        dirnames[:] = [name for name in dirnames if name not in skip]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = Path(dirpath) / filename
            text = path.read_text(encoding="utf-8", errors="ignore")
            if _BUNDLED_CSS_DECLARATION_RE.search(text):
                matched.add(path.relative_to(_PACKAGE_ROOT).as_posix())

    assert matched == declaring, (
        "the staleness check and the builder disagree about which modules are "
        f"stylesheet inputs.\n  only the check sees: {sorted(matched - declaring)}"
        f"\n  only the builder sees: {sorted(declaring - matched)}"
    )
