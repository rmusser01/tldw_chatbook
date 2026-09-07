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
import importlib
import os
import re
from pathlib import Path

import pytest
from textual.app import App
from textual.css.errors import UnresolvedVariableError
from textual.css.parse import parse
from textual.css.stylesheet import Stylesheet, StylesheetParseError
from textual.css.tokenize import tokenize_values

from tldw_chatbook.css import build_css, widget_css

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPO_ROOT / "tldw_chatbook"
_CSS_ROOT = _PACKAGE_ROOT / "css"

#: TASK-15997: the other two trees the class-level-CSS parse guard covers.
#: Neither is a "vendored/standalone" case the way ``widget_css.EXCLUDED_DIRS``
#: means it (that exclusion is about code with its own App that never loads
#: *this* app's bundle) -- a harness ``App`` under ``Tests/`` or a runnable
#: script under ``Helper_Scripts/`` both parse their own CSS at their own
#: runtime, so an invalid property there is worth catching on exactly the same
#: grounds, and the walk below applies no directory exclusions to either.
_TESTS_ROOT = _REPO_ROOT / "Tests"
_HELPER_SCRIPTS_ROOT = _REPO_ROOT / "Helper_Scripts"

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


def _class_css_blocks(
    root: Path | None = None,
    *,
    excluded_dirs: tuple[str, ...] = widget_css.EXCLUDED_DIRS,
) -> list[tuple[str, str, str]]:
    """Every class-level CSS string literal under ``root``.

    The one block-extraction helper every parse-guard test shares (TASK-15997):
    called with no arguments it walks the ``tldw_chatbook`` package exactly as
    it always did; passing a different ``root`` (and, typically, no exclusions)
    reuses the identical AST scan for ``Tests/`` or ``Helper_Scripts/`` instead
    of forking a copy of it.

    Args:
        root: Directory to walk. Defaults to the ``tldw_chatbook`` package.
        excluded_dirs: Path components that exclude a file from the walk.
            Defaults to ``widget_css.EXCLUDED_DIRS`` (vendored/standalone code
            under the package); pass ``()`` for trees with no such carve-out.

    Returns:
        ``(module, class_name, css)`` triples, covering the consolidated
        attributes and any ``DEFAULT_CSS``/``CSS`` that has not moved.
    """
    root = _PACKAGE_ROOT if root is None else root
    wanted = {"DEFAULT_CSS", "CSS", widget_css.WIDGET_ATTR, widget_css.SCREEN_ATTR}
    blocks: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(part in excluded_dirs for part in relative.parts):
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
    leaves ``A`` matching app-wide. ``split_scoped_css``'s default keeps that
    quirk so ``test_scope_rewrite_matches_textuals_own_scoping`` can pin the
    transform against Textual's own parser; ``scope_every_selector`` opts out,
    and BOTH build-time streams now use it -- the screen sheets from TASK-15450
    and the widget-defaults sheets since TASK-15998 -- because consolidation
    made every generated sheet live from boot rather than from a first mount.
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


def test_isolate_local_variables_inlines_and_drops_the_declaration():
    """A block-local ``$var`` is substituted into its own rule, then removed.

    Mirrors ``EmojiPickerScreen``'s real shape (a local fallback aliasing a
    real app/theme variable): the app variable reference itself must survive
    untouched, only the local name disappears.
    """
    css = "$fallback: $surface;\n.thing { background: $fallback; }\n"
    isolated = widget_css.isolate_local_variables(css)
    assert "$fallback" not in isolated
    assert "background: $surface;" in isolated
    assert ".thing" in isolated


def test_isolate_local_variables_resolves_chained_local_references():
    """A local variable's value may itself reference an earlier local variable."""
    css = "$a: red;\n$b: $a;\nFoo { color: $b; }\n"
    isolated = widget_css.isolate_local_variables(css)
    assert "$a" not in isolated and "$b" not in isolated
    assert "color: red;" in isolated


def test_isolate_local_variables_leaves_unbalanced_css_an_error():
    """Malformed CSS fails loud here too, matching ``split_scoped_css``."""
    with pytest.raises(ValueError):
        widget_css.isolate_local_variables("$x: 1; Foo { color: red;")


def test_isolate_local_variables_rejects_a_forward_reference():
    """TASK-15993 review gap 2b: a forward reference must fail loudly, not
    silently accept a shape Textual itself rejects.

    Textual, parsing ``$a: $b;\\n$b: blue;\\nFoo { color: $a; }\\n`` standalone,
    raises ``UnresolvedVariableError`` immediately -- its single left-to-right
    scan hits ``$b`` inside ``$a``'s own value before ``$b: blue;`` has been
    seen. The resolver must reject the same shape rather than silently
    resolving to a dangling ``$b`` (which either defers the failure to
    sheet-parse time, or -- if an unrelated global happens to share the name
    -- resolves to the WRONG value with no error anywhere; born-red evidence
    below).
    """
    css = "$a: $b;\n$b: blue;\nFoo { color: $a; }\n"
    with pytest.raises(ValueError, match=r"\$a.*\$b"):
        widget_css.isolate_local_variables(css, scope="Foo")

    # Born-red for the *silent-wrong-value* half of the gap: a global variable
    # happens to share the forward-referenced name. Before the fix this
    # produced `Alpha { color: #800080; }` (purple) instead of "blue", with no
    # error anywhere -- exactly the silent-misresolution class this task
    # exists to eliminate, relocated into the new resolver. The fix must
    # reject it at build time rather than let it reach that state.
    shared_name_css = "$a: $fwd-shared;\n$fwd-shared: blue;\nAlpha { color: $a; }\n"
    with pytest.raises(ValueError, match=r"\$a.*\$fwd-shared"):
        widget_css.isolate_local_variables(shared_name_css, scope="Alpha")


def test_isolate_local_variables_leaves_quoted_content_untouched():
    """TASK-15993 review gap 2c-i: a ``$name``-shaped sequence inside a quoted
    string must not be rewritten -- Textual's tokenizer emits a quoted string
    as a single opaque ``string`` token, never re-scanned for a
    ``variable_ref`` (confirmed against ``textual.css.tokenize``).
    """
    css = '$a: red;\nFoo { note: "price is $a dollars"; color: $a; }\n'
    isolated = widget_css.isolate_local_variables(css, scope="Foo")
    assert 'note: "price is $a dollars";' in isolated, (
        "the quoted string's contents must survive verbatim, $a and all"
    )
    assert "color: red;" in isolated


def test_isolate_local_variables_handles_semicolon_inside_a_quoted_value():
    """TASK-15993 review gap 2c-ii: a ``;`` inside a quoted variable *value*
    must not end the declaration early and corrupt the rest of the block.
    """
    css = '$sep: "a;b";\nFoo { color: red; }\n'
    isolated = widget_css.isolate_local_variables(css, scope="Foo")
    assert "$sep" not in isolated
    assert "Foo { color: red; }" in isolated
    assert 'b";' not in isolated, "the string's tail must not leak as raw text"


def test_local_variable_definitions_do_not_leak_across_blocks():
    """TASK-15993: a block-local ``$var`` cannot silently apply to a later
    block's rules once both land in the same generated sheet.

    Textual resolves ``$variable`` references with a single left-to-right
    scan over whatever ONE STRING is handed to its parser, and a generated
    sheet concatenates every block's CSS into one such string -- so a local
    fallback meant only for parsing its own block in isolation used to stay
    "defined" for every block rendered after it (verified: this fixture
    parses *silently* -- no error -- against ``split_scoped_css`` output with
    no ``isolate_local_variables`` pre-pass, exactly reproducing the bug this
    guard pins).

    ``render_stylesheets`` (which now runs ``isolate_local_variables`` per
    block before splitting/scoping) must instead leave Bravo's reference
    genuinely unresolved: Alpha's local definition is inlined into Alpha's
    own rule and dropped, so it never appears in the emitted text for
    Bravo to inherit. Parsing with Textual's own real parser -- and real
    theme variables, so a coincidental app-var name would not mask the
    leak -- must therefore raise for the undefined name.
    """
    alpha = widget_css.BundledBlock(
        module="a.py",
        class_name="Alpha",
        lineno=1,
        css="$leak-var: red;\nAlpha { color: $leak-var; }\n",
    )
    bravo = widget_css.BundledBlock(
        module="b.py",
        class_name="Bravo",
        lineno=1,
        css="Bravo { color: $leak-var; }\n",
    )
    variables = App().get_css_variables()
    assert "leak-var" not in variables, (
        "fixture sanity: 'leak-var' must not coincide with a real theme "
        "variable, or a genuine leak could hide behind it resolving anyway"
    )

    own, scoped = widget_css.render_stylesheets([alpha, bravo], "fixture")
    # Neither block's own selector needed scoping (each already names its own
    # class), so with the default `scope_every_selector=False` everything
    # lands in the "self" stream and "scoped" is just its (non-blank) header
    # -- checking `sheet.strip()` alone would not catch that, so require an
    # actual WIDGET banner before exercising a stream.
    exercised = 0
    for stream_name, sheet in (("self", own), ("scoped", scoped)):
        if "===== WIDGET:" not in sheet:
            continue
        exercised += 1
        stylesheet = Stylesheet(variables=variables)
        stylesheet.add_source(sheet, read_from=(f"fixture-{stream_name}", ""))
        with pytest.raises(UnresolvedVariableError):
            stylesheet.parse()
    assert exercised, (
        "neither stream carried the fixture blocks -- the guard is vacuous"
    )


_BANNER_RE = re.compile(r"/\* ===== WIDGET: (\S+) \(\S+\) ===== \*/")


@pytest.mark.parametrize("filename", _GENERATED_SHEETS)
def test_generated_sheets_scope_every_selector(filename: str):
    """Every top-level selector in every generated sheet names its class first.

    TASK-15998. Textual's scoped-DEFAULT_CSS parser prefixes only the LAST
    selector of a comma list, so ``A, .b {…}`` leaves ``A`` matching app-wide.
    Per-class registration confined that leak to first-mount time; the
    consolidated sheets are live from boot, so the builder now writes the scope
    onto EVERY selector in both tiers (``scope_every_selector=True`` in
    ``build_css.py`` -- see the decision comment there for the parity evidence).
    Born red against the quirked widget-defaults build: the self sheet carried
    56 leaked selectors across 6 classes (LibraryScreen, MCPAuditMode,
    MCPToolsMode, MCPScreen, MainNavigationBar, SyncStatusWidget), and this
    guard is what keeps that set from silently growing back.
    """
    text = (_CSS_ROOT / filename).read_text(encoding="utf-8")
    parts = _BANNER_RE.split(text)
    blocks = list(zip(parts[1::2], parts[2::2]))  # (class_name, block css)
    assert blocks, f"{filename}: no WIDGET banners found -- the split is broken"

    variables = _css_variables()
    leaks = []
    checked = 0
    for class_name, css in blocks:
        for chain, _specificity, _declarations in _selector_entries(css, "", variables):
            checked += 1
            first_name, first_type, _combinator, _pseudo = chain[0]
            if not (first_type == "TYPE" and first_name == class_name):
                leaks.append(f"{filename}::{class_name}: {chain}")
    assert checked, f"{filename}: no selectors parsed -- the guard is vacuous"
    assert not leaks, (
        f"{len(leaks)} selector(s) not scoped to their declaring class -- these "
        "match app-wide from boot (Textual's comma-list quirk has crept back "
        "into the build; see build_css.build_widget_defaults):\n" + "\n".join(leaks)
    )


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


def _stream_order(path: Path) -> dict[str, int]:
    """Banner index of each class's block within ONE generated sheet.

    ``render_stylesheets`` writes one ``/* ===== WIDGET: ... */`` banner per
    class, in the order its block was rendered into this stream. That text
    order is what decides an exact-specificity tie *within this stream*: see
    ``test_base_class_blocks_precede_their_subclasses`` for why.
    """
    order: dict[str, int] = {}
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.startswith("/* ===== WIDGET: "):
            order.setdefault(line.split()[3], index)
    return order


def _module_name(module: str) -> str:
    """``iter_blocks``' package-relative POSIX path -> a dotted module name."""
    return "tldw_chatbook." + module[:-3].replace("/", ".")


def _transitive_base_pairs(
    blocks: list[widget_css.BundledBlock],
) -> list[tuple[str, str]]:
    """``(class_name, ancestor_name)`` for every real ancestor relationship
    between two consolidated widget classes.

    Imports each class and walks its actual ``__mro__`` rather than its
    syntactic bases, so a grandparent inversion is not invisible just because
    an intermediate class in the chain declares no ``BUNDLED_CSS`` of its own
    -- a syntactic-direct-bases-only scan only ever fires from a class that
    itself has CSS to check, so it can never reach past such a class.
    """
    consolidated = {block.class_name for block in blocks}
    pairs: list[tuple[str, str]] = []
    for block in blocks:
        module = importlib.import_module(_module_name(block.module))
        cls = getattr(module, block.class_name)
        for ancestor in cls.__mro__[1:]:
            if (
                ancestor.__name__ in consolidated
                and ancestor.__name__ != block.class_name
            ):
                pairs.append((block.class_name, ancestor.__name__))
    return pairs


def _ordering_problems(
    pairs: list[tuple[str, str]], streams: list[tuple[str, dict[str, int]]]
) -> list[str]:
    """Flag ``(class, base)`` pairs where ``base`` is emitted after ``class``
    *within the same stream*.

    Comparing across streams pins nothing: the self stream's tie-breaker (0)
    and the scoped stream's (``SCOPED_DEFAULTS_TIE_BREAKER``, -1,000,000)
    already decide any cross-stream tie outright, regardless of either
    block's text position, so only a same-stream comparison is load-bearing.
    """
    problems: list[str] = []
    for class_name, base_name in pairs:
        for stream_name, order in streams:
            if base_name not in order or class_name not in order:
                continue
            if order[base_name] > order[class_name]:
                problems.append(
                    f"[{stream_name}] {base_name} is a base of {class_name} but "
                    "is emitted after it, inverting the tie-breaker Textual "
                    "gave them"
                )
    return problems


def test_base_class_blocks_precede_their_subclasses():
    """Base-class CSS must be emitted before a subclass's, as Textual ordered it.

    Textual gave each class's own ``DEFAULT_CSS`` tie-breaker 0 and its bases
    ``-(depth)``: a subclass won a specificity tie against its base outright,
    by that numeric comparison, regardless of source order
    (``Styles.extract_rules``/``Stylesheet._check_and_refresh``).

    The consolidated scheme collapses every class's self-stream rules onto
    ONE shared tie-breaker (0, ``build_css.widget_defaults_sources``) and
    every scoped-stream rule onto another shared one
    (``SCOPED_DEFAULTS_TIE_BREAKER``). Two same-stream rules that still tie on
    specificity therefore fall through to Textual's *next* tie-break: on an
    exact tie, the LAST rule in source order wins (the stylesheet scans rules
    in reverse and ``max()`` keeps the first-seen maximum). So within one
    stream, a base's block must sit *before* its subclass's -- and only a
    same-stream comparison means anything: a pair that straddles streams is
    already decided outright by the differing tie-breakers, so comparing
    their raw text positions (as this test used to, via a naive concatenation
    of both streams) pins nothing. See TASK-15994.
    """
    blocks = widget_css.iter_blocks(_PACKAGE_ROOT, widget_css.WIDGET_ATTR)
    self_order = _stream_order(_CSS_ROOT / build_css.WIDGET_DEFAULTS_SELF_FILENAME)
    scoped_order = _stream_order(_CSS_ROOT / build_css.WIDGET_DEFAULTS_SCOPED_FILENAME)
    pairs = _transitive_base_pairs(blocks)
    problems = _ordering_problems(
        pairs, [("self", self_order), ("scoped", scoped_order)]
    )
    assert not problems, "\n".join(problems)


def test_ordering_check_catches_a_cross_stream_conflation_the_old_index_missed():
    """TASK-15994 AC3, defect 1 (born-red): the retired algorithm merged both
    streams into ONE index by scanning their concatenation and keeping only
    each class's FIRST occurrence (``order.setdefault``). Since the self
    stream was concatenated whole before the scoped stream, any class with a
    self-stream block had its scoped-stream position silently discarded.

    Seed exactly that: ``BaseWidget``/``SubWidget`` are correctly ordered in
    the self stream, but ``SubWidget``'s scoped block sits BEFORE
    ``BaseWidget``'s -- a real inversion the old algorithm could never see.
    """
    self_order = {"BaseWidget": 1, "SubWidget": 5}
    scoped_order = {"SubWidget": 8, "BaseWidget": 20}
    pairs = [("SubWidget", "BaseWidget")]

    # Reconstruct the retired algorithm's merged index: every self-stream
    # line preceded every scoped-stream one in the concatenation, so a class
    # present in the self stream permanently shadowed its own scoped-stream
    # position via `order.setdefault(class_name, index)`.
    old_order: dict[str, int] = {}
    for name, index in self_order.items():
        old_order.setdefault(name, index)
    for name, index in scoped_order.items():
        old_order.setdefault(name, index)
    old_problems = [
        (base, cls)
        for cls, base in pairs
        if base in old_order and cls in old_order and old_order[base] > old_order[cls]
    ]
    assert old_problems == [], (
        "setup invalid -- the retired merged-index algorithm should pass this "
        f"over silently, but flagged {old_problems}"
    )

    new_problems = _ordering_problems(
        pairs, [("self", self_order), ("scoped", scoped_order)]
    )
    assert new_problems, (
        "the per-stream check must catch the scoped-stream inversion the "
        "retired merged-index check missed"
    )


def test_ordering_check_catches_a_transitive_base_inversion_the_old_scan_missed():
    """TASK-15994 AC3, defect 2 (born-red): the retired algorithm only
    inspected a class's own SYNTACTIC (direct) bases, so a base that is only
    a base-of-a-base was invisible whenever the intermediate class declared
    no CSS of its own -- there was never a ``class_name`` entry for it to
    check its own direct bases from. Seed exactly that with a real
    inheritance chain.
    """

    class Grandparent:
        pass

    class Middle(Grandparent):
        """Declares no BUNDLED_CSS of its own -- invisible to a scan that only
        ever inspects the direct bases of a class that DOES have CSS."""

    class Grandchild(Middle):
        pass

    consolidated = {"Grandparent", "Grandchild"}  # Middle is not consolidated

    # The retired algorithm's check, reconstructed: for the one class in
    # `consolidated` that even has an ancestor in the chain (Grandchild),
    # look only at its syntactic __bases__ -- equivalent to what `ast` would
    # see, since these classes are declared with ordinary Python syntax.
    old_flagged_bases = {
        base.__name__ for base in Grandchild.__bases__ if base.__name__ in consolidated
    }
    assert old_flagged_bases == set(), (
        "setup invalid -- Grandparent must not be a direct base of Grandchild"
    )

    # The new transitive walk finds Grandparent regardless.
    new_pairs = [
        (Grandchild.__name__, ancestor.__name__)
        for ancestor in Grandchild.__mro__[1:]
        if ancestor.__name__ in consolidated
    ]
    assert ("Grandchild", "Grandparent") in new_pairs, (
        "the transitive MRO walk must still find Grandparent as an ancestor "
        "of Grandchild even though it is not a direct base"
    )

    # Base emitted AFTER its (transitive) subclass -- a real inversion within
    # one stream -- is exactly what the strengthened pairs must let us catch.
    order = {"Grandparent": 10, "Grandchild": 2}
    problems = _ordering_problems(new_pairs, [("self", order)])
    assert problems, "must flag Grandparent emitted after its descendant Grandchild"


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

    TASK-15997 swept ``Tests/`` and ``Helper_Scripts/`` for the same defect
    class -- see ``test_class_level_css_blocks_outside_the_package_parse``,
    which shares ``_assert_class_css_blocks_parse`` with this test rather than
    forking the check.
    """
    _assert_class_css_blocks_parse(_class_css_blocks(), allowlist={})


#: TASK-15997: deliberate invalid-CSS fixtures that exist to negative-test the
#: CSS machinery itself (e.g. a harness asserting Textual raises
#: ``StylesheetParseError`` when the fixture is pushed). This is an EXPLICIT
#: per-``(module, class_name)`` allowlist, not a directory or filename skip --
#: a fixture that needs to stay invalid must be named here, with a reason, or
#: the sweep below fails on it. Empty today: no such fixture exists yet under
#: ``Tests/`` or ``Helper_Scripts/``; ``test_css_parse_guard_catches_seeded_
#: invalid_blocks_in_newly_covered_trees`` proves the mechanism itself works
#: without needing a real one committed.
_KNOWN_INVALID_CSS_FIXTURES: dict[tuple[str, str], str] = {}


def _assert_class_css_blocks_parse(
    blocks: list[tuple[str, str, str]],
    *,
    allowlist: dict[tuple[str, str], str],
) -> None:
    """Run every ``(module, class_name, css)`` block through ``Stylesheet.parse()``.

    Shared by every tree the guard covers (the package, ``Tests/``,
    ``Helper_Scripts/``) so a fix to the check itself applies everywhere at
    once, rather than three copies drifting apart.

    Args:
        blocks: ``(module, class_name, css)`` triples from ``_class_css_blocks``.
        allowlist: ``(module, class_name) -> reason`` for fixtures that are
            *deliberately* invalid CSS (negative tests of the CSS machinery
            itself). An allowlisted block is excluded from the failure list,
            but must actually still fail to parse today -- a block that starts
            parsing cleanly again (fixed, or the fixture rewritten) makes its
            entry stale, which is also asserted here rather than left to rot.
    """
    variables = App().get_css_variables()
    failures: list[str] = []
    stale_allowlist_entries: list[str] = []
    seen_keys: set[tuple[str, str]] = set()
    for module, class_name, css in blocks:
        key = (module, class_name)
        seen_keys.add(key)
        stylesheet = Stylesheet(variables=variables)
        stylesheet.add_source(
            css, read_from=(module, class_name), is_default_css=True, scope=class_name
        )
        try:
            stylesheet.parse()
        except Exception as exc:  # noqa: BLE001 - report every offender at once
            if key not in allowlist:
                failures.append(f"{module}::{class_name}: {type(exc).__name__}")
        else:
            if key in allowlist:
                stale_allowlist_entries.append(f"{module}::{class_name}")
    assert not failures, (
        "class-level CSS that Textual cannot parse -- this fails the whole "
        "stylesheet at runtime, not just the offending rule:\n" + "\n".join(failures)
    )
    assert not stale_allowlist_entries, (
        "allowlisted invalid-CSS fixture now parses cleanly -- remove its "
        "entry from _KNOWN_INVALID_CSS_FIXTURES:\n" + "\n".join(stale_allowlist_entries)
    )
    unused_allowlist_entries = [
        f"{module}::{class_name}: {reason}"
        for (module, class_name), reason in allowlist.items()
        if (module, class_name) not in seen_keys
    ]
    assert not unused_allowlist_entries, (
        "allowlist entry does not match any scanned block -- the fixture was "
        "renamed, moved, or removed; update or delete the entry:\n"
        + "\n".join(unused_allowlist_entries)
    )


@pytest.mark.parametrize(
    "root",
    [
        pytest.param(_TESTS_ROOT, id="Tests"),
        pytest.param(_HELPER_SCRIPTS_ROOT, id="Helper_Scripts"),
    ],
)
def test_class_level_css_blocks_outside_the_package_parse(root: Path):
    """TASK-15997: the same parse guard, swept over ``Tests/`` and ``Helper_Scripts/``.

    ``test_every_class_level_css_block_parses_as_a_stylesheet`` only ever
    walked the ``tldw_chatbook`` package -- nothing checked a test harness's
    own ``App.CSS`` or a ``Helper_Scripts/`` example for the identical defect
    class (an invalid property poisons the *whole* stylesheet, not just its
    own declaration). Swept once while adding this test: 28 class-level CSS
    blocks under ``Tests/`` (test-harness ``App``/``Screen`` subclasses'
    ``CSS`` -- none declare ``BUNDLED_CSS``/``BUNDLED_SCREEN_CSS``, since the
    consolidation only applies inside the package), 0 under
    ``Helper_Scripts/`` (its custom-splash-card examples are ``.toml`` data,
    not Python classes, and the one ``.py`` helper there declares no
    class-level CSS attribute at all). All 28 parsed cleanly -- no crashers
    found in either tree.
    """
    blocks = _class_css_blocks(root, excluded_dirs=())
    _assert_class_css_blocks_parse(blocks, allowlist=_KNOWN_INVALID_CSS_FIXTURES)


def test_css_parse_guard_catches_seeded_invalid_blocks_in_newly_covered_trees(
    tmp_path,
):
    """Born-red proof: the Tests/Helper_Scripts sweep is not a no-op.

    Both real trees currently come back clean (see
    ``test_class_level_css_blocks_outside_the_package_parse``), which on its
    own does not distinguish "nothing is broken" from "the check never ran".
    Seed one throwaway module per newly-covered tree with the exact defect
    class TASK-15450 found in the package (``font-size:`` is not a Textual
    property) and assert the shared check -- the same
    ``_assert_class_css_blocks_parse`` the real sweep uses -- raises for both.
    """
    invalid_css_module = (
        "from textual.app import App\n\n\n"
        "class _SeededInvalidCssHarness(App):\n"
        '    CSS = """\n'
        "    Widget { font-size: 10; }\n"
        '    """\n'
    )
    for tree_name in ("Tests", "Helper_Scripts"):
        tree_root = tmp_path / tree_name
        tree_root.mkdir()
        (tree_root / "seeded_invalid_css.py").write_text(invalid_css_module)

        blocks = _class_css_blocks(tree_root, excluded_dirs=())
        assert [b[:2] for b in blocks] == [
            ("seeded_invalid_css.py", "_SeededInvalidCssHarness")
        ], f"extractor did not find the seeded block under {tree_name}/"

        with pytest.raises(AssertionError, match="StylesheetParseError"):
            _assert_class_css_blocks_parse(blocks, allowlist={})


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


# ---------------------------------------------------------------------------
# TASK-21115: the static allowlist ratchet.
#
# Every class-level ``DEFAULT_CSS`` (and every Screen's class-level ``CSS``)
# registers one more stylesheet source at its first mount, and Textual's parse
# cache is an ``LRUCache(64)``: past that cliff every first-mount of a
# not-yet-seen class re-pays a full cold ``Stylesheet.parse()`` (measured
# 127-378 ms) for the rest of the session. The consolidation exists to keep
# the live source count under that cliff, but the only guard that noticed new
# declarations was the *integration tour* above -- slow, and red for stretches
# (TASK-21106) exactly when accretion was fastest (34 new declarations in the
# 2 months after TASK-15450 shipped; the 2026-08-22 holistic review measured a
# feature-rich session crossing the cliff).
#
# This ratchet is the fast, boot-free version of the invariant: an AST walk
# over the package fails on any class-level ``DEFAULT_CSS``/``CSS`` binding
# that is not explicitly allowlisted below. The allowlist is a snapshot of the
# declarations that existed when the ratchet landed -- it may only SHRINK
# (entries whose declaration was converted or deleted are flagged as stale so
# the recorded debt cannot silently rot).
# ---------------------------------------------------------------------------

_RATCHETED_CSS_ATTRS = ("DEFAULT_CSS", "CSS")


def _textual_css_declarations(
    root: Path | None = None,
    *,
    excluded_dirs: tuple[str, ...] = widget_css.EXCLUDED_DIRS,
) -> list[tuple[str, str, str]]:
    """Every class-level ``DEFAULT_CSS``/``CSS`` *binding* under ``root``.

    Deliberately broader than ``_class_css_blocks``: that helper collects only
    plain-string ``ast.Assign`` values (all it needs to parse-check CSS text),
    which would let an annotated assignment (``DEFAULT_CSS: str = ...``) or a
    non-literal value (an f-string, a concatenation, a name) slip past the
    ratchet -- and a non-literal ``DEFAULT_CSS`` still registers a stylesheet
    source at runtime exactly like a literal one.

    Returns:
        ``(module, class_name, attr)`` triples, sorted.
    """
    root = _PACKAGE_ROOT if root is None else root
    found: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(part in excluded_dirs for part in relative.parts):
            continue
        source = path.read_text(encoding="utf-8")
        if not any(name in source for name in _RATCHETED_CSS_ATTRS):
            continue
        for node in ast.walk(ast.parse(source, filename=str(path))):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                names: set[str] = set()
                if isinstance(stmt, ast.Assign):
                    names = {t.id for t in stmt.targets if isinstance(t, ast.Name)}
                elif isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    names = {stmt.target.id}
                for attr in names & set(_RATCHETED_CSS_ATTRS):
                    found.append((relative.as_posix(), node.name, attr))
    return sorted(found)


#: The class-level CSS declarations that existed when the ratchet landed
#: (TASK-21115) -- the pre-TASK-15450 residue the consolidation deliberately
#: left in place. Additions require review: the sanctioned default for new
#: widget CSS is ``BUNDLED_CSS`` / ``BUNDLED_SCREEN_CSS`` (see the module
#: docstring), which costs zero live stylesheet sources. Removals (after
#: converting or deleting a declaration) are mandatory -- a stale entry fails
#: the ratchet.
_UNCONSOLIDATED_CSS_ALLOWLIST: frozenset[tuple[str, str, str]] = frozenset([
    ("UI/CCP_Modules/ccp_loading_indicators.py", "InlineLoadingIndicator", "DEFAULT_CSS"),
    ("UI/ChatbookCreationWindow.py", "ChatbookCreationWindow", "DEFAULT_CSS"),
    ("UI/ChatbookExportManagementWindow.py", "ChatbookExportManagementWindow", "DEFAULT_CSS"),
    ("UI/Chatbooks_Window_Improved.py", "ChatbookCard", "DEFAULT_CSS"),
    ("UI/Chatbooks_Window_Improved.py", "ChatbooksWindowImproved", "DEFAULT_CSS"),
    ("UI/Chatbooks_Window_Improved.py", "EmptyStateWidget", "DEFAULT_CSS"),
    ("UI/Chatbooks_Window.py", "ChatbooksWindow", "DEFAULT_CSS"),
    ("UI/ChatbookTemplatesWindow.py", "ChatbookTemplatesWindow", "DEFAULT_CSS"),
    ("UI/Console_Modules/provider_continuation_recovery.py", "ProviderContinuationRecoveryCallout", "DEFAULT_CSS"),
    ("UI/Dictation_Window_Improved.py", "ImprovedDictationWindow", "DEFAULT_CSS"),
    ("UI/Library_Modules/prompt_collection_manager_modal.py", "PromptCollectionManagerModal", "DEFAULT_CSS"),
    ("UI/MCP_Modules/mcp_profile_form.py", "MCPImportPanel", "DEFAULT_CSS"),
    ("UI/MCP_Modules/mcp_profile_form.py", "MCPProfileForm", "DEFAULT_CSS"),
    ("UI/MCP_Modules/mcp_schema_form.py", "MCPSchemaForm", "DEFAULT_CSS"),
    ("UI/MCP_Modules/mcp_server_mutations.py", "MCPServerMutationsPanel", "DEFAULT_CSS"),
    ("UI/MediaWindow_v2.py", "MediaWindow", "DEFAULT_CSS"),
    ("UI/Navigation/nav_overflow_menu.py", "NavOverflowMenu", "DEFAULT_CSS"),
    ("UI/Outputs_Panel.py", "OutputsPanel", "DEFAULT_CSS"),
    ("UI/Screens/scheduling/forms/reminder_form.py", "ReminderForm", "DEFAULT_CSS"),
    ("UI/Screens/skills_screen.py", "SkillTrustBootstrapModal", "DEFAULT_CSS"),
    ("UI/Screens/skills_screen.py", "SkillTrustPassphraseModal", "DEFAULT_CSS"),
    ("UI/Sharing_Panel.py", "SharingPanel", "DEFAULT_CSS"),
    ("UI/Speech/speech_clone_setup.py", "SpeechCloneSetup", "DEFAULT_CSS"),
    ("UI/stts_profile_library.py", "STTSProfileLibrary", "DEFAULT_CSS"),
    ("UI/stts_profile_library.py", "TTSCloneProfileSaveReviewModal", "DEFAULT_CSS"),
    ("UI/stts_profile_library.py", "TTSProfileDeleteModal", "DEFAULT_CSS"),
    ("UI/stts_profile_library.py", "TTSProfileEditorModal", "DEFAULT_CSS"),
    ("UI/stts_profile_library.py", "TTSProfileNameModal", "DEFAULT_CSS"),
    ("UI/STTS_Window.py", "AudioBookGenerationWidget", "DEFAULT_CSS"),
    ("UI/STTS_Window.py", "STTSWindow", "DEFAULT_CSS"),
    ("UI/STTS_Window.py", "VoiceProfilePickerModal", "DEFAULT_CSS"),
    ("UI/Study_Window.py", "AnkiFlashcardsWidget", "DEFAULT_CSS"),
    ("UI/Study_Window.py", "CourseCreationWidget", "DEFAULT_CSS"),
    ("UI/Study_Window.py", "LearningMapWidget", "DEFAULT_CSS"),
    ("UI/Study_Window.py", "MindmapsWidget", "DEFAULT_CSS"),
    ("UI/Study_Window.py", "QuizzesWidget", "DEFAULT_CSS"),
    ("UI/Study_Window.py", "StructuredLearningWidget", "DEFAULT_CSS"),
    ("UI/Study_Window.py", "StudyGuideWidget", "DEFAULT_CSS"),
    ("UI/Study_Window.py", "StudyWindow", "DEFAULT_CSS"),
    ("UI/Tools_Settings_Window.py", "ConfirmDisableDialog", "DEFAULT_CSS"),
    ("UI/Tools_Settings_Window.py", "ToolsSettingsWindow", "DEFAULT_CSS"),
    ("UI/Voice_Cloning_Window.py", "VoiceCloningWindow", "DEFAULT_CSS"),
    ("UI/Wizards/BaseWizard.py", "WizardContainer", "DEFAULT_CSS"),
    ("UI/Wizards/BaseWizard.py", "WizardProgress", "DEFAULT_CSS"),
    ("UI/Wizards/BaseWizard.py", "WizardScreen", "DEFAULT_CSS"),
    ("UI/Workbench/help.py", "WorkbenchHelpPanel", "DEFAULT_CSS"),
    ("Utils/widget_helpers.py", "FeatureNotAvailableDialog", "DEFAULT_CSS"),
    ("Widgets/audio_troubleshooting_dialog.py", "AudioTroubleshootingDialog", "DEFAULT_CSS"),
    ("Widgets/base_components.py", "ActionButtonRow", "DEFAULT_CSS"),
    ("Widgets/base_components.py", "ConfigurationForm", "DEFAULT_CSS"),
    ("Widgets/base_components.py", "NavigationButton", "DEFAULT_CSS"),
    ("Widgets/base_components.py", "SectionContainer", "DEFAULT_CSS"),
    ("Widgets/base_components.py", "StatusDisplay", "DEFAULT_CSS"),
    ("Widgets/cancel_confirmation_dialog.py", "CancelConfirmationDialog", "DEFAULT_CSS"),
    ("Widgets/Chat_Widgets/chat_handoff_card.py", "ChatHandoffCard", "DEFAULT_CSS"),
    ("Widgets/Chat_Widgets/chat_message_enhanced.py", "ChatMessageEnhanced", "DEFAULT_CSS"),
    ("Widgets/Chat_Widgets/chat_message.py", "ChatMessage", "DEFAULT_CSS"),
    ("Widgets/Chat_Widgets/chat_shell_bar.py", "ChatShellBar", "DEFAULT_CSS"),
    ("Widgets/chunk_preview_modal.py", "ChunkPreviewModal", "DEFAULT_CSS"),
    ("Widgets/Coding_Widgets/repo_tree_widgets.py", "TreeNode", "DEFAULT_CSS"),
    ("Widgets/Coding_Widgets/repo_tree_widgets.py", "TreeView", "DEFAULT_CSS"),
    ("Widgets/confirmation_dialog.py", "ConfirmationDialog", "DEFAULT_CSS"),
    ("Widgets/Console/console_character_picker_modal.py", "ConsoleCharacterPickerModal", "DEFAULT_CSS"),
    ("Widgets/Console/console_composer_menu_modal.py", "ConsoleComposerMenuModal", "DEFAULT_CSS"),
    ("Widgets/Console/console_edit_message_modal.py", "ConsoleEditMessageModal", "DEFAULT_CSS"),
    ("Widgets/Console/console_generate_image_modal.py", "ConsoleGenerateImageModal", "DEFAULT_CSS"),
    ("Widgets/Console/console_image_viewer_modal.py", "ConsoleImageViewerModal", "DEFAULT_CSS"),
    ("Widgets/Console/console_model_popover.py", "ConsoleModelPopover", "DEFAULT_CSS"),
    ("Widgets/Console/console_prompt_improve_view.py", "ConsolePromptImproveView", "DEFAULT_CSS"),
    ("Widgets/Console/console_prompt_queue_modal.py", "ConsolePromptQueueModal", "DEFAULT_CSS"),
    ("Widgets/Console/console_prompts_modal.py", "ConsolePromptsModal", "DEFAULT_CSS"),
    ("Widgets/Console/console_rag_settings_modal.py", "ConsoleRagSettingsModal", "DEFAULT_CSS"),
    ("Widgets/Console/console_rename_session_modal.py", "ConsoleRenameSessionModal", "DEFAULT_CSS"),
    ("Widgets/Console/console_rewind_modal.py", "ConsoleRewindModal", "DEFAULT_CSS"),
    ("Widgets/Console/console_save_as_modal.py", "ConsoleSaveAsModal", "DEFAULT_CSS"),
    ("Widgets/Console/console_session_switcher_modal.py", "ConsoleSessionSwitcherModal", "DEFAULT_CSS"),
    ("Widgets/Console/console_settings_modal.py", "ConsoleSettingsModal", "DEFAULT_CSS"),
    ("Widgets/Console/console_transcript.py", "ConsoleMarkdownMessage", "DEFAULT_CSS"),
    ("Widgets/Console/console_video_capacity_modal.py", "ConsoleVideoCapacityModal", "DEFAULT_CSS"),
    ("Widgets/Console/console_workspace_switcher_modal.py", "ConsoleWorkspaceRenameModal", "DEFAULT_CSS"),
    ("Widgets/Console/console_workspace_switcher_modal.py", "ConsoleWorkspaceSwitcherModal", "DEFAULT_CSS"),
    ("Widgets/delete_confirmation_dialog.py", "DeleteConfirmationDialog", "DEFAULT_CSS"),
    ("Widgets/destination_workbench.py", "DestinationWorkbench", "DEFAULT_CSS"),
    ("Widgets/dictation_performance_widget.py", "DictationPerformanceWidget", "DEFAULT_CSS"),
    ("Widgets/document_generation_modal.py", "DocumentGenerationModal", "DEFAULT_CSS"),
    ("Widgets/enhanced_file_picker.py", "DirectorySearch", "DEFAULT_CSS"),
    ("Widgets/enhanced_file_picker.py", "EnhancedFileDialog", "DEFAULT_CSS"),
    ("Widgets/enhanced_file_picker.py", "PathBreadcrumbs", "DEFAULT_CSS"),
    ("Widgets/feedback_dialog.py", "FeedbackDialog", "DEFAULT_CSS"),
    ("Widgets/Library/library_file_notes_git_panel.py", "LibraryFileNotesGitPanel", "DEFAULT_CSS"),
    ("Widgets/Library/library_file_notes_git_panel.py", "PushDestinationAuthorizationDialog", "DEFAULT_CSS"),
    ("Widgets/Library/library_file_notes_git_panel.py", "PushEndpointDetailsDialog", "DEFAULT_CSS"),
    ("Widgets/Library/library_file_notes_workspace.py", "FileNotesConflictCompareDialog", "DEFAULT_CSS"),
    ("Widgets/Library/library_file_notes_workspace.py", "FileNotesRootDetailsDialog", "DEFAULT_CSS"),
    ("Widgets/Library/library_file_notes_workspace.py", "LibraryFileNotesWorkspace", "DEFAULT_CSS"),
    ("Widgets/Library/library_ingest_canvas.py", "LibraryIngestCanvas", "DEFAULT_CSS"),
    ("Widgets/Library/library_media_content.py", "LibraryMediaContentSearchControls", "DEFAULT_CSS"),
    ("Widgets/Library/library_media_viewer.py", "LibraryMediaViewer", "DEFAULT_CSS"),
    ("Widgets/Library/prompt_delete_confirmation_modal.py", "PromptDeleteConfirmationModal", "DEFAULT_CSS"),
    ("Widgets/Media/media_list_panel.py", "MediaListPanel", "DEFAULT_CSS"),
    ("Widgets/Media/media_navigation_panel.py", "MediaNavigationPanel", "DEFAULT_CSS"),
    ("Widgets/Media/media_search_panel.py", "MediaSearchPanel", "DEFAULT_CSS"),
    ("Widgets/Media/media_viewer_panel.py", "DeleteConfirmDialog", "DEFAULT_CSS"),
    ("Widgets/Media/media_viewer_panel.py", "MediaViewerPanel", "DEFAULT_CSS"),
    ("Widgets/model_search_picker.py", "ModelSearchPicker", "DEFAULT_CSS"),
    ("Widgets/ModelArtifacts/activation_controls.py", "ModelActivationControls", "DEFAULT_CSS"),
    ("Widgets/ModelArtifacts/install_modal.py", "ModelInstallModal", "DEFAULT_CSS"),
    ("Widgets/ModelArtifacts/local_gguf_import.py", "LocalGGUFImportConsentModal", "DEFAULT_CSS"),
    ("Widgets/ModelArtifacts/local_gguf_import.py", "LocalGGUFImportControls", "DEFAULT_CSS"),
    ("Widgets/Note_Widgets/note_creation_modal.py", "NoteCreationModal", "DEFAULT_CSS"),
    ("Widgets/password_dialog.py", "EncryptionSetupDialog", "DEFAULT_CSS"),
    ("Widgets/password_dialog.py", "PasswordDialog", "DEFAULT_CSS"),
    ("Widgets/Persona_Widgets/character_tts_portability_dialogs.py", "CharacterTTSExistingAssignmentDialog", "DEFAULT_CSS"),
    ("Widgets/Persona_Widgets/character_tts_portability_dialogs.py", "CharacterTTSProfileCollisionDialog", "DEFAULT_CSS"),
    ("Widgets/Persona_Widgets/conversation_attach_picker.py", "ConversationAttachPicker", "DEFAULT_CSS"),
    ("Widgets/Persona_Widgets/dictionary_attach_picker.py", "DictionaryAttachPicker", "DEFAULT_CSS"),
    ("Widgets/Persona_Widgets/dictionary_picker.py", "DictionaryPicker", "DEFAULT_CSS"),
    ("Widgets/Persona_Widgets/tag_filter_picker.py", "TagFilterPicker", "DEFAULT_CSS"),
    ("Widgets/Persona_Widgets/world_book_picker.py", "WorldBookPicker", "DEFAULT_CSS"),
    ("Widgets/Prompts/prompt_block_editor.py", "PromptBlockEditor", "DEFAULT_CSS"),
    ("Widgets/Settings_Widgets/server_switch_modal.py", "ServerSwitchModal", "DEFAULT_CSS"),
    ("Widgets/status_widget.py", "EnhancedStatusWidget", "DEFAULT_CSS"),
    ("Widgets/Study/quiz_session_widget.py", "QuizSessionWidget", "DEFAULT_CSS"),
    ("Widgets/Study/study_dashboard.py", "StudyDashboard", "DEFAULT_CSS"),
    ("Widgets/Tamagotchi/base_tamagotchi.py", "BaseTamagotchi", "DEFAULT_CSS"),
    ("Widgets/TTS/chapter_editor_widget.py", "ChapterEditorWidget", "DEFAULT_CSS"),
    ("Widgets/TTS/character_voice_widget.py", "CharacterVoiceWidget", "DEFAULT_CSS"),
    ("Widgets/voice_input_widget.py", "VoiceInputWidget", "DEFAULT_CSS"),
    ("Widgets/voice_profile_dialog.py", "VoiceProfileDialog", "DEFAULT_CSS"),
])


def test_class_level_css_stays_within_the_allowlist():
    """No new ``DEFAULT_CSS``/``CSS`` outside the allowlist; no stale entries.

    TASK-21115. Static and boot-free on purpose: the integration tour above is
    the *measurement*, but it is slow and has spent red stretches during which
    34 new ``DEFAULT_CSS`` declarations accreted unnoticed -- enough that a
    feature-rich session (~10 distinct modal opens) crossed the LRUCache(64)
    parse-cache cliff. This test makes the invariant cheap enough to run on
    every change.
    """
    declared = set(_textual_css_declarations())
    assert declared, "no class-level CSS declarations found -- the walk is broken"

    offenders = sorted(declared - _UNCONSOLIDATED_CSS_ALLOWLIST)
    assert not offenders, (
        f"{len(offenders)} class-level CSS declaration(s) outside the "
        "allowlist:\n"
        + "\n".join(f"  {module}::{name}.{attr}" for module, name, attr in offenders)
        + "\n\nEvery class-level DEFAULT_CSS/CSS registers one more stylesheet "
        "source at first mount, and Textual's parse cache holds 64 -- past the "
        "cliff every first-mount of an unseen class re-pays a full cold parse "
        "(~150-450 ms) for the rest of the session. Two sanctioned options:\n"
        "  1. (default) Ride the bundle: rename DEFAULT_CSS -> BUNDLED_CSS "
        "(widget-defaults tier), or a Screen's CSS -> BUNDLED_SCREEN_CSS "
        "(app-CSS tier), keep the block a plain string literal, then run "
        "`python tldw_chatbook/css/build_css.py` and commit the regenerated "
        "sheets in css/ together with your source change.\n"
        "  2. (reviewed exception only) add the (module, class, attr) triple "
        "to _UNCONSOLIDATED_CSS_ALLOWLIST in this file, with the reason in "
        "your PR -- e.g. the CSS genuinely cannot be a plain string literal, "
        "or the widget belongs to a standalone app that never loads this "
        "app's bundle."
    )

    stale = sorted(_UNCONSOLIDATED_CSS_ALLOWLIST - declared)
    assert not stale, (
        "allowlist entries with no matching declaration (converted, renamed "
        "or deleted) -- the ratchet only moves down, remove them from "
        "_UNCONSOLIDATED_CSS_ALLOWLIST:\n"
        + "\n".join(f"  {module}::{name}.{attr}" for module, name, attr in stale)
    )


def test_css_ratchet_walker_sees_declarations_the_block_extractor_skips(tmp_path):
    """Born-red proof the ratchet cannot be dodged with a non-literal block.

    ``_class_css_blocks`` collects only plain-string ``ast.Assign`` values, so
    an annotated assignment or an f-string ``DEFAULT_CSS`` is invisible to it
    -- but still registers a live stylesheet source at runtime. The ratchet's
    own walker must see all three shapes.
    """
    (tmp_path / "dodgy.py").write_text(
        "class AnnotatedDeclaration:\n"
        "    DEFAULT_CSS: str = 'X { height: 1; }'\n\n\n"
        "class FStringDeclaration:\n"
        "    DEFAULT_CSS = f'X {{ width: {1}; }}'\n\n\n"
        "class ScreenCssDeclaration:\n"
        "    CSS = 'X { height: 1; }'\n",
        encoding="utf-8",
    )
    seen = _textual_css_declarations(tmp_path, excluded_dirs=())
    assert seen == [
        ("dodgy.py", "AnnotatedDeclaration", "DEFAULT_CSS"),
        ("dodgy.py", "FStringDeclaration", "DEFAULT_CSS"),
        ("dodgy.py", "ScreenCssDeclaration", "CSS"),
    ], f"ratchet walker missed a declaration shape: {seen}"
    # And the narrow extractor really does skip the two dodges -- if this ever
    # starts seeing them, the walkers have converged and this proof is moot.
    narrow = {name for _m, name, _css in _class_css_blocks(tmp_path, excluded_dirs=())}
    assert "AnnotatedDeclaration" not in narrow
    assert "FStringDeclaration" not in narrow
