#!/usr/bin/env python3
"""Collect class-level widget CSS into the built stylesheets (TASK-15450).

Textual registers one *stylesheet source* per widget class that declares
``DEFAULT_CSS``, and its parse cache is an ``LRUCache(64)``
(``textual/css/stylesheet.py``).  A full 13-destination tour of this app used to
end at 94 live sources -- past the cliff, every ``Stylesheet.parse()`` runs
fully cold (measured 125-380 ms) and Textual re-runs it whenever a widget class
not yet seen this session first mounts.

The fix is to stop those classes registering their own source: a widget declares
``BUNDLED_CSS`` (or a screen ``BUNDLED_SCREEN_CSS``) instead of
``DEFAULT_CSS``/``CSS``, and this module lifts every such block into a single
generated stylesheet at build time.  ``build_css.py`` writes:

* ``widget_defaults.tcss`` -- the ``BUNDLED_CSS`` blocks, loaded by the app as
  *one* default-CSS source (``TldwCli.DEFAULT_CSS``), which keeps them in
  Textual's low-priority "widget defaults" origin tier exactly as before.
* ``components/_bundled_screen_css.tcss`` -- the ``BUNDLED_SCREEN_CSS`` blocks,
  concatenated last into the app bundle, which keeps them in the high-priority
  "app CSS" tier exactly as before (screen ``CSS`` is *not* default CSS) while
  removing the full cold ``Stylesheet.reparse()`` Textual runs on a modal's
  first open.

Scoping is preserved by rewriting the selectors the same way Textual would.
``Widget.SCOPED_CSS`` defaults to ``True``, so Textual prefixes every top-level
selector of a class's CSS with that class's own type name unless the selector
already starts with it (``textual/css/parse.py`` ``parse_rule_set``).  We bake
that prefix into the text, so the generated sheets can be loaded unscoped.

There is one unavoidable wrinkle, and handling it is why each tier produces
*two* sheets rather than one.  Textual's **injected** scope selector carries
specificity ``(0, 0, 0)``, while the same type name **written out** carries
``(0, 0, 1)``, so every rewritten selector gains ``+1`` in the lowest-order
specificity component.  That is not cosmetic: measured on the real app, the nav
buttons' ``MainNavigationBar .nav-button`` rule (0,1,0) used to *lose* to
Textual's own ``Button.-style-default`` (0,1,1) and the naive rewrite tied it,
flipping 179 nodes' computed styles across the destination tour.

The fix is exact rather than approximate.  Selectors split into two streams:

* **self** -- selectors Textual would *not* have prefixed (they already start
  with the class's own type name).  Unchanged text, unchanged specificity.
* **scoped** -- selectors that gain the prefix, and therefore the ``+1``.

A rewritten rule ties another rule exactly when it used to sit one point below
it, i.e. exactly when it used to lose.  So the scoped stream only has to lose
every tie it now finds itself in, which is arranged per tier:

* widget defaults: the scoped sheet is registered with a tie-breaker below every
  other default-CSS source (Textual's own are ``-(MRO depth)``), so it loses
  ties on the tie-breaker.
* app CSS (screens): the scoped sheet is concatenated near the *top* of the
  bundle and the self sheet at the very *bottom*, so the scoped rules lose ties
  on source order while the self rules keep winning them as they did when
  Textual appended a screen's ``CSS`` at first open.

**That compensation is one-directional, and the limit is worth stating.**  It
handles LOSE -> TIE.  It cannot handle TIE -> WIN: a rule shifted from ``S`` to
``S+1`` now *strictly* outranks anything that used to sit level with it at ``S``,
and no tie-breaker can undo a strict specificity win -- ``tie_breaker`` is the
last element of the comparison key (``textual/css/styles.py`` ``extract_rules``),
so it only ever resolves exact ties.  The mitigating facts, measured rather than
assumed: the app bundle sits in a strictly higher origin tier than every one of
these rules, so it owns any property it declares regardless; and a dev-vs-branch
computed-style comparison found **0** differences over 2,528 nodes on 14 screens
in the resting state, plus **0** over 3,135 node-states with ``:hover``,
``:focus`` and ``:disabled`` each forced on every node of the Console, Personas
and MCP screens (117 of those were nav buttons -- the family whose shifted
``:hover`` / ``:focus`` rules are the most exposed).  No TIE -> WIN flip has been
demonstrated; none is claimed to be impossible.

``Tests/UI/test_widget_css_consolidation.py`` pins the whole scheme by parsing
both forms with Textual's own parser and asserting the rules match.  The
computed-style comparisons above are one-off dev-parity measurements, not tests:
they need a dev checkout to diff against, which CI does not have.

Stdlib-only, so the CSS guard can run without the app's dependencies.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

#: Class attribute holding CSS destined for the widget-defaults stylesheet
#: (Textual's low-priority default-CSS tier, formerly ``DEFAULT_CSS``).
WIDGET_ATTR = "BUNDLED_CSS"

#: Class attribute holding CSS destined for the app bundle (Textual's app-CSS
#: tier, formerly a ``Screen``/``ModalScreen`` class-level ``CSS``).
SCREEN_ATTR = "BUNDLED_SCREEN_CSS"

#: Directories under the package that keep their own widget CSS.  Vendored
#: third-party code and runnable examples are excluded on purpose: they are
#: mounted by their own standalone apps (which never load this app's bundle),
#: so consolidating them would strip their styling for no source-count win.
EXCLUDED_DIRS = ("Third_Party", "examples")

_FIRST_NAME_RE = re.compile(r"[.#]?([A-Za-z_\-][A-Za-z0-9_\-]*|\*)")


@dataclass(frozen=True)
class BundledBlock:
    """One class's CSS, lifted out of its Python declaration.

    Attributes:
        module: Package-relative path of the declaring module.
        class_name: Declaring class name, which is also its CSS scope (Textual
            sets ``_css_type_name`` to the class's own name).
        lineno: Line of the declaration, used only for stable ordering.
        css: The CSS exactly as written in the class body.
    """

    module: str
    class_name: str
    lineno: int
    css: str


def _split_top_level(selector_text: str) -> list[str]:
    """Split a selector list on its top-level commas.

    Args:
        selector_text: Raw text of a selector list (no trailing brace).

    Returns:
        The comma-separated selectors, with their surrounding whitespace intact.
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in selector_text:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        if char == "," and depth <= 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return parts


def _scope_one_selector(selector: str, scope: str) -> str:
    """Prefix a single selector with ``scope`` the way Textual's parser would.

    Textual inserts the scope selector unless the selector's first token already
    names the scope (comparing the token with its leading ``.``/``#`` stripped).

    Note that Textual applies this to the **last** selector of a comma-separated
    list only: ``parse_rule_set`` flushes each earlier group to ``rule_selectors``
    when it meets the comma, and scopes just the group left over when the loop
    ends (``textual/css/parse.py``).  So in scoped ``DEFAULT_CSS``, ``A, B {…}``
    scopes ``B`` but leaves ``A`` matching app-wide.  ``split_scoped_css``'s
    default reproduces that quirk exactly, so the exactness test can pin this
    transform against Textual's own parser -- but **both** build-time callers
    now opt out via ``scope_every_selector=True``: the screen sheets from the
    start (TASK-15450) and the widget-defaults sheets since TASK-15998, because
    consolidation made every sheet live from boot and the leak was measured
    cascade-neutral to close (see ``build_css.build_widget_defaults``).

    Args:
        selector: One selector, possibly with leading/trailing whitespace.
        scope: The scope type name.

    Returns:
        The selector with the scope prefix applied where Textual would apply it.
    """
    stripped = selector.strip()
    if not stripped:
        return selector
    match = _FIRST_NAME_RE.match(stripped)
    if match is not None and match.group(1) == scope:
        return selector
    # Insert the prefix at the first non-whitespace character so the original
    # indentation (and therefore the generated sheet's readability) survives.
    lead = len(selector) - len(selector.lstrip())
    return f"{selector[:lead]}{scope} {selector[lead:]}"


def split_scoped_css(
    css: str, scope: str, *, scope_every_selector: bool = False
) -> tuple[str, str]:
    """Bake Textual's ``SCOPED_CSS`` prefixing into CSS text, split by stream.

    Only top-level selectors are rewritten; nested rule sets are left alone,
    matching Textual, which recurses into nested rules with an empty scope.

    A rule whose selector list mixes both kinds (``MyWidget, .other {…}``) is
    emitted into both streams, each keeping the selectors that belong to it and
    both keeping the declaration body.

    Args:
        css: The CSS as written in the class body.
        scope: Type name to scope the CSS to (the declaring class's name).

    Returns:
        ``(self_css, scoped_css)`` -- the selectors Textual would have left
        alone, and the ones it would have prefixed (here written out).  Either
        may be empty. Neither needs a ``scope`` argument when parsed.

    Raises:
        ValueError: If the CSS has unbalanced braces or an unterminated string,
            which would silently mis-scope the remainder of the sheet.
    """
    streams: tuple[list[str], list[str]] = ([], [])  # (self, scoped)
    # Selectors for the rule currently being read, per stream.
    heads: list[list[str]] = [[], []]
    body: list[str] = []
    pending: list[str] = []  # text since the last top-level '}' (or the start)
    depth = 0
    quote: str | None = None
    index = 0
    length = len(css)

    def sink() -> list[str]:
        """Where raw characters go right now."""
        return body if depth else pending

    def flush_rule() -> None:
        """Emit the finished rule into whichever streams claimed a selector."""
        text = "".join(body)
        for stream, head in zip(streams, heads):
            if head:
                stream.append(",".join(head))
                stream.append(text)
        heads[0].clear()
        heads[1].clear()
        body.clear()

    while index < length:
        char = css[index]

        if quote is not None:
            sink().append(char)
            if char == quote:
                quote = None
            index += 1
            continue

        if css[index : index + 2] == "/*":
            end = css.find("*/", index + 2)
            end = length if end == -1 else end + 2
            sink().append(css[index:end])
            index = end
            continue

        if char in "\"'":
            quote = char
            sink().append(char)
            index += 1
            continue

        if char == "{":
            if depth == 0:
                # `pending` is trivia (comments, blank lines, and top-level
                # `$variable:` declarations) plus the selector list.  Trivia goes
                # to both streams so variables stay defined in each sheet.
                text = "".join(pending)
                pending = []
                split_at = _selector_start(text)
                trivia, selectors = text[:split_at], text[split_at:]
                streams[0].append(trivia)
                streams[1].append(trivia)
                # Textual scopes only the FINAL selector of a comma-separated
                # list -- see `_scope_one_selector`'s note.  Reproduce that.
                parts = _split_top_level(selectors)
                for position, part in enumerate(parts):
                    if not part.strip():
                        continue  # stray comma; contributes no selector
                    is_last = position == len(parts) - 1
                    prefix = scope_every_selector or is_last
                    scoped = _scope_one_selector(part, scope) if prefix else part
                    heads[0 if scoped == part else 1].append(scoped)
            body.append(char)
            depth += 1
            index += 1
            continue

        if char == "}":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unbalanced '}}' in CSS scoped to {scope!r}")
            body.append(char)
            if depth == 0:
                flush_rule()
            index += 1
            continue

        sink().append(char)
        index += 1

    if depth != 0:
        raise ValueError(f"unbalanced '{{' in CSS scoped to {scope!r}")
    if quote is not None:
        raise ValueError(f"unterminated string in CSS scoped to {scope!r}")
    trailing = "".join(pending)
    streams[0].append(trailing)
    streams[1].append(trailing)
    return "".join(streams[0]), "".join(streams[1])


def _selector_start(text: str) -> int:
    """Find where the selector list begins inside a top-level chunk.

    The chunk between two top-level rule sets is trivia -- blank lines, comments
    and ``$variable: value;`` declarations -- followed by the next selector list.
    Splitting them keeps comments attached to the rule they document instead of
    being swallowed by the scope prefix, and keeps variable declarations (which
    are legal at the top level of a sheet) out of the selector entirely.

    Args:
        text: Text between the previous ``}`` and the next ``{``.

    Returns:
        Index of the first character of the selector list.
    """
    # Everything up to the last top-level ';' is a variable declaration.
    base = 0
    index = 0
    length = len(text)
    while index < length:
        if text[index : index + 2] == "/*":
            end = text.find("*/", index + 2)
            index = length if end == -1 else end + 2
            continue
        if text[index] == ";":
            base = index + 1
        index += 1

    index = base
    last_break = base
    while index < length:
        if text[index : index + 2] == "/*":
            end = text.find("*/", index + 2)
            index = length if end == -1 else end + 2
            last_break = index
            continue
        if text[index] == "\n":
            last_break = index + 1
        elif not text[index].isspace():
            return last_break
        index += 1
    return last_break


def iter_blocks(package_root: Path, attr: str) -> list[BundledBlock]:
    """Collect every class-level ``attr`` string literal under ``package_root``.

    Args:
        package_root: The ``tldw_chatbook`` package directory.
        attr: Class attribute to collect (``WIDGET_ATTR`` or ``SCREEN_ATTR``).

    Returns:
        The blocks, ordered by module path then declaration line so the
        generated stylesheet is byte-stable across machines.

    Raises:
        ValueError: If a declaration is not a plain string literal (it could not
            be lifted out of Python) or if two classes share a name (their CSS
            scopes would collide in the generated sheet).
    """
    blocks: list[BundledBlock] = []
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        source = path.read_text(encoding="utf-8")
        if attr not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                target_names: list[str] = []
                value: ast.expr | None = None
                if isinstance(stmt, ast.Assign):
                    target_names = [
                        t.id for t in stmt.targets if isinstance(t, ast.Name)
                    ]
                    value = stmt.value
                elif isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    target_names = [stmt.target.id]
                    value = stmt.value
                if attr not in target_names:
                    continue
                if not isinstance(value, ast.Constant) or not isinstance(
                    value.value, str
                ):
                    raise ValueError(
                        f"{relative}::{node.name}.{attr} is not a plain string "
                        "literal; it cannot be lifted into the built stylesheet."
                    )
                blocks.append(
                    BundledBlock(
                        module=relative.as_posix(),
                        class_name=node.name,
                        lineno=stmt.lineno,
                        css=value.value,
                    )
                )
    blocks.sort(key=lambda block: (block.module, block.lineno))

    seen: dict[str, str] = {}
    for block in blocks:
        if block.class_name in seen:
            raise ValueError(
                f"{block.module}::{block.class_name} collides with "
                f"{seen[block.class_name]}::{block.class_name}: two classes with "
                "the same name cannot both bundle their CSS (Textual scopes by "
                "class name, so the rules would cross-apply)."
            )
        seen[block.class_name] = block.module
    return blocks


def render_stylesheets(
    blocks: list[BundledBlock], title: str, *, scope_every_selector: bool = False
) -> tuple[str, str]:
    """Render collected blocks as the two generated stylesheets.

    Args:
        blocks: Blocks to render, already in their final cascade order.
        title: Human-readable description for the generated headers.
        scope_every_selector: Scope every selector of a comma-separated list
            rather than reproducing Textual's last-selector-only quirk.

    Returns:
        ``(self_sheet, scoped_sheet)`` -- see :func:`split_scoped_css` for what
        separates them and the module docstring for why they must stay apart.
    """

    def header(stream: str) -> str:
        return (
            "/* ========================================\n"
            " * GENERATED FILE - DO NOT EDIT DIRECTLY\n"
            " * ========================================\n"
            f" * {title} -- {stream} selectors\n"
            " *\n"
            " * Generated by tldw_chatbook/css/build_css.py from the class-level\n"
            f" * {WIDGET_ATTR}/{SCREEN_ATTR} declarations in the Python sources.\n"
            " * Edit those declarations, then re-run build_css.py.\n"
            " * ======================================== */\n"
        )

    rendered = [header("self"), header("scoped")]
    for block in blocks:
        banner = f"\n/* ===== WIDGET: {block.class_name} ({block.module}) ===== */\n"
        split = split_scoped_css(
            block.css, block.class_name, scope_every_selector=scope_every_selector
        )
        for stream, text in enumerate(split):
            if not text.strip():
                continue
            rendered[stream] += banner + text
            if not text.endswith("\n"):
                rendered[stream] += "\n"
    return rendered[0], rendered[1]
