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

The one intentional difference: Textual's injected scope selector carries
specificity ``(0, 0, 0)`` while a written type selector carries ``(0, 0, 1)``,
so every rewritten rule gains ``+1`` in the lowest-order specificity component.
That shift is uniform across the rewritten rules and only ever raises them
against *other* widgets' ``DEFAULT_CSS`` (which they already outranked via the
tie-breaker) -- never against the app bundle, which sits in a strictly higher
origin tier.  ``Tests/UI/test_widget_css_consolidation.py`` pins this by parsing
both forms with Textual's own parser and asserting the rules are identical
apart from that documented delta.

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
    scopes ``B`` but leaves ``A`` matching app-wide.  That is upstream behaviour
    this app's styling already depends on, so the transform reproduces it rather
    than quietly "fixing" it (which would silently unstyle whatever those leaked
    selectors are currently reaching).  Filed as a follow-up, not changed here.

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


def scope_css(css: str, scope: str) -> str:
    """Bake Textual's ``SCOPED_CSS`` selector prefixing into CSS text.

    Only top-level selectors are rewritten; nested rule sets are left alone,
    matching Textual, which recurses into nested rules with an empty scope.

    Args:
        css: The CSS as written in the class body.
        scope: Type name to scope the CSS to (the declaring class's name).

    Returns:
        Equivalent CSS that needs no ``scope`` argument when parsed.

    Raises:
        ValueError: If the CSS has unbalanced braces, which would silently
            mis-scope the remainder of the sheet.
    """
    out: list[str] = []
    pending: list[str] = []  # text since the last top-level '}' (or the start)
    depth = 0
    quote: str | None = None
    index = 0
    length = len(css)

    while index < length:
        char = css[index]
        pair = css[index : index + 2]

        if quote is not None:
            (out if depth else pending).append(char)
            if char == quote:
                quote = None
            index += 1
            continue

        if pair == "/*":
            end = css.find("*/", index + 2)
            end = length if end == -1 else end + 2
            (out if depth else pending).append(css[index:end])
            index = end
            continue

        if char in "\"'":
            quote = char
            (out if depth else pending).append(char)
            index += 1
            continue

        if char == "{":
            if depth == 0:
                # `pending` is trivia (comments/blank lines) plus the selector
                # list.  Only the selector list is rewritten.
                text = "".join(pending)
                pending = []
                split_at = _selector_start(text)
                trivia, selectors = text[:split_at], text[split_at:]
                out.append(trivia)
                # Textual scopes only the FINAL selector of a comma-separated
                # list -- see `_scope_one_selector`'s note.  Reproduce that.
                parts = _split_top_level(selectors)
                parts[-1] = _scope_one_selector(parts[-1], scope)
                out.append(",".join(parts))
            out.append(char)
            depth += 1
            index += 1
            continue

        if char == "}":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unbalanced '}}' in CSS scoped to {scope!r}")
            out.append(char)
            index += 1
            continue

        (out if depth else pending).append(char)
        index += 1

    if depth != 0:
        raise ValueError(f"unbalanced '{{' in CSS scoped to {scope!r}")
    if quote is not None:
        raise ValueError(f"unterminated string in CSS scoped to {scope!r}")
    out.append("".join(pending))
    return "".join(out)


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


def render_stylesheet(blocks: list[BundledBlock], title: str) -> str:
    """Render collected blocks as one generated stylesheet.

    Args:
        blocks: Blocks to render, already in their final cascade order.
        title: Human-readable description for the generated header.

    Returns:
        The stylesheet text.
    """
    parts = [
        "/* ========================================\n"
        " * GENERATED FILE - DO NOT EDIT DIRECTLY\n"
        " * ========================================\n"
        f" * {title}\n"
        " *\n"
        " * Generated by tldw_chatbook/css/build_css.py from the class-level\n"
        f" * {WIDGET_ATTR}/{SCREEN_ATTR} declarations in the Python sources.\n"
        " * Edit those declarations, then re-run build_css.py.\n"
        " * ======================================== */\n"
    ]
    for block in blocks:
        parts.append(
            f"\n/* ===== WIDGET: {block.class_name} ({block.module}) ===== */\n"
        )
        scoped = scope_css(block.css, block.class_name)
        parts.append(scoped)
        if not scoped.endswith("\n"):
            parts.append("\n")
    return "".join(parts)
