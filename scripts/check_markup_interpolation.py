#!/usr/bin/env python3
"""Guard interpolated names reaching markup-parsing surfaces (TASK-1513 AC1).

Textual parses Rich markup by default in three widget-facing places --
``App.notify()`` messages, widget ``tooltip`` strings (rendered through the
``Tooltip`` Static), and ``Button(label=...)`` -- so an interpolated
user-derived value containing a stray ``[/]`` or ``[b]`` raises
``MarkupError`` at render time and can crash the app. The 2026-07-30 Evals
UAT batch (task-1476/1482) fixed the Evals package's instances; this guard
is the repo-wide half: the convention is

- ``escape_markup(...)`` (``tldw_chatbook.Utils.input_validation``) on every
  interpolated runtime value inside a tooltip or Button label (they have no
  ``markup=False`` seam), and
- ``markup=False`` on ``notify()`` calls whose message interpolates runtime
  text (renders literally; no escape needed -- one convention per seam).

Kinds matched (an "unsafe atom" is a runtime value interpolated into the
string that is not wrapped in ``escape_markup(...)``):

1. ``notify_interp`` -- ``<anything>.notify(<msg>, ...)`` whose ``msg``
   interpolates an unsafe atom AND the call does not pass ``markup=False``.
2. ``button_label_interp`` -- ``Button(<label>)`` / ``Button(label=...)``
   whose label interpolates an unsafe atom. A whole-expression
   ``escape_markup(...)`` or ``Text(...)`` wrap is exempt (``Text(str)``
   does not parse markup).
3. ``tooltip_interp`` -- any call's ``tooltip=`` keyword, or an
   ``<obj>.tooltip =`` assignment, interpolating an unsafe atom.

Interpolation shapes covered: f-strings (``JoinedStr``), ``%`` formatting,
``"...".format(...)``, and ``+`` concatenation.

All three kinds are RATCHETED, not gated: the sites that already exist are
pinned in ``scripts/markup_interpolation_census.tsv`` and only a new or grown
one fails, so screens can adopt the convention incrementally without a
repo-wide rewrite. Many pinned rows interpolate low-risk values (counts,
glyphs, internal enums); the census stops NEW sites, and each pinned row is
a work item for the adopting task.

The census is keyed on ``module<TAB>symbol<TAB>kind<TAB>count`` (the
enclosing function/method qualname, not line numbers, so ordinary edits do
not churn it). A key that is new, or whose count has grown, fails; removing
occurrences is always allowed -- regenerate with ``--write`` to shrink.

Usage:
    python scripts/check_markup_interpolation.py            # verify (preflight/CI)
    python scripts/check_markup_interpolation.py --write    # regenerate the census
"""

from __future__ import annotations

import argparse
import ast
import sys
import warnings
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = REPO_ROOT / "tldw_chatbook"
CENSUS = Path(__file__).resolve().parent / "markup_interpolation_census.tsv"

KIND_NOTIFY = "notify_interp"
KIND_BUTTON_LABEL = "button_label_interp"
KIND_TOOLTIP = "tooltip_interp"

RATCHETED_KINDS = (KIND_NOTIFY, KIND_BUTTON_LABEL, KIND_TOOLTIP)

Occurrence = tuple[str, str, str]  # (module, symbol, kind)


def _is_escape_markup_call(node: ast.AST) -> bool:
    """Whether ``node`` is a call to ``escape_markup`` (imported or dotted)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "escape_markup"
    return isinstance(func, ast.Attribute) and func.attr == "escape_markup"


def _is_text_wrap(node: ast.AST) -> bool:
    """Whether ``node`` wraps its content in a markup-inert ``Text`` object.

    ``Text("...")`` and ``Text.from_markup("...")`` both yield parsed
    content: a plain-string ``Text`` never interprets brackets, and a
    ``from_markup`` wrap is the author explicitly choosing markup with no
    further interpolation hazard *of its own string* (the f-string inside
    is still scanned by the JoinedStr predicate, so ``Text(f"{name}")``
    stays safe while ``Button(f"{name}")`` does not).
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "Text"
    if isinstance(func, ast.Attribute):
        return func.attr == "from_markup" or (
            func.attr == "Text" and isinstance(func.value, ast.Name)
        )
    return False


def _atom_is_safe(node: ast.AST) -> bool:
    """Whether one interpolated value cannot carry markup metacharacters.

    Safe: literals; ``escape_markup(...)``-wrapped values; and boolean /
    conditional expressions whose every branch is safe. Unsafe: names,
    attributes, subscripts, calls (other than ``escape_markup``), etc.
    """
    if isinstance(node, ast.Constant):
        return True
    if _is_escape_markup_call(node):
        return True
    if isinstance(node, ast.BoolOp):
        return all(_atom_is_safe(value) for value in node.values)
    if isinstance(node, ast.IfExp):
        return _atom_is_safe(node.body) and _atom_is_safe(node.orelse)
    if isinstance(node, ast.JoinedStr):
        # A nested f-string is as safe as its own atoms.
        return not _joinedstr_unsafe_atoms(node)
    return False


def _joinedstr_unsafe_atoms(node: ast.JoinedStr) -> list[ast.AST]:
    """Unsafe interpolated atoms of an f-string (incl. format specs)."""
    unsafe: list[ast.AST] = []
    for value in node.values:
        if isinstance(value, ast.JoinedStr):
            unsafe.extend(_joinedstr_unsafe_atoms(value))
        elif isinstance(value, ast.FormattedValue):
            if value.format_spec is not None:
                unsafe.extend(_joinedstr_unsafe_atoms(value.format_spec))
            if not _atom_is_safe(value.value):
                unsafe.append(value.value)
    return unsafe


def _add_leaves(node: ast.AST, leaves: list[ast.AST]) -> None:
    """Flatten an ``a + b + c`` chain into its leaf operands."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        _add_leaves(node.left, leaves)
        _add_leaves(node.right, leaves)
    else:
        leaves.append(node)


def _interpolates_unsafe_atom(expr: ast.AST) -> bool:
    """Whether ``expr`` builds a string that embeds a runtime value unescaped.

    Covers f-strings, ``%`` formatting, ``"...".format(...)`` and ``+``
    concatenation, at any nesting depth of the expression.
    """
    found = False

    class _V(ast.NodeVisitor):
        def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
            nonlocal found
            if _joinedstr_unsafe_atoms(node):
                found = True
            self.generic_visit(node)

        def visit_BinOp(self, node: ast.BinOp) -> None:
            nonlocal found
            if isinstance(node.op, ast.Mod):
                if (
                    isinstance(node.left, ast.Constant)
                    and isinstance(node.left.value, str)
                    and not isinstance(node.right, ast.Constant)
                ):
                    found = True
            elif isinstance(node.op, ast.Add):
                leaves: list[ast.AST] = []
                _add_leaves(node, leaves)
                if any(
                    not isinstance(leaf, ast.Constant)
                    and not _is_escape_markup_call(leaf)
                    and not _is_text_wrap(leaf)
                    for leaf in leaves
                ):
                    found = True
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            nonlocal found
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "format"
                and isinstance(func.value, ast.Constant)
                and isinstance(func.value.value, str)
                and (node.args or node.keywords)
            ):
                found = True
            self.generic_visit(node)

    _V().visit(expr)
    return found


def _has_markup_false(call: ast.Call) -> bool:
    """Whether a call passes the literal ``markup=False`` keyword."""
    return any(
        keyword.arg == "markup"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in call.keywords
    )


def _button_label_expr(call: ast.Call) -> ast.AST | None:
    """The label expression of a ``Button(...)`` call, positional or keyword."""
    label: ast.AST | None = None
    if call.args:
        label = call.args[0]
    for keyword in call.keywords:
        if keyword.arg == "label":
            label = keyword.value
    if label is None:
        return None
    # Whole-expression wraps that are markup-inert by construction.
    if _is_escape_markup_call(label) or _is_text_wrap(label):
        return None
    return label


def _is_button_call(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "Button"
    return isinstance(func, ast.Attribute) and func.attr == "Button"


class _Visitor(ast.NodeVisitor):
    """Collect markup-interpolation occurrences with their enclosing qualname."""

    def __init__(self, module: str) -> None:
        self.module = module
        self._stack: list[str] = []
        self.hits: Counter[Occurrence] = Counter()

    @property
    def _symbol(self) -> str:
        return ".".join(self._stack) if self._stack else "<module>"

    def _enter(self, node: ast.AST) -> None:
        self._stack.append(node.name)  # type: ignore[attr-defined]
        self.generic_visit(node)
        self._stack.pop()

    visit_FunctionDef = _enter
    visit_AsyncFunctionDef = _enter
    visit_ClassDef = _enter

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func

        # <anything>.notify(<message>) — Textual's App.notify parses markup
        # unless the call opts out with markup=False.
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "notify"
            and node.args
            and not _is_escape_markup_call(node.args[0])
            and not _is_text_wrap(node.args[0])
            and _interpolates_unsafe_atom(node.args[0])
            and not _has_markup_false(node)
        ):
            self.hits[(self.module, self._symbol, KIND_NOTIFY)] += 1

        # Button(<label>) / Button(label=...)
        if _is_button_call(node):
            label = _button_label_expr(node)
            if label is not None and _interpolates_unsafe_atom(label):
                self.hits[(self.module, self._symbol, KIND_BUTTON_LABEL)] += 1

        # tooltip=<...> keyword on any widget constructor / helper call.
        for keyword in node.keywords:
            if (
                keyword.arg == "tooltip"
                and not _is_escape_markup_call(keyword.value)
                and not _is_text_wrap(keyword.value)
                and _interpolates_unsafe_atom(keyword.value)
            ):
                self.hits[(self.module, self._symbol, KIND_TOOLTIP)] += 1

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # <widget>.tooltip = <...>
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "tooltip"
                and not _is_escape_markup_call(node.value)
                and not _is_text_wrap(node.value)
                and _interpolates_unsafe_atom(node.value)
            ):
                self.hits[(self.module, self._symbol, KIND_TOOLTIP)] += 1
        self.generic_visit(node)


def scan_tree() -> Counter[Occurrence]:
    hits: Counter[Occurrence] = Counter()
    for path in sorted(PRODUCTION_ROOT.rglob("*.py")):
        module = path.relative_to(REPO_ROOT).with_suffix("").as_posix()
        try:
            with warnings.catch_warnings():
                # Some scanned files carry pre-existing invalid-escape strings;
                # their SyntaxWarning is not this guard's concern.
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        visitor = _Visitor(module)
        visitor.visit(tree)
        hits.update(visitor.hits)
    return hits


def read_census() -> Counter[Occurrence]:
    census: Counter[Occurrence] = Counter()
    if not CENSUS.exists():
        return census
    for lineno, raw in enumerate(CENSUS.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            print(f"FAIL: {CENSUS.name}:{lineno}: expected 4 TAB-separated fields")
            raise SystemExit(2)
        module, symbol, kind, count = parts
        census[(module, symbol, kind)] += int(count)
    return census


def write_census(hits: Counter[Occurrence]) -> None:
    """Overwrite the committed census with ``hits``, discarding the old pins.

    Destructive by design and only reached via ``--write``: every previous
    pin is replaced, so a shape that regressed since the last regeneration
    is silently re-baselined. Read the rows the check named before running
    it.
    """
    lines = [
        "# Markup-interpolation census (TASK-1513 AC1). Every row is a",
        "# notify/tooltip/Button-label site that interpolates a runtime value",
        "# without escaping. Ratchet: counts only shrink as screens adopt",
        "# escape_markup / markup=False. Regenerate with:",
        "# python scripts/check_markup_interpolation.py --write",
        "# module\tsymbol\tkind\tcount",
    ]
    for (module, symbol, kind), count in sorted(hits.items()):
        lines.append(f"{module}\t{symbol}\t{kind}\t{count}")
    CENSUS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Scan the package and compare it against the committed census.

    Returns:
        0 when clean (or when ``--write`` regenerated the census), 1 when a
        ratcheted occurrence is new or has grown. A malformed census row
        exits 2 from :func:`read_census`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="regenerate the census from the tree"
    )
    args = parser.parse_args()

    hits = scan_tree()

    if args.write:
        write_census(hits)
        total = sum(hits.values())
        print(f"wrote {CENSUS.relative_to(REPO_ROOT)}: {total} occurrence(s)")
        return 0

    census = read_census()

    grown = sorted(k for k, n in hits.items() if n > census.get(k, 0))

    totals = {
        kind: sum(n for k, n in hits.items() if k[2] == kind)
        for kind in RATCHETED_KINDS
    }
    print(
        "markup interpolation: "
        + ", ".join(f"{totals[kind]} {kind}" for kind in RATCHETED_KINDS)
        + f" occurrence(s) ({sum(census.values())} pinned)."
    )

    if not grown:
        print("check_markup_interpolation: OK")
        return 0

    print(
        f"\nFAIL: {len(grown)} new/grown markup-interpolation site(s). Textual "
        "parses Rich markup in notify messages, tooltips and Button labels; "
        "an unescaped user-derived name containing [/] raises MarkupError at "
        "render time. Wrap interpolated values in escape_markup(...) "
        "(tldw_chatbook.Utils.input_validation), or pass markup=False on the "
        "notify call:"
    )
    for module, symbol, kind in grown:
        print(
            f"    {module}\t{symbol}\t{kind} "
            f"(now {hits[(module, symbol, kind)]}, "
            f"pinned {census.get((module, symbol, kind), 0)})"
        )
    print(
        f"  If a pinned site must grow legitimately, regenerate the census "
        f"with --write and say why in the PR. Do NOT regenerate to hide a "
        f"new unescaped user-name site."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
