# test_progress_widget_clock_guard.py
# Description: Package-wide guard: stock textual progress widgets are forbidden (TASK-23022)
"""
TASK-23022: textual's stock ``ProgressBar``/``LoadingIndicator`` arm repeating
clocks (15 Hz indeterminate refresh + 1 Hz ETA sampler; 16 Hz indicator) that
``display = False`` does not stop -- measured as 88% of the Lab screen's idle
CPU. The repo's timer census cannot see this class of clock at all: the
``set_interval`` calls live inside textual, not in any package file.

The durable defense is therefore structural: **no package file may construct
or subclass the stock widgets**. Every use must go through
``tldw_chatbook/Widgets/pausable_progress.py``, whose drop-in replacements
pause all their clocks while hidden. With that invariant, a hidden
indeterminate progress widget arming a permanent clock cannot be written by
accident -- new code that tries fails this guard with a pointer to the house
classes.

Importing the stock names stays legal (they are the right type arguments for
``query_one``/``isinstance``); only *constructing* and *subclassing* are
flagged. The scanner unwraps ``ast.Subscript`` on callees, because the
subscripted-generic spelling of a call is otherwise invisible to a
``Name``/``Attribute``-only match (the trap recorded in
``backlog/docs/lessons-textual.md``, TASK-15771), and it resolves
``import`` aliases so ``from textual.widgets import ProgressBar as PB`` or
``textual.widgets.ProgressBar(...)`` cannot slip past. Positive-control
fixtures below hold the scanner itself to account: each forbidden spelling
must be flagged, so the guard cannot rot into a vacuous pass.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"
REPO_ROOT = PACKAGE_ROOT.parent

#: The one module allowed to touch the stock classes: it exists to wrap them.
HOUSE_MODULE = "tldw_chatbook/Widgets/pausable_progress.py"

#: Stock textual widget classes that arm hide-proof clocks.
STOCK_CLASSES = {"ProgressBar", "LoadingIndicator", "Bar"}

#: Modules the stock classes may be imported from.
TEXTUAL_WIDGET_MODULES = {
    "textual.widgets",
    "textual.widgets._progress_bar",
    "textual.widgets._loading_indicator",
}

GUIDANCE = (
    "construct/subclass PausableProgressBar / PausableLoadingIndicator from "
    "tldw_chatbook.Widgets.pausable_progress instead: the stock widgets arm "
    "repeating clocks that `display = False` does not stop (TASK-23022)"
)


def _unwrap_subscript(node: ast.expr) -> ast.expr:
    while isinstance(node, ast.Subscript):
        node = node.value
    return node


def _scan_module(source: str, filename: str) -> list[str]:
    """Return violation descriptions for one module's source."""
    tree = ast.parse(source, filename=filename)

    # -- pass 1: aliases ----------------------------------------------------
    # names bound to a stock class in this module
    class_aliases: set[str] = set()
    # names bound to a textual widgets module (import textual.widgets as tw /
    # from textual import widgets)
    module_aliases: set[str] = set()
    # names bound to the top-level ``textual`` package
    textual_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module in TEXTUAL_WIDGET_MODULES:
                for alias in node.names:
                    if alias.name in STOCK_CLASSES:
                        class_aliases.add(alias.asname or alias.name)
            elif node.module == "textual":
                for alias in node.names:
                    if alias.name == "widgets":
                        module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in TEXTUAL_WIDGET_MODULES:
                    if alias.asname:
                        # import textual.widgets as tw -> tw.ProgressBar(...)
                        module_aliases.add(alias.asname)
                    else:
                        # import textual.widgets binds the name ``textual``;
                        # the use is textual.widgets.ProgressBar(...)
                        textual_aliases.add("textual")
                elif alias.name == "textual":
                    textual_aliases.add(alias.asname or "textual")

    def is_stock_reference(node: ast.expr) -> str | None:
        node = _unwrap_subscript(node)
        if isinstance(node, ast.Name) and node.id in class_aliases:
            return node.id
        if isinstance(node, ast.Attribute) and node.attr in STOCK_CLASSES:
            value = _unwrap_subscript(node.value)
            if isinstance(value, ast.Name) and value.id in module_aliases:
                return f"{value.id}.{node.attr}"
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "widgets"
                and isinstance(_unwrap_subscript(value.value), ast.Name)
                and _unwrap_subscript(value.value).id in textual_aliases
            ):
                return f"textual.widgets.{node.attr}"
        return None

    # -- pass 2: constructions and subclassings -----------------------------
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            reference = is_stock_reference(node.func)
            if reference is not None:
                violations.append(
                    f"{filename}:{node.lineno}: constructs stock {reference}(...)"
                )
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                reference = is_stock_reference(base)
                if reference is not None:
                    violations.append(
                        f"{filename}:{node.lineno}: class {node.name} subclasses "
                        f"stock {reference}"
                    )
    return violations


def _scan_package() -> list[str]:
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel == HOUSE_MODULE:
            continue
        violations.extend(_scan_module(path.read_text(encoding="utf-8"), rel))
    return violations


# ---------------------------------------------------------------------------
# positive controls: every forbidden spelling must be caught, or the package
# assertion below proves nothing
# ---------------------------------------------------------------------------

_FORBIDDEN_FIXTURES = {
    "plain call": (
        "from textual.widgets import ProgressBar\n"
        "def f():\n"
        "    return ProgressBar(total=None)\n"
    ),
    "aliased call": (
        "from textual.widgets import LoadingIndicator as LI\n"
        "def f():\n"
        "    return LI()\n"
    ),
    "attribute call": (
        "import textual.widgets\n"
        "def f():\n"
        "    return textual.widgets.ProgressBar()\n"
    ),
    "widgets-module alias call": (
        "from textual import widgets as tw\n"
        "def f():\n"
        "    return tw.LoadingIndicator()\n"
    ),
    "subscripted-generic call": (
        "from textual.widgets import ProgressBar\n"
        "def f():\n"
        "    return ProgressBar[int](total=None)\n"
    ),
    "subclass": (
        "from textual.widgets import ProgressBar\n"
        "class Mine(ProgressBar):\n"
        "    pass\n"
    ),
    "private-module Bar import call": (
        "from textual.widgets._progress_bar import Bar\n"
        "def f():\n"
        "    return Bar()\n"
    ),
}

_COMPLIANT_FIXTURE = (
    "from textual.widgets import ProgressBar, LoadingIndicator\n"
    "from tldw_chatbook.Widgets.pausable_progress import PausableProgressBar\n"
    "def f(app):\n"
    "    bar = PausableProgressBar(total=None)\n"
    "    typed = app.query_one('#x', ProgressBar)\n"  # type reference: legal
    "    ok = isinstance(typed, LoadingIndicator)\n"  # type reference: legal
    "    return bar, ok\n"
)


def test_scanner_flags_every_forbidden_spelling() -> None:
    for label, fixture in _FORBIDDEN_FIXTURES.items():
        assert _scan_module(fixture, f"<fixture:{label}>"), (
            f"scanner missed the forbidden {label!r} spelling -- the package "
            "assertion below is now vacuous"
        )


def test_scanner_allows_type_references_and_house_classes() -> None:
    assert _scan_module(_COMPLIANT_FIXTURE, "<fixture:compliant>") == []


def test_house_module_is_scanned_nowhere_else() -> None:
    """The allowlist entry must exist -- a rename would exempt nothing."""
    assert (REPO_ROOT / HOUSE_MODULE).is_file(), (
        f"{HOUSE_MODULE} moved or was deleted; update HOUSE_MODULE or the "
        "guard exempts a ghost"
    )


def test_no_stock_progress_widget_use_outside_the_house_module() -> None:
    violations = _scan_package()
    assert not violations, (
        "Stock textual progress widgets found outside "
        f"{HOUSE_MODULE}; {GUIDANCE}:\n  " + "\n  ".join(violations)
    )
