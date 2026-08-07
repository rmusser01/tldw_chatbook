"""Imports kept alive only by tests that patch them through a module alias.

**Why this test exists.** The Console decomposition keeps moving code out of
`chat_screen.py`, which keeps leaving imports behind that the module's own body
no longer references. Deleting them looks like tidying. It is not always:
tests reach symbols through whatever namespace they can, and

    monkeypatch.setattr(chat_screen_module, "ConsoleDictationController", ...)
    chat_screen_module.ConsoleStreamingDictationSession

keep a `chat_screen`-level import load-bearing with **no reference any
import-grep can find** -- the alias hides it, and in the quoted-`setattr` form
the symbol is not even an identifier.

Wave 4 deleted five such imports and turned 28 tests red across five files. The
fix at the time was a block comment plus `# noqa: F401` markers -- and the very
next review found the marked set was wrong in *both* directions: four classes
were marked that no test reaches, while six symbols that tests DO reach sat
unmarked, one of them in the same import block. A comment cannot keep that set
honest as the decomposition continues. This test can.

**What to do when this fails.** Either add `# noqa: F401` to the named import
(it is load-bearing -- a test patches it through the module), or repoint the
test at the module that actually defines the symbol and drop the import.
Repointing is the better fix and is tracked as task-3023; the marker is the
holding position.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TARGET = "tldw_chatbook/UI/Screens/chat_screen.py"


def _imported_but_unreferenced(source: str) -> dict[str, int]:
    """Imported names the module's own AST never references.

    Deliberately AST-based: a name that appears only in a comment or docstring
    does NOT count as referenced, which is exactly the trap that made an
    earlier hand-audit miss `ConsoleDictationController` (it is named in a
    dozen comments and used in zero expressions).

    Args:
        source: Full text of the module.

    Returns:
        dict[str, int]: Unreferenced imported name -> its import line number.
    """
    tree = ast.parse(source)
    imported: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue
                imported[alias.asname or alias.name.split(".")[0]] = node.lineno

    # Quoted annotations (`x: "ChatScreen"`) are real uses, and they are
    # `ast.Constant` strings -- but so is every docstring. Scanning ALL string
    # constants therefore silently marks a name "referenced" because some
    # docstring mentions it, which is the exact false negative this function
    # exists to avoid: `ConsoleDictationController` is named in a dozen
    # docstrings in the target module and used in zero expressions. So string
    # scanning is restricted to annotation subtrees only.
    annotation_nodes: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.AnnAssign, ast.arg)) and node.annotation is not None:
            annotation_nodes.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                annotation_nodes.append(node.returns)

    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                referenced.add(base.id)
    for annotation in annotation_nodes:
        for node in ast.walk(annotation):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                referenced.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", node.value))

    return {name: line for name, line in imported.items() if name not in referenced}


def _alias_reached(symbols: set[str]) -> dict[str, int]:
    """Symbols reached through the `chat_screen` module namespace in `Tests/`.

    Covers both spellings: attribute access (`chat_screen_module.Foo`) and the
    quoted `setattr(chat_screen_module, "Foo", ...)` form, which contains no
    identifier at all and is therefore invisible to a name-based search.

    Args:
        symbols: Candidate names to look for.

    Returns:
        dict[str, int]: Symbol -> number of reaching sites found.
    """
    counts: dict[str, int] = {}
    for path in (_REPO_ROOT / "Tests").rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "chat_screen" not in text:
            continue
        for symbol in symbols:
            escaped = re.escape(symbol)
            hits = len(re.findall(rf"chat_screen[A-Za-z_]*\.{escaped}\b", text))
            hits += len(
                re.findall(
                    rf"setattr\([^,]*chat_screen[A-Za-z_]*\s*,\s*[\"']{escaped}[\"']",
                    text,
                )
            )
            if hits:
                counts[symbol] = counts.get(symbol, 0) + hits
    return counts


@pytest.mark.unit
def test_alias_reached_imports_are_marked_load_bearing() -> None:
    source = (_REPO_ROOT / _TARGET).read_text(encoding="utf-8")
    unreferenced = _imported_but_unreferenced(source)
    at_risk = _alias_reached(set(unreferenced))
    assert at_risk, (
        "no alias-reached imports found at all -- either the tests were "
        "repointed (delete this test and the markers, see task-3023) or this "
        "detector stopped working. Do not leave it silently passing on an "
        "empty set."
    )

    lines = source.splitlines()
    unmarked = sorted(
        f"{name} (line {unreferenced[name]}, {at_risk[name]} test site(s))"
        for name in at_risk
        if "noqa: F401" not in lines[unreferenced[name] - 1]
        and not any(
            "noqa: F401" in line and re.match(rf"\s*{re.escape(name)}\b", line)
            for line in lines
        )
    )
    assert unmarked == [], (
        f"{_TARGET} imports these symbols, never references them in its own "
        f"code, and tests reach them through the module alias: {unmarked}.\n"
        "A `ruff --fix` / autoflake pass will delete them and those tests will "
        "fail with AttributeError. Add `# noqa: F401` to each, or repoint the "
        "tests at the defining module and drop the import (task-3023)."
    )


@pytest.mark.unit
def test_a_docstring_mention_does_not_count_as_a_reference() -> None:
    """The false negative this detector shipped with, pinned.

    The first cut scanned every string `ast.Constant` for identifier tokens so
    that quoted annotations would count as uses. Docstrings are string
    constants too, so a symbol named only in prose looked referenced -- and
    `ConsoleDictationController`, whose deletion turns 28 tests red, is named
    in a dozen docstrings and used in zero expressions. The guard passed with
    its marker removed. Real annotations must still count, so both halves are
    asserted here rather than only the bug.
    """
    source = (
        "from x import DocstringOnly, RealAnnotation, Genuinely\n"
        "def f(a: 'RealAnnotation') -> None:\n"
        "    \"\"\"Mentions DocstringOnly in prose only.\"\"\"\n"
        "    Genuinely()\n"
    )
    unreferenced = _imported_but_unreferenced(source)
    assert "DocstringOnly" in unreferenced, (
        "a name mentioned only in a docstring was treated as referenced -- "
        "the detector is scanning docstrings again."
    )
    assert "RealAnnotation" not in unreferenced, "quoted annotations are real uses"
    assert "Genuinely" not in unreferenced, "expression uses are real uses"
