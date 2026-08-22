# floor_syntax.py
# Description: Interpreter-independent detection of PEP 701 floor breaks
"""Detect f-string syntax that the project's Python FLOOR cannot parse.

`Tests/Architecture/test_python_floor_syntax.py` compiles the package under a
real floor interpreter, which is authoritative and complete. But it SKIPS when
no such interpreter is installed, and a skip reports exactly as much as a pass
-- so on a machine or CI runner without Python 3.11 the guard says nothing
while a module that cannot import on the declared floor sails through.
Demonstrated, not assumed: with a genuine PEP 701 construct injected into
`Utils/egress.py`, that guard FAILS with 3.11 reachable and SKIPS GREEN with
`PATH` stripped of `uv` and `python3.11`.

This module is the always-runs half. It is deliberately PARTIAL: it covers the
one class that actually shipped a broken module (TASK-19560 -- `TTS/backends/
kokoro.py` could not be imported at all on 3.11 while every test passed on
3.14), not every possible floor incompatibility. When a floor interpreter is
available the compile check remains the real gate; this is the floor under the
floor.

WHAT IT DETECTS. PEP 701 (Python 3.12) lifted two restrictions on the
expression part of an f-string. Both were verified against a real 3.11 and a
real 3.14 rather than taken from the PEP text:

* **Same delimiter.** A nested string using the f-string's own quote delimiter.
  `f"{ d["k"] }"` is a 3.11 SyntaxError; `f"{ d['k'] }"` is fine, and so is
  `f\"\"\"{ d["k"] }\"\"\"` -- the delimiter there is `\"\"\"`, not `"`, which is
  why this compares the full delimiter rather than a single character.
* **Backslash in the expression.** `f"{ 'a\\nb'.strip() }"` is a 3.11
  SyntaxError even though the quotes differ. A backslash in a FORMAT SPEC
  (`f"{v:\\>10}"`) is accepted by 3.11, and is not flagged, because a spec is
  emitted as `FSTRING_MIDDLE` rather than as a nested `STRING` token.

Both reduce to inspecting the `STRING` tokens nested inside an f-string, which
is why this needs no parser of its own. `Tests/Architecture/
test_python_floor_syntax.py` pins every case in the table above against the
real interpreters' verdicts, so the two can never drift apart silently.

NOTE ON THE RUNNING INTERPRETER: `FSTRING_START`/`FSTRING_END` tokens exist
only from 3.12. Below that the tokenizer cannot produce the constructs this
looks for -- and does not need to, because an interpreter at or below the
floor rejects them itself at import.
"""

from __future__ import annotations

import io
import tokenize
from dataclasses import dataclass
from pathlib import Path

#: Directory names never swept for source. `.gitignore` does not affect
#: `Path.rglob`, and a nested virtualenv under the package root will happily
#: serve up third-party modules written for a newer Python -- reported as
#: findings against this project (TASK-19906 recorded exactly that failure for
#: a sibling AST sweep).
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".venv",
        "venv",
        "site-packages",
        "node_modules",
        "__pycache__",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
    }
)


@dataclass(frozen=True)
class FloorBreak:
    """One construct that the declared floor cannot parse."""

    path: Path
    line: int
    reason: str
    text: str

    def __str__(self) -> str:  # pragma: no cover - human-readable only
        return f"{self.path}:{self.line}: {self.reason}: {self.text}"


def _delimiter_of(token_text: str) -> str:
    """Return the quote delimiter of a string token, prefix stripped.

    Args:
        token_text: The raw source text of a STRING or FSTRING_START token,
            e.g. ``rb'x'`` or ``f\"\"\"``.

    Returns:
        The delimiter -- one of ``'``, ``"``, ``'''``, ``\"\"\"`` -- or ``""``
        if the token has no recognisable quote (which should not happen for a
        well-formed token, and is treated as "nothing to compare" rather than
        raising, so one odd token cannot take down a whole sweep).
    """
    index = 0
    while index < len(token_text) and token_text[index] not in "\"'":
        index += 1
    if index >= len(token_text):
        return ""
    quote = token_text[index]
    if token_text[index : index + 3] == quote * 3:
        return quote * 3
    return quote


def find_floor_breaks(source: str, *, path: Path) -> list[FloorBreak]:
    """Find PEP 701 constructs in ``source`` that the floor cannot parse.

    Args:
        source: The module source text.
        path: The path to attribute findings to; not read.

    Returns:
        Every finding, in source order. Empty means "nothing of the covered
        class", NOT "parses on the floor" -- see the module docstring.

    Raises:
        Nothing. A file that fails to tokenize returns no findings: on the
        running interpreter that means it is already broken in a way the
        ordinary suite reports far more clearly than this sweep would.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return []

    findings: list[FloorBreak] = []
    # A stack, not a flag: f-strings nest, and `f"{f"{x}"}"` is a floor break
    # precisely BECAUSE the inner one reuses the outer's delimiter.
    open_delimiters: list[str] = []
    for token in tokens:
        name = tokenize.tok_name[token.type]
        if name == "FSTRING_START":
            nested = _delimiter_of(token.string)
            # An inner f-string is not a STRING token, so it needs its own
            # check: `f"{f"{x}"}"` breaks the floor for exactly the same
            # reason a nested plain string does -- it reuses the delimiter.
            if nested and nested in open_delimiters:
                findings.append(
                    FloorBreak(
                        path=path,
                        line=token.start[0],
                        reason=(
                            f"nested f-string reuses the enclosing f-string's "
                            f"{nested} delimiter (PEP 701, needs >= 3.12)"
                        ),
                        text=token.string,
                    )
                )
            open_delimiters.append(nested)
        elif name == "FSTRING_END":
            if open_delimiters:
                open_delimiters.pop()
        elif name == "STRING" and open_delimiters:
            nested = _delimiter_of(token.string)
            if nested and nested in open_delimiters:
                findings.append(
                    FloorBreak(
                        path=path,
                        line=token.start[0],
                        reason=(
                            f"nested string reuses the enclosing f-string's "
                            f"{nested} delimiter (PEP 701, needs >= 3.12)"
                        ),
                        text=token.string,
                    )
                )
            elif "\\" in token.string:
                findings.append(
                    FloorBreak(
                        path=path,
                        line=token.start[0],
                        reason=(
                            "backslash inside an f-string expression "
                            "(PEP 701, needs >= 3.12)"
                        ),
                        text=token.string,
                    )
                )
    return findings


def iter_source_files(root: Path):
    """Yield every ``.py`` file under ``root``, skipping environment dirs.

    Args:
        root: Directory to walk.

    Yields:
        Paths to source files, excluding anything under a directory named in
        ``EXCLUDED_DIRECTORY_NAMES``.
    """
    for path in sorted(root.rglob("*.py")):
        if EXCLUDED_DIRECTORY_NAMES.isdisjoint(path.parts):
            yield path
