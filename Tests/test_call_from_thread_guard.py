"""Repo-wide guard: bare ``self.call_from_thread(...)`` is not a real method.

``call_from_thread`` exists only on ``textual.app.App``:

    Widget False | Container False | Screen False | ModalScreen False | App True

Any ``Widget``/``Container``/``Screen``/``ModalScreen`` subclass that spells a
threaded-worker callback as ``self.call_from_thread(...)`` raises
``AttributeError`` at runtime instead of marshaling the callback onto the UI
thread. Because these calls live inside ``@work(thread=True)`` workers and
are almost always wrapped in a broad ``except Exception`` handler, the
failure is invisible -- the intended notification simply never appears.

TASK-899/927 found and fixed this in ``Tools_Settings_Window.py`` (39 sites;
see the file-scoped guard in ``Tests/UI/test_tools_settings_window.py``).
TASK-929 swept the rest of ``tldw_chatbook/`` and found 31 more sites across
six files, all now using ``self.app.call_from_thread(...)``. This test is
the repo-wide backstop that keeps the bug class from coming back anywhere in
the package.

``tldw_chatbook/app.py`` is special-cased -- see ``ALLOWLISTED_RELATIVE_PATHS``
and ``test_app_py_allowlisted_sites_are_still_safe`` below for why its
remaining bare sites are not bugs.
"""
from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "tldw_chatbook"

# app.py defines `LibraryIngestQueueMixin`, a mixin that is *only* ever
# combined with `App` -- production: `TldwCli(LibraryIngestQueueMixin,
# App[None])`; tests: `_LibraryIngestCanvasHarness(LibraryIngestQueueMixin,
# App)` (Tests/UI/test_library_shell.py) and `_IngestRunnerHarness(
# LibraryIngestQueueMixin, App)` (Tests/Library/test_library_ingest_runner.py).
# Python attribute lookup resolves `self.call_from_thread` through the
# *instance's* full MRO, not the class whose body contains the call, so
# those sites are safe despite the mixin itself not subclassing App. The
# file's remaining bare sites sit directly on `TldwCli(App)`, also safe.
# Everywhere else in the package a bare call is a real bug -- see TASK-929.
ALLOWLISTED_RELATIVE_PATHS = {"app.py"}

# Non-structural token types to drop before pattern-matching. Comments are
# excluded so a `#` remark that merely *mentions* `self.call_from_thread(`
# (documenting why the bare form is wrong, or restating the correct
# `self.app.call_from_thread(` form) can never be mistaken for real code in
# either direction -- this is the trap that made an earlier draft of this
# guard pass against broken code. String/docstring tokens are excluded for
# the same reason: several modules discuss `call_from_thread` in prose.
_IGNORED_TOKEN_TYPES = frozenset(
    {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.STRING,
    }
)


def _bare_call_from_thread_lines(path: Path) -> list[int]:
    """1-based line numbers where `self.call_from_thread(` appears as code.

    Tokenizes the file and looks for the exact NAME('self') OP('.')
    NAME('call_from_thread') OP('(') token sequence, rather than doing a
    substring/regex scan over raw source text. This is comment- and
    string-immune by construction (see `_IGNORED_TOKEN_TYPES`), and it
    naturally does not match `self.app.call_from_thread(` -- the token
    right after `self` `.` there is `app`, not `call_from_thread`.
    """
    source = path.read_text(encoding="utf-8")
    try:
        tokens = [
            tok
            for tok in tokenize.generate_tokens(io.StringIO(source).readline)
            if tok.type not in _IGNORED_TOKEN_TYPES
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        pytest.fail(f"could not tokenize {path}: {exc}")

    offenders: list[int] = []
    for a, b, c, d in zip(tokens, tokens[1:], tokens[2:], tokens[3:]):
        if (
            a.type == tokenize.NAME
            and a.string == "self"
            and b.type == tokenize.OP
            and b.string == "."
            and c.type == tokenize.NAME
            and c.string == "call_from_thread"
            and d.type == tokenize.OP
            and d.string == "("
        ):
            offenders.append(a.start[0])
    return offenders


def test_no_bare_self_call_from_thread_outside_app() -> None:
    """Every `self.call_from_thread(` in the package must be
    `self.app.call_from_thread(`, except the documented app.py sites."""
    failures: dict[str, list[int]] = {}
    for py_file in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = py_file.relative_to(PACKAGE_ROOT).as_posix()
        if rel in ALLOWLISTED_RELATIVE_PATHS:
            continue
        lines = _bare_call_from_thread_lines(py_file)
        if lines:
            failures[rel] = lines

    assert not failures, (
        "found bare 'self.call_from_thread(' outside App subclasses -- "
        "only App defines call_from_thread, so this raises AttributeError "
        "at runtime (usually swallowed by a broad except Exception). Use "
        "'self.app.call_from_thread(' instead (see TASK-929):\n"
        + "\n".join(
            f"  {path}: lines {lines}" for path, lines in sorted(failures.items())
        )
    )


def test_app_py_allowlisted_sites_are_still_safe() -> None:
    """Pin down *why* app.py is exempt so the exemption cannot silently
    rot into a hole. Every bare `self.call_from_thread(` line remaining in
    app.py must sit inside `LibraryIngestQueueMixin` (always mixed with App)
    or `TldwCli` itself (a genuine App subclass) -- never inside a class
    that can be instantiated on its own, such as `TabDropdown(Widget)`.
    """
    app_py = PACKAGE_ROOT / "app.py"
    lines = _bare_call_from_thread_lines(app_py)
    assert lines, (
        "app.py no longer contains any bare 'self.call_from_thread(' sites "
        "-- remove it from ALLOWLISTED_RELATIVE_PATHS and this test"
    )

    tree = ast.parse(app_py.read_text(encoding="utf-8"))
    class_defs = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    safe_class_names = {"LibraryIngestQueueMixin", "TldwCli"}

    def enclosing_class(lineno: int) -> str | None:
        # Innermost (highest start-line) enclosing ClassDef that spans lineno.
        candidates = [
            node
            for node in class_defs
            if node.lineno <= lineno <= (node.end_lineno or node.lineno)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda n: n.lineno).name

    for lineno in lines:
        cls = enclosing_class(lineno)
        assert cls in safe_class_names, (
            f"app.py:{lineno} 'self.call_from_thread(' sits inside "
            f"{cls!r}, which is not one of the documented safe classes "
            f"{sorted(safe_class_names)} -- this may be a real bug, not "
            f"the LibraryIngestQueueMixin/App exemption"
        )
