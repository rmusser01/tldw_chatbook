"""stdlib-logging calls must use %-style, not loguru's {}.

task-15512. `logging.getLogger(...)` formats with `%`, so a call written in
loguru's style -- `logger.warning("... {} ...", value)` -- raises
`TypeError: not all arguments converted during string formatting` when the
record is formatted.

In production stdlib swallows that: it prints "--- Logging error ---" to stderr
and carries on, so the message is simply lost -- a diagnostic that is missing
exactly when something has gone wrong enough to log a warning. Under pytest it
is fatal: `_pytest.logging.LogCaptureHandler.handleError` deliberately re-raises
so bad log calls fail tests. That is how this was found -- one such call in the
Settings save path killed the Textual worker mid-save, and three tests reported
a mysterious timeout instead of the real assertion underneath.

Nineteen sites existed when this guard was written. It exists so the twentieth
fails here.
"""

from __future__ import annotations

import ast
import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[1] / "tldw_chatbook"
_LOG_METHODS = frozenset({"debug", "info", "warning", "error", "exception", "critical"})
_STDLIB_LOGGER_ASSIGNMENT = re.compile(
    r"(?m)^[ \t]*logger[ \t]*=[ \t]*logging\.getLogger\("
)


def _uses_stdlib_logger(path: pathlib.Path) -> bool:
    return bool(_STDLIB_LOGGER_ASSIGNMENT.search(path.read_text(errors="replace")))


def _modules_using_stdlib_logger() -> list[pathlib.Path]:
    return [path for path in sorted(_ROOT.rglob("*.py")) if _uses_stdlib_logger(path)]


def _brace_style_calls(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return (lineno, message) for `logger.x("...{}...", arg)` calls."""
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:  # pragma: no cover - a syntax error is another test's job
        return []
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "logger"
            and func.attr in _LOG_METHODS
        ):
            continue
        if len(node.args) < 2:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        if "{}" in first.value:
            found.append((node.lineno, first.value[:80].replace("\n", " ")))
    return found


def test_the_scan_finds_a_planted_violation(tmp_path):
    """The guard must be able to fail -- otherwise it proves nothing."""
    planted = tmp_path / "planted.py"
    planted.write_text(
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        'logger.warning("broken (value={}).", 1)\n'
    )

    assert _brace_style_calls(planted) == [(3, "broken (value={}).")]


def test_percent_style_is_not_flagged(tmp_path):
    """A correct call must not be reported."""
    ok = tmp_path / "ok.py"
    ok.write_text(
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        'logger.warning("fine (value=%s).", 1)\n'
    )

    assert _brace_style_calls(ok) == []


def test_stdlib_logger_module_detection_requires_exact_target(tmp_path):
    stdlib = tmp_path / "stdlib.py"
    stdlib.write_text("import logging\nlogger = logging.getLogger(__name__)\n")
    loguru = tmp_path / "loguru.py"
    loguru.write_text(
        "import logging\n"
        "from loguru import logger\n"
        "root_logger = logging.getLogger()\n"
        'logger.error("valid loguru value={}", 1)\n'
    )

    assert _uses_stdlib_logger(stdlib) is True
    assert _uses_stdlib_logger(loguru) is False


def test_no_stdlib_logger_uses_loguru_brace_style():
    modules = _modules_using_stdlib_logger()
    assert modules, (
        "found no stdlib-logging modules; the scan is looking in the wrong place"
    )

    offenders = []
    for path in modules:
        for lineno, message in _brace_style_calls(path):
            offenders.append(f"{path.relative_to(_ROOT.parent)}:{lineno}  {message}")

    assert not offenders, (
        "these stdlib-logging calls use loguru's {} style and will raise "
        "TypeError when formatted -- losing the message in production and "
        "failing the test that triggers them. Use %s:\n  " + "\n  ".join(offenders)
    )
