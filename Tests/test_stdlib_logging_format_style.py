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
import string

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
    for node, loguru_names in _calls_with_loguru_names(tree):
        if "logger" in loguru_names:
            continue  # a loguru `logger` in this scope (TASK-41714)
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


def test_a_loguru_module_with_one_stdlib_function_is_scanned_by_scope(tmp_path):
    """Client_Media_DB_v2 binds a stdlib `logger` in one function only."""
    mixed = tmp_path / "mixed.py"
    mixed.write_text(
        "import logging\n"
        "from loguru import logger\n"
        'logger.info("loguru value={}", 1)\n'
        "def stdlib_scope():\n"
        "    logger = logging.getLogger(__name__)\n"
        '    logger.warning("broken value={}", 2)\n'
    )

    assert _uses_stdlib_logger(mixed) is True
    assert _brace_style_calls(mixed) == [(6, "broken value={}")]


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


# --- the opposite case: loguru calls in %-style (TASK-41714) -----------------
#
# Loguru formats with `str.format`, so `logger.info("route: %s", route)` logs a
# literal "route: %s" and drops `route`. Forty-nine such calls existed when
# this guard was written. A loguru logger here is a name imported from loguru,
# a name assigned from one (`logger = logger.bind(...)`), or an
# `*.loguru_logger` attribute; a `logger` rebound by `logging.getLogger` in its
# function or module is stdlib. Other attribute receivers (`module.logger`)
# are not resolved.

_LOGURU_METHODS = _LOG_METHODS | {"trace", "success"}
_LOGGER_CHAIN = frozenset({"opt", "bind", "patch"})
_PERCENT_PLACEHOLDER = re.compile(r"%(?:\(\w+\))?[-#0 +]*\d*(?:\.\d+)?[sdrifgeExXoc]")


def _own_nodes(scope: ast.AST) -> list[ast.AST]:
    """Return the nodes of ``scope``, not entering nested function scopes."""
    nodes, stack = [], list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        nodes.append(node)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.extend(ast.iter_child_nodes(node))
    return nodes


def _loguru_names(nodes: list[ast.AST], inherited: frozenset[str]) -> frozenset[str]:
    names = set(inherited)
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and node.module == "loguru":
            names.update(a.asname or a.name for a in node.names if a.name == "logger")
    for node in nodes:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            value = node.value
            if isinstance(value, ast.Call) and ast.unparse(value.func).endswith("getLogger"):
                names.discard(target.id)
            elif any(isinstance(n, ast.Name) and n.id in names for n in ast.walk(value)):
                names.add(target.id)
    return frozenset(names)


def _is_loguru_receiver(receiver: ast.expr, names: frozenset[str]) -> bool:
    while (
        isinstance(receiver, ast.Call)
        and isinstance(receiver.func, ast.Attribute)
        and receiver.func.attr in _LOGGER_CHAIN
    ):
        receiver = receiver.func.value
    if isinstance(receiver, ast.Name):
        return receiver.id in names
    return isinstance(receiver, ast.Attribute) and receiver.attr == "loguru_logger"


def _percent_message(call: ast.Call) -> str | None:
    """Return the message of a `%`-style call whose `{}` fields miss an argument."""
    func = call.func
    if not isinstance(func, ast.Attribute):
        return None
    index = 1 if func.attr == "log" else 0 if func.attr in _LOGURU_METHODS else None
    if index is None or len(call.args) <= index:
        return None
    message = call.args[index]
    if not (isinstance(message, ast.Constant) and isinstance(message.value, str)):
        return None
    text = message.value
    positional = len(call.args) - index - 1
    if not (positional or call.keywords) or not _PERCENT_PLACEHOLDER.search(text):
        return None
    try:
        parsed = string.Formatter().parse(text)
        fields = sum(1 for _, name, _, _ in parsed if name is not None)
    except ValueError:
        fields = 0  # not a valid str.format template at all
    # str.format drops positional arguments beyond its fields. Loguru binds
    # keywords no field names to `extra`, so keywords only count when the
    # message has no field at all: then a `%s` was meant for them (TASK-41714).
    return text if positional > fields or not fields else None


def _calls_with_loguru_names(tree: ast.AST) -> list[tuple[ast.Call, frozenset[str]]]:
    """Return each call with the loguru logger names of its scope."""
    found = []

    def visit(scope: ast.AST, inherited: frozenset[str]) -> None:
        nodes = _own_nodes(scope)
        names = _loguru_names(nodes, inherited)
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(node, names)
            elif isinstance(node, ast.Call):
                found.append((node, names))

    visit(tree, frozenset())
    return found


def _percent_style_loguru_calls(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return (lineno, message) for loguru calls written in `%`-style."""
    source = path.read_text(errors="replace")
    if "loguru" not in source or not _PERCENT_PLACEHOLDER.search(source):
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - a syntax error is another test's job
        return []
    found = []
    for call, names in _calls_with_loguru_names(tree):
        message = _percent_message(call)
        if message and _is_loguru_receiver(call.func.value, names):
            found.append((call.lineno, message[:80].replace("\n", " ")))
    return sorted(found)


def test_the_loguru_scan_finds_a_planted_percent_style_call(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text(
        "import logging\n"
        "from loguru import logger\n"
        "def stdlib_scope():\n"
        "    logger = logging.getLogger(__name__)\n"
        '    logger.warning("fine (value=%s).", 1)\n'
        'logger.warning("dropped (value=%s).", 1)\n'
        'logger.opt(exception=True).error("dropped too: %d", 2)\n'
        'logger.info("fine (value={}).", 3)\n'
        'logger.info("fine, no arguments: %s")\n'
        "def bound(app):\n"
        "    log = logger.bind(x=1)\n"
        '    log.debug("dropped: %r", 4)\n'
        '    app.loguru_logger.info("dropped: %s", 5)\n'
    )

    assert _percent_style_loguru_calls(planted) == [
        (6, "dropped (value=%s)."),
        (7, "dropped too: %d"),
        (12, "dropped: %r"),
        (13, "dropped: %s"),
    ]


def test_the_loguru_scan_flags_a_mixed_brace_and_percent_call(tmp_path):
    """A `{}` does not excuse a `%s` whose argument str.format drops."""
    planted = tmp_path / "mixed.py"
    planted.write_text(
        "from loguru import logger\n"
        'logger.info("{} rows to %s", 3, "path")\n'
        'logger.info("{}% done", 50)\n'
        'logger.info("{} rows to {}", 3, "path")\n'
    )

    assert _percent_style_loguru_calls(planted) == [(2, "{} rows to %s")]


def test_the_loguru_scan_flags_keyword_only_percent_calls(tmp_path):
    """Keyword arguments are format arguments too; `%s` reaches none of them."""
    planted = tmp_path / "keywords.py"
    planted.write_text(
        "from loguru import logger\n"
        'logger.info("x %s", value=3)\n'
        'logger.error("Failed: %s", exc_info=True)\n'
        'logger.info("x {value}", value=3)\n'
    )

    assert _percent_style_loguru_calls(planted) == [(2, "x %s"), (3, "Failed: %s")]


def test_the_loguru_scan_spares_unused_keywords_beside_brace_fields(tmp_path):
    """Loguru binds keywords no field names to `extra`; a literal `%` is text."""
    planted = tmp_path / "extra.py"
    planted.write_text(
        "from loguru import logger\n"
        'logger.info("50% of {}", 3, user="u")\n'
        'logger.info("{} at 5% rate", 3, extra=1)\n'
    )

    assert _percent_style_loguru_calls(planted) == []


def test_no_loguru_call_uses_percent_style_with_arguments():
    offenders = [
        f"{path.relative_to(_ROOT.parent)}:{lineno}  {message}"
        for path in sorted(_ROOT.rglob("*.py"))
        for lineno, message in _percent_style_loguru_calls(path)
    ]

    assert not offenders, (
        "these loguru calls use stdlib's %-style; loguru formats with "
        "str.format, so they log a literal %s and drop their values. Use {}:\n  "
        + "\n  ".join(offenders)
    )
