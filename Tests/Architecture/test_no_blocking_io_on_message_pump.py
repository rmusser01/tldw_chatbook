"""Repo-wide guard: blocking I/O must not be reachable from a message handler.

Textual runs message handlers on a serialized pump. Anything blocking that a
handler reaches -- directly, or through the functions it calls -- stops the app
processing clicks, keys and navigation for the whole duration.

TASK-1320 fixed four mount-path instances, with measured stalls of 1030ms
(chatbook directory scan), 1140ms (character library read) and up to 300s
(unreachable server). TASK-1373 added this guard plus two more instances it
found outside the mount path: `subprocess.run(["xdg-open", ...])` from a button
handler in the Chatbook wizard and in the export-management window.

**This guard walks the call graph, and that is the whole point.** The blocking
call is almost always a level or more below the handler --
`on_mount -> _refresh_chatbooks -> .glob(` -- so a scan that reads only each
handler's own body reports a clean result against code known to be broken. That
is not hypothetical: the first draft of this scan returned zero findings against
the pre-TASK-1320 chatbooks source. ``test_guard_detects_a_known_blocking_shape``
below exists so this can never silently regress into a scan that always passes.

Scope is deliberately narrow. Only calls whose cost is user-visible are flagged.
Small local filesystem operations are not: `mkdir(exist_ok=True)` plus a small
JSON write measures 0.049ms and `read_text` of a config file 0.014ms, four
orders of magnitude below the cases above. Flagging them would produce a large
noisy baseline, and "fixing" them would be churn with real risk -- deferring work
into a worker has its own failure modes (`run_worker` defaults to
`exit_on_error=True`, which turns a load error into an app exit).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent / "tldw_chatbook"
SCANNED_SUBDIRS = ("UI", "Widgets")

#: Handlers Textual invokes on a message pump.
HANDLER_PREFIXES = ("on_", "watch_", "action_")

#: Attribute calls that walk the filesystem. Deliberately excludes `mkdir`,
#: `stat`, `read_text` and `write_text` -- see the module docstring.
BLOCKING_ATTRS = frozenset({"glob", "iterdir"})

#: Bare names whose call blocks.
BLOCKING_NAMES = frozenset(
    {"fetch_all_characters", "fetch_character_names", "ZipFile", "urlopen"}
)

#: Dotted calls whose cost is a network round trip or a child process.
BLOCKING_QUALIFIED = frozenset(
    {
        "requests.get",
        "requests.post",
        "requests.put",
        "requests.delete",
        "httpx.get",
        "httpx.post",
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_output",
        "time.sleep",
    }
)

#: A hop that moves work off the pump ends the walk -- everything below it is
#: already someone else's thread or worker.
HANDOFFS = frozenset(
    {
        "to_thread",
        "run_worker",
        "call_from_thread",
        "set_timer",
        "call_after_refresh",
        "call_later",
    }
)

# Accepted pre-existing paths, each with its own reason. Never a blanket
# suppression: a new entry here is a decision that has to be argued.
#
# Keyed on the BLOCKING CALL as well as the file and handler. Keying only on
# (file, handler) left a hole exactly where it mattered: this window's
# `on_button_pressed` is baselined for a cheap glob, and under the coarser key a
# newly added `subprocess.run` reachable from that same handler -- the very bug
# TASK-1373 fixed here -- would have been silently accepted.
BASELINE: dict[tuple[str, str, str], str] = {
    # Dead file: nothing imports `Chatbooks_Window` (the pre-"Improved" copy).
    # Verified by grep for `Chatbooks_Window import` across the package --
    # only `Chatbooks_Window_Improved` has importers. Left flagged rather than
    # deleted because retiring the file is a separate decision.
    (
        "UI/Chatbooks_Window.py",
        "on_mount",
        "self._export_path.glob",
    ): "dead file, no importers",
    (
        "UI/Chatbooks_Window.py",
        "on_button_pressed",
        "self._export_path.glob",
    ): "dead file, no importers",
    (
        "UI/Chatbooks_Window.py",
        "action_refresh",
        "self._export_path.glob",
    ): "dead file, no importers",
    # `glob("*.zip")` plus one `stat` per entry -- no per-file archive open or
    # parse, unlike the pre-TASK-1320 scan that measured 1030ms. A stat per
    # chatbook is sub-millisecond for any realistic export directory.
    (
        "UI/ChatbookExportManagementWindow.py",
        "on_mount",
        "self.chatbooks_dir.glob",
    ): "glob + one stat per entry, no archive read",
    (
        "UI/ChatbookExportManagementWindow.py",
        "on_button_pressed",
        "self.chatbooks_dir.glob",
    ): "glob + one stat per entry, no archive read",
    # `glob("*.toml")` plus a `toml.load` per user-created theme, when the theme
    # editor is shown. Bounded by how many themes the user has authored by hand,
    # realistically single digits.
    (
        "Widgets/settings_theme_editor.py",
        "on_show",
        "self.custom_themes_path.glob",
    ): "one toml parse per hand-authored user theme",
}


def _qualified(node: ast.AST) -> str:
    """Render a call target as a dotted string (`subprocess.run`, `x.y.glob`)."""
    if isinstance(node, ast.Attribute):
        base = _qualified(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


class _Function:
    """One function: what it calls, what it blocks on, whether it hands off."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: set[str] = set()
        #: Callees passed INTO a hand-off (`run_worker(self._refresh())`). They
        #: run on a worker or thread, so they are not on the pump and must not
        #: be walked -- tracked separately rather than by a blanket "this
        #: function hands off somewhere" flag, which would also excuse blocking
        #: work the handler still does directly.
        self.deferred: set[str] = set()
        self.blocking: list[str] = []


def _is_worker(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether Textual's ``@work`` runs this function off the message pump.

    Matches both ``@work`` and ``@work(thread=True)``. Without this the guard
    reports a false positive for every handler that calls a decorated worker:
    `HuggingFace/local_models_widget.on_button_pressed -> _delete_model ->
    iterdir` was flagged even though `_delete_model` is `@work(thread=True)` and
    never touches the pump. A hand-off is a property of the CALLEE as much as of
    the call site.
    """
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if _qualified(target).rsplit(".", 1)[-1] == "work":
            return True
    return False


def _analyse(source: str) -> tuple[dict[str, _Function], list[str]]:
    """Build the intra-module call graph and list the handler entry points."""
    tree = ast.parse(source)
    functions: dict[str, _Function] = {}
    handlers: list[str] = []
    workers: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _is_worker(node):
            workers.add(node.name)
        info = _Function(node.name)
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            qualified = _qualified(sub.func)
            leaf = qualified.rsplit(".", 1)[-1] if qualified else ""
            if leaf in HANDOFFS:
                for arg in sub.args:
                    for inner in ast.walk(arg):
                        if isinstance(inner, (ast.Call, ast.Attribute, ast.Name)):
                            target = _qualified(
                                inner.func if isinstance(inner, ast.Call) else inner
                            )
                            if target:
                                info.deferred.add(target.rsplit(".", 1)[-1])
            if (
                qualified in BLOCKING_QUALIFIED
                or leaf in BLOCKING_NAMES
                or leaf in BLOCKING_ATTRS
            ):
                info.blocking.append(qualified or leaf)
            if leaf:
                info.calls.add(leaf)
        functions[node.name] = info
        if node.name.startswith(HANDLER_PREFIXES):
            handlers.append(node.name)

    # A call to a `@work`-decorated function is a hand-off wherever it appears.
    for info in functions.values():
        info.deferred |= workers
    return functions, handlers


def _paths_to_blocking(functions: dict[str, _Function], entry: str) -> list[list[str]]:
    """Every path from `entry` down to a blocking call, stopping at a hand-off."""
    found: list[list[str]] = []

    def walk(name: str, path: list[str], seen: frozenset[str]) -> None:
        info = functions.get(name)
        if info is None or name in seen:
            return
        if info.blocking:
            found.append(path + [f"<{info.blocking[0]}>"])
            return
        # Anything handed to a worker/thread is no longer on the pump.
        for callee in sorted(info.calls - info.deferred):
            if callee in functions:
                walk(callee, path + [callee], seen | {name})

    walk(entry, [entry], frozenset())
    return found


def _scan_package() -> dict[tuple[str, str, str], list[str]]:
    """Map (relative path, handler, blocking call) -> the chain that reaches it."""
    findings: dict[tuple[str, str, str], list[str]] = {}
    for subdir in SCANNED_SUBDIRS:
        for path in sorted((PACKAGE_ROOT / subdir).rglob("*.py")):
            try:
                functions, handlers = _analyse(path.read_text())
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
                continue
            relative = path.relative_to(PACKAGE_ROOT).as_posix()
            for handler in handlers:
                for chain in _paths_to_blocking(functions, handler):
                    call = chain[-1].strip("<>")
                    findings[(relative, handler, call)] = chain
    return findings


@pytest.mark.unit
def test_guard_detects_a_known_blocking_shape():
    """The guard must catch a handler that reaches blocking I/O one level down.

    Load-bearing. A body-only scan of this exact shape reports nothing, and that
    is how the first draft of this guard came to pass against the real
    pre-TASK-1320 chatbooks code. Without this test, a future refactor could
    quietly reduce the guard to something that always succeeds.
    """
    source = """
class Thing:
    def on_mount(self):
        self._refresh()

    def _refresh(self):
        for item in self._dir.glob("*.zip"):
            pass
"""
    functions, handlers = _analyse(source)
    assert handlers == ["on_mount"]
    chains = _paths_to_blocking(functions, "on_mount")
    assert chains, "guard is blind to blocking I/O one call below the handler"
    assert chains[0] == ["on_mount", "_refresh", "<self._dir.glob>"]


@pytest.mark.unit
def test_guard_respects_a_handoff():
    """Work moved off the pump must not be reported."""
    source = """
class Thing:
    def on_mount(self):
        self.run_worker(self._refresh())

    async def _refresh(self):
        for item in self._dir.glob("*.zip"):
            pass
"""
    functions, _ = _analyse(source)
    assert not _paths_to_blocking(functions, "on_mount"), (
        "a handler that hands the work to a worker is not blocking the pump"
    )


@pytest.mark.unit
def test_no_new_blocking_io_is_reachable_from_a_message_handler():
    """No handler may reach blocking I/O unless it is baselined with a reason."""
    findings = _scan_package()
    unexpected = {key: chain for key, chain in findings.items() if key not in BASELINE}

    assert not unexpected, "blocking I/O is reachable from a message handler:\n" + "\n".join(
        f"  {path}::{handler}\n      {' -> '.join(chain)}"
        for (path, handler, _call), chain in sorted(unexpected.items())
    ) + (
        "\n\nTextual runs handlers on a serialized pump, so this freezes the whole "
        "app for the duration. Move the work off the pump (a thread for blocking "
        "calls, a worker for awaited ones) -- and if you defer into a worker, pass "
        "exit_on_error=False so a failure cannot exit the app. If the cost is "
        "genuinely negligible, add it to BASELINE with the measurement."
    )


@pytest.mark.unit
def test_baseline_has_no_stale_entries():
    """A baselined path that is now clean must be removed, not left to rot."""
    findings = _scan_package()
    stale = sorted(key for key in BASELINE if key not in findings)

    assert not stale, (
        "these BASELINE entries no longer report anything and should be deleted:\n"
        + "\n".join(f"  {path}::{handler} ({call})" for path, handler, call in stale)
    )


@pytest.mark.unit
def test_guard_respects_a_work_decorated_callee():
    """Calling a `@work` function is a hand-off, even without run_worker.

    A hand-off is a property of the callee as much as of the call site. Missing
    this reported a false positive against
    `HuggingFace/local_models_widget.on_button_pressed`, whose `_delete_model`
    is `@work(thread=True)` and never touches the pump.
    """
    source = """
class Thing:
    def on_button_pressed(self, event):
        self._delete(event.path)

    @work(thread=True)
    def _delete(self, path):
        if not any(path.parent.iterdir()):
            pass
"""
    functions, _ = _analyse(source)
    assert not _paths_to_blocking(functions, "on_button_pressed"), (
        "a @work-decorated callee runs off the pump and must not be reported"
    )
