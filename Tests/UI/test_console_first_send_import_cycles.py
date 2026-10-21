"""The first-send warm-up imports no module-level import cycle (TASK-33620.15.16).

The post-UI-ready pre-import thread imports ``FIRST_SEND_MODULES`` while a
send on the UI thread may import the same modules. Python gives each module
its own import lock. If two modules import each other at module level and the
two threads enter the cycle at different modules, each holds one lock and
waits for the other; CPython breaks that deadlock by handing one thread the
other's partially initialised module, which surfaces as an ``AttributeError``
or ``ImportError`` in the send.

So the test takes the modules the warm-up really imports (a fresh interpreter
imports the app and the initial Console route, as at UI-ready, then the list
in order), reads each one's module-level imports from its source, and fails
on any cycle among them, naming its modules. Imports inside functions do not
count: they run after every module involved has finished initialising.

Static reading has two blind spots, both named here so nobody trusts the test
for them: an import performed by a function called at module level, and an
``importlib.import_module`` call at module level. Neither occurs among the
warm-up's modules today (the second is checked below).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# The per-test sandbox recovers a raw source selection the subprocess never
# reads; keep the collection-time profile, as the sibling pre-import tests do.
pytestmark = pytest.mark.bootstrap_profile

REPO_ROOT = Path(__file__).resolve().parents[2]

_WARMUP_SCRIPT = """
import json
import sys

import tldw_chatbook.app  # noqa: F401  (the boot closure)
import tldw_chatbook.UI.Screens.chat_screen  # noqa: F401  (the initial route)
from tldw_chatbook.Chat.console_first_send_modules import FIRST_SEND_MODULES

before = set(sys.modules)
for name in FIRST_SEND_MODULES:
    __import__(name)
added = {}
for name in sorted(set(sys.modules) - before):
    module = sys.modules[name]
    path = getattr(module, "__file__", None)
    if name.startswith("tldw_chatbook") and path and path.endswith(".py"):
        added[name] = {"file": path, "package": bool(getattr(module, "__path__", None))}
print("WARMUP_JSON:" + json.dumps({"listed": list(FIRST_SEND_MODULES), "added": added}))
"""


@pytest.fixture(scope="module")
def warm_up_modules(tmp_path_factory) -> dict:
    """The tldw modules the warm-up imports beyond UI-ready, with their files."""
    tmp_path = tmp_path_factory.mktemp("first_send_cycles")
    home, data, config_dir = (tmp_path / n for n in ("home", "data", "config"))
    for directory in (home, data, config_dir):
        directory.mkdir(mode=0o700)
    env = {
        **os.environ,
        "HOME": str(home),
        "USERPROFILE": str(home),
        "XDG_DATA_HOME": str(data),
        "XDG_CONFIG_HOME": str(config_dir),
        "TLDW_TEST_MODE": "1",
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)
    result = subprocess.run(
        [sys.executable, "-c", _WARMUP_SCRIPT],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    line = next(x for x in result.stdout.splitlines() if x.startswith("WARMUP_JSON:"))
    return json.loads(line[len("WARMUP_JSON:") :])


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _module_level_statements(body: list[ast.stmt]):
    """Yield the statements that run when the module body runs.

    Function and lambda bodies are skipped (they run later); class bodies,
    ``try``/``with``/``if`` bodies run at import and are walked, except the
    ``TYPE_CHECKING`` branch of an ``if``.
    """
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        yield node
        if isinstance(node, ast.If):
            if not _is_type_checking(node.test):
                yield from _module_level_statements(node.body)
            yield from _module_level_statements(node.orelse)
        elif isinstance(node, ast.ClassDef):
            yield from _module_level_statements(node.body)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            yield from _module_level_statements(node.body)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            yield from _module_level_statements(node.body)
            yield from _module_level_statements(node.orelse)
        elif isinstance(node, ast.Try) or type(node).__name__ == "TryStar":
            for block in (node.body, node.orelse, node.finalbody):
                yield from _module_level_statements(block)
            for handler in node.handlers:
                yield from _module_level_statements(handler.body)


_COMPOUND = (
    ast.If,
    ast.ClassDef,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.For,
    ast.AsyncFor,
    ast.While,
    getattr(ast, "TryStar", ast.Try),
)


def _module_level_imports(name: str, row: dict) -> tuple[set[str], list[str]]:
    """Modules ``name``'s body imports while it runs, and any dynamic import calls."""
    tree = ast.parse(Path(row["file"]).read_text(encoding="utf-8"))
    package = name if row["package"] else name.rpartition(".")[0]
    targets: set[str] = set()
    dynamic: list[str] = []
    for node in _module_level_statements(tree.body):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = (
                importlib.util.resolve_name(
                    "." * node.level + (node.module or ""), package
                )
                if node.level
                else node.module or ""
            )
            targets.add(base)
            # ``from pkg import sub`` imports the submodule too, when it is one.
            targets.update(f"{base}.{alias.name}" for alias in node.names)
        if isinstance(node, _COMPOUND):
            continue  # its body's statements are yielded on their own
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, (ast.Attribute, ast.Name))
                and getattr(inner.func, "attr", getattr(inner.func, "id", ""))
                == "import_module"
            ):
                dynamic.append(f"{name}:{inner.lineno}")
    return targets, dynamic


def _cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Strongly connected components with more than one module (Tarjan, iterative)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    found: list[list[str]] = []
    counter = 0
    for root in sorted(graph):
        if root in index:
            continue
        work = [(root, iter(sorted(graph[root])))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, edges = work[-1]
            advanced = False
            for nxt in edges:
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, iter(sorted(graph[nxt]))))
                    advanced = True
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    found.append(sorted(component))
    return found


def _graph(modules: dict) -> tuple[dict[str, set[str]], list[str]]:
    graph: dict[str, set[str]] = {}
    dynamic: list[str] = []
    for name, row in modules.items():
        targets, calls = _module_level_imports(name, row)
        graph[name] = {t for t in targets if t in modules and t != name}
        dynamic.extend(calls)
    return graph, dynamic


def test_the_cycle_finder_names_a_cycle_through_an_unlisted_module():
    """The finder is not vacuous: A -> B -> C -> A is one cycle, D is not in one."""
    graph = {"a": {"b"}, "b": {"c"}, "c": {"a", "d"}, "d": set()}
    assert _cycles(graph) == [["a", "b", "c"]]


@pytest.mark.integration
def test_the_warm_up_imports_no_module_level_cycle(warm_up_modules):
    """Red if two modules the warm-up imports import each other at module level."""
    modules = warm_up_modules["added"]
    # Anti-vacuity: the warm-up's closure beyond UI-ready is about ninety
    # modules (94 on 2026-10-20), and the hooks package a first send runs
    # must be in it.
    assert len(modules) > 60, sorted(modules)
    assert "tldw_chatbook.Agents.hooks_v2.lifecycle" in modules
    graph, _dynamic = _graph(modules)
    assert sum(len(edges) for edges in graph.values()) > 60, "no edges read"
    cycles = _cycles(graph)
    assert not cycles, (
        "module-level import cycles among the first-send warm-up's modules; "
        "the pre-import thread and a send can each hold one module's import "
        "lock and hand the other a partially initialised module: "
        + "; ".join(" <-> ".join(cycle) for cycle in cycles)
    )


@pytest.mark.integration
def test_the_warm_up_modules_make_no_dynamic_import_at_module_level(warm_up_modules):
    """The static graph's blind spot is empty: no module-level ``import_module`` call."""
    _graph_edges, dynamic = _graph(warm_up_modules["added"])
    assert not dynamic, (
        f"module-level dynamic imports the cycle check cannot see: {dynamic}"
    )
