"""No `async def` in the watchlists service may touch sqlite directly.

task-19562 AC7. Part B moved 22 `async def` methods off the event loop by
routing every synchronous `SubscriptionsDB` call through
`db_offload.run_db_off_loop`. `Tests/Subscriptions/
test_watchlists_service_off_loop.py` pins that per method, by behaviour --
it proves the calls that exist today run on a worker thread. What it cannot
do is fail when someone adds a *new* `async def` next month with a fresh
inline `db.get_subscription(...)` in it: a per-method test only covers the
methods it names.

This is that guard. It parses the module and rejects any synchronous
database call reached directly from an `async def` body, whatever the method
is called. The check is structural, so it applies to code that does not exist
yet -- which is the whole point.

What counts as "directly": a `Call` whose attribute chain is rooted at a
value this module knows to be a database -- the `db` local produced by
`self._db()`, a parameter annotated `SubscriptionsDB`, or anything derived
from one (`db.conn`, a cursor taken from it). Passing a *bound method* to the
offload helper (`run_db_off_loop(db, db.get_subscription, source_id)`) is an
`Attribute`, not a `Call`, so the correct shape is accepted by construction
rather than by an exemption list.

Synchronous helpers (`def`, not `async def`) are deliberately NOT checked:
they are the bodies handed to `run_db_off_loop`, so sqlite is exactly what
they are supposed to contain. The guard is about which *thread* the work
lands on, and only an `async def` body runs on the event loop.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import tldw_chatbook.Subscriptions.local_watchlists_service as service_module

pytestmark = pytest.mark.unit

SERVICE_PATH = pathlib.Path(service_module.__file__)

#: The helper that makes a sqlite call legitimate inside an `async def`.
OFFLOAD_HELPER = "run_db_off_loop"

#: Names that are a `SubscriptionsDB` (or a cursor/connection taken from one)
#: wherever they appear in this module. `db` is the near-universal local for
#: `self._db()`; the others are the parameter names the offloaded helpers use.
_DB_ROOT_NAMES = frozenset({"db", "database", "conn", "cursor"})


def _attribute_root(node: ast.AST) -> str | None:
    """The leftmost `Name` id of an attribute chain, or None.

    `db.conn.execute` -> "db"; `self._db().get_subscription` -> None (the
    chain is rooted at a Call, which `_call_roots_at_db` handles separately).
    """
    while isinstance(node, ast.Attribute):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def _is_self_db_call(node: ast.AST) -> bool:
    """True for `self._db()` -- the module's one way to obtain the database."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_db"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    )


def _db_rooted(func: ast.AST) -> bool:
    """True when this call's receiver is (or came from) the database."""
    if not isinstance(func, ast.Attribute):
        return False
    # `self._db().get_subscription(...)` -- the chain bottoms out in the
    # accessor itself rather than in a name.
    inner = func.value
    while isinstance(inner, ast.Attribute):
        inner = inner.value
    if _is_self_db_call(inner):
        return True
    return _attribute_root(func) in _DB_ROOT_NAMES


def _offloaded_call_nodes(tree: ast.AST) -> set[int]:
    """Every node id sitting inside a `run_db_off_loop(...)` argument list.

    A call passed as an *argument* to the offload helper is not executed on
    the loop -- the helper is what runs it, on a worker thread. In practice
    the arguments are bound methods rather than calls, but a lambda or an
    inline expression there is equally offloaded, so the whole argument
    subtree is exempt.
    """
    exempt: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else None
        )
        if name != OFFLOAD_HELPER:
            continue
        for argument in [*node.args, *(kw.value for kw in node.keywords)]:
            for inner in ast.walk(argument):
                exempt.add(id(inner))
    return exempt


def _loop_body_nodes(function: ast.AST) -> list[ast.AST]:
    """Every node that actually runs on the event loop for this `async def`.

    Descends the body but STOPS at a nested `def`/`async def`/`lambda`: a
    synchronous closure defined inside an `async def` is an offload body --
    the shape `run_db_off_loop` is handed -- so sqlite is exactly what it is
    supposed to contain, and only the thread it lands on matters.

    Review note (task-19562): this replaced an `ast.walk` loop that
    `continue`d on a nested `FunctionDef`. `ast.walk` yields descendants
    regardless, so that skipped the `def` node and then walked its body
    anyway -- the guard rejected the correct offload shape:

        async def list_sources(self):
            db = self._db()
            def work():
                return db.get_all_subscriptions()   # <- was flagged
            return await run_db_off_loop(db, work)

    A guard that fails on correct code gets weakened by the next person, so
    the exclusion is done by descent rather than by a `continue` that cannot
    perform it.
    """
    collected: list[ast.AST] = []

    def descend(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            ):
                continue
            collected.append(child)
            descend(child)

    descend(function)
    return collected


def _blocking_db_calls_in_async_defs(source: str) -> list[str]:
    """Every direct database call reachable from an `async def` body."""
    tree = ast.parse(source)
    exempt = _offloaded_call_nodes(tree)
    findings: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        for inner in _loop_body_nodes(node):
            if not isinstance(inner, ast.Call) or id(inner) in exempt:
                continue
            if _db_rooted(inner.func):
                rendered = ast.unparse(inner.func)
                findings.append(
                    f"{node.name} (line {inner.lineno}): {rendered}(...)"
                )
    return findings


def test_no_async_def_in_the_service_calls_sqlite_directly():
    """The guard: a new `async def` doing inline sqlite fails this test."""
    findings = _blocking_db_calls_in_async_defs(SERVICE_PATH.read_text())
    assert findings == [], (
        "these `async def` bodies call SubscriptionsDB directly, blocking the "
        "event loop for the query's duration. Route the call through "
        f"`{OFFLOAD_HELPER}` (see `Subscriptions/db_offload.py`):\n  "
        + "\n  ".join(findings)
    )


def test_the_guard_detects_an_inline_call():
    """The guard must be able to fail -- proven on a synthetic module.

    Without this, `findings == []` above is indistinguishable from a checker
    that never finds anything (the failure mode this repo has shipped before:
    an inert guard that passes forever).
    """
    offending = (
        "class S:\n"
        "    async def list_sources(self):\n"
        "        db = self._db()\n"
        "        return db.get_all_subscriptions()\n"
    )
    assert _blocking_db_calls_in_async_defs(offending) == [
        "list_sources (line 4): db.get_all_subscriptions(...)"
    ]


def test_the_guard_accepts_the_offloaded_shape():
    """The correct shape must not be flagged, or the guard is unusable."""
    accepted = (
        "class S:\n"
        "    async def list_sources(self):\n"
        "        db = self._db()\n"
        "        return await run_db_off_loop(db, db.get_all_subscriptions)\n"
    )
    assert _blocking_db_calls_in_async_defs(accepted) == []


def test_the_guard_accepts_a_nested_synchronous_offload_body():
    """The other correct shape: a sync closure handed to the helper.

    Review of task-19562. The guard used to flag this -- its walker
    `continue`d on the nested `def` but `ast.walk` had already queued that
    def's children, so the closure's sqlite was reported as loop code. A
    guard that rejects the correct pattern is worse than no guard: it gets
    deleted or loosened the first time someone writes the pattern.
    """
    accepted = (
        "class S:\n"
        "    async def list_sources(self):\n"
        "        db = self._db()\n"
        "        def work():\n"
        "            return db.get_all_subscriptions()\n"
        "        return await run_db_off_loop(db, work)\n"
    )
    assert _blocking_db_calls_in_async_defs(accepted) == []


def test_the_guard_still_flags_an_inline_call_beside_a_nested_def():
    """Excluding nested bodies must not blind the guard to its own scope."""
    offending = (
        "class S:\n"
        "    async def list_sources(self):\n"
        "        db = self._db()\n"
        "        def work():\n"
        "            return db.get_all_subscriptions()\n"
        "        db.mark_all_read(1)\n"
        "        return await run_db_off_loop(db, work)\n"
    )
    findings = _blocking_db_calls_in_async_defs(offending)
    assert findings == ["list_sources (line 6): db.mark_all_read(...)"], findings


def test_the_guard_sees_through_a_bare_connection_read():
    """`db.conn.execute(...)` is the shape the original AST sweep missed.

    Three of the 22 methods (`get_alert_rule`, `list_runs`,
    `list_alert_rules`) read through a bare `db.conn.cursor()` and were
    invisible to a scan that only matched named `SubscriptionsDB` methods.
    """
    offending = (
        "class S:\n"
        "    async def list_runs(self):\n"
        "        db = self._db()\n"
        "        return db.conn.execute('SELECT 1').fetchall()\n"
    )
    findings = _blocking_db_calls_in_async_defs(offending)
    assert findings and "list_runs" in findings[0], findings
