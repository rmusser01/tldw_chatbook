#!/usr/bin/env python3
"""Guard: Textual's worker contract, and DOM lookups that resume after an await.

TASK-32800.4. Three of the four P0s in the 2026-09-17 core-runtime review
(`qa/core-code-review-2026-09-17/report.md`) were the same defect class, and
nothing in the repo could catch it:

* `Widgets/audio_troubleshooting_dialog.py` ran a plain ``def`` as an async
  worker. ``Worker._run_async`` raises ``WorkerError("Request to run a
  non-async function as an async worker")`` for a non-coroutine target,
  ``Worker._run`` catches that generically, and ``exit_on_error`` defaults to
  ``True`` -- so opening the dialog exited the whole application.
* `UI/MCP_Modules/mcp_inspector.py` dereferenced ``#mcp-adv-content`` after
  awaiting a section load. The sibling handler removes that subtree, and
  ``exclusive=True`` does not help (it cancels another worker in the same
  *group*, never the running one), so the await resumed into a removed
  subtree, raised ``NoMatches``, and again exited the application.
* `Widgets/Library/library_prompts_canvas.py` kept a flag describing a region
  that a recompose had removed, and the next key press dereferenced it.

Two rules, deliberately of different strengths:

W001 (hard gate, zero tolerance)
    ``run_worker(<target>)`` whose target statically resolves to a
    non-coroutine method, without ``thread=True``. This is precise: the whole
    package currently has zero violations, so any new one is a real defect
    and there is no allowlist to hide behind.

W002 (census ratchet, not a gate)
    A ``query_one``/``query_exactly_one`` that is reached after an ``await``
    inside an ``async def``, with no enclosing ``try`` **body whose handlers
    can catch it**. Both qualifiers are load-bearing, and the check shipped
    with neither:

    * "body" -- a lookup in an ``except``/``else``/``finally`` clause is a
      child of the same ``Try`` node while sitting outside the region its own
      handlers cover; an exception there propagates straight out. Treating any
      ``ast.Try`` ancestor as protection hid 50 sites across 21 functions
      (269 reported vs 319 real) behind a green check.
    * "can catch it" -- ``try``/``finally`` with no handler, and
      ``except ValueError``, do not catch the ``NoMatches`` a failed lookup
      raises. Counting them as protection hid a further 24 occurrences across
      14 functions (319 vs 343), among them
      ``Widgets/Persona_Widgets/petdex_import_review.py::_load``, whose five
      post-``push_screen_wait`` lookups sit under ``except (ValueError,
      OSError, RuntimeError)``.

    Whether this crashes depends on whether the awaited work can remove the
    subtree, which is not statically decidable -- there are 346 such sites
    today and the vast majority are fine. Failing on all of them would be
    exactly the guard that cries wolf and gets muted, which
    ``scripts/preflight.sh`` warns about in its own header. So the existing
    sites are recorded in a census and the check fails only when a site
    appears that is not in it. Shrinking the
    census is always allowed; growing it is a deliberate act.

    The census rows are a *baseline*, not an endorsement: they were captured
    mechanically and have not been individually reviewed.

W003 (census ratchet, TASK-33621.13)
    A wait-for-dismiss screen push -- ``push_screen_wait(...)`` or
    ``push_screen(..., wait_for_dismiss=True)`` -- reachable from a coroutine
    that is not a worker. Textual APPENDS the screen to the stack and only
    then raises ``NoActiveWorker``; the exception kills the message loop of
    whatever pump was dispatching, with the pushed screen already painted on
    top. GAP4-01 (Console UX review 2026-09-29) was this: the Conversation
    Inspector's ``@on`` handler awaited a recovery callable that awaited a
    folder picker, and the whole app froze -- Ctrl+Q included.

    The same wait by hand counts too: ``push_screen(..., callback=done)``
    in a function that also awaits a future or event it created (matched by
    shape: a push is left out only when ``done`` visibly settles nothing but
    OTHER fresh futures that function created, and those futures reach
    nothing -- no done-callback, relay, handoff or rebinding could chain
    them to the awaited one, for a ``self`` future anywhere in the package;
    see ``_settled_by``. Settling a different future alone is not enough,
    and anything not shown counts. The check is syntactic: ``getattr`` and
    ``__dict__`` access are not seen). No
    ``NoActiveWorker`` -- but Textual runs ``done`` through the requester
    pump's ``call_next``, and from a handler that pump is the one blocked on
    the await, so it deadlocks. PR #2922's
    ``request_hook_review`` was this, and froze the Console's Send
    (TASK-33621.28). It now awaits a future the modal settles in its own
    ``dismiss``, which no pump has to flush -- a shape W003 does not match.
    That is not a licence to await it on the APP pump: the app pump delivers
    the keys that dismiss the modal, so awaiting any screen's answer there
    still freezes the app, and W003 cannot see that either. The Console hands
    a Send's review to a worker unless the caller IS a worker's own task
    (``hooks.in_worker_task``: a screen pushed from a worker inherits that
    worker's contextvar, so ``get_current_worker()`` alone answers "worker"
    on its pump), and ``request_hook_review`` logs an ERROR when awaited off
    one; the runtime proof is
    ``Tests/UI/test_console_hook_review_send_freeze.py``. The wait may be
    split across two functions (TASK-33621.33): a helper that creates the
    future and pushes with ``callback=``, then RETURNS the future, is a site
    wherever it is awaited -- ``await helper()``, ``fut = helper(); await
    fut``, ``await helper().wait()`` -- and so is a helper that STORES it on
    ``self`` for a method of the same object (its class, a base, a
    subclass) to await. Not followed: a future created by the caller and
    completed by a push in a helper it calls.

    The roots are message handlers (``@on``, ``on_*``/``_on_*``, ``key_*``),
    actions (``action_*``) and watchers (``watch_*``), none of which Textual
    runs in a worker, plus a callable handed to ``call_later``/``call_next``/
    ``call_after_refresh``/``set_timer``/``set_interval``, a ``push_screen``
    result ``callback`` (run through ``call_next``), or a coroutine handed to
    ``create_task``/``ensure_future`` (a separate task, so only a real
    ``push_screen_wait`` counts there: the scheduling pump is free to run a
    hand-rolled wait's callback). A root is reported when it pushes directly,
    or ``await``s -- transitively -- something that does. ``@work``
    functions, and coroutines handed to ``run_worker`` (a push built inline as
    its argument included), are workers and stop the propagation.
    ``call_from_thread`` targets are not roots: the loop runs them in a copy
    of the calling thread worker's context, where that worker is active.

    "Transitively" follows callables passed as values, deliberately, because
    the real defect crossed three modules that way: ``partial(recover,
    select_binding=controller._select_binding)`` in a dict whose key became
    the Inspector's ``__init__`` parameter and then its
    ``self._project_instruction_recovery`` attribute. So a keyword argument,
    a string-keyed dict entry, or an assignment whose value refers to a
    waiting callable (``partial`` and a one-call ``lambda`` unwrapped) makes
    its keyword/key/target name an alias for one; an ``await`` of an alias
    waits too. A POSITIONAL argument binds the parameter at its position,
    exactly as the keyword form binds it by name: ``HooksController(_pick)``
    is ``HooksController(request_review=_pick)`` once the callee resolves (a
    class to its package ``__init__``, a function, a ``self``/``super()``
    method, ``obj.f`` by name). A waiting callable whose positional
    parameter W003 cannot name -- the callee is outside the package, or
    takes ``*args`` -- is reported as its own census row, ``<function> ->
    <callee>#<n> => <site>``, rather than dropped (TASK-33621.33).
    ``self.x()`` resolves as dispatch does, on every class
    ``self`` can be: the enclosing class AND each of its in-package
    subclasses, each through its own package base classes and what they
    assign to ``self.x``. So a base-class template method reaches a
    subclass's override, and a mixin reaches the class that mixes it in and
    that class's other mixins -- but never an unrelated class's ``x``. A def
    that a later def of the same name rebinds in the same statement list,
    before anything reads it, is dead code: never a root, never reached by
    name, and neither is anything defined inside it, nor any binding its
    body makes. Anything else stays live: alternatives in ``if``/``else`` or
    ``try``/``except`` branches (whichever runs binds the name), a property
    getter that its own ``@x.setter`` reads, a def handed on before it was
    rebound, a def whose own decorator may have registered it. A bare
    ``x()`` resolves lexically first -- a nested def or local alias of
    an enclosing function (through a class defined inside one, too), then a
    module-level ``x``. A bare name written in a CLASS BODY (``choose =
    _pick``, not a lambda's body) reads that class's own namespace -- never
    its bases' -- before the enclosing function's and the module's, as
    Python does, so it can name one of that class's methods; anywhere else a
    bare name never reaches a method, nor any ATTRIBUTE binding (a
    class-body name, ``self.x = ...``, ``obj.x = ...``) anywhere.
    ``obj.x()`` is resolved by NAME against
    every definition, methods included, and an imported ``x()`` against every
    module-level function: two unrelated functions sharing a
    name are one to it. That over-approximation
    is why W003 is a census like W002 rather than a zero-tolerance gate: the
    pre-existing rows are pinned in ``scripts/textual_wait_push_census.tsv``
    and only a NEW one fails. A row is an entry point AND the function
    holding the push it reaches, so a new push reachable from a censused
    entry point is a new row. Those rows are an unreviewed baseline -- each
    may be a real freeze of the GAP4-01 kind -- not an endorsement. A row
    that HAS been reviewed carries a third, tab-separated column: its
    verdict, evidence and follow-up, which ``--write`` preserves while the
    row survives (in both censuses).

Stdlib-only, like the other derived-artifact checkers, so it runs with no
dependency install.

Usage:
    python scripts/check_textual_worker_contract.py
    python scripts/check_textual_worker_contract.py --write   # re-pin W002+W003
"""

from __future__ import annotations

import argparse
import ast
import warnings
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "tldw_chatbook"
CENSUS = REPO_ROOT / "scripts" / "textual_await_dom_census.tsv"
WAIT_PUSH_CENSUS = REPO_ROOT / "scripts" / "textual_wait_push_census.tsv"

#: Directories that never ship a Textual widget.
SKIP_PARTS = {".venv", "Third_Party", "__pycache__", "node_modules"}

#: W002 only looks at the packages that define screens and widgets. Service
#: and database modules have no DOM to lose.
UI_PACKAGES = {"UI", "Widgets"}

DOM_LOOKUPS = {"query_one", "query_exactly_one"}


def _source_files() -> list[Path]:
    return [
        path
        for path in sorted(PACKAGE.rglob("*.py"))
        if not SKIP_PARTS & set(path.parts)
    ]


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _is_coroutine_def(node: ast.AST) -> bool:
    """True for ``async def``, and for ``@work``-decorated methods.

    A ``@work`` decorator returns a function that schedules a Worker; Textual
    accepts it either way, and ``inspect.iscoroutinefunction`` reports False on
    the decorated attribute, which is precisely the false positive that fooled
    the review's own first enumeration of this defect.
    """
    if isinstance(node, ast.AsyncFunctionDef):
        return True
    if not isinstance(node, ast.FunctionDef):
        return False
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = getattr(target, "attr", None) or getattr(target, "id", None)
        if name == "work":
            return True
    return False


def _is_property(node: ast.AST) -> bool:
    """A ``@property`` hands back a value we cannot resolve statically.

    ``UI/Console_Modules/agent.py`` has a property whose value is an ``async
    def`` on the screen; treating the property itself as the target reported a
    false positive, so these are skipped rather than guessed at.
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    return any(
        getattr(decorator, "id", None) == "property"
        for decorator in node.decorator_list
    )


def _class_methods(cls: ast.ClassDef) -> dict[str, ast.AST]:
    return {
        node.name: node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _worker_target(call: ast.Call) -> ast.AST | None:
    """The callable ``run_worker`` will invoke, or None when unresolvable.

    ``run_worker(self.method())`` passes an already-created coroutine or Worker
    and is always valid; only a bare reference names something to be called.
    ``partial(self.method, ...)`` is unwrapped because it is the shape the MCP
    inspector uses.
    """
    if not call.args:
        return None
    target = call.args[0]
    if isinstance(target, ast.Call):
        func_name = getattr(target.func, "id", None) or getattr(
            target.func, "attr", None
        )
        if func_name == "partial" and target.args:
            target = target.args[0]
        else:
            return None
    if not isinstance(target, ast.Attribute):
        return None
    if not (isinstance(target.value, ast.Name) and target.value.id == "self"):
        return None
    return target


def _declares_thread(call: ast.Call) -> bool:
    return any(
        keyword.arg == "thread"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in call.keywords
    )


def collect_w001(tree: ast.Module, path: Path) -> list[str]:
    """Synchronous ``run_worker`` targets that do not declare ``thread=True``."""
    violations: list[str] = []
    for cls in [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]:
        methods = _class_methods(cls)
        for node in ast.walk(cls):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run_worker"
            ):
                continue
            if _declares_thread(node):
                continue
            target = _worker_target(node)
            if target is None:
                continue
            method = methods.get(target.attr)
            if method is None:
                continue  # inherited or assigned elsewhere; not resolvable here
            if _is_property(method) or _is_coroutine_def(method):
                continue
            violations.append(
                f"{_rel(path)}:{node.lineno}\t{cls.name}.{target.attr}"
            )
    return violations


#: Exception names whose handler can catch a failed DOM lookup. ``query_one``
#: raises ``NoMatches``/``TooManyMatches``/``WrongType`` -- all ``QueryError``
#: and so ``Exception`` subclasses -- and nothing else. An ``except
#: ValueError`` around the lookup is not protection; it is noise between the
#: failure and the worker boundary.
LOOKUP_CATCHERS = {
    "Exception",
    "BaseException",
    "QueryError",
    "NoMatches",
    "TooManyMatches",
    "WrongType",
}


def _catches_lookup_failure(node: ast.Try | ast.TryStar) -> bool:
    """Whether this ``try``'s own handlers can catch a failed DOM lookup.

    Args:
        node: The ``try`` statement whose ``body`` holds the lookup.

    Returns:
        True for a bare ``except:``, or any handler naming a class in
        :data:`LOOKUP_CATCHERS` (including inside an ``except (A, B):``
        tuple). A ``try``/``finally`` with no handlers, or one handling only
        unrelated exceptions, returns False: the lookup propagates out of it
        exactly as if the ``try`` were not there.
    """
    for handler in node.handlers:
        if handler.type is None:  # bare `except:`
            return True
        caught = (
            handler.type.elts
            if isinstance(handler.type, ast.Tuple)
            else [handler.type]
        )
        for entry in caught:
            if isinstance(entry, ast.Attribute):
                name: str | None = entry.attr
            elif isinstance(entry, ast.Name):
                name = entry.id
            else:
                name = None
            if name in LOOKUP_CATCHERS:
                return True
    return False


def _own_nodes(func: ast.AST) -> tuple[list[ast.AST], dict[ast.AST, ast.AST]]:
    """The nodes belonging to ``func`` itself, and each one's parent.

    A nested ``async def`` is walked *out*: the module-level scan enumerates
    it separately, so descending into it here emitted every unguarded lookup
    inside it twice -- once under the inner function, once misattributed to
    this one. A nested plain ``def`` or ``lambda`` is deliberately **not**
    excluded: nothing else ever scans those, and a dialog callback
    dereferencing a screen the await let go is the W002 defect class itself.

    Args:
        func: The ``async def`` being scanned.

    Returns:
        ``(nodes, parents)`` -- the reachable nodes, and a child-to-parent
        map over exactly those nodes for the guard-ancestor walk.
    """
    nodes: list[ast.AST] = [func]
    parents: dict[ast.AST, ast.AST] = {}
    stack: list[ast.AST] = [func]
    while stack:
        node = stack.pop()
        for child in ast.iter_child_nodes(node):
            parents[child] = node
            nodes.append(child)
            if not isinstance(child, ast.AsyncFunctionDef):
                stack.append(child)
    return nodes, parents


def collect_w002(tree: ast.Module, path: Path) -> list[str]:
    """DOM lookups reached after an await, with no ``try`` able to catch them.

    Args:
        tree: The parsed module to scan.
        path: That module's path -- both the non-UI-package skip and the
            census key are derived from it.

    Returns:
        One ``"<relative path>::<function name>"`` census key per offending
        lookup, repeated when one function holds several (the census pins a
        count per function). Empty for a module outside :data:`UI_PACKAGES`.
    """
    if not UI_PACKAGES & set(path.parts):
        return []
    sites: list[str] = []
    for func in [
        node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
    ]:
        own, parents = _own_nodes(func)
        awaits = [node for node in own if isinstance(node, ast.Await)]
        if not awaits:
            continue
        # Positions, not line numbers: a lookup INSIDE the first await's own
        # expression (`await save(self.query_one(...))`) runs before the
        # suspension however the call is wrapped. Comparing lines made the
        # verdict depend on formatting -- a reflow added one row and removed
        # five with no statement changed (PR #2993).
        first = min(awaits, key=lambda node: (node.lineno, node.col_offset))
        first_await_end = (first.end_lineno, first.end_col_offset)
        for node in own:
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in DOM_LOOKUPS
            ):
                continue
            if (node.lineno, node.col_offset) <= first_await_end:
                continue
            guarded = False
            child: ast.AST = node
            cursor = parents.get(child)
            while cursor is not None and cursor is not func:
                # Two conditions, and the check shipped with neither.
                #
                # `Try.body`: a lookup in an `except`/`else`/`finally` clause
                # is a *child* of the same `Try` node while sitting OUTSIDE
                # the region its handlers cover -- an exception there
                # propagates straight out. Treating any `Try` ancestor as
                # protection hid 50 sites across 21 functions.
                #
                # ...and a handler that can catch it: `try`/`finally` with no
                # handler at all, and `except ValueError`, are both "no `try`"
                # as far as a `NoMatches` is concerned.
                #
                # Keep ascending when either fails: an OUTER try may still
                # legitimately cover this lookup.
                if (
                    isinstance(cursor, (ast.Try, ast.TryStar))
                    and any(child is stmt for stmt in cursor.body)
                    and _catches_lookup_failure(cursor)
                ):
                    guarded = True
                    break
                child = cursor
                cursor = parents.get(cursor)
            if guarded:
                continue
            # Keyed by enclosing function, NOT by line: an edit anywhere above
            # a site shifts its line number, and a census that churns on every
            # unrelated edit is one people re-pin without reading. The count
            # still catches a NEW lookup added to an already-censused function.
            sites.append(f"{_rel(path)}::{func.name}")
    return sites


#: W003: pump-run entry points, by Textual's naming conventions. ``@on`` is
#: detected from the decorator.
HANDLER_PREFIXES = ("on_", "_on_", "action_", "watch_", "_watch_", "key_")

#: W003: schedulers that run a callable (or a coroutine) on a message pump or a
#: plain task -- never inside a worker.
PUMP_SCHEDULERS = {
    "call_later",
    "call_next",
    "call_after_refresh",
    "set_timer",
    "set_interval",
    "create_task",
    "ensure_future",
}

#: Of those, the ones that run their coroutine as a separate asyncio TASK.
#: The pump that scheduled it is free again, so a hand-rolled wait (below)
#: completes there; only a real ``push_screen_wait`` (NoActiveWorker) fails.
_SCHEDULED_COROUTINE = {"create_task", "ensure_future"}

#: ``set_timer(delay, callback)`` / ``set_interval(interval, callback)`` take
#: the callable second; every other scheduler takes it first.
_TIMER_SCHEDULERS = {"set_timer", "set_interval"}

#: Calls that make an awaitable a ``push_screen`` result callback can complete.
_FUTURE_FACTORIES = {"create_future", "Future", "Event"}

#: ``await asyncio.wait_for(fut, t)`` / ``asyncio.shield(fut)`` await ``fut``.
_FUTURE_WAITERS = {"wait_for", "shield"}

#: ``fut.set_result(...)``, ``event.set()``: what settles a future or event.
_COMPLETERS = {"set_result", "set_exception", "set", "cancel"}

#: A future's or event's methods that only read it: a callback may call these
#: on what it settles and still settle nothing else. Anywhere else a read is a
#: way to hand the future's state on (a relay polling ``done()``), so it is
#: treated like any other use.
_FUTURE_READERS = {"done", "cancelled", "result", "exception", "is_set"}

#: Calls that hand on every local of the function making them
#: (``chain(**locals())``, ``vars()``; ``vars(self)`` hands on ``self``'s).
_SCOPE_READERS = {"locals", "vars"}

#: The shared, never-mutated default of ``_Function.fresh_self_stores``.
_NO_STORES: dict[str, int] = {}

#: A callable reference: ``("self", name)`` for ``self.name``, ``("name",
#: name)`` for a bare name, ``("attr", name)`` for ``anything.name``, and
#: ``("lambda", name)`` for a bare name called in a lambda's body -- read when
#: the lambda runs, from its own scope, so a class body's namespace (which
#: ``("name", name)`` written there reads first) is not on its path.
_Ref = tuple[str, str]


def _decorator_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for decorator in getattr(node, "decorator_list", ()):
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = getattr(target, "attr", None) or getattr(target, "id", None)
        if name:
            names.add(name)
    return names


def _is_wait_push(call: ast.Call) -> bool:
    """``push_screen_wait(...)``, or ``push_screen(..., wait_for_dismiss=True)``.

    Matched by the called name alone, so ``getattr(app, "push_screen_wait")``
    bound to a local of the same name is caught too.
    """
    name = getattr(call.func, "attr", None) or getattr(call.func, "id", None)
    if name == "push_screen_wait":
        return True
    if name != "push_screen":
        return False
    if any(
        keyword.arg == "wait_for_dismiss"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in call.keywords
    ):
        return True
    return (
        len(call.args) >= 3
        and isinstance(call.args[2], ast.Constant)
        and call.args[2].value is True
    )


def _ref(node: ast.AST) -> _Ref | None:
    """The callable ``node`` names, unwrapping ``partial(target, ...)`` and a
    ``lambda`` whose body is one call (``lambda r: self._after(r)`` returns
    ``_after``'s coroutine, which Textual's ``invoke`` -- and any ``await`` of
    the lambda's result -- then awaits)."""
    if isinstance(node, ast.Lambda):
        if not isinstance(node.body, ast.Call):
            return None
        ref = _ref(node.body.func)
        if ref is not None and ref[0] == "name":
            return ("lambda", ref[1])
        return ref
    if isinstance(node, ast.Call):
        func_name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if func_name == "partial" and node.args:
            return _ref(node.args[0])
        return None
    if isinstance(node, ast.Name):
        return ("name", node.id)
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return ("self", node.attr)
        return ("attr", node.attr)
    return None


def _future_name(node: ast.AST) -> str | None:
    """The local (``fut``) or own attribute (``self.fut``) an await waits on:
    ``await fut``, ``await fut.wait()``, ``await asyncio.wait_for(fut, t)``."""
    if isinstance(node, ast.Call):
        name = _call_name(node)
        if name == "wait" and isinstance(node.func, ast.Attribute) and not node.args:
            return _future_name(node.func.value)
        if name in _FUTURE_WAITERS and node.args:
            return _future_name(node.args[0])
        return None
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return f"self.{node.attr}"
    return None


def _result_awaited(node: ast.AST) -> ast.Call | None:
    """The call whose AWAITED result an await waits on: ``await (await
    helper())``, ``await (await helper()).wait()``, ``await
    asyncio.wait_for(await helper(), t)`` -- for an ``async def helper``,
    the future it hands back rather than its coroutine."""
    if isinstance(node, ast.Call):
        name = _call_name(node)
        if name == "wait" and isinstance(node.func, ast.Attribute) and not node.args:
            return _result_awaited(node.func.value)
        if name in _FUTURE_WAITERS and node.args:
            return _result_awaited(node.args[0])
        return None
    if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
        return node.value
    return None


def _completion(callback: ast.AST) -> "frozenset[str] | _Ref | None":
    """Which futures a ``push_screen`` result callback settles, as far as
    the callback expression alone shows.

    This says only WHAT the callback settles. Whether those futures are
    independent of the one awaited -- nothing chains them to it -- is
    decided later (:func:`_settled_by`, ``_WaitGraph._contained``).

    Args:
        callback: The callback expression handed to ``push_screen``.

    Returns:
        The futures it settles, named in the pushing function's own scope,
        when that is visible here -- ``fut.set_result`` itself, or a
        ``partial``/one-call ``lambda`` of it (``lambda r:
        fut.set_result(r)``, ``lambda _: event.set()``). Otherwise a
        reference to the callable, whose own body says what it settles once
        the module is collected (see :func:`_settled_by`), or ``None`` when
        even that is unknown -- including a lambda whose parameter shadows
        the name it settles or calls (``lambda r, fut=answer:
        fut.set_result(r)``, ``lambda r, self=self._peer: self._settle(r)``),
        or whose arguments make a call of their own.
    """
    target = callback
    shadowed: set[str] = set()
    if isinstance(target, ast.Call) and _call_name(target) == "partial" and target.args:
        target = target.args[0]
    elif isinstance(target, ast.Lambda) and isinstance(target.body, ast.Call):
        args = target.args
        shadowed = {
            arg.arg
            for arg in (
                *args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg
            )
            if arg is not None
        }
        body = target.body
        for value in (*body.args, *(keyword.value for keyword in body.keywords)):
            if any(isinstance(node, ast.Call) for node in ast.walk(value)):
                return None
        target = body.func
        if isinstance(target, ast.Name) and target.id in shadowed:
            return None
    if isinstance(target, ast.Attribute) and target.attr in _COMPLETERS:
        future = _future_name(target.value)
        if future is not None:
            if future.partition(".")[0] in shadowed:
                return None
            return frozenset((future,))
    ref = _ref(callback)
    if ref is not None and ref[0] == "self" and "self" in shadowed:
        return None  # `self` is the lambda's own default, not the method's
    return ref


class _Function:
    """One ``def``/``async def`` and what W003 needs to know about it."""

    def __init__(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        module: "_Module",
        cls: str | None,
        parent: "_Function | None",
        scope: "_Function | None" = None,
    ) -> None:
        # The name only, never the node: holding every module's AST alive
        # until the graph is solved made the cyclic GC rescan them all and
        # more than doubled the checker's parse time.
        self.name = node.name
        # Which of two same-named definitions Python binds: the later one.
        self.lineno = node.lineno
        self.module = module
        self.cls = cls
        # The def this one is nested in (it binds the name there) ...
        self.parent = parent
        # ... and the function whose locals a free name here reads next:
        # the parent, or -- for a method of a class defined inside a
        # function -- that enclosing function, past the class scope.
        self.scope_parent = parent if parent is not None else scope
        self.nested: dict[str, _Function] = {}
        # `local = <callable>` inside this function: scoped, never global.
        self.local_aliases: dict[str, list[_Ref]] = {}
        decorators = _decorator_names(node)
        self.is_worker = "work" in decorators
        self.is_root = not self.is_worker and (
            "on" in decorators or node.name.startswith(HANDLER_PREFIXES)
        )
        # What a positional argument binds: `params[position + offset]`.
        args = node.args
        self.params = [arg.arg for arg in (*args.posonlyargs, *args.args)]
        self.is_static = "staticmethod" in decorators
        self.is_classmethod = "classmethod" in decorators
        # Awaiting an `async def`'s call runs it to its `return`; awaiting a
        # plain def's call awaits whatever it returned.
        self.is_async = isinstance(node, ast.AsyncFunctionDef)
        # `push_screen_wait(...)` / `push_screen(..., wait_for_dismiss=True)`.
        self.wait_pushes = 0
        # `push_screen(..., callback=...)`, and the futures/events this
        # function creates and awaits: together, a hand-rolled wait. Each
        # push's entry names the futures its callback settles when they are
        # fresh futures of this function that its body hands on nowhere
        # (`None`: unknown or unshown; a `("self", name)` reference: resolved
        # per graph -- see `_settled_by`). `push_settles` is that list as one
        # graph resolved it, with each `self.X` entry also checked against
        # every use of `.X` in the package (`_WaitGraph._contained`).
        self.callback_pushes = 0
        self.push_callbacks: list[frozenset[str] | _Ref | None] = []
        self.push_settles: list[frozenset[str] | _Ref | None] = []
        # `fut.set_result(...)`/`event.set()` in this body: what it settles
        # when it is itself a push's callback ...
        self.completes: set[str] = set()
        # ... whether settling (and inspecting) those is ALL its body
        # visibly does -- `None` if not -- ...
        self.settles_only: frozenset[str] | None = None
        # ... and what that amounts to in its caller's names: called as a
        # nested def (every name it settles is free, read from the enclosing
        # def) or as a method (every name is `self.X`). `None`: not shown.
        # Set by `_collect_module`.
        self.settles_as_nested: frozenset[str] | None = None
        self.settles_as_method: frozenset[str] | None = None
        self.futures: set[str] = set()
        # The created futures that are this function's alone: every binding
        # of the name is a single-target fresh-future assignment, and its
        # body (nested defs and lambdas included) reads it nowhere but to
        # settle it -- so no other name, object or relay can reach one.
        self.fresh_futures: frozenset[str] = frozenset()
        # How many of those bindings each `self.X` among them has: the graph
        # accepts such a future only when they are every use of `.X` in the
        # package. Shared and empty until `_collect_module` sets it.
        self.fresh_self_stores: dict[str, int] = _NO_STORES
        self.awaited_futures: set[str] = set()
        # The futures it hands back: `return fut` (TASK-33621.33).
        self.returned_futures: set[str] = set()
        # `pending = helper(...)`: awaiting `pending` awaits `helper(...)`.
        self.call_results: dict[str, list[_Ref]] = {}
        # `pending = await helper(...)`, and `await (await helper(...))`: the
        # awaited RESULT of a call -- for an `async def helper`, the future
        # it returned.
        self.await_results: dict[str, list[_Ref]] = {}
        self.awaited_results: list[_Ref] = []
        # `return await helper(...)`: hands on the awaited RESULT of a call --
        # for an `async def helper`, the pending future it returned.
        self.returned_results: list[_Ref] = []
        self.awaited: list[_Ref] = []
        self.scheduled: list[tuple[_Ref, str]] = []
        # False for dead code: rebound before anything read it, or defined
        # inside a def that is (set by the collector, see `_dead_defs`).
        self.live = True
        # Solved by _WaitGraph.
        self.targets: list[_Target] = []
        # The functions whose callback push an await here waits on without
        # going through a call of them: a future they stored on `self`, or
        # one an `async def` handed back.
        self.publishers: list[_Function] = []
        # For an `async def`: the publishers of the future it RETURNS, which
        # a caller waits on by awaiting the call's result.
        self.handed_back: list[_Function] = []
        self.waiting = False
        self.sites: frozenset[str] = frozenset()

    @property
    def key(self) -> str:
        owner = f"{self.cls}." if self.cls else ""
        return f"{self.module.rel}::{owner}{self.name}"

    def _pushes_settling(self, futures: set[str]) -> int:
        """This body's callback pushes that may settle one of ``futures``,
        or a future that one of them is chained to.

        Every push counts unless its callback visibly settles nothing but
        other fresh futures this function created, named in its own scope,
        and none of those can reach anything else: read nowhere but to
        settle them (or, inside a callback that settles one alone, to
        inspect it), and for a ``self`` future, no other use of that
        attribute name in the package (see :func:`_settled_by` and
        ``_WaitGraph._contained``). Settling a DIFFERENT future is not
        enough: a done-callback, a relay or a handoff can chain it to the
        awaited one, which then hangs just the same (PR #2987 review,
        round 3). A callback whose effect is unknown, or that settles a name
        this function cannot show is a separate, contained future -- a
        helper's parameter, a local alias -- counts by shape (recall over
        precision, round 2). The check is syntactic: dynamic access
        (``getattr``, ``__dict__``) is not seen."""
        if not futures:
            return 0
        return sum(
            1
            for settles in self.push_settles
            if settles.__class__ is not frozenset or settles & futures
        )

    @property
    def own_pushes(self) -> int:
        """Wait pushes made in this function's own body.

        A ``push_screen(..., callback=done)`` in a function that then awaits
        a future or event it created is ``push_screen_wait`` by hand (PR
        #2922's ``request_hook_review``): Textual queues ``done`` on the
        requester pump through ``call_next``, and from a handler that pump is
        the one blocked on the await -- it can never run the callback.

        A plain ``def`` RETURNING that future is the same wait one call
        later: the caller's ``await helper()`` (and Textual's own
        ``invoke``, which awaits a handler's return value once) blocks on it,
        so the helper is a site that waits wherever it is awaited
        (TASK-33621.33). An ``async def`` returning it is not: ``await
        helper()`` only obtains the future (see :attr:`pending_pushes`).

        Only a push whose callback may settle THAT future, or one chained to
        it, counts: one whose callback settles only a different, contained
        future leaves this await to be settled elsewhere (PR #2987 review;
        :meth:`_pushes_settling`).
        """
        settled = self.awaited_futures
        if not self.is_async:
            settled = settled | self.returned_futures
        return self.wait_pushes + self._pushes_settling(self.futures & settled)

    @property
    def pending_pushes(self) -> int:
        """For an ``async def``, the callback pushes that may settle a future
        it created and RETURNS: a caller waits on them by awaiting the
        call's result -- ``await (await helper())``."""
        if not self.is_async:
            return 0
        return self._pushes_settling(self.futures & self.returned_futures)

    @property
    def self_futures(self) -> set[str]:
        """The ``self`` attributes this function stores a NEW future or event
        in, with a ``callback=`` push that may settle it: a caller that
        awaits one of them waits on this function's push."""
        if not self.callback_pushes:
            return set()
        return {
            name[5:]
            for name in self.futures
            if name.startswith("self.") and self._pushes_settling({name})
        }

    @property
    def published_pushes(self) -> int:
        """The callback pushes behind what this function publishes -- futures
        it stores on ``self`` or returns -- counted at its census key."""
        published = {f"self.{attr}" for attr in self.self_futures}
        return self._pushes_settling(
            self.futures & (published | self.returned_futures)
        )


#: What a reference can resolve to: one definition; every top-level
#: definition -- methods included -- (and alias) sharing a name, ``("defs",
#: name)``; every module-level FUNCTION (and alias) sharing a name, ``("funcs",
#: name)``; what one class assigns to one of its own attributes, ``("bound",
#: "<path>::<Class>.<attr>")``; or an alias bound outside the class that reads
#: it, ``("alias", name)``.
#: The four tuple kinds are graph nodes solved alongside the functions, so
#: resolution never recurses through them (expanding ``self.a = self.b``
#: chains in place went exponential and tripled the checker's runtime).
_Target = "_Function | tuple[str, str]"


class _Module:
    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.functions: dict[str, _Function] = {}  # module-level defs
        self.classes: dict[str, dict[str, _Function]] = {}
        self.bases: dict[str, list[str]] = {}
        # Each class body's enclosing function, when the class is defined
        # inside one: a free name in the body reads that function's locals
        # after the class's own namespace. Keyed by class name, like
        # `classes`, so two classes of one name in a module share an entry.
        self.class_scopes: dict[str, _Function | None] = {}
        # (alias name, value reference, class context, function context,
        # is an ATTRIBUTE binding). An attribute -- a class-body name,
        # `self.x = ...`, `obj.x = ...` -- is reachable only through an
        # attribute (`obj.x()`), never through a bare name.
        self.aliases: list[
            tuple[str, _Ref, str | None, _Function | None, bool]
        ] = []
        # (class, attribute) -> (value reference, binding function): a
        # `self.attr = <callable>` in that class's methods, or `attr =
        # <callable>` in its body. Resolves that class's own `self.attr()`.
        self.class_aliases: dict[
            tuple[str, str], list[tuple[_Ref, _Function | None]]
        ] = {}
        # `callee(..., <callable>, ...)`: (callee, position, value reference,
        # class context, function context, the bare name `X` of an `X.f`
        # callee) -- bound to the callee's parameter once every module is
        # known.
        self.handoffs: list[
            tuple[_Ref, int, _Ref, str | None, _Function | None, str | None]
        ] = []
        # Every `.X` attribute node in the module, by `X`, except the
        # receiver of a settle call (`self.X.set_result`) and an inspection
        # (`self.X.done()`) inside a def that settles `self.X` alone: the
        # uses that can hand a future stored at `.X` on, or rebind it.
        self.attr_uses: dict[str, int] = {}
        # Names an instance or a class can bind OVER a method of the same
        # name: every attribute stored or deleted (`self.x = ...`, `obj.x =
        # ...`) and every name a class body binds other than by `def`
        # (assignment, import, nested class -- a comprehension's or a
        # lambda's own names too, which only over-counts).
        self.rebound_members: set[str] = set()


def _call_name(call: ast.Call) -> str | None:
    return getattr(call.func, "attr", None) or getattr(call.func, "id", None)


def _is_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _record_call(sink: _Function, call: ast.Call) -> None:
    """A direct wait push, or a callable handed to a pump scheduler.

    ``push_screen``'s result ``callback`` counts as scheduled: Textual runs it
    through the requester's ``call_next``, on a pump, never in a worker.
    """
    if _is_wait_push(call):
        sink.wait_pushes += 1
    name = _call_name(call)
    if name in _COMPLETERS and isinstance(call.func, ast.Attribute):
        future = _future_name(call.func.value)
        if future is not None:
            sink.completes.add(future)
    if name == "push_screen":
        callbacks = [
            *call.args[1:2],
            *(kw.value for kw in call.keywords if kw.arg == "callback"),
        ]
        given = [cb for cb in callbacks if not _is_none(cb)]
        if not _is_wait_push(call) and given:
            sink.callback_pushes += 1
            sink.push_callbacks.append(_completion(given[0]))
        for value in callbacks:
            ref = _ref(value)
            if ref is not None:
                sink.scheduled.append((ref, name))
        return
    if name not in PUMP_SCHEDULERS:
        return
    # Only the CALLABLE: `call_after_refresh(self.run_worker, self._load)`
    # hands `_load` to `run_worker` as an argument, and it runs in a worker.
    # Scanning every argument reported ChunkingLabScreen.on_mount->_load.
    position = 1 if name in _TIMER_SCHEDULERS else 0
    candidates = [
        *call.args[position : position + 1],
        *(kw.value for kw in call.keywords if kw.arg == "callback"),
    ]
    for arg in candidates:
        target = (
            arg.func
            if isinstance(arg, ast.Call) and name in _SCHEDULED_COROUTINE
            else arg
        )
        ref = _ref(target)
        if ref is not None:
            sink.scheduled.append((ref, name))


def _record_await(sink: _Function, node: ast.Await) -> None:
    """The awaited call, and any coroutine built inline as its argument
    (``await asyncio.wait_for(self.pick(), 5)`` still runs ``pick`` here --
    except ``run_worker(self.pick())``, whose argument runs in a worker).
    ``await helper().wait()`` waits on what ``helper`` hands back, so it
    awaits ``helper`` too; ``await (await helper())`` waits on what an
    ``async def helper`` hands back."""
    future = _future_name(node.value)
    if future is not None:
        sink.awaited_futures.add(future)
    result = _result_awaited(node.value)
    if result is not None:
        ref = _ref(result.func)
        if ref is not None:
            sink.awaited_results.append(ref)
    if not isinstance(node.value, ast.Call):
        return
    func = node.value.func
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "wait"
        and not node.value.args
        and isinstance(func.value, ast.Call)
    ):
        ref = _ref(func.value.func)
        if ref is not None:
            sink.awaited.append(ref)
    inline = (
        ()
        if (getattr(func, "attr", None) or getattr(func, "id", None)) == "run_worker"
        else node.value.args
    )
    for call in (node.value, *inline):
        if isinstance(call, ast.Call):
            ref = _ref(call.func)
            if ref is not None:
                sink.awaited.append(ref)


def _callee(func: ast.AST) -> _Ref | None:
    """What a call's positional arguments are handed to: ``("self", x)``,
    ``("name", x)``, ``("attr", x)``, or ``("super", x)`` for
    ``super().x(...)`` -- resolved through the class's bases, not by name."""
    if isinstance(func, ast.Name):
        return ("name", func.id)
    if not isinstance(func, ast.Attribute):
        return None
    owner = func.value
    if isinstance(owner, ast.Name) and owner.id == "self":
        return ("self", func.attr)
    if (
        isinstance(owner, ast.Call)
        and isinstance(owner.func, ast.Name)
        and owner.func.id == "super"
    ):
        return ("super", func.attr)
    return ("attr", func.attr)


def _owner(func: ast.AST) -> str | None:
    """``X`` of an ``X.f(...)`` callee whose receiver is a bare name other
    than ``self``: when ``X`` is a package class, ``X.f(self, ...)`` is an
    UNBOUND call, and its first argument binds ``f``'s ``self``."""
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id != "self"
    ):
        return func.value.id
    return None


#: Callees whose positional callables W003 already models (a worker, a pump
#: scheduler, a push's result callback) or rules out (``call_from_thread``
#: runs its callable in the calling worker's context): not a parameter
#: handoff. ``partial``'s first argument is its target, which ``_ref``
#: follows; the rest bind the target's parameters.
_HANDOFF_MODELLED = PUMP_SCHEDULERS | {
    "run_worker",
    "call_from_thread",
    "push_screen",
    "push_screen_wait",
    "partial",
}


def _record_handoffs(
    module: "_Module", call: ast.Call, cls: str | None, fn: _Function | None
) -> None:
    """Each callable passed POSITIONALLY: it binds the callee's parameter at
    that position, exactly as the keyword form binds it by name (PR #2945:
    ``HooksController(_pick)`` hid the push its keyword twin reports).
    ``partial(target, a, b)`` hands ``a`` and ``b`` to ``target``'s first
    two parameters."""
    args = call.args
    if not args:
        return
    func = call.func
    callee = _callee(func)
    if callee is not None and callee[1] == "partial":
        func, args = args[0], args[1:]
        callee = _callee(func)
    if callee is None or callee[1] in _HANDOFF_MODELLED:
        return
    owner = _owner(func)
    for position, arg in enumerate(args):
        if isinstance(arg, ast.Starred):
            return  # every later position is unknown
        ref = _ref(arg)
        if ref is not None:
            module.handoffs.append((callee, position, ref, cls, fn, owner))


def _alias_pairs(node: ast.AST) -> list[tuple[str, ast.AST, str]]:
    """Names a callable is handed on under: keyword, dict key, assignment.

    Returns ``(name, value, form)``, ``form`` being ``"name"`` for an
    assignment to a bare name (a LOCAL alias inside a function, a class
    attribute in a class body), ``"self"`` for ``self.name = ...``,
    ``"param"`` for a keyword or a dict key (which can become another
    function's parameter), and ``"attr"`` for ``obj.name = ...``. All but a
    function's locals go into the package-wide table.
    """
    if isinstance(node, ast.Call):
        return [(kw.arg, kw.value, "param") for kw in node.keywords if kw.arg]
    if isinstance(node, ast.Dict):
        return [
            (key.value, value, "param")
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
    if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        pairs = []
        for target in targets:
            if isinstance(target, ast.Name):
                pairs.append((target.id, node.value, "name"))
            elif isinstance(target, ast.Attribute):
                own = isinstance(target.value, ast.Name) and target.value.id == "self"
                pairs.append((target.attr, node.value, "self" if own else "attr"))
        return pairs
    return []


_ALIAS_SOURCES = (ast.Call, ast.Dict, ast.Assign, ast.AnnAssign)


def _base_names(node: ast.ClassDef) -> list[str]:
    names = []
    for base in node.bases:
        if isinstance(base, ast.Subscript):  # Generic[T], Base[Screen]
            base = base.value
        name = getattr(base, "attr", None) or getattr(base, "id", None)
        if name:
            names.append(name)
    return names


#: The fields holding a statement list: a def is dead only when a SIBLING in
#: the same list rebinds its name (``handlers``/``cases`` hold nodes whose
#: own ``body`` is such a list).
_STATEMENT_LISTS = frozenset({"body", "orelse", "finalbody"})
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)

#: Decorators that wrap or describe a def without handing it anywhere, so a
#: later def of the same name still leaves it unreachable. Any OTHER decorator
#: may have registered it first (the package's ``@self.mcp.tool()``), and the
#: registration outlives the name. Textual's ``@on`` qualifies: Textual reads
#: handlers from the finished class namespace, where only the last is left.
_NON_REGISTERING = frozenset(
    {
        "overload",
        "property",
        "setter",
        "getter",
        "deleter",
        "cached_property",
        "staticmethod",
        "classmethod",
        "abstractmethod",
        "override",
        "final",
        "on",
        "work",
        "wraps",
        "lru_cache",
        "cache",
        "contextmanager",
        "asynccontextmanager",
    }
)


def _reads_name(nodes: list[ast.AST], name: str) -> bool:
    """Whether any of ``nodes`` loads the bare name ``name``."""
    return any(
        isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, ast.Load)
        for root in nodes
        for node in ast.walk(root)
    )


def _dead_defs(body: list[ast.stmt]) -> list[ast.AST]:
    """The defs in one statement list that Python never runs.

    A def is dead when a LATER def of the same name in the same list rebinds
    it before anything reads it. Read in between -- ``READY = {"ready":
    on_ready}`` -- it is live wherever it went; so is one whose own
    decorator may have registered it (anything outside
    ``_NON_REGISTERING``). Read only by the rebinding def's header (its
    decorators or parameters' defaults), its value lives on in that def:
    ``@service.setter`` keeps the ``@property`` getter, so the getter is
    live exactly when the setter is. A def in another list -- an
    ``if``/``else`` or ``try``/``except`` alternative -- is never rebound
    here: whichever branch runs binds the name (TASK-33621.33 checkpoint
    review: "the last definition in the scope table" called every one of
    those dead and lost rows dev reported).
    """
    defs = [(index, stmt) for index, stmt in enumerate(body) if isinstance(stmt, _DEFS)]
    if len(defs) < 2:
        return []
    live: dict[int, bool] = {}
    next_def: dict[str, int] = {}
    dead: list[ast.AST] = []
    for index, stmt in reversed(defs):
        later = next_def.get(stmt.name)
        if (
            later is None
            or _decorator_names(stmt) - _NON_REGISTERING
            or _reads_name(body[index + 1 : later], stmt.name)
        ):
            live[index] = True
        else:
            rebinding = body[later]
            header = [*rebinding.decorator_list, rebinding.args]
            live[index] = _reads_name(header, stmt.name) and live[later]
        if not live[index]:
            dead.append(stmt)
        next_def[stmt.name] = index
    return dead


def _bind(table: dict[str, _Function], new: _Function) -> None:
    """Register ``new`` under its name unless a LATER definition holds it.

    Python binds a name defined twice in one scope to its last definition.
    The collector's LIFO stack delivers siblings last-first, so plain
    assignment let the first definition -- dead code, as far as Python is
    concerned -- overwrite the live one (TASK-33621.13 review). Compared by
    line rather than by arrival order, so the rule holds whatever order a
    traversal visits them in.
    """
    current = table.get(new.name)
    if current is None or new.lineno > current.lineno:
        table[new.name] = new


_ScopeFacts = tuple[
    set[str], dict[str, int], dict[str, int], set[tuple[str | None, str | None]], bool
]


def _scope_facts(node: ast.FunctionDef | ast.AsyncFunctionDef) -> _ScopeFacts:
    """What a def's OWN scope binds and calls: the facts that show which
    future a push's callback settles, and that the name is only ever that
    future in this scope (PR #2987 review, round 2). Whether the future
    reaches anything else is :func:`_escaping_names`'s question.

    Read only for the defs that push with a callback or settle a future --
    a few hundred in the package (206 defs push with a callback) -- while
    their module's tree is still alive.

    Args:
        node: The ``def``/``async def``.

    Returns:
        ``(params, bound, fresh, calls, opaque)``: its parameter names; how
        many times its body binds each bare name and ``self.attr``
        (assignment, ``for``/``with``/``except``/``import``/``match``
        targets, nested ``def``/``class`` names, ``global``/``nonlocal``,
        and a nested def's ``nonlocal`` rebinding -- a nested def's
        ``self.attr`` store is left to the package-wide use count); how many
        of those bindings are one fresh future alone (``x =
        loop.create_future()``, not ``x = y = loop.create_future()``); each
        call's ``(receiver, method)`` -- ``(None, None)`` for a bare call;
        and whether calling it runs code this does not read (a class body,
        or a decorator, which may replace the def or wrap it in anything).
    """
    args = node.args
    params = {
        arg.arg
        for arg in (
            *args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg
        )
        if arg is not None
    }
    bound: dict[str, int] = {}
    fresh: dict[str, int] = {}
    calls: set[tuple[str | None, str | None]] = set()
    opaque = bool(node.decorator_list)

    def bind(name: str) -> None:
        bound[name] = bound.get(name, 0) + 1

    stack: list[ast.AST] = list(node.body)
    while stack:
        child = stack.pop()
        kind = type(child)
        if kind in _DEFS or kind is ast.Lambda:
            # Its name, decorators and defaults belong here; its body runs
            # later -- but a `nonlocal` in it rebinds a name of this scope.
            if kind is not ast.Lambda:
                bind(child.name)
                stack.extend(child.decorator_list)
                for inner in ast.walk(child):
                    if type(inner) is ast.Nonlocal:
                        for name in inner.names:
                            bind(name)
            stack.extend(child.args.defaults)
            stack.extend(value for value in child.args.kw_defaults if value is not None)
            continue
        if kind is ast.ClassDef:
            bind(child.name)
            opaque = True
            continue
        if kind is ast.Name:
            if child.ctx.__class__ is not ast.Load:
                bind(child.id)
            continue
        if kind is ast.Attribute:
            if (
                child.ctx.__class__ is not ast.Load
                and isinstance(child.value, ast.Name)
                and child.value.id == "self"
            ):
                bind(f"self.{child.attr}")
        elif kind is ast.Assign or kind is ast.AnnAssign:
            targets = child.targets if kind is ast.Assign else [child.target]
            value = child.value
            if (
                len(targets) == 1
                and isinstance(value, ast.Call)
                and _call_name(value) in _FUTURE_FACTORIES
            ):
                name = _future_name(targets[0])
                if name is not None:
                    fresh[name] = fresh.get(name, 0) + 1
        elif kind is ast.Call:
            func = child.func
            if isinstance(func, ast.Attribute):
                calls.add((_future_name(func.value), func.attr))
            else:
                calls.add((None, None))
        elif kind is ast.Import or kind is ast.ImportFrom:
            for alias in child.names:
                bind((alias.asname or alias.name).partition(".")[0])
        elif kind is ast.Global or kind is ast.Nonlocal:
            for name in child.names:
                bind(name)
        elif kind is ast.ExceptHandler or kind is ast.MatchAs or kind is ast.MatchStar:
            if child.name:
                bind(child.name)
        elif kind is ast.MatchMapping and child.rest:
            bind(child.rest)
        stack.extend(ast.iter_child_nodes(child))
    return params, bound, fresh, calls, opaque


def _prove_settles(function: _Function, facts: _ScopeFacts) -> None:
    """Set what ``function``, used as a push's callback, settles -- when
    its body shows that settling is all it does.

    That is: every call it makes settles or inspects a future it settles,
    and it has no class body and no decorator. Then, called as a nested
    def, it settles the enclosing def's futures of those names when every
    one is free here -- not a parameter, local or default of its own;
    called as a method, it settles ``self`` futures when ``self`` is its
    untouched first parameter. ``settles_only`` records either answer, so
    an inspection there (``if not fut.done()``) is not counted as handing
    ``fut`` on (:func:`_escaping_names`, ``_Module.attr_uses``).

    This shows WHAT the callback settles, not that the awaited future is
    independent of it: that needs the settled futures to be contained too
    (:func:`_settled_by`).
    """
    params, bound, _, calls, opaque = facts
    settled = function.completes
    if opaque or any(
        receiver not in settled
        or (method not in _COMPLETERS and method not in _FUTURE_READERS)
        for receiver, method in calls
    ):
        return
    if all(
        name.partition(".")[0] not in params and name.partition(".")[0] not in bound
        for name in settled
    ):
        function.settles_as_nested = frozenset(settled)
    if (
        not function.is_static
        and not function.is_classmethod
        and function.params[:1] == ["self"]
        and "self" not in bound
        and all(name.startswith("self.") for name in settled)
    ):
        function.settles_as_method = frozenset(settled)
    if function.settles_as_nested is not None or function.settles_as_method is not None:
        function.settles_only = frozenset(settled)


def _escaping_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    settles_only: Callable[[ast.AST], frozenset[str] | None],
) -> set[str] | None:
    """The bare names a def's subtree may hand on: every name it loads,
    nested defs, lambdas and class bodies included, except as the receiver
    of a settle call (``fut.set_result``, ``event.set``, called or not),
    and of an inspection (``fut.done()``) inside a nested def whose whole
    effect is settling that one name (:func:`_prove_settles`).

    A future the push's callback settles must not be among them. Textual
    runs the callback through the requester's ``call_next``, so while the
    handler awaits, that future is never settled -- and an awaited future
    chained to it by a done-callback, a relay that awaits or polls it, or a
    handoff (``watch(closed, answer)``) hangs just the same (PR #2987
    review, round 3). Settling it again elsewhere (``fut.cancel()``) hands
    nothing on, so it does not count.

    Args:
        node: The pushing ``def``/``async def``.
        settles_only: A nested def node's ``_Function.settles_only``.

    Returns:
        Those names, or ``None`` when the body hands on EVERY local --
        ``locals()``, ``vars()``.
    """
    escaped: set[str] = set()
    stack: list[tuple[ast.AST, ast.AST | None]] = [(stmt, node) for stmt in node.body]
    while stack:
        child, owner = stack.pop()
        kind = type(child)
        if kind is ast.Name:
            if child.ctx.__class__ is ast.Load:
                escaped.add(child.id)
            continue
        if kind is ast.Attribute:
            value = child.value
            if value.__class__ is ast.Name:
                if child.attr in _COMPLETERS:
                    continue
                if child.attr in _FUTURE_READERS:
                    if owner is None or settles_only(owner) != {value.id}:
                        escaped.add(value.id)
                    continue
            stack.append((value, owner))
            continue
        if (
            kind is ast.Call
            and child.func.__class__ is ast.Name
            and child.func.id in _SCOPE_READERS
        ):
            return None
        # A nested def's body reads for that def; a lambda's or a class
        # body's for no def, so an inspection there always hands on.
        scoped = kind in _DEFS or kind is ast.Lambda or kind is ast.ClassDef
        inner = child if kind in _DEFS else None
        for field in child._fields:
            if field == "ctx":
                continue
            value = getattr(child, field, None)
            reader = inner if scoped and field == "body" else owner
            if value.__class__ is list:
                stack.extend(
                    (item, reader) for item in value if isinstance(item, ast.AST)
                )
            elif isinstance(value, ast.AST):
                stack.append((value, reader))
    return escaped


def _collect_module(tree: ast.Module, rel: str) -> tuple[_Module, list[_Function]]:
    """One iterative pass: defs, their pushes/awaits/schedules, and aliases.

    Hand-rolled rather than ``ast.walk`` + ``iter_child_nodes`` twice: this
    runs over the whole package inside ``preflight.sh``, and the two-pass
    first cut tripled the checker's runtime. ``ctx`` children (``Load``/
    ``Store``) are skipped for the same reason.

    Each stack entry carries the enclosing class, the enclosing function
    (the alias context), the "sink" that owns pushes and awaits -- which
    is ``None`` inside a ``lambda``: its body runs later, not in the
    enclosing function's await chain -- and the function enclosing the
    current class body, if any (``fn`` resets at a ``class`` statement, but
    a free name in that class still reads the enclosing function's locals).
    """
    module = _Module(rel)
    functions: list[_Function] = []
    # Each def's node, for this pass only (see `_scope_facts`).
    nodes: dict[_Function, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    # `id()` of each dead def node: the tree outlives this pass.
    dead: set[int] = set()
    # `self.X.done()` and the like: (the def it is in, `self.X`, `X`) --
    # not a use of `.X` only if that def settles `self.X` alone.
    inspected: list[tuple[_Function, str, str]] = []
    AST = ast.AST
    stack: list[
        tuple[
            ast.AST, str | None, _Function | None, _Function | None, _Function | None
        ]
    ] = [(tree, None, None, None, None)]
    while stack:
        node, cls, fn, sink, outer = stack.pop()
        kind = type(node)
        if kind is ast.ClassDef:
            module.classes.setdefault(node.name, {})
            module.bases.setdefault(node.name, []).extend(_base_names(node))
            if fn is None and cls is not None:
                module.rebound_members.add(node.name)  # a class in a class body
            if fn is not None:
                outer = fn
            module.class_scopes[node.name] = outer
            cls, fn, sink = node.name, None, None
        elif kind is ast.FunctionDef or kind is ast.AsyncFunctionDef:
            new = _Function(node, module, cls if fn is None else None, fn, outer)
            # Dead when rebound unread, or defined inside a dead def -- for a
            # method of a class defined inside a function, that function.
            enclosing = fn if fn is not None else outer
            new.live = id(node) not in dead and (
                enclosing is None or enclosing.live
            )
            functions.append(new)
            nodes[new] = node
            if fn is not None:
                _bind(fn.nested, new)
            elif cls is not None:
                _bind(module.classes[cls], new)
            else:
                _bind(module.functions, new)
            fn = sink = new
        elif kind is ast.Lambda:
            sink = None
        else:
            if kind is ast.Attribute:
                # Every `.X`, dead code included (counting more only keeps
                # a push), less the receivers of a settle -- `.X.set_result`
                # -- whose own node is counted when it is visited next.
                attr = node.attr
                uses = module.attr_uses
                uses[attr] = uses.get(attr, 0) + 1
                if node.ctx.__class__ is not ast.Load:
                    module.rebound_members.add(attr)
                value = node.value
                if value.__class__ is ast.Attribute:
                    if attr in _COMPLETERS:
                        uses[value.attr] = uses.get(value.attr, 0) - 1
                    elif attr in _FUTURE_READERS and sink is not None:
                        future = _future_name(value)
                        if future is not None:
                            inspected.append((sink, future, value.attr))
            elif fn is None and cls is not None:
                # A class body's own binding, in any branch: `_settle =
                # make_settler()` there is what `self._settle` dispatches to,
                # whatever a base class's `def _settle` does.
                if kind is ast.Name:
                    if node.ctx.__class__ is not ast.Load:
                        module.rebound_members.add(node.id)
                elif kind is ast.alias:
                    module.rebound_members.add(
                        (node.asname or node.name).partition(".")[0]
                    )
            # A binding made in dead code never happens.
            inert = (fn is not None and not fn.live) or (
                outer is not None and not outer.live
            )
            if kind is ast.Call and not inert:
                _record_handoffs(module, node, cls, fn)
            if sink is not None:
                if kind is ast.Call:
                    _record_call(sink, node)
                elif kind is ast.Await:
                    _record_await(sink, node)
                elif kind is ast.Return:
                    value = node.value
                    future = value is not None and _future_name(value)
                    if future:
                        sink.returned_futures.add(future)
                    elif isinstance(value, ast.Await) and isinstance(
                        value.value, ast.Call
                    ):
                        ref = _ref(value.value.func)
                        if ref is not None:
                            sink.returned_results.append(ref)
                elif kind is ast.Assign or kind is ast.AnnAssign:
                    value = node.value
                    # `pending = await helper()`: the awaited result.
                    awaited = isinstance(value, ast.Await)
                    if awaited:
                        value = value.value
                    if isinstance(value, ast.Call):
                        made = not awaited and _call_name(value) in _FUTURE_FACTORIES
                        result = None if made else _ref(value.func)
                        results = sink.await_results if awaited else sink.call_results
                        for target in (
                            node.targets if kind is ast.Assign else [node.target]
                        ):
                            future = _future_name(target)
                            if future is None:
                                continue
                            if made:
                                sink.futures.add(future)
                            elif result is not None:
                                results.setdefault(future, []).append(result)
            if not inert and isinstance(node, _ALIAS_SOURCES):
                local = fn is not None and kind is not ast.Call and kind is not ast.Dict
                for alias, value, form in _alias_pairs(node):
                    ref = _ref(value)
                    if ref is None:
                        continue
                    if local and form == "name":
                        # `pick = pick` (or a partial/lambda of it) only
                        # re-reads the outer binding; `pick = self.pick`
                        # names a different value (PR #2987 review).
                        if ref[1] != alias or ref[0] not in ("name", "lambda"):
                            fn.local_aliases.setdefault(alias, []).append(ref)
                        continue
                    if form == "param":
                        # A keyword or dict key hands the value to a
                        # parameter; whether it is that parameter's own name
                        # is decided on the resolved value (`_binding`).
                        module.aliases.append((alias, ref, cls, fn, False))
                        continue
                    if ref[1] == alias:
                        continue
                    class_body = form == "name" and fn is None and cls is not None
                    if cls is not None and (
                        (form == "self" and fn is not None) or class_body
                    ):
                        module.class_aliases.setdefault((cls, alias), []).append(
                            (ref, fn)
                        )
                    attribute = class_body or form == "self" or form == "attr"
                    module.aliases.append((alias, ref, cls, fn, attribute))
        # A push built inline as `run_worker(...)`'s argument -- the fix this
        # check recommends -- runs in the worker, not in this function.
        in_worker = kind is ast.Call and _call_name(node) == "run_worker"
        for field in node._fields:
            if field == "ctx":
                continue
            child_sink = None if in_worker and field != "func" else sink
            value = getattr(node, field, None)
            if value.__class__ is list:
                if field in _STATEMENT_LISTS:
                    dead.update(map(id, _dead_defs(value)))
                for item in value:
                    if isinstance(item, AST):
                        stack.append((item, cls, fn, child_sink, outer))
            elif isinstance(value, AST):
                stack.append((value, cls, fn, child_sink, outer))
    facts = {
        function: _scope_facts(nodes[function])
        for function in functions
        if function.completes or function.push_callbacks
    }
    for function, scope in facts.items():
        if function.completes:
            _prove_settles(function, scope)
    # An inspection inside a def that settles that `self` future alone hands
    # nothing on: it is not a use of the attribute.
    uses = module.attr_uses
    for sink, future, attr in inspected:
        if sink.settles_only == {future}:
            uses[attr] -= 1
    pushers = [function for function in facts if function.push_callbacks]
    by_node = {id(node): owner for owner, node in nodes.items()} if pushers else {}

    def settles_only(node: ast.AST) -> frozenset[str] | None:
        return by_node[id(node)].settles_only

    for function in pushers:
        params, bound, fresh, _, _ = facts[function]
        escaped = _escaping_names(nodes[function], settles_only)
        rebinds_self = "self" in bound
        function.fresh_futures = frozenset(
            name
            for name in function.futures
            if escaped is not None
            and name not in params
            and name not in escaped
            and bound.get(name, 0) == fresh.get(name, 0)
            and not (rebinds_self and name.startswith("self."))
        )
        stores = {
            name: fresh[name]
            for name in function.fresh_futures
            if name.startswith("self.")
        }
        if stores:
            function.fresh_self_stores = stores
        function.push_callbacks = [
            _settled_by(settles, function, params, bound)
            for settles in function.push_callbacks
        ]
    return module, functions


def _settled_by(
    settles: "frozenset[str] | _Ref | None",
    fn: _Function,
    params: set[str],
    bound: dict[str, int],
) -> "frozenset[str] | _Ref | None":
    """What a push's callback settles, when it can only be futures that
    ``fn`` created and nothing else can reach, now that every def in its
    module is known.

    A push is left out of a future's wait only on that showing, so anything
    short of it is ``None`` -- unknown, matched by shape (recall over
    precision). Settling a different future is not enough on its own: a
    done-callback, a relay or a handoff can chain that future to the
    awaited one, and Textual never settles it while the handler awaits
    (PR #2987 review, round 3) -- hence :attr:`_Function.fresh_futures`
    requires the future to escape nowhere. The first cut compared the names
    a callback's body settles with ``fn``'s own future names across scopes,
    and so took a helper's parameter (``partial(settle, answer)``), a
    callback's local alias (``fut = answer``) or a pusher-side alias
    (``answer = self._answer``) for "a different future", and dropped
    pushes 5918cfd1df caught while the real-tree census stayed
    byte-identical (round 2).

    Args:
        settles: The push's entry from :func:`_completion`.
        fn: The function holding the push.
        params: ``fn``'s parameter names (:func:`_scope_facts`).
        bound: How often ``fn``'s body binds each name (:func:`_scope_facts`).

    Returns:
        The futures settled, named in ``fn``'s scope, when every one is in
        :attr:`_Function.fresh_futures` -- every binding of the name is a
        single-target fresh-future assignment, not a parameter, and the
        body hands it on nowhere -- and settling them is all the callback
        visibly does: a future's own ``set_result``, or a def nested in
        ``fn`` that is the name's only binding (not a parameter it may
        leave in place) and whose settled names are all free
        (:func:`_prove_settles`). A ``("self", name)`` reference is returned
        as is: what ``self.name`` dispatches to depends on the subclasses in
        the whole package, so each graph resolves it, and checks every
        ``self.X`` it settles against the package's uses of ``.X``
        (``_WaitGraph._contained``). ``None`` for anything else -- a module
        function's or a method's bare names are never ``fn``'s locals.
    """
    if settles is None:
        return None
    if settles.__class__ is frozenset:
        return settles if settles <= fn.fresh_futures else None
    kind, name = settles
    if kind == "self":
        return settles
    if kind != "name" and kind != "lambda":
        return None
    callee = fn.nested.get(name)
    if callee is None or name in params or bound.get(name, 0) != 1:
        return None
    proved = callee.settles_as_nested
    if proved is None or not proved <= fn.fresh_futures:
        return None
    return proved


class _WaitGraph:
    """Which functions await a wait push, and WHICH push each one reaches.

    References are resolved to targets once, statically:

    * ``self.x()`` -- for the enclosing class and for each in-package
      subclass of it: that class's own ``x``, else the first package base
      class (by name) that defines one, else what that class (or a base)
      assigns to ``self.x``. The union of those; an alias ``x`` bound from
      outside only when none of them has one. Subclasses count because
      ``self`` may be one of their instances: resolving upward only hid a
      mixin's call into its host class and a template method's call into a
      subclass override (TASK-33621.13, round 3). Never an unrelated class's
      method that happens to be called ``x``: resolving by name there made
      ``ConsoleHooksController``'s ``self._review`` (a constructor-injected
      callable) wait because ``BuddyManagementModal`` has a waiting
      ``_review``, and so censused the Console's Send dispatchers through a
      collision (TASK-33621.13 review).
    * a bare ``x()`` -- a nested def or a local alias of this function or a
      lexically enclosing one (a method of a class defined inside a function
      reads that function, past the class scope), a module-level def in
      scope, else every module-level function and non-attribute alias named
      ``x`` (an import, a module global or a parameter). Never a method from
      a function body, a lambda body or module level: there only an
      attribute reaches one, and falling back to methods too made an
      imported helper wait through an unrelated class's same-named method
      (PR #2944 review). Never an ATTRIBUTE binding either -- a class-body
      ``choose = _pick``, ``self.choose = ...``, ``obj.choose = ...`` -- for
      the same reason (PR #2944 round 6). The exception is a bare name
      written in a CLASS BODY, which Python looks up in that class's own
      namespace first -- its method, else its class-body assignment, never a
      base class's -- then in the function enclosing the class, if any, and
      only then the module: ``choose = _pick`` there names the class's
      ``_pick``;
    * ``obj.x()`` -- every live top-level def (methods included) and alias
      named ``x``: the real defect's chain ran through ``controller._select_
      project_instruction_binding``, a name shared with a non-waiting method
      of the Console runtime, and ``obj``'s type is not statically known.

    A NAME waits when ANY live definition of it waits. An ALIAS waits only
    when EVERY place it is bound -- by keyword, dict key, assignment or
    positional argument -- hands on a waiting callable: with "any" there
    too, one ``callback=<waiting>`` keyword made every ``await callback()`` in
    the package wait, and the first cut of this check reported 517 roots,
    nearly all of them that cascade.

    Then, over the (few) waiting functions only, each one's push SITES: the
    functions holding a wait push it reaches. A census row is keyed by root
    AND site, so a new push reachable from an already-censused entry point
    is a new row -- keyed by entry point alone, one noted row exempted the
    Console's two main dispatchers from W003 entirely.
    """

    def __init__(self, collected: list[tuple[_Module, list[_Function]]]) -> None:
        self.modules: list[_Module] = []
        self.functions: list[_Function] = []
        for module, functions in collected:
            self.modules.append(module)
            self.functions.extend(functions)
        for fn in self.functions:
            # Solved state lives on the functions; a graph built again from
            # the same collection must not inherit the last one's answer.
            fn.targets, fn.waiting, fn.sites = [], False, frozenset()
            fn.publishers, fn.handed_back = [], []
        self.defs_by_name: dict[str, list[_Function]] = {}
        for fn in self.functions:
            if fn.parent is None and fn.live:
                self.defs_by_name.setdefault(fn.name, []).append(fn)
        self.classes_by_name: dict[str, list[tuple[_Module, str]]] = {}
        for module in self.modules:
            for cls in module.classes:
                self.classes_by_name.setdefault(cls, []).append((module, cls))
        self._mro_cache: dict[tuple[str, str], list[tuple[_Module, str]]] = {}
        # Each class's in-package descendants: every class whose MRO holds it,
        # found through the same base resolution as the MRO itself. A mixin's
        # descendants are the classes that mix it in.
        self._subclasses: dict[tuple[str, str], list[tuple[_Module, str]]] = {}
        for module in self.modules:
            for cls in module.classes:
                for owner, base in self._mro(module, cls)[1:]:
                    self._subclasses.setdefault((owner.rel, base), []).append(
                        (module, cls)
                    )
        self._self_cache: dict[tuple[str, str, str], list[_Target]] = {}
        # Package-wide facts about attribute names (see `_Module.attr_uses`
        # and `rebound_members`), read for the few callback pushes that
        # settle a `self` future or call a `self` method.
        self._attr_uses: dict[str, int] = {}
        self._rebound_members: set[str] = set().union(
            *(module.rebound_members for module in self.modules)
        )
        # What each callback push settles, with `callback=self.x` resolved
        # through every class `self` can be -- per graph, since a module
        # swapped into another collection can add an override or a use.
        for fn in self.functions:
            fn.push_settles = fn.push_callbacks
            if any(entry is not None for entry in fn.push_callbacks):
                fn.push_settles = [
                    self._contained(entry, fn) for entry in fn.push_callbacks
                ]
        # `obj.f(x)` matched by name: the same answer for every call of `f`
        # at one position, and most of the package's handoffs are this shape.
        self._by_name_params: dict[tuple[str, int], tuple[set[str], bool]] = {}
        # Every node's targets, resolved once. An alias waits when EVERY
        # binding (one target list each) reaches a push; a class-bound
        # attribute when ANY of its bindings does -- one class's own
        # assignments are few and deliberate, unlike a package-wide keyword.
        # A BARE name reads only the non-attribute bindings (keywords, dict
        # keys, module globals, positional parameters): no bare name can
        # ever be some class's attribute (PR #2944 round-6 review).
        self.alias_targets: dict[str, list[list[_Target]]] = {}
        self.bare_alias_targets: dict[str, list[list[_Target]]] = {}
        # Positional bindings to a parameter GUESSED by name -- `obj.f(x)`,
        # whose `f` is every package def of that name. Each counts only
        # while it waits, so it can add a wait but never cancel one: a
        # guessed binding that does not wait is more likely a different
        # `f`'s than evidence against this one (PR #2987 review).
        self.weak_alias_targets: dict[str, list[list[_Target]]] = {}
        # Positional handoffs W003 could not bind to a parameter, reported
        # (when the callable waits) by `unresolved()`.
        self._unresolved: list[tuple[str, str, int, list[_Target]]] = []
        for module in self.modules:
            for alias, ref, cls, fn, attribute in module.aliases:
                targets = self._binding(alias, ref, self._targets(ref, module, cls, fn))
                if targets is not None:
                    self._add_alias(alias, targets, attribute)
            for callee, position, ref, cls, fn, owner in module.handoffs:
                params, guessed = self._handoff_params(
                    callee, position, module, cls, fn, owner
                )
                targets = self._targets(ref, module, cls, fn)
                if params:
                    for param in params:
                        bound = self._binding(param, ref, targets)
                        if bound is None:
                            continue
                        if guessed:
                            self.weak_alias_targets.setdefault(param, []).append(bound)
                        else:
                            self._add_alias(param, bound, False)
                    continue
                holder = (
                    fn.key
                    if fn is not None
                    else f"{module.rel}::{f'{cls}.' if cls else ''}<body>"
                )
                self._unresolved.append((holder, callee[1], position, targets))
        self.bound_targets: dict[str, list[_Target]] = {}
        for module in self.modules:
            for (cls, attr), bindings in module.class_aliases.items():
                self.bound_targets[f"{module.rel}::{cls}.{attr}"] = [
                    target
                    for ref, binder in bindings
                    for target in self._targets(ref, module, cls, binder)
                ]
        # A future stored on `self` by a function that pushes with
        # `callback=`: (module, class, attribute) -> those functions.
        publishers: dict[tuple[str, str, str], list[_Function]] = {}
        for fn in self.functions:
            stored = fn.self_futures if fn.live else ()
            cls = self._class_of(fn) if stored else None
            for attr in stored if cls is not None else ():
                publishers.setdefault((fn.module.rel, cls, attr), []).append(fn)
        # What each `async def` hands back -- its own pending future, one
        # stored on `self` that it returns, or one another `async def` handed
        # it (`return await helper()`) -- before any caller reads it.
        relays: list[tuple[_Function, list[_Function]]] = []
        for fn in self.functions:
            if not fn.is_async or fn.is_worker or not fn.live:
                continue
            handed = [fn] if fn.pending_pushes else []
            cls = self._class_of(fn)
            if publishers and cls is not None:
                handed += self._self_future_publishers(
                    fn, cls, publishers, fn.returned_futures
                )
            fn.handed_back = handed
            relayed = fn.returned_results + [
                ref
                for name in fn.returned_futures - fn.futures
                for ref in fn.await_results.get(name, ())
            ]
            helpers = [
                helper
                for ref in relayed
                for target in self._targets(ref, fn.module, cls, fn)
                for helper in self._defs_of(target)
            ]
            if helpers:
                relays.append((fn, helpers))
        # A relay of a relay: until nothing new is handed on.
        changed = bool(relays)
        while changed:
            changed = False
            for fn, helpers in relays:
                for helper in helpers:
                    for publisher in helper.handed_back:
                        if publisher not in fn.handed_back:
                            fn.handed_back.append(publisher)
                            changed = True
        for fn in self.functions:
            if fn.is_worker:
                continue
            cls = self._class_of(fn)
            foreign = fn.awaited_futures - fn.futures
            # `pending = helper(); await pending` awaits `helper()`.
            awaited = fn.awaited + [
                ref for name in foreign for ref in fn.call_results.get(name, ())
            ]
            if awaited:
                fn.targets = [
                    target
                    for ref in awaited
                    for target in self._targets(ref, fn.module, cls, fn)
                ]
            if publishers and cls is not None:
                # A plain def that RETURNS a `self` future hands its caller
                # the same wait as awaiting it here (a getter).
                fn.publishers = self._self_future_publishers(
                    fn,
                    cls,
                    publishers,
                    foreign if fn.is_async else foreign | fn.returned_futures,
                )
            # `await (await helper())`, `pending = await helper(); await
            # pending`: the future an `async def helper` hands back.
            results = fn.awaited_results + [
                ref for name in foreign for ref in fn.await_results.get(name, ())
            ]
            for ref in results:
                for target in self._targets(ref, fn.module, cls, fn):
                    for helper in self._defs_of(target):
                        for publisher in helper.handed_back:
                            if publisher not in fn.publishers:
                                fn.publishers.append(publisher)
        self.waiting_def_names: set[str] = set()
        # The module-level functions among them: what a bare name can reach.
        self.waiting_func_names: set[str] = set()
        self.waiting_aliases: set[str] = set()
        self.waiting_bare_aliases: set[str] = set()
        self.waiting_bound: set[str] = set()
        self.site_pushes: dict[str, int] = {}
        self.callback_only_sites: set[str] = set()
        self._def_sites: dict[str, frozenset[str]] = {}
        self._func_sites: dict[str, frozenset[str]] = {}
        self._alias_sites: dict[str, frozenset[str]] = {}
        self._bare_alias_sites: dict[str, frozenset[str]] = {}
        self._bound_sites: dict[str, frozenset[str]] = {}
        self._solve()
        self._solve_sites()

    def _add_alias(self, alias: str, targets: list[_Target], attribute: bool) -> None:
        self.alias_targets.setdefault(alias, []).append(targets)
        if not attribute:
            self.bare_alias_targets.setdefault(alias, []).append(targets)

    @staticmethod
    def _binding(alias: str, ref: _Ref, targets: list[_Target]) -> list[_Target] | None:
        """What binding ``ref`` to the name ``alias`` contributes, or ``None``
        when it binds the name to nothing but itself.

        Only a SAME-named value can do that: ``run(pick=pick)`` forwarding a
        parameter, ``run(pick=other.pick)`` (every ``pick`` by name). Its
        targets that are ``alias``'s own node are dropped; recorded, they
        would make ``alias`` wait only if it already did, so one forwarding
        call anywhere stopped it ever waiting. What remains -- the method
        ``run(self.pick)`` names -- is a real binding: skipping every
        same-named value on the name alone lost it (PR #2987 review).
        """
        if ref[1] != alias:
            return targets
        own = [
            target
            for target in targets
            if isinstance(target, _Function)
            or target[1] != alias
            or target[0] not in ("funcs", "alias", "defs")
        ]
        return own or None

    def _defs_of(self, target: _Target) -> list[_Function]:
        """The definitions a resolved target names directly: itself, or
        every live def (``"defs"``) or module-level function (``"funcs"``)
        of its name. Aliases and class-bound attributes name none."""
        if isinstance(target, _Function):
            return [target]
        kind, name = target
        if kind == "defs":
            return self.defs_by_name.get(name, [])
        if kind == "funcs":
            return [fn for fn in self.defs_by_name.get(name, ()) if fn.cls is None]
        return []

    def _self_future_publishers(
        self,
        fn: _Function,
        cls: str,
        publishers: dict[tuple[str, str, str], list[_Function]],
        names: set[str],
    ) -> list[_Function]:
        """The functions that store, on ``self``, one of the futures
        ``names`` (which ``fn`` awaits or returns) that ``fn`` did not
        create, while pushing with a ``callback=`` that may settle it -- on
        any class ``self`` can be (``cls``, its bases, its subclasses and
        theirs), as ``self.x()`` resolves. Awaiting it is the hand-rolled
        wait split across two methods (TASK-33621.33)."""
        foreign = {
            name[5:] for name in names - fn.futures if name.startswith("self.")
        }
        if not foreign:
            return []
        found: dict[_Function, None] = {}
        module = fn.module
        for owner, owner_cls in (
            (module, cls),
            *self._subclasses.get((module.rel, cls), ()),
        ):
            for base_module, base in self._mro(owner, owner_cls):
                for attr in foreign:
                    for publisher in publishers.get((base_module.rel, base, attr), ()):
                        if publisher is not fn:
                            found.setdefault(publisher, None)
        return list(found)

    def _handoff_params(
        self,
        callee: _Ref,
        position: int,
        module: _Module,
        cls: str | None,
        fn: _Function | None,
        owner: str | None = None,
    ) -> tuple[set[str], bool]:
        """The parameter name(s) a positional argument binds.

        Args:
            callee: The called reference (see :func:`_callee`).
            position: The argument's position in the call.
            module: The module holding the call.
            cls: The enclosing class, if any.
            fn: The enclosing function; ``None`` in a class body or at
                module level.
            owner: ``X`` of an ``X.f(...)`` callee (see :func:`_owner`).

        Returns:
            ``(params, guessed)``. ``params`` is empty when no definition of
            the callee in the package names one: a callee defined outside
            it, an attribute bound to a callable, a class with no package
            ``__init__``, or a ``*args``. ``guessed`` is True when the
            callee was matched by NAME alone (``obj.f(x)``), so the binding
            may belong to an unrelated ``f``.
        """
        key = None
        if callee[0] == "attr" and not self._owner_classes(module, owner):
            key = (callee[1], position)
            cached = self._by_name_params.get(key)
            if cached is not None:
                return cached
        candidates, guessed = self._callee_defs(callee, module, cls, fn, owner)
        params = set()
        for target, offset in candidates:
            index = position + offset
            if index < len(target.params):
                params.add(target.params[index])
        if key is not None:
            self._by_name_params[key] = (params, guessed)
        return params, guessed

    def _owner_classes(
        self, module: _Module, owner: str | None
    ) -> list[tuple[_Module, str]]:
        """The package classes ``X`` of an ``X.f(...)`` callee can be: the
        module's own class of that name, else every package class of it."""
        if owner is None:
            return []
        if owner in module.classes:
            return [(module, owner)]
        return self.classes_by_name.get(owner, [])

    def _callee_defs(
        self,
        callee: _Ref,
        module: _Module,
        cls: str | None,
        fn: _Function | None,
        owner: str | None,
        seen: set[tuple] | None = None,
    ) -> tuple[list[tuple[_Function, int]], bool]:
        """The definitions a call can run, each with the offset of its first
        positional parameter, and whether they were matched by name alone.

        Resolution follows the rest of W003: ``self.f(x)`` through the
        classes ``self`` can be, ``super().f(x)`` through the bases, a bare
        ``f(x)`` lexically -- in a class body, that class's own namespace
        first, as ``_targets`` reads it -- and then as an import. A local
        alias (``go = run; go(x)``) or class-body alias is followed to what
        it names, each once (``seen``), so a cycle ends. ``X.f(self, x)``
        with ``X`` a package class is an UNBOUND call of ``X``'s ``f``: its
        first argument is ``self``. When no class in ``X``'s package MRO
        defines ``f`` (it comes from a base outside the package), that call
        binds nothing, and a waiting argument is reported as an unresolved
        handoff rather than guessed by name. Any other ``obj.f(x)`` is every
        package ``f`` by name. A class binds its ``__init__``'s parameters, after
        ``self``; so does a bound method (a ``@staticmethod`` has no
        ``self``).
        """
        kind, name = callee
        candidates: list[tuple[_Function, int]] = []
        if kind == "self" or kind == "super":
            if cls is None:
                return [], False
            if kind == "self":
                targets = self._self_targets(module, cls, name)
            else:
                targets = []
                for base_module, base in self._mro(module, cls)[1:]:
                    method = base_module.classes[base].get(name)
                    if method is not None:
                        targets = [method]
                        break
            for target in targets:
                if isinstance(target, _Function):
                    candidates.append((target, 0 if target.is_static else 1))
            return candidates, False
        if kind == "name":
            scope = fn
            if fn is None and cls is not None:
                method = module.classes.get(cls, {}).get(name)
                if method is not None:
                    # Called in the class body it is still a plain function.
                    return [(method, 0)], False
                body = [
                    ref
                    for ref, binder in module.class_aliases.get((cls, name), ())
                    if binder is None
                ]
                if body:
                    return self._follow(body, module, cls, None, (module.rel, cls, name), seen)
                scope = module.class_scopes.get(cls)
            while scope is not None:
                if name in scope.nested:
                    return [(scope.nested[name], 0)], False
                if name in scope.local_aliases:
                    return self._follow(
                        scope.local_aliases[name], module, cls, scope, (id(scope), name), seen
                    )
                scope = scope.scope_parent
            if name in module.functions:
                candidates.append((module.functions[name], 0))
            elif name in module.classes:
                candidates.extend(self._init_of(module, name))
            elif name in self.classes_by_name:
                for owner_module, owner_cls in self.classes_by_name[name]:
                    candidates.extend(self._init_of(owner_module, owner_cls))
            else:
                candidates.extend(
                    (target, 0)
                    for target in self.defs_by_name.get(name, ())
                    if target.cls is None
                )
            return candidates, False
        if kind == "attr":
            classes = self._owner_classes(module, owner)
            if classes:
                for owner_module, owner_cls in classes:
                    for base_module, base in self._mro(owner_module, owner_cls):
                        method = base_module.classes[base].get(name)
                        if method is not None:
                            candidates.append((method, 1 if method.is_classmethod else 0))
                            break
                return candidates, False
            # A class is matched by its (CapWords) name -- `mod.Cls(x)` --
            # while a def name like `run` is shared by unrelated methods.
            for owner_module, owner_cls in self.classes_by_name.get(name, ()):
                candidates.extend(self._init_of(owner_module, owner_cls))
            defs = self.defs_by_name.get(name, ())
            candidates.extend(
                (target, 0 if target.cls is None or target.is_static else 1)
                for target in defs
            )
            return candidates, bool(defs)
        return [], False

    def _follow(
        self,
        refs: list[_Ref],
        module: _Module,
        cls: str | None,
        fn: _Function | None,
        key: tuple,
        seen: set[tuple] | None,
    ) -> tuple[list[tuple[_Function, int]], bool]:
        """:meth:`_callee_defs` of each value an alias is bound to. A
        ``lambda`` is skipped: calling it binds ITS parameters, not those of
        the call in its body."""
        seen = set() if seen is None else seen
        if key in seen:
            return [], False
        seen.add(key)
        candidates: list[tuple[_Function, int]] = []
        guessed = False
        for ref in refs:
            if ref[0] == "lambda":
                continue
            found, by_name = self._callee_defs(ref, module, cls, fn, None, seen)
            candidates.extend(found)
            guessed = guessed or by_name
        return candidates, guessed

    def _init_of(self, module: _Module, cls: str) -> list[tuple[_Function, int]]:
        """``cls(...)``'s ``__init__``: the nearest one in its package MRO."""
        for owner, owner_cls in self._mro(module, cls):
            init = owner.classes[owner_cls].get("__init__")
            if init is not None:
                return [(init, 1)]
        return []

    @staticmethod
    def _class_of(fn: _Function) -> str | None:
        while fn.parent is not None:
            fn = fn.parent
        return fn.cls

    def _mro(self, module: _Module, cls: str) -> list[tuple[_Module, str]]:
        """``cls`` and its package base classes, nearest first. A base is
        found in its own module first, else anywhere in the package by name
        (a mixin imported from elsewhere); a base outside the package
        (Textual's ``Screen``) simply ends that line."""
        cached = self._mro_cache.get((module.rel, cls))
        if cached is not None:
            return cached
        order: list[tuple[_Module, str]] = []
        seen: set[tuple[str, str]] = set()
        queue = [(module, cls)]
        while queue:
            owner, name = queue.pop(0)
            if (owner.rel, name) in seen:
                continue
            seen.add((owner.rel, name))
            order.append((owner, name))
            for base in owner.bases.get(name, ()):
                if base in owner.classes:
                    queue.append((owner, base))
                else:
                    queue.extend(self.classes_by_name.get(base, ()))
        self._mro_cache[(module.rel, cls)] = order
        return order

    def _targets(
        self,
        ref: _Ref,
        module: _Module,
        cls: str | None,
        fn: _Function | None,
        seen: set[tuple[int, str]] | None = None,
    ) -> list[_Target]:
        kind, name = ref
        if kind == "self" and cls is not None:
            return self._self_targets(module, cls, name)
        if kind == "name" or kind == "lambda":
            scope = fn
            if fn is None and cls is not None:
                # A bare name in a CLASS BODY reads that class's namespace
                # first, as Python does: `choose = _pick` there names the
                # class's own `_pick`. Resolving it like a function body's
                # bare name missed that method's push (PR #2944 review). A
                # lambda's body skips the class scope.
                if kind == "name":
                    own = self._class_namespace(module, cls, name)
                    if own:
                        return own
                # Then -- for a class defined inside a function -- that
                # function's locals, before the module (PR #2944 round 6).
                scope = module.class_scopes.get(cls)
            while scope is not None:
                if name in scope.nested:
                    return [scope.nested[name]]
                if name in scope.local_aliases:
                    # `recovery = self._project_instruction_recovery` then
                    # `await recovery(...)`: follow the local to its end.
                    # Each local is followed once per resolution, so an
                    # `a = b; b = a` cycle stops at the repeat; a fixed depth
                    # cap silently dropped any chain longer than the cap.
                    seen = set() if seen is None else seen
                    if (id(scope), name) in seen:
                        return []
                    seen.add((id(scope), name))
                    return [
                        target
                        for local in scope.local_aliases[name]
                        for target in self._targets(local, module, cls, scope, seen)
                    ]
                # Lexically outward: a method of a class defined inside a
                # function reads that function next, past the class scope.
                scope = scope.scope_parent
            if name in module.functions:
                return [module.functions[name]]
            # An import, a module global or a parameter: never a method.
            return [("funcs", name)]
        return [("defs", name)]

    @staticmethod
    def _class_namespace(module: _Module, cls: str, name: str) -> list[_Target]:
        """What a bare ``name`` read in ``cls``'s own BODY is, if the class
        binds it: its method of that name, else its class-body assignment.

        Only the class's own namespace, never its bases: a class body does
        not see its base classes' attributes. Empty when the class binds no
        callable ``name``, and Python then reads the enclosing function (for
        a class defined inside one) and then the module.

        Two simplifications: the ``("bound", ...)`` node also carries what
        the class's methods assign to ``self.name`` (an over-approximation),
        and definition order is not checked -- a ``choose = _pick`` written
        ABOVE ``def _pick`` reads the module in Python, and the class's
        ``_pick`` here. None of the package's class-body aliases does that.
        """
        method = module.classes.get(cls, {}).get(name)
        if method is not None:
            return [method]
        if any(
            binder is None for _, binder in module.class_aliases.get((cls, name), ())
        ):
            return [("bound", f"{module.rel}::{cls}.{name}")]
        return []

    def _resolve_on(self, module: _Module, cls: str, name: str) -> list[_Target]:
        """``self.name`` on an instance whose class is exactly ``cls``: the
        nearest method in its MRO, else what that MRO assigns to
        ``self.name`` -- empty when neither exists."""
        mro = self._mro(module, cls)
        for owner, owner_cls in mro:
            method = owner.classes[owner_cls].get(name)
            if method is not None:
                return [method]
        return [
            ("bound", f"{owner.rel}::{owner_cls}.{name}")
            for owner, owner_cls in mro
            if (owner_cls, name) in owner.class_aliases
        ]

    def _self_targets(self, module: _Module, cls: str, name: str) -> list[_Target]:
        """Everything ``self.name`` can be when the call sits in ``cls``.

        ``self`` is an instance of ``cls`` OR of any in-package subclass, and
        dispatch resolves ``name`` on the instance's own class. So this is the
        union, over ``cls`` and each descendant, of what ``name`` resolves
        to there: a subclass's override (a base-class template method), the
        host class of a mixin, or a sibling mixin of that host. 2ebe5b1a7e
        resolved upward only -- ``cls`` and its bases -- and both shapes went
        invisible. Only when
        none of them has ``name`` does it fall back to an alias bound from
        outside -- never to an unrelated class's method of that name.
        """
        key = (module.rel, cls, name)
        cached = self._self_cache.get(key)
        if cached is None:
            found: dict[_Target, None] = {}
            for owner, owner_cls in (
                (module, cls),
                *self._subclasses.get((module.rel, cls), ()),
            ):
                for target in self._resolve_on(owner, owner_cls, name):
                    found.setdefault(target, None)
            cached = list(found) or [("alias", name)]
            self._self_cache[key] = cached
        return cached

    def _contained(
        self, entry: "frozenset[str] | _Ref | None", fn: _Function
    ) -> frozenset[str] | None:
        """A push's entry (:func:`_settled_by`) as this graph settles it:
        the futures its callback settles, when nothing else can reach them,
        else ``None``.

        A ``callback=self.x`` reference is resolved first
        (:meth:`_self_settles`). Then each ``self.X`` it settles must have no
        use of ``.X`` anywhere in the package beyond ``fn``'s own fresh
        stores -- settle calls aside (``_Module.attr_uses``). Another
        method's ``self.X = self._answer``, a nested def's, a done-callback
        on ``self.X`` or a relay awaiting it each chain it to what the
        handler awaits, and ``fn``'s own scope shows none of them (PR #2987
        review, round 3). Matched by name, not by class: an unrelated
        class's ``.X`` keeps the push too (recall over precision).
        """
        if entry is None:
            return None
        if entry.__class__ is tuple:
            entry = self._self_settles(entry, fn)
            if entry is None:
                return None
        for name in entry:
            if name.startswith("self.") and self._attr_use_count(
                name[5:]
            ) != fn.fresh_self_stores.get(name):
                return None
        return entry

    def _attr_use_count(self, attr: str) -> int:
        count = self._attr_uses.get(attr)
        if count is None:
            count = sum(module.attr_uses.get(attr, 0) for module in self.modules)
            self._attr_uses[attr] = count
        return count

    def _self_settles(self, ref: _Ref, fn: _Function) -> frozenset[str] | None:
        """What a ``callback=self.name`` push in ``fn`` settles, when every
        definition ``self.name`` can dispatch to settles ``self`` futures
        alone.

        ``self.name`` runs whatever the instance's class dispatches it to,
        so this is the union over everything :meth:`_self_targets` resolves
        -- ``fn``'s class, its bases and every in-package subclass -- and
        only when each one is a method whose whole visible effect is
        settling ``self`` futures (:func:`_prove_settles`) and all of them
        are fresh futures ``fn`` created. Reading only the class's own
        method missed a subclass override that settles the awaited future
        (PR #2987 review, round 2). ``None`` otherwise, and whenever
        anything in the package can bind ``name`` over those methods: an
        attribute store (``self._settle = self._finish`` in ``__init__``)
        or a class body's non-``def`` binding (a subclass's ``_settle =
        make_settler()``), which dispatch reads before the class's method
        (round 3). Whether those futures are contained is
        :meth:`_contained`'s question.
        """
        cls = self._class_of(fn)
        if cls is None or ref[1] in self._rebound_members:
            return None
        settled: set[str] = set()
        for target in self._self_targets(fn.module, cls, ref[1]):
            proved = target.settles_as_method if isinstance(target, _Function) else None
            if proved is None:
                return None
            settled |= proved
        return frozenset(settled) if settled <= fn.fresh_futures else None

    def _target_waits(self, target: _Target) -> bool:
        if isinstance(target, _Function):
            return target.waiting
        kind, name = target
        if kind == "bound":
            return name in self.waiting_bound
        if kind == "funcs":
            return (
                name in self.waiting_bare_aliases or name in self.waiting_func_names
            )
        if name in self.waiting_aliases:
            return True
        return kind == "defs" and name in self.waiting_def_names

    def _target_sites(self, target: _Target) -> frozenset[str]:
        if isinstance(target, _Function):
            return target.sites
        kind, name = target
        if kind == "bound":
            return self._bound_sites.get(name, frozenset())
        if kind == "funcs":
            return self._bare_alias_sites.get(name, frozenset()) | self._func_sites.get(
                name, frozenset()
            )
        sites = self._alias_sites.get(name, frozenset())
        if kind == "defs":
            sites = sites | self._def_sites.get(name, frozenset())
        return sites

    def _solve(self) -> None:
        pending = [
            fn
            for fn in self.functions
            if not fn.is_worker and (fn.own_pushes or fn.targets or fn.publishers)
        ]
        weak = self.weak_alias_targets
        # Every alias a table can make wait: its own bindings' names, and
        # the names only guessed bindings reach.
        alias_tables = [
            (table, waiting, [*table, *(alias for alias in weak if alias not in table)])
            for table, waiting in (
                (self.alias_targets, self.waiting_aliases),
                (self.bare_alias_targets, self.waiting_bare_aliases),
            )
        ]
        changed = True
        while changed:
            changed = False
            still_pending: list[_Function] = []
            for fn in pending:
                if (
                    fn.own_pushes
                    or fn.publishers
                    or any(self._target_waits(target) for target in fn.targets)
                ):
                    fn.waiting = True
                    if fn.parent is None and fn.live:
                        self.waiting_def_names.add(fn.name)
                        if fn.cls is None:
                            self.waiting_func_names.add(fn.name)
                    changed = True
                else:
                    still_pending.append(fn)
            pending = still_pending
            for table, waiting, names in alias_tables:
                for alias in names:
                    if alias in waiting:
                        continue
                    # EVERY binding must wait; a guessed binding counts only
                    # once it does, so with no other binding one waiting
                    # guess is enough and a non-waiting guess changes nothing.
                    strong = table.get(alias)
                    if strong:
                        waits = all(
                            any(self._target_waits(target) for target in targets)
                            for targets in strong
                        )
                    else:
                        waits = any(
                            any(self._target_waits(target) for target in targets)
                            for targets in weak[alias]
                        )
                    if waits:
                        waiting.add(alias)
                        changed = True
            for bound, targets in self.bound_targets.items():
                if bound in self.waiting_bound:
                    continue
                if any(self._target_waits(target) for target in targets):
                    self.waiting_bound.add(bound)
                    changed = True

    def _solve_sites(self) -> None:
        """Each waiting function's push sites, by a second fixpoint over the
        waiting functions, aliases and class-bound attributes only."""
        waiting = [fn for fn in self.functions if fn.waiting]
        for fn in waiting:
            pushes = fn.own_pushes
            if pushes:
                fn.sites = frozenset((fn.key,))
                self.site_pushes[fn.key] = max(self.site_pushes.get(fn.key, 0), pushes)
                if not fn.wait_pushes:
                    self.callback_only_sites.add(fn.key)
            for publisher in fn.publishers:
                # The push is the publisher's, but only this await waits on it.
                fn.sites = fn.sites | {publisher.key}
                self.site_pushes[publisher.key] = max(
                    self.site_pushes.get(publisher.key, 0), publisher.published_pushes
                )
                if not publisher.wait_pushes:
                    self.callback_only_sites.add(publisher.key)

        def union(targets: list[_Target]) -> frozenset[str]:
            return frozenset().union(
                *(self._target_sites(target) for target in targets)
            )

        def alias_union(
            table: dict[str, list[list[_Target]]], waiting_names: set[str]
        ) -> dict[str, frozenset[str]]:
            weak = self.weak_alias_targets
            return {
                alias: frozenset().union(
                    *(
                        union(targets)
                        for targets in (*table.get(alias, ()), *weak.get(alias, ()))
                    )
                )
                for alias in waiting_names
            }

        changed = True
        while changed:
            changed = False
            def_sites: dict[str, frozenset[str]] = {}
            func_sites: dict[str, frozenset[str]] = {}
            for fn in waiting:
                if fn.parent is None and fn.live:
                    def_sites[fn.name] = def_sites.get(fn.name, frozenset()) | fn.sites
                    if fn.cls is None:
                        func_sites[fn.name] = (
                            func_sites.get(fn.name, frozenset()) | fn.sites
                        )
            self._def_sites = def_sites
            self._func_sites = func_sites
            alias_sites = alias_union(self.alias_targets, self.waiting_aliases)
            bare_alias_sites = alias_union(
                self.bare_alias_targets, self.waiting_bare_aliases
            )
            bound_sites = {
                bound: union(self.bound_targets[bound]) for bound in self.waiting_bound
            }
            if (
                alias_sites != self._alias_sites
                or bare_alias_sites != self._bare_alias_sites
                or bound_sites != self._bound_sites
            ):
                changed = True
            self._alias_sites = alias_sites
            self._bare_alias_sites = bare_alias_sites
            self._bound_sites = bound_sites
            for fn in waiting:
                sites = fn.sites | union(fn.targets)
                if sites != fn.sites:
                    fn.sites = sites
                    changed = True

    def _rows(self, root: str, sites: frozenset[str], on_pump: bool) -> list[str]:
        rows: list[str] = []
        for site in sorted(sites):
            if not on_pump and site in self.callback_only_sites:
                continue
            rows.extend([f"{root} => {site}"] * self.site_pushes.get(site, 1))
        return rows

    def unresolved(self) -> list[str]:
        """Positional handoffs of a WAITING callable whose parameter W003
        could not name, as ``"<holder> -> <callee>#<position> => <site>"``:
        whatever the callee does with it is invisible here, so they are
        reported rather than silently dropped (TASK-33621.33).

        Only a value W003 resolved as a definition, a bare name or a
        ``self`` attribute counts. An ``obj.x`` value matches every ``x`` in
        the package by name; on the real tree every one of those was a
        collision (``Stylesheet.apply`` passed to ``getattr`` "waited"
        through an unrelated dialog's ``apply``), so they are not reported.
        """
        rows: list[str] = []
        for holder, callee, position, every in self._unresolved:
            targets = [
                target
                for target in every
                if isinstance(target, _Function) or target[0] != "defs"
            ]
            if not any(self._target_waits(target) for target in targets):
                continue
            sites = frozenset().union(
                *(self._target_sites(target) for target in targets)
            )
            rows.extend(
                f"{holder} -> {callee}#{position} => {site}" for site in sorted(sites)
            )
        return rows

    def roots(self) -> list[str]:
        sites: list[str] = []
        for fn in self.functions:
            if fn.is_worker or not fn.live:
                continue
            if fn.is_root and fn.waiting:
                sites.extend(self._rows(fn.key, fn.sites, on_pump=True))
            if not fn.scheduled:
                continue
            cls = self._class_of(fn)
            for ref, scheduler in fn.scheduled:
                reached = frozenset().union(
                    *(
                        self._target_sites(target)
                        for target in self._targets(ref, fn.module, cls, fn)
                    )
                )
                sites.extend(
                    self._rows(
                        f"{fn.key}->{ref[1]}",
                        reached,
                        on_pump=scheduler not in _SCHEDULED_COROUTINE,
                    )
                )
        return sites


def collect_w003(modules: list[tuple[ast.Module, Path]]) -> list[str]:
    """Non-worker entry points that reach a wait-for-dismiss screen push.

    Args:
        modules: Every parsed module with its path -- the resolution is
            cross-module by design.

    Returns:
        One ``"<root> => <site>"`` census key per wait push each offending
        root reaches. ``<root>`` is ``"<path>::<Class.>function"`` (a
        pump-scheduled callable is ``"<scheduler key>-><callable>"``);
        ``<site>`` is the key of the function holding the push. A key repeats
        once per wait push in that site and once per same-named LIVE root
        (nested functions of one name in different parents share a key; a
        dead earlier definition is not a root). ``main`` adds the rows of
        :func:`collect_w003_unresolved` to these.
    """
    return _WaitGraph(
        [_collect_module(tree, _rel(path)) for tree, path in modules]
    ).roots()


def collect_w003_unresolved(modules: list[tuple[ast.Module, Path]]) -> list[str]:
    """Positional handoffs of a waiting callable that W003 cannot bind.

    Args:
        modules: Every parsed module with its path, as for
            :func:`collect_w003` -- a handoff's callee and its callable
            argument may be defined in different modules.

    Returns:
        One ``"<holder> -> <callee>#<position> => <site>"`` census key per
        wait-push site the handed-over callable reaches, for each positional
        argument whose parameter no package definition of the callee names
        (see :meth:`_WaitGraph.unresolved`). ``<holder>`` is the key of the
        function making the call (``"<path>::<Class.><body>"`` for a class
        body or module level); ``<site>`` is the key of the function holding
        the push. ``main`` reports these with the rows of :func:`collect_w003`.
    """
    return _WaitGraph(
        [_collect_module(tree, _rel(path)) for tree, path in modules]
    ).unresolved()


def _read_census(census: Path | None = None) -> dict[str, int]:
    census = CENSUS if census is None else census
    if not census.exists():
        return {}
    rows: dict[str, int] = {}
    for line in census.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, rest = line.partition("\t")
        # A reviewed row, in either census, carries a third column: its note.
        count = rest.partition("\t")[0].strip()
        rows[key] = int(count) if count.isdigit() else 1
    return rows


def _read_census_notes(census: Path) -> dict[str, str]:
    """Each reviewed row's note: the third column, verdict and follow-up."""
    if not census.exists():
        return {}
    notes: dict[str, str] = {}
    for line in census.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t", 2)
        if len(fields) == 3 and fields[2].strip():
            notes[fields[0].strip()] = fields[2].strip()
    return notes


def _tally(sites: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for site in sites:
        counts[site] = counts.get(site, 0) + 1
    return counts


#: Both censuses share one reader, so both carry the note column the same
#: way: W002 once accepted a third column that its own --write dropped.
_NOTE_HEADER = (
    "# A third column is a REVIEWED row's note -- its verdict, evidence and\n"
    "# follow-up. --write carries a note forward for as long as its row\n"
    "# survives, so a re-pin never erases a review.\n"
)


def _pin(census: Path, header: str, sites: list[str]) -> None:
    """Rewrite ``census`` from ``sites``, keeping each surviving row's note."""
    notes = _read_census_notes(census)
    counts = _tally(sites)
    body = "\n".join(
        f"{key}\t{counts[key]}" + (f"\t{notes[key]}" if key in notes else "")
        for key in sorted(counts)
    )
    census.write_text(header + body + "\n", encoding="utf-8")


def _write_census(sites: list[str]) -> None:
    header = (
        "# Baseline census for W002: `query_one` reached after an `await` with no\n"
        "# enclosing `try` body whose handlers can catch a NoMatches, in UI/ and\n"
        "# Widgets/. Generated by\n"
        "# scripts/check_textual_worker_contract.py --write\n"
        "#\n"
        "# These rows are a BASELINE, not an endorsement: they were captured\n"
        "# mechanically and have not been individually reviewed. Whether one of\n"
        "# them can actually crash depends on whether the awaited work can remove\n"
        "# the subtree, which is not statically decidable -- see TASK-32800.1 for\n"
        "# one that could.\n"
        "#\n"
        "# Removing a row is always fine. Adding one is a deliberate act: prefer\n"
        "# guarding the lookup (`query()` + `first()`, or an enclosing `try`) or\n"
        "# re-checking that the widget is still mounted after the await.\n"
        "#\n"
        "# Rows are keyed by ENCLOSING FUNCTION, not by line number, so an edit\n"
        "# elsewhere in the file does not churn the census.\n"
        "#\n"
        + _NOTE_HEADER
        + "#\n"
        "# path::async def\tunguarded lookups in it[\tnote]\n"
    )
    _pin(CENSUS, header, sites)


def _write_wait_push_census(sites: list[str]) -> None:
    header = (
        "# Baseline census for W003: a non-worker entry point (message handler,\n"
        "# action, watcher, a `push_screen` result callback, or a callable handed\n"
        "# to call_later/call_next/call_after_refresh/set_timer/set_interval/\n"
        "# create_task/ensure_future) that reaches `push_screen_wait` or\n"
        "# `push_screen(..., wait_for_dismiss=True)`. Textual pushes the screen\n"
        "# and THEN raises NoActiveWorker, which kills the dispatching pump under\n"
        "# the painted screen -- GAP4-01, the Console Inspector 'Choose folder'\n"
        "# freeze (TASK-33621.13). The same wait by hand -- `push_screen(...,\n"
        "# callback=done)` and then `await` a future `done` completes --\n"
        "# deadlocks instead: Textual queues `done` on the pump that is blocked\n"
        "# on that await (TASK-33621.28). Generated by\n"
        "# scripts/check_textual_worker_contract.py --write\n"
        "#\n"
        "# These rows are a BASELINE, not an endorsement: they were captured\n"
        "# mechanically, `obj.x()` is resolved by NAME (so some rows are two\n"
        "# unrelated functions sharing a name), and a row without a note (see\n"
        "# below) has not been individually reviewed. Any of them may be a real\n"
        "# freeze.\n"
        "#\n"
        "# Each row is an entry point AND the function holding the push it\n"
        "# reaches, so a new push reachable from a censused entry point is still\n"
        "# a new row. Its count is the wait pushes in that function, times the\n"
        "# same-named entry points.\n"
        "#\n"
        "# Removing a row is always fine. Adding one is a deliberate act: run the\n"
        "# flow in a worker (`run_worker(coro)` / `@work`), or push with a\n"
        "# `callback=` and return -- not await a future that callback completes.\n"
        "#\n"
        + _NOTE_HEADER
        + "#\n"
        "# A row `path::[Class.]function -> callee#N => ...` is instead a waiting\n"
        "# callable handed as positional argument N to a callee W003 cannot bind\n"
        "# to a parameter (defined outside the package, or a `*args`): what the\n"
        "# callee does with it is invisible here.\n"
        "#\n"
        "# path::[Class.]entry point[->scheduled callable] => path::[Class.]function\n"
        "# holding the push\toccurrences[\tnote]\n"
    )
    _pin(WAIT_PUSH_CENSUS, header, sites)


def _entry_points(keys) -> set[str]:
    """The distinct entry points among W003 ``"<root> => <site>"`` keys."""
    return {key.partition(" => ")[0] for key in keys}


def _added(known: dict[str, int], current: dict[str, int]) -> list[str]:
    """Census keys the tree now holds more of than the census pins."""
    return sorted(key for key, count in current.items() if count > known.get(key, 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write",
        action="store_true",
        help="re-pin the W002 and W003 censuses to what the tree currently contains",
    )
    args = parser.parse_args()

    w001: list[str] = []
    w002: list[str] = []
    collected: list[tuple[_Module, list[_Function]]] = []
    for path in _source_files():
        try:
            with warnings.catch_warnings():
                # A few modules carry invalid escape sequences; they are a
                # separate finding, and their SyntaxWarnings are not this
                # check's output.
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        w001.extend(collect_w001(tree, path))
        w002.extend(collect_w002(tree, path))
        collected.append(_collect_module(tree, _rel(path)))
    graph = _WaitGraph(collected)
    w003 = graph.roots() + graph.unresolved()

    if args.write:
        _write_census(w002)
        print(f"W002 census re-pinned: {len(w002)} site(s) -> {_rel(CENSUS)}")
        _write_wait_push_census(w003)
        print(
            f"W003 census re-pinned: {len(w003)} wait push(es) from "
            f"{len(_entry_points(w003))} entry point(s) -> {_rel(WAIT_PUSH_CENSUS)}"
        )
        if w001:
            print("W001 violations are NOT allowlistable; fix them:")
            for row in w001:
                print(f"  {row}")
            return 1
        return 0

    failed = False

    if w001:
        failed = True
        print(
            f"::error::{len(w001)} synchronous run_worker target(s) without thread=True."
        )
        print(
            "Textual's Worker._run_async raises WorkerError for a non-coroutine "
            "target, and exit_on_error defaults to True, so each of these exits "
            "the application when it runs:"
        )
        for row in w001:
            site, name = row.split("\t")
            print(f"  {site}  ({name})")
        print("Pass thread=True, or make the target a coroutine.")
        print()

    known = _read_census()
    current = _tally(w002)
    added = _added(known, current)
    if added:
        failed = True
        print(f"::error::{len(added)} new post-await DOM lookup(s) with no guard.")
        print(
            "A `query_one` that resumes after an `await` can find its subtree "
            "removed -- NoMatches propagates out of the worker and, with "
            "exit_on_error defaulting to True, exits the application "
            "(TASK-32800.1 was exactly this). Guard the lookup with `query()` + "
            "`first()`, or an enclosing `try`, or re-check the widget is still "
            "mounted after the await:"
        )
        for key in added:
            print(f"  {key}  ({known.get(key, 0)} in census, {current[key]} now)")
        print()
        print(
            "If the await genuinely cannot remove the subtree, re-pin the census "
            "with:  python scripts/check_textual_worker_contract.py --write"
        )
        print()

    known_waits = _read_census(WAIT_PUSH_CENSUS)
    current_waits = _tally(w003)
    added_waits = _added(known_waits, current_waits)
    if added_waits:
        failed = True
        print(
            f"::error::{len(added_waits)} new wait-for-dismiss screen push(es) "
            "reachable from a non-worker entry point."
        )
        print(
            "`push_screen_wait` / `push_screen(..., wait_for_dismiss=True)` "
            "appends the screen and THEN raises NoActiveWorker outside a "
            "worker: the dispatching pump dies under the painted screen and "
            "the app stops responding (TASK-33621.13, the Inspector 'Choose "
            "folder' freeze). Awaiting a future that a `push_screen` callback "
            "completes is the same wait by hand, and deadlocks: Textual queues "
            "the callback on the pump that is blocked on the await "
            "(TASK-33621.28). Run the flow in a worker (`run_worker(coro)` or "
            "`@work`), or push with a `callback=` and return. Each row is "
            "`<entry point> => <function holding the push>`; a row "
            "`<function> -> <callee>#<n> => ...` is a waiting callable handed "
            "as positional argument <n> to a callee W003 cannot see into -- "
            "pass it by keyword to a parameter W003 can follow, or check what "
            "the callee does with it:"
        )
        for key in added_waits:
            print(
                f"  {key}  ({known_waits.get(key, 0)} in census, "
                f"{current_waits[key]} now)"
            )
        print()
        print(
            "If the entry point genuinely always runs inside a worker, re-pin "
            "with:  python scripts/check_textual_worker_contract.py --write"
        )
        print()

    if failed:
        return 1

    resolved = sum(
        max(0, count - current.get(key, 0)) for key, count in known.items()
    )
    note = f"; {resolved} baseline lookup(s) resolved" if resolved else ""
    resolved_waits = sum(
        max(0, count - current_waits.get(key, 0))
        for key, count in known_waits.items()
    )
    wait_note = (
        f"; {resolved_waits} baseline push(es) resolved" if resolved_waits else ""
    )
    print(
        f"textual worker contract: no synchronous run_worker targets; "
        f"{sum(current.values())} post-await DOM lookup(s) in "
        f"{len(current)} function(s), none new{note}; "
        f"{sum(current_waits.values())} wait-for-dismiss push(es) reachable from "
        f"{len(_entry_points(current_waits))} non-worker entry point(s), "
        f"none new{wait_note}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
