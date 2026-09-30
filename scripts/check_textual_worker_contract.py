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

    The roots are message handlers (``@on``, ``on_*``/``_on_*``, ``key_*``),
    actions (``action_*``) and watchers (``watch_*``), none of which Textual
    runs in a worker, plus a callable handed to ``call_later``/``call_next``/
    ``call_after_refresh``/``set_timer``/``set_interval`` or a coroutine handed
    to ``create_task``/``ensure_future``. A root is reported when it pushes
    directly, or ``await``s -- transitively -- something that does. ``@work``
    functions, and coroutines handed to ``run_worker``, are workers and stop
    the propagation.

    "Transitively" is resolved by NAME, deliberately, because the real defect
    crossed three modules through callables passed as arguments:
    ``partial(recover, select_binding=controller._select_binding)`` in a dict
    whose key became the Inspector's ``__init__`` parameter and then its
    ``self._project_instruction_recovery`` attribute. So a keyword argument,
    a string-keyed dict entry, or an assignment whose value refers to a
    waiting callable makes its keyword/key/target name an alias for one; an
    ``await`` of an alias waits too. ``self.x()`` resolves to the enclosing
    class's own ``x`` when it defines one; a bare ``x()`` to a nested or
    module-level ``x`` in scope first. This over-approximates (two unrelated
    functions sharing a name are one to it), which is why W003 is a census
    like W002 rather than a zero-tolerance gate: the pre-existing roots are
    pinned in ``scripts/textual_wait_push_census.tsv`` and only a NEW root
    fails. Those rows are an unreviewed baseline -- each may be a real freeze
    of the GAP4-01 kind -- not an endorsement.

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
        awaits = [node.lineno for node in own if isinstance(node, ast.Await)]
        if not awaits:
            continue
        first_await = min(awaits)
        for node in own:
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in DOM_LOOKUPS
            ):
                continue
            if node.lineno <= first_await:
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

#: A callable reference: ``("self", name)`` for ``self.name``, ``("name",
#: name)`` for a bare name, ``("attr", name)`` for ``anything.name``.
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
    """The callable ``node`` names, unwrapping ``partial(target, ...)``."""
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


class _Function:
    """One ``def``/``async def`` and what W003 needs to know about it."""

    def __init__(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        module: "_Module",
        cls: str | None,
        parent: "_Function | None",
    ) -> None:
        # The name only, never the node: holding every module's AST alive
        # until the graph is solved made the cyclic GC rescan them all and
        # more than doubled the checker's parse time.
        self.name = node.name
        self.module = module
        self.cls = cls
        self.parent = parent
        self.nested: dict[str, _Function] = {}
        # `local = <callable>` inside this function: scoped, never global.
        self.local_aliases: dict[str, list[_Ref]] = {}
        decorators = _decorator_names(node)
        self.is_worker = "work" in decorators
        self.is_root = not self.is_worker and (
            "on" in decorators or node.name.startswith(HANDLER_PREFIXES)
        )
        self.direct = False
        self.awaited: list[_Ref] = []
        self.scheduled: list[_Ref] = []
        self.waiting = False

    @property
    def key(self) -> str:
        owner = f"{self.cls}." if self.cls else ""
        return f"{self.module.rel}::{owner}{self.name}"


class _Module:
    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.functions: dict[str, _Function] = {}  # module-level defs
        self.classes: dict[str, dict[str, _Function]] = {}
        # (alias name, value reference, class context, function context)
        self.aliases: list[tuple[str, _Ref, str | None, _Function | None]] = []


_SCHEDULED_COROUTINE = {"create_task", "ensure_future"}


def _record_call(sink: _Function, call: ast.Call) -> None:
    """A direct wait push, or a callable handed to a pump scheduler."""
    if _is_wait_push(call):
        sink.direct = True
    name = getattr(call.func, "attr", None) or getattr(call.func, "id", None)
    if name not in PUMP_SCHEDULERS:
        return
    for arg in call.args:
        target = (
            arg.func
            if isinstance(arg, ast.Call) and name in _SCHEDULED_COROUTINE
            else arg
        )
        ref = _ref(target)
        if ref is not None:
            sink.scheduled.append(ref)


def _record_await(sink: _Function, node: ast.Await) -> None:
    """The awaited call, and any coroutine built inline as its argument
    (``await asyncio.wait_for(self.pick(), 5)`` still runs ``pick`` here --
    except ``run_worker(self.pick())``, whose argument runs in a worker)."""
    if not isinstance(node.value, ast.Call):
        return
    func = node.value.func
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


def _alias_pairs(node: ast.AST) -> list[tuple[str, ast.AST, bool]]:
    """Names a callable is handed on under: keyword, dict key, assignment.

    Returns ``(name, value, is_bare_name_target)``: an assignment to a bare
    name inside a function is a LOCAL alias, resolved in that function's
    scope only; everything else can cross into another function's
    parameter or attribute, so it goes into the package-wide table.
    """
    if isinstance(node, ast.Call):
        return [(kw.arg, kw.value, False) for kw in node.keywords if kw.arg]
    if isinstance(node, ast.Dict):
        return [
            (key.value, value, False)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
    if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        pairs = []
        for target in targets:
            if isinstance(target, ast.Name):
                pairs.append((target.id, node.value, True))
            elif isinstance(target, ast.Attribute):
                pairs.append((target.attr, node.value, False))
        return pairs
    return []


_ALIAS_SOURCES = (ast.Call, ast.Dict, ast.Assign, ast.AnnAssign)


def _collect_module(tree: ast.Module, rel: str) -> tuple[_Module, list[_Function]]:
    """One iterative pass: defs, their pushes/awaits/schedules, and aliases.

    Hand-rolled rather than ``ast.walk`` + ``iter_child_nodes`` twice: this
    runs over the whole package inside ``preflight.sh``, and the two-pass
    first cut tripled the checker's runtime. ``ctx`` children (``Load``/
    ``Store``) are skipped for the same reason.

    Each stack entry carries the enclosing class, the enclosing function
    (the alias context), and the "sink" that owns pushes and awaits -- which
    is ``None`` inside a ``lambda``: its body runs later, not in the
    enclosing function's await chain.
    """
    module = _Module(rel)
    functions: list[_Function] = []
    AST = ast.AST
    stack: list[tuple[ast.AST, str | None, _Function | None, _Function | None]] = [
        (tree, None, None, None)
    ]
    while stack:
        node, cls, fn, sink = stack.pop()
        kind = type(node)
        if kind is ast.ClassDef:
            module.classes.setdefault(node.name, {})
            cls, fn, sink = node.name, None, None
        elif kind is ast.FunctionDef or kind is ast.AsyncFunctionDef:
            new = _Function(node, module, cls if fn is None else None, fn)
            functions.append(new)
            if fn is not None:
                fn.nested[node.name] = new
            elif cls is not None:
                module.classes[cls][node.name] = new
            else:
                module.functions[node.name] = new
            fn = sink = new
        elif kind is ast.Lambda:
            sink = None
        else:
            if sink is not None:
                if kind is ast.Call:
                    _record_call(sink, node)
                elif kind is ast.Await:
                    _record_await(sink, node)
            if isinstance(node, _ALIAS_SOURCES):
                local = fn is not None and kind is not ast.Call and kind is not ast.Dict
                for alias, value, is_name in _alias_pairs(node):
                    ref = _ref(value)
                    if ref is None or ref[1] == alias:
                        continue
                    if local and is_name:
                        fn.local_aliases.setdefault(alias, []).append(ref)
                    else:
                        module.aliases.append((alias, ref, cls, fn))
        for field in node._fields:
            if field == "ctx":
                continue
            value = getattr(node, field, None)
            if value.__class__ is list:
                for item in value:
                    if isinstance(item, AST):
                        stack.append((item, cls, fn, sink))
            elif isinstance(value, AST):
                stack.append((value, cls, fn, sink))
    return module, functions


class _WaitGraph:
    """Name-resolved fixpoint of "awaits a wait-for-dismiss push".

    A function or method name reached through the fallback (``obj.x()``, an
    inherited ``self.x()``, an imported ``x()``) waits when ANY definition of
    that name waits -- the real defect's chain ran through
    ``controller._select_project_instruction_binding``, a name shared with a
    non-waiting method of the Console runtime. An ALIAS waits only when EVERY
    place it is bound hands on a waiting callable: with "any" there too, one
    ``callback=<waiting>`` keyword made every ``await callback()`` in the
    package wait, and the first cut of this check reported 517 roots,
    nearly all of them that cascade.
    """

    def __init__(self, collected: list[tuple[_Module, list[_Function]]]) -> None:
        self.modules: list[_Module] = []
        self.functions: list[_Function] = []
        for module, functions in collected:
            self.modules.append(module)
            self.functions.extend(functions)
        self.defs_by_name: dict[str, list[_Function]] = {}
        for fn in self.functions:
            if fn.parent is None:
                self.defs_by_name.setdefault(fn.name, []).append(fn)
        self.alias_bindings: dict[
            str, list[tuple[_Ref, _Module, str | None, _Function | None]]
        ] = {}
        for module in self.modules:
            for alias, ref, cls, fn in module.aliases:
                self.alias_bindings.setdefault(alias, []).append(
                    (ref, module, cls, fn)
                )
        self.waiting_def_names: set[str] = set()
        self.waiting_aliases: set[str] = set()
        self._solve()

    def _name_waits(self, name: str) -> bool:
        return name in self.waiting_def_names or name in self.waiting_aliases

    def _waits(
        self,
        ref: _Ref,
        module: _Module,
        cls: str | None,
        fn: _Function | None,
        depth: int = 0,
    ) -> bool:
        kind, name = ref
        if kind == "self" and cls is not None:
            method = module.classes.get(cls, {}).get(name)
            if method is not None:
                return method.waiting
        elif kind == "name":
            scope = fn
            while scope is not None:
                if name in scope.nested:
                    return scope.nested[name].waiting
                if name in scope.local_aliases:
                    # `recovery = self._project_instruction_recovery` then
                    # `await recovery(...)`: follow the local, bounded so an
                    # `a = b; b = a` cycle cannot recurse forever.
                    return depth < 16 and any(
                        self._waits(local, module, cls, scope, depth + 1)
                        for local in scope.local_aliases[name]
                    )
                scope = scope.parent
            if name in module.functions:
                return module.functions[name].waiting
        return self._name_waits(name)

    @staticmethod
    def _class_of(fn: _Function) -> str | None:
        while fn.parent is not None:
            fn = fn.parent
        return fn.cls

    def _solve(self) -> None:
        pending = [
            fn
            for fn in self.functions
            if not fn.is_worker and (fn.direct or fn.awaited)
        ]
        changed = True
        while changed:
            changed = False
            still_pending: list[_Function] = []
            for fn in pending:
                cls = self._class_of(fn)
                if fn.direct or any(
                    self._waits(ref, fn.module, cls, fn) for ref in fn.awaited
                ):
                    fn.waiting = True
                    if fn.parent is None:
                        self.waiting_def_names.add(fn.name)
                    changed = True
                else:
                    still_pending.append(fn)
            pending = still_pending
            for alias, bindings in self.alias_bindings.items():
                if alias in self.waiting_aliases:
                    continue
                if all(
                    self._waits(ref, module, cls, fn)
                    for ref, module, cls, fn in bindings
                ):
                    self.waiting_aliases.add(alias)
                    changed = True

    def roots(self) -> list[str]:
        sites: list[str] = []
        for fn in self.functions:
            if fn.is_worker:
                continue
            if fn.is_root and fn.waiting:
                sites.append(fn.key)
            cls = self._class_of(fn)
            sites.extend(
                f"{fn.key}->{ref[1]}"
                for ref in fn.scheduled
                if self._waits(ref, fn.module, cls, fn)
            )
        return sites


def collect_w003(modules: list[tuple[ast.Module, Path]]) -> list[str]:
    """Non-worker entry points that reach a wait-for-dismiss screen push.

    Args:
        modules: Every parsed module with its path -- the resolution is
            cross-module by design.

    Returns:
        One ``"<path>::<Class.>function"`` census key per offending root (a
        pump-scheduled callable is keyed ``"<scheduler key>-><callable>"``),
        repeated when one file holds several same-named roots.
    """
    return _WaitGraph(
        [_collect_module(tree, _rel(path)) for tree, path in modules]
    ).roots()


def _read_census(census: Path | None = None) -> dict[str, int]:
    census = CENSUS if census is None else census
    if not census.exists():
        return {}
    rows: dict[str, int] = {}
    for line in census.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, count = line.partition("\t")
        rows[key] = int(count) if count.strip().isdigit() else 1
    return rows


def _tally(sites: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for site in sites:
        counts[site] = counts.get(site, 0) + 1
    return counts


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
        "# path::async def\tunguarded lookups in it\n"
    )
    counts = _tally(sites)
    body = "\n".join(f"{key}\t{counts[key]}" for key in sorted(counts))
    CENSUS.write_text(header + body + "\n", encoding="utf-8")


def _write_wait_push_census(sites: list[str]) -> None:
    header = (
        "# Baseline census for W003: a non-worker entry point (message handler,\n"
        "# action, watcher, or a callable handed to call_later/call_next/\n"
        "# call_after_refresh/set_timer/set_interval/create_task/ensure_future)\n"
        "# that reaches `push_screen_wait` or `push_screen(..., wait_for_dismiss=\n"
        "# True)`. Textual pushes the screen and THEN raises NoActiveWorker, which\n"
        "# kills the dispatching pump under the painted screen -- GAP4-01, the\n"
        "# Console Inspector 'Choose folder' freeze (TASK-33621.13). Generated by\n"
        "# scripts/check_textual_worker_contract.py --write\n"
        "#\n"
        "# These rows are a BASELINE, not an endorsement: they were captured\n"
        "# mechanically, reachability is resolved by NAME (so some rows are two\n"
        "# unrelated functions sharing a name), and none has been individually\n"
        "# reviewed. Any of them may be a real freeze.\n"
        "#\n"
        "# Removing a row is always fine. Adding one is a deliberate act: run the\n"
        "# flow in a worker (`run_worker(coro)` / `@work`) or push with a\n"
        "# `callback=` instead of awaiting the dismissal.\n"
        "#\n"
        "# path::[Class.]entry point[->scheduled callable]\toccurrences\n"
    )
    counts = _tally(sites)
    body = "\n".join(f"{key}\t{counts[key]}" for key in sorted(counts))
    WAIT_PUSH_CENSUS.write_text(header + body + "\n", encoding="utf-8")


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
    w003 = _WaitGraph(collected).roots()

    if args.write:
        _write_census(w002)
        print(f"W002 census re-pinned: {len(w002)} site(s) -> {_rel(CENSUS)}")
        _write_wait_push_census(w003)
        print(
            f"W003 census re-pinned: {len(w003)} root(s) -> {_rel(WAIT_PUSH_CENSUS)}"
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
    added = sorted(
        key for key, count in current.items() if count > known.get(key, 0)
    )
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
    added_waits = sorted(
        key
        for key, count in current_waits.items()
        if count > known_waits.get(key, 0)
    )
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
            "folder' freeze). Run the flow in a worker (`run_worker(coro)` or "
            "`@work`), or push with a `callback=` instead of awaiting:"
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
        f"; {resolved_waits} baseline root(s) resolved" if resolved_waits else ""
    )
    print(
        f"textual worker contract: no synchronous run_worker targets; "
        f"{sum(current.values())} post-await DOM lookup(s) in "
        f"{len(current)} function(s), none new{note}; "
        f"{sum(current_waits.values())} non-worker wait-for-dismiss root(s), "
        f"none new{wait_note}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
