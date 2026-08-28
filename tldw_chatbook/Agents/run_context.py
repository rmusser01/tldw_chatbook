"""The dispatching run actor, bound around a run's tool-facing work.

Two bindings exist, both established by ``AgentService`` and readable through
``current_run_actor()``/``current_run_id()``. They are not redundant: they cover
different THREADS, because the work they guard runs in different places.

1. **Per-invocation** (PR2a Task 5, ``_make_invoke_tool``) -- bound around
   each ``ToolProvider.invoke``, so the permission gates can find this
   run's own approval stamps when a tool executes. It must be established
   inside the callable handed to ``_call_with_timeout``, because that
   helper runs the tool on a fresh per-call daemon thread which does NOT
   inherit the caller's context; a binding set on the loop thread would
   simply be invisible there.

2. **Loop-wide** (PR2a Task 7, around ``run_agent_loop`` in ``_run_one``)
   -- bound for the whole run, on the loop thread. The per-invocation
   binding above cannot cover this: the two things that ARM HUMAN
   APPROVAL CARDS run on the loop thread rather than inside a provider
   invoke -- ``review_tool_calls`` (one batch-approval round trip per
   turn) and the in-loop runtime tools, of which ``run_skill_script``
   raises a confirm card of its own. Both record ``current_run_id()`` at
   arm time so a cancelled child's card can be revoked without touching a
   live sibling's, and neither can be handed the id as a parameter (the
   approval bridge is a pre-bound partial shared with
   ``MCPToolProvider.approval_callback``; the runtime-tool closures are
   built one layer below any run identity). One binding there covers both
   -- and every future loop-thread consumer -- with no signature churn.

The two nest harmlessly: the inner binding sets the same actor the outer one
holds, on a different thread.

WHY A ContextVar AT ALL (PR2a Task 5). Both permission gates key this
turn's verdict stamps by ``(run_id, tool_name)`` instead of by tool name
alone, so that concurrent sub-agent runs sharing ONE gate instance cannot
clear or overwrite each other's verdicts. The WRITE side gets its run id
explicitly: the review hook is a callable ``AgentService`` wires per run,
so ``run_id`` is just a parameter (see ``AgentService.review_tool_calls``).

The READ side cannot. A stamp is consumed inside ``ToolProvider.invoke``
(``BuiltinToolProvider.invoke`` -> ``BuiltinToolGate.check``,
``MCPToolProvider.invoke`` -> ``stamped_decision``,
``LocalToolProvider.invoke`` -> ``_verdict_for``), and that method's
signature is a Protocol (``tool_catalog.ToolProvider``) implemented by
every provider and called generically by
``ToolCatalogRegistry.invoke_by_name``; widening it would touch every
provider and hundreds of call sites for a value only three of them want.
So the run actor rides a ``ContextVar`` that ``AgentService`` binds around
each invocation instead -- the same shape, and for the same reason, as
``Tools/workspace_file_roots.run_workspace``, which already binds THIS
run's workspace around ``BuiltinToolProvider.invoke``'s tool execution.

Why either binding is sound under concurrency: each is reset in its own
``finally``, so nested inline runs unwind LIFO on one thread, and a run on
its own thread (PR2a Task 6, ON by default since Task 6.5) simply sets its
own value in its own context -- siblings never see each other's.

``current_run_id()`` returns ``""`` outside any agent run -- e.g. a direct
``provider.invoke()`` from the MCP workbench's Test Tool. That is a
distinct key no review hook ever writes, so such a caller finds no stamp
and falls through to the provider's own gate, which is exactly what it did
before this module existed.

Stdlib only, no project imports: this is consumed by ``tool_catalog`` and
the providers, which are all deliberately dependency-light.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Literal


@dataclass(frozen=True, slots=True)
class CurrentRunActor:
    """Attribution for the primary or subagent invoking a provider tool."""

    kind: Literal["primary", "subagent"]
    run_id: str
    parent_run_id: str | None


_CURRENT_RUN_ACTOR: ContextVar[CurrentRunActor | None] = ContextVar(
    "tldw_agent_run_actor", default=None
)
_CURRENT_TOOL_CALL_ID: ContextVar[str] = ContextVar(
    "tldw_agent_tool_call_id", default=""
)


def current_run_actor() -> CurrentRunActor | None:
    """Return the actor bound to this tool thread, if any.

    Returns:
        Exact bound actor, or ``None`` outside an agent run.
    """
    return _CURRENT_RUN_ACTOR.get()


def current_run_id() -> str:
    """The run id whose tool call is executing here.

    Returns:
        The bound run id, or ``""`` when no agent run bound one on this
        thread (a direct provider call outside any run). Never raises.
    """
    actor = current_run_actor()
    return actor.run_id if actor is not None else ""


def current_tool_call_id() -> str:
    """Return the native id of the tool call currently being dispatched."""
    return _CURRENT_TOOL_CALL_ID.get()


@contextmanager
def use_run_actor(actor: CurrentRunActor) -> Iterator[None]:
    """Bind exact provider-call attribution for the duration of a block.

    Args:
        actor: Non-empty primary or subagent attribution to bind.

    Yields:
        None while the actor is bound on the current context.

    Raises:
        ValueError: If ``actor`` is not a valid non-empty attribution.
    """
    if not isinstance(actor, CurrentRunActor) or not actor.run_id:
        raise ValueError("a non-empty CurrentRunActor is required")
    token = _CURRENT_RUN_ACTOR.set(actor)
    try:
        yield
    finally:
        _CURRENT_RUN_ACTOR.reset(token)


@contextmanager
def use_run_id(run_id: str) -> Iterator[None]:
    """Bind ``run_id`` as the dispatching run for the duration of the block.

    Args:
        run_id: The run whose tool call is about to execute. A falsy value
            binds ``""`` (the no-run key), never ``None``.

    Yields:
        None. The previous binding is restored on exit, including on an
        exception, so nested inline runs on one thread unwind correctly.
    """
    actor = CurrentRunActor("primary", run_id, None) if run_id else None
    token = _CURRENT_RUN_ACTOR.set(actor)
    try:
        yield
    finally:
        _CURRENT_RUN_ACTOR.reset(token)


@contextmanager
def use_tool_call_id(call_id: str) -> Iterator[None]:
    """Bind one tool-call id without changing its run attribution."""
    token = _CURRENT_TOOL_CALL_ID.set(str(call_id or ""))
    try:
        yield
    finally:
        _CURRENT_TOOL_CALL_ID.reset(token)
