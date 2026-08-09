"""The dispatching run's id, bound for the duration of one tool invocation.

PR2a Task 5. Both permission gates now key this turn's verdict stamps by
``(run_id, tool_name)`` instead of by tool name alone, so that concurrent
sub-agent runs sharing ONE gate instance cannot clear or overwrite each
other's verdicts. The WRITE side gets its run id explicitly: the review
hook is a callable ``AgentService`` wires per run, so ``run_id`` is just a
parameter (see ``AgentService.review_tool_calls``).

The READ side cannot. A stamp is consumed inside ``ToolProvider.invoke``
(``BuiltinToolProvider.invoke`` -> ``BuiltinToolGate.check``,
``MCPToolProvider.invoke`` -> ``stamped_decision``,
``LocalToolProvider.invoke`` -> ``_verdict_for``), and that method's
signature is a Protocol (``tool_catalog.ToolProvider``) implemented by
every provider and called generically by
``ToolCatalogRegistry.invoke_by_name``; widening it would touch every
provider and hundreds of call sites for a value only three of them want.
So the run id rides a ``ContextVar`` that ``AgentService`` binds around
each invocation instead -- the same shape, and for the same reason, as
``Tools/workspace_file_roots.run_workspace``, which already binds THIS
run's workspace around ``BuiltinToolProvider.invoke``'s tool execution.

Why a per-invocation binding is sound: the binding is established inside
the callable ``AgentService._make_invoke_tool`` hands to
``_call_with_timeout``, i.e. *on the thread that actually runs the tool*
(``_call_with_timeout`` runs it on a fresh per-call daemon thread, which
does NOT inherit the caller's context), and it is reset in its own
``finally``. Nested inline runs are therefore LIFO-safe on one thread, and
a run on its own thread (PR2a Task 6) simply sets its own value.

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
from typing import Iterator

#: The run id whose tool call is executing on this thread, or "" for none.
_CURRENT_RUN_ID: ContextVar[str] = ContextVar("tldw_agent_run_id", default="")


def current_run_id() -> str:
    """The run id whose tool call is executing here.

    Returns:
        The bound run id, or ``""`` when no agent run bound one on this
        thread (a direct provider call outside any run). Never raises.
    """
    return _CURRENT_RUN_ID.get()


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
    token = _CURRENT_RUN_ID.set(run_id or "")
    try:
        yield
    finally:
        _CURRENT_RUN_ID.reset(token)
