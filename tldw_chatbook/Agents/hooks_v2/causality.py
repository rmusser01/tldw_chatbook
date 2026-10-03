"""Host-only causal ancestry propagated across nested hook/tool work."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class CausalChain:
    """Exact host identities; never reconstructed from public event data."""

    visits: tuple[tuple[str, str, str], ...] = ()

    def enter(self, event_id: str, handler_id: str, tool_id: str) -> CausalChain:
        """Reject repeated handler/tool identity or more than four nested calls."""
        if not all((event_id, handler_id, tool_id)):
            raise ValueError("hook_causal_identity_missing")
        if any(h == handler_id or t == tool_id for _, h, t in self.visits):
            raise ValueError("hook_dependency_cycle")
        if len(self.visits) >= 4:
            raise ValueError("hook_causal_depth")
        return CausalChain(self.visits + ((event_id, handler_id, tool_id),))

    @contextmanager
    def scope(self):
        token = _chain.set(self)
        try:
            yield self
        finally:
            _chain.reset(token)


_chain: ContextVar[CausalChain | None] = ContextVar("hook_causal_chain", default=None)


def current_chain() -> CausalChain:
    return _chain.get() or CausalChain()
