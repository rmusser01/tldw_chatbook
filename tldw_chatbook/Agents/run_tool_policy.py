"""Per-run tool call caps from persona policy rules.

Workspace assistant defaults (Task 7). ``max_calls_per_turn`` rules from a
persona's policy are enforced HERE rather than inside any single provider:
``ToolCatalogRegistry.invoke_by_name`` -- the one choke point every
provider's ``invoke()`` is reached through -- consults a ``RunToolPolicy``
before dispatch, so the cap holds for MCP, skill, local, library, and any
provider added later without each of them opting in.

Narrowing-only: a policy instance can only refuse; it never widens access
and never replaces the permission gates that decide whether a call may run
at all. Counters key on ``(run_id, tool name)`` so concurrent sub-agent
runs sharing one registry cannot consume each other's caps, and the policy
object itself is per-run (built fresh by ``_compose_run_registry_and_
allowed``), which is what makes the counts per-turn rather than per-session.
"""

from __future__ import annotations

import threading
from typing import Mapping

PERSONA_POLICY_CALL_CAP_REFUSAL = "persona_policy_call_cap_reached: {name}"


class RunToolPolicy:
    """Counts invocations per (run_id, tool name); refuses past the cap."""

    def __init__(self, caps: Mapping[str, int]) -> None:
        self._caps = dict(caps)
        self._counts: dict[tuple[str, str], int] = {}
        # invoke_by_name dispatches on per-call daemon threads (see
        # AgentService._call_with_timeout), so sibling tool calls in one run
        # can check() concurrently; the count read-modify-write must be
        # atomic or two racing calls could both pass a cap of 1.
        self._lock = threading.Lock()

    def check(self, run_id: str, name: str) -> tuple[bool, str | None]:
        """Consume one allowed invocation of ``name`` in ``run_id``, if any.

        Args:
            run_id: The dispatching run's id (``run_context.current_run_id``
                -- ``""`` outside any agent run, a distinct key no run
                competes with).
            name: The LLM-facing tool name being invoked.

        Returns:
            ``(True, None)`` when the call may proceed (the counter was
            advanced); ``(False, refusal_message)`` once the cap is reached
            -- and on every later check for that ``(run_id, name)``, since a
            refused call does not consume budget. Uncapped names always
            proceed and are never counted.
        """
        cap = self._caps.get(name)
        if cap is None:
            return True, None
        key = (run_id, name)
        with self._lock:
            count = self._counts.get(key, 0)
            if count >= cap:
                return False, PERSONA_POLICY_CALL_CAP_REFUSAL.format(name=name)
            self._counts[key] = count + 1
        return True, None
