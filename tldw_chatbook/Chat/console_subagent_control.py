"""Targeted subagent control through the bridge's existing run ownership."""

from __future__ import annotations

from . import console_agent_bridge as host


def steer_subagent(
    self,
    conversation_id: str,
    row_id: str,
    text: str,
    *,
    expected_run_id: str | None = None,
) -> bool:
    """Queue USER steering for ONE live child of this conversation's fleet.

    PR3b Task 3 (spec §6's second path into Task 1's per-child
    mailbox; §7's drill-in steering input). The USER twin of
    ``AgentService``'s ``send_to_agent`` closure (Task 2), sharing its
    resolution SHAPE but not its plumbing:

    * **No service hop, deliberately** — unlike ``cancel_subagent``
      just above, whose retained-owner walk exists only because cancel
      Events are service-local, the mailbox lives on the
      conversation-lifetime ``FleetCoordinator`` this bridge owns
      (``_fleet_coordinators``), reachable from the UI thread under
      the coordinator's own brief lock. A live survivor another turn's
      service spawned is steerable for free.
    * **Both id vocabularies, handle id FIRST** (Task 2's pinned
      order): a live fleet row's ``row_id`` IS the handle id, but the
      drill-in target the panel actually holds
      (``_console_agent_drilldown_run_id``) is a RUN id — both must
      reach the same mailbox, and a pathological collision lands on
      the handle-id owner.
    * **Validation at THIS boundary** (non-empty after strip,
      ``MAX_STEERING_CHARS``) — ``post_steering`` deliberately does
      not validate (Task 1's pinned decision); each producer owes its
      own refusal. This producer's user-facing copy lives in the
      steering bar widget, which refuses before posting — the checks
      here are the boundary's own, so no caller can bypass them.
    * **Steering never cancels** (spec §3 invariant 4): the post
      touches the mailbox and nothing else — no cancel Event, no run
      row, no handle status.

    Args:
        conversation_id: The conversation whose fleet to look up.
        row_id: A live child's handle id, or its run id.
        text: The steering message body (posted stripped; the drain
            point prepends the ``[Steering from user]`` label).

    Returns:
        ``True`` when the entry was queued for a LIVE child; ``False``
        — never raises — for empty/oversize text, a conversation with
        no coordinator, an unknown id, or a finished child (including
        losing the race where the child goes terminal between the
        snapshot and the post — ``post_steering`` refuses terminal
        handles itself, and that refusal is returned honestly).
    """
    stripped = (text or "").strip()
    if not stripped or len(stripped) > host.MAX_STEERING_CHARS:
        return False
    coordinator = self._fleet_coordinators.get(conversation_id)
    if coordinator is None:
        return False
    live = [
        handle
        for handle in coordinator.snapshot()
        if handle.status not in host.TERMINAL_RUN_STATUSES
    ]
    target = next((h for h in live if h.handle_id == row_id), None) or next(
        (h for h in live if h.run_id == row_id), None
    )
    if target is None or (
        expected_run_id is not None and target.run_id != expected_run_id
    ):
        return False
    return coordinator.post_steering(
        target.handle_id, host.STEERING_SOURCE_USER, stripped
    )
