"""Refusal sentences shared by every tool-dispatch seam (task-32285).

Deliberately IMPORT-FREE. The five sites that report a kill-switch block
sit on both sides of this repo's lazy-import boundaries -- the built-in
gate, the two dispatched providers, the Console controller's pre-dispatch
review block, and the transcript classifier in ``console_agent_bridge``,
which hand-copied the sentence rather than take a dependency edge on
``Agents.builtin_tool_gate`` (see ``_blocked_provider_refusals()`` there).
A module with no imports of its own can be pulled in from any of them
without dragging anything else along, so the sentence has exactly one
definition and downstream classifiers keying on it by identity or prefix
cannot drift apart.
"""

from __future__ import annotations

#: TASK-631 / task-32285: the result every tool call gets while the chat
#: tool kill switch is on. Names the switch, so the model (and a user
#: reading the transcript) can tell this from a per-call denial.
TOOL_KILL_SWITCH_REFUSAL = "tool call blocked: the chat tool kill switch is on"

#: TASK-33620.1.12: the start of the result every agent tool call gets while
#: the global emergency stop is active. Owner decision, 2026-11-23: an
#: in-flight run's tool calls are held too, not only new runs. Names the stop,
#: so the model and a user reading the transcript can tell it from a per-call
#: denial and from the kill switch above.
TOOL_EMERGENCY_STOP_REFUSAL = "tool call blocked: the emergency stop is active"

#: ``tldw_chatbook.emergency_stop.EMERGENCY_STOP_CLEAR_COMMAND``, copied
#: because this module takes no imports; a test pins that the two agree.
_EMERGENCY_STOP_CLEAR_COMMAND = "/emergency-stop clear"

#: The stop's reason is free text the user typed. The model and the call's
#: transcript row get it on one line, bounded.
_EMERGENCY_STOP_REASON_LIMIT = 200


def emergency_stop_tool_refusal(reason: str = "") -> str:
    """The emergency stop's refusal for one agent tool call.

    Args:
        reason: The stop's reason: what the user typed after
            ``/emergency-stop``, or the fail-safe cause when the stop's
            state could not be read. Blank leaves the parentheses out.

    Returns:
        One line naming the stop, its reason and how to clear it.
    """
    why = " ".join(str(reason or "").split())
    if len(why) > _EMERGENCY_STOP_REASON_LIMIT:
        why = why[: _EMERGENCY_STOP_REASON_LIMIT - 1].rstrip() + "\u2026"
    detail = f" ({why})" if why else ""
    return (
        f"{TOOL_EMERGENCY_STOP_REFUSAL}{detail}. No tool runs until the user "
        f"clears it with {_EMERGENCY_STOP_CLEAR_COMMAND}."
    )


__all__ = [
    "TOOL_EMERGENCY_STOP_REFUSAL",
    "TOOL_KILL_SWITCH_REFUSAL",
    "emergency_stop_tool_refusal",
]
