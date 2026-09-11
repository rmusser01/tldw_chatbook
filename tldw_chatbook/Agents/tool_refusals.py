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

__all__ = ["TOOL_KILL_SWITCH_REFUSAL"]
