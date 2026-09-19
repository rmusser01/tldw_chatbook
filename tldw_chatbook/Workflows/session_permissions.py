"""Exact-effect approval handoff over the existing built-in permission gate."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

from tldw_chatbook.Agents.builtin_tool_gate import BuiltinGateDecision, BuiltinToolGate
from tldw_chatbook.Tools.note_management_tools import CreateNoteTool
from tldw_chatbook.Tools.tool_executor import Tool


@dataclass(frozen=True)
class EffectRequest:
    """Resolved effect identity; payloads are private and carry no approval grant."""

    run_id: str
    step_id: str
    kind: Literal["file", "model", "note"]
    payload_json: str = field(repr=False)


class _EffectTool(Tool):
    """Metadata for permission resolution only; never an executable LLM tool."""

    def __init__(self, name: str, description: str, risk: str) -> None:
        self._name = name
        self._description = description
        self._risk = risk

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    @property
    def risk_tags(self) -> tuple[str, ...]:
        return (self._risk,)

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Workflow permission descriptors cannot execute")


_EFFECT_TOOLS: dict[str, Tool] = {
    "file": _EffectTool(
        "workflow_read_file", "Read the selected workflow file", "reads"
    ),
    "model": _EffectTool(
        "workflow_local_model",
        "Send workflow input to the selected local model",
        "network",
    ),
    "note": CreateNoteTool(),
}


class WorkflowPermissions:
    """Scope temporary approvals to a single check; the gate owns all policy."""

    def __init__(self, gate: BuiltinToolGate) -> None:
        self._gate = gate

    def check(
        self, effect: EffectRequest, *, approved_once: bool = False
    ) -> BuiltinGateDecision:
        """Recheck fresh authority for the exact resolved effect.

        Args:
            effect: Run, step and resolved payload including captured destination.
            approved_once: Only the coordinator may supply this, after matching
                and consuming its pending effect identity. Never import it from
                workflow JSON. It may be reused for immediate pre-write checks
                only while that same physical operation remains owned.

        Returns:
            The gate's structured verdict. No approval survives this call.
        """
        tool = _EFFECT_TOOLS[effect.kind]
        digest = hashlib.sha256(effect.payload_json.encode()).hexdigest()
        effect_key = f"{effect.run_id}:{effect.step_id}:{digest}"
        self._gate.begin_turn(effect_key)
        try:
            if approved_once:
                self._gate.stamp(effect_key, tool.name, "approve_once")
            return self._gate.check_detailed(
                tool, effect_key, strict=True, allow_session_approvals=False
            )
        finally:
            self._gate.begin_turn(effect_key)
