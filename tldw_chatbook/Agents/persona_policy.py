"""Persona-local tool policy evaluation — narrowing-only.

Mirrors the server's persona policy semantics: deny-by-default when rules
exist for a kind, bounded (prefix-only) wildcards, explicit deny precedence,
confirmation floors, and per-run call caps. No rule can widen access; callers
layer this after every gate and floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from loguru import logger

from tldw_chatbook.MCP.permission_store import EffectiveToolState


@dataclass(frozen=True)
class PersonaToolPolicy:
    rules: tuple[dict, ...] = ()
    kinds: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ToolPolicyVerdict:
    advertised: bool
    requires_confirmation: bool
    max_calls_per_turn: int | None


def parse_persona_policy_from_rules(rules: Iterable[Mapping] | None) -> PersonaToolPolicy:
    cleaned: list[dict] = []
    for entry in rules or ():
        if not isinstance(entry, Mapping):
            logger.warning("Dropping non-mapping persona policy rule: {!r}", entry)
            continue
        try:
            from tldw_chatbook.tldw_api.character_persona_schemas import PersonaPolicyRule

            cleaned.append(
                PersonaPolicyRule.model_validate(dict(entry)).model_dump(mode="json")
            )
        except Exception:
            logger.warning("Dropping malformed persona policy rule: {!r}", entry)
    return PersonaToolPolicy(
        rules=tuple(cleaned), kinds=frozenset(r["rule_kind"] for r in cleaned)
    )


def parse_persona_policy(record: Mapping) -> PersonaToolPolicy:
    return parse_persona_policy_from_rules(
        record.get("policy_rules") if isinstance(record, Mapping) else None
    )


def _matches(rule_name: str, tool_name: str) -> bool:
    if rule_name.endswith("*"):
        return tool_name.startswith(rule_name[:-1])
    return rule_name == tool_name


def evaluate_tool_policy(
    policy: PersonaToolPolicy, *, rule_kind: str, tool_name: str
) -> ToolPolicyVerdict:
    if rule_kind not in policy.kinds:
        return ToolPolicyVerdict(True, False, None)
    kind_rules = [r for r in policy.rules if r["rule_kind"] == rule_kind]
    matched = [r for r in kind_rules if _matches(r["rule_name"], tool_name)]
    if not matched:
        # Deny-by-default is an allowlist posture: it activates only once the
        # persona allows at least one tool for the kind (a tool matching no
        # allowed=true rule is not advertised). A pure deny rule set only
        # carves out its matches and never unadvertises the whole kind.
        if any(r.get("allowed") is not False for r in kind_rules):
            return ToolPolicyVerdict(False, False, None)
        return ToolPolicyVerdict(True, False, None)
    if any(r.get("allowed") is False for r in matched):
        # A denied tool still reports requires_confirmation=True so downstream
        # refusal copy stays informative even when unadvertised.
        return ToolPolicyVerdict(False, True, None)
    caps = [r["max_calls_per_turn"] for r in matched if r.get("max_calls_per_turn")]
    return ToolPolicyVerdict(
        advertised=True,
        requires_confirmation=any(r.get("require_confirmation") for r in matched),
        max_calls_per_turn=min(caps) if caps else None,
    )


def persona_floor_state(
    state: EffectiveToolState, policy: PersonaToolPolicy, tool_name: str
) -> EffectiveToolState:
    verdict = evaluate_tool_policy(policy, rule_kind="mcp_tool", tool_name=tool_name)
    if verdict.requires_confirmation and state.state == "allow":
        return EffectiveToolState(state="ask", origin="persona_policy")
    return state
