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
    """A persona's validated, narrowing-only tool policy rules.

    Attributes:
        rules: Cleaned rule dicts (``PersonaPolicyRule`` model-dump shape).
        kinds: The ``rule_kind`` values present, used for deny-by-default.
    """

    rules: tuple[dict[str, object], ...] = ()
    kinds: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ToolPolicyVerdict:
    """The outcome of evaluating a policy against one tool.

    Attributes:
        advertised: Whether the tool may be advertised at all.
        requires_confirmation: Whether an ``allow`` floors to ``ask``.
        max_calls_per_turn: Smallest per-run call cap among matches, if any.
    """

    advertised: bool
    requires_confirmation: bool
    max_calls_per_turn: int | None


def parse_persona_policy_from_rules(
    rules: Iterable[Mapping[str, object]] | None,
) -> PersonaToolPolicy:
    """Parse raw ``policy_rules`` entries into a ``PersonaToolPolicy``.

    Args:
        rules: Raw rule payloads (may be ``None``); non-mapping or
            malformed entries are logged and dropped.

    Returns:
        The policy with only the validatable rules retained.
    """
    cleaned: list[dict[str, object]] = []
    for entry in rules or ():
        if not isinstance(entry, Mapping):
            logger.warning(
                "Dropping non-mapping persona policy rule; entry_type={}",
                type(entry).__name__,
            )
            continue
        try:
            from tldw_chatbook.tldw_api.character_persona_schemas import PersonaPolicyRule

            cleaned.append(
                PersonaPolicyRule.model_validate(dict(entry)).model_dump(mode="json")
            )
        except Exception as exc:
            logger.warning(
                "Dropping malformed persona policy rule; error_type={}",
                type(exc).__name__,
            )
    return PersonaToolPolicy(
        rules=tuple(cleaned), kinds=frozenset(str(r["rule_kind"]) for r in cleaned)
    )


def parse_persona_policy(record: Mapping[str, object]) -> PersonaToolPolicy:
    """Parse a persona record's ``policy_rules`` into a policy.

    Args:
        record: A persona record mapping (anything with a ``policy_rules``
            key); a non-mapping value degrades to an empty policy.

    Returns:
        The parsed ``PersonaToolPolicy`` (empty when no valid rules exist).
    """
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
    """Evaluate a policy for one tool of the given rule kind.

    Args:
        policy: The persona's parsed policy.
        rule_kind: Rule kind to evaluate (e.g. ``"mcp_tool"``); kinds with
            no rules are fully advertised.
        tool_name: The tool name to match (bounded ``prefix*`` wildcards).

    Returns:
        The verdict: advertised/ask/capped, deny-by-default when the kind
        has at least one allow rule.
    """
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
    """Apply the persona confirmation floor to a resolved tool state.

    Args:
        state: The post-gates resolved tool state.
        policy: The persona's parsed policy.
        tool_name: The MCP tool name to evaluate.

    Returns:
        ``ask`` (origin ``persona_policy``) when the policy requires
        confirmation and the state was ``allow``; otherwise ``state``.
    """
    verdict = evaluate_tool_policy(policy, rule_kind="mcp_tool", tool_name=tool_name)
    if verdict.requires_confirmation and state.state == "allow":
        return EffectiveToolState(state="ask", origin="persona_policy")
    return state
