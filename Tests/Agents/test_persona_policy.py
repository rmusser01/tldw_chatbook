import copy
from dataclasses import replace

from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_chatbook.Agents.persona_policy import (
    PersonaToolPolicy,
    evaluate_tool_policy,
    parse_persona_policy,
    parse_persona_policy_from_rules,
    persona_floor_state,
)
from tldw_chatbook.MCP.permission_store import EffectiveToolState

RULES = st.lists(
    st.fixed_dictionaries(
        {
            "rule_kind": st.sampled_from(["mcp_tool", "skill"]),
            "rule_name": st.sampled_from(["fs_write", "web_search", "fs_*", "web_*", "x*"]),
            "allowed": st.booleans(),
            "require_confirmation": st.booleans(),
            "max_calls_per_turn": st.one_of(st.none(), st.integers(min_value=1, max_value=9)),
        }
    ),
    max_size=6,
)
NAMES = st.sampled_from(["fs_write", "fs_read", "web_search", "web_fetch", "unrelated"])


def baseline(name):
    return evaluate_tool_policy(PersonaToolPolicy(), rule_kind="mcp_tool", tool_name=name)


@given(rules=RULES, name=NAMES)
@settings(max_examples=300)
def test_rules_never_widen(rules, name):
    verdict = evaluate_tool_policy(
        parse_persona_policy_from_rules(rules), rule_kind="mcp_tool", tool_name=name
    )
    base = baseline(name)
    assert verdict.advertised <= base.advertised
    assert verdict.requires_confirmation >= base.requires_confirmation


def test_no_rules_is_identity_posture():
    verdict = evaluate_tool_policy(
        parse_persona_policy({}), rule_kind="mcp_tool", tool_name="fs_write"
    )
    assert (verdict.advertised, verdict.requires_confirmation) == (True, False)


def test_deny_by_default_when_kind_rules_present():
    policy = parse_persona_policy_from_rules(
        [{"rule_kind": "mcp_tool", "rule_name": "fs_read", "allowed": True}]
    )
    unlisted = evaluate_tool_policy(policy, rule_kind="mcp_tool", tool_name="fs_write")
    assert unlisted.advertised is False  # kinds with rules deny unlisted tools


def test_explicit_denial_wins_and_confirmation_ors():
    policy = parse_persona_policy_from_rules(
        [
            {"rule_kind": "mcp_tool", "rule_name": "web_*", "allowed": True,
             "require_confirmation": True, "max_calls_per_turn": 4},
            {"rule_kind": "mcp_tool", "rule_name": "web_search", "allowed": False},
        ]
    )
    verdict = evaluate_tool_policy(policy, rule_kind="mcp_tool", tool_name="web_search")
    assert verdict.advertised is False
    other = evaluate_tool_policy(policy, rule_kind="mcp_tool", tool_name="web_fetch")
    assert other.advertised and other.requires_confirmation and other.max_calls_per_turn == 4


def test_bounded_wildcard_is_prefix_only():
    policy = parse_persona_policy_from_rules(
        [{"rule_kind": "mcp_tool", "rule_name": "fs_*", "allowed": False}]
    )
    assert not evaluate_tool_policy(policy, rule_kind="mcp_tool", tool_name="fs_list").advertised
    assert evaluate_tool_policy(policy, rule_kind="mcp_tool", tool_name="git_status").advertised


def test_skill_rules_do_not_affect_mcp_tools():
    policy = parse_persona_policy_from_rules(
        [{"rule_kind": "skill", "rule_name": "deep-research", "allowed": False}]
    )
    assert evaluate_tool_policy(policy, rule_kind="mcp_tool", tool_name="fs_read").advertised


def test_floor_state_only_lowers_allow():
    policy = parse_persona_policy_from_rules(
        [{"rule_kind": "mcp_tool", "rule_name": "web_*", "require_confirmation": True}]
    )
    allowed = EffectiveToolState(state="allow", origin="tool_override")
    floored = persona_floor_state(allowed, policy, "web_search")
    assert (floored.state, floored.origin) == ("ask", "persona_policy")
    # deny/ask pass through untouched; non-matching tool untouched
    assert persona_floor_state(
        EffectiveToolState(state="deny", origin="tool_override"), policy, "web_search"
    ).state == "deny"
    assert persona_floor_state(allowed, policy, "fs_read") is allowed


def test_invalid_rules_are_dropped_with_metadata_only_warnings():
    """Invalid rule diagnostics must not retain user-authored rule values."""
    from loguru import logger

    class _Handler:
        def __init__(self):
            self.records = []

        def __call__(self, message):
            self.records.append(str(message))

    handler = _Handler()
    logger_id = logger.add(
        handler, level="WARNING"
    )
    try:
        policy = parse_persona_policy_from_rules(
            [
                "PRIVATE_NON_MAPPING_RULE",
                {"rule_kind": "bogus", "rule_name": "PRIVATE_MALFORMED_RULE"},
                {"rule_kind": "mcp_tool", "rule_name": "fs_read"},
            ]
        )
    finally:
        logger.remove(logger_id)

    assert policy.kinds == frozenset({"mcp_tool"})
    assert len(policy.rules) == 1
    dropped = [record for record in handler.records if "Dropping" in record]
    assert len(dropped) == 2, handler.records
    rendered = "".join(dropped)
    assert "PRIVATE_NON_MAPPING_RULE" not in rendered
    assert "PRIVATE_MALFORMED_RULE" not in rendered
