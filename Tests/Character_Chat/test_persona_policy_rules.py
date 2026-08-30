"""Persona policy rules mirror the tldw_server PersonaPolicyRule contract."""

import pytest
from tldw_chatbook.tldw_api.character_persona_schemas import PersonaPolicyRule


def test_rule_shape_matches_server_contract():
    rule = PersonaPolicyRule.model_validate(
        {"rule_kind": "mcp_tool", "rule_name": "fs_write", "allowed": False}
    )
    assert rule.rule_kind == "mcp_tool"
    assert rule.require_confirmation is False
    assert rule.max_calls_per_turn is None


def test_rejects_unknown_kind_and_extras():
    with pytest.raises(Exception):
        PersonaPolicyRule.model_validate({"rule_kind": "syscall", "rule_name": "x"})
    with pytest.raises(Exception):
        PersonaPolicyRule.model_validate(
            {"rule_kind": "skill", "rule_name": "x", "grant": True}
        )


def test_caps_minimum_is_one():
    with pytest.raises(Exception):
        PersonaPolicyRule.model_validate(
            {"rule_kind": "mcp_tool", "rule_name": "x", "max_calls_per_turn": 0}
        )


def test_normalize_drops_malformed_rules_without_logging_private_values():
    from loguru import logger

    from tldw_chatbook.Character_Chat.local_character_persona_service import (
        normalize_policy_rules,
    )

    records = []
    sink_id = logger.add(records.append, level="WARNING", format="{message}")
    try:
        cleaned = normalize_policy_rules(
            [
                {"rule_kind": "mcp_tool", "rule_name": "ok"},
                {"rule_kind": "bogus", "rule_name": "PRIVATE_POLICY_RULE"},
                "PRIVATE_NON_MAPPING_RULE",
            ]
        )
    finally:
        logger.remove(sink_id)

    assert cleaned == [
        {"rule_kind": "mcp_tool", "rule_name": "ok", "allowed": True,
         "require_confirmation": False, "max_calls_per_turn": None}
    ]
    rendered = "".join(str(record) for record in records)
    assert rendered.count("Dropping malformed persona policy rule") == 2
    assert "PRIVATE_POLICY_RULE" not in rendered
    assert "PRIVATE_NON_MAPPING_RULE" not in rendered
