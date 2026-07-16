"""Tests for MCP permission effective-state resolution (Phase 4, Task 2).

Covers: `definition_hash` determinism, `EffectiveToolState.ui_label`,
`resolve_effective_state` precedence (tool override -> server default ->
global default), the rug-pull hash guard (mismatch and persisted
`config_changed` marker, both independently downgrading an explicit allow),
the high-risk floor (inherited-allow-only, tag-gated), and the two
Space-cycle helpers.
"""
from __future__ import annotations

import json

from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import (
    HIGH_RISK_TAGS,
    EffectiveToolState,
    cycle_global,
    cycle_ui_state,
    definition_hash,
    resolve_effective_state,
)


def _tool(
    *,
    server_key: str = "local:demo",
    name: str = "search",
    description: str = "Search docs",
    input_schema: dict | None = None,
    tags: tuple[str, ...] = (),
) -> HubTool:
    return HubTool(
        server_key=server_key,
        server_label="demo",
        source="local",
        name=name,
        description=description,
        input_schema=input_schema,
        tags=tags,
        stale=False,
        executable=True,
    )


def _payload(*, global_default: str = "ask", servers: dict | None = None) -> dict:
    return {
        "schema_version": 1,
        "kill_switch": False,
        "profiles": {
            "default": {
                "global_default": global_default,
                "servers": servers or {},
            }
        },
    }


# -- definition_hash ---------------------------------------------------------


def test_definition_hash_matches_manual_canonical_json():
    expected_canonical = json.dumps(
        {"description": "desc", "inputSchema": {"a": 1}},
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    import hashlib

    expected = hashlib.sha256(expected_canonical.encode("utf-8")).hexdigest()

    assert definition_hash("desc", {"a": 1}) == expected


def test_definition_hash_is_order_independent():
    assert definition_hash("desc", {"a": 1, "b": 2}) == definition_hash("desc", {"b": 2, "a": 1})


def test_definition_hash_defaults_none_description_and_schema():
    assert definition_hash("", None) == definition_hash(None, None)  # type: ignore[arg-type]


def test_definition_hash_differs_for_different_inputs():
    assert definition_hash("desc", {"a": 1}) != definition_hash("desc", {"a": 2})
    assert definition_hash("desc-a", None) != definition_hash("desc-b", None)


# -- EffectiveToolState.ui_label ---------------------------------------------


def test_ui_label_maps_states_to_display_text():
    assert EffectiveToolState(state="allow", origin="tool_override").ui_label == "Allow"
    assert EffectiveToolState(state="ask", origin="global_default").ui_label == "Ask"
    assert EffectiveToolState(state="deny", origin="server_default").ui_label == "Off"


# -- HIGH_RISK_TAGS ------------------------------------------------------------


def test_high_risk_tags_constant():
    assert HIGH_RISK_TAGS == frozenset({"mutates", "process"})


# -- resolve_effective_state: precedence --------------------------------------


def test_resolve_effective_state_tool_override_wins_over_server_and_global():
    tool = _tool()
    current_hash = definition_hash(tool.description, tool.input_schema)
    payload = _payload(
        global_default="deny",
        servers={
            tool.server_key: {
                "default": "deny",
                "tools": {tool.name: {"state": "allow", "definition_hash": current_hash}},
            }
        },
    )

    result = resolve_effective_state(payload, tool)

    assert result.state == "allow"
    assert result.origin == "tool_override"
    assert result.config_changed is False
    assert result.risk_floored is False


def test_resolve_effective_state_falls_back_to_server_default_when_no_tool_entry():
    tool = _tool()
    payload = _payload(global_default="deny", servers={tool.server_key: {"default": "ask"}})

    result = resolve_effective_state(payload, tool)

    assert result.state == "ask"
    assert result.origin == "server_default"


def test_resolve_effective_state_falls_back_to_global_default_when_no_server_or_tool_entry():
    tool = _tool()
    payload = _payload(global_default="deny", servers={})

    result = resolve_effective_state(payload, tool)

    assert result.state == "deny"
    assert result.origin == "global_default"


def test_resolve_effective_state_falls_back_to_global_default_when_server_entry_has_no_default():
    tool = _tool()
    payload = _payload(
        global_default="allow",
        servers={tool.server_key: {"tools": {"other-tool": {"state": "ask"}}}},
    )

    result = resolve_effective_state(payload, tool)

    assert result.state == "allow"
    assert result.origin == "global_default"


def test_resolve_effective_state_tool_override_ask_and_deny_pass_through_unchanged():
    tool = _tool()
    payload = _payload(servers={tool.server_key: {"tools": {tool.name: {"state": "ask"}}}})

    result = resolve_effective_state(payload, tool)

    assert result.state == "ask"
    assert result.origin == "tool_override"
    assert result.config_changed is False


# -- resolve_effective_state: rug-pull hash guard ------------------------------


def test_resolve_effective_state_matching_hash_does_not_downgrade():
    tool = _tool()
    current_hash = definition_hash(tool.description, tool.input_schema)
    payload = _payload(servers={tool.server_key: {"tools": {tool.name: {"state": "allow", "definition_hash": current_hash}}}})

    result = resolve_effective_state(payload, tool)

    assert result.state == "allow"
    assert result.config_changed is False


def test_resolve_effective_state_hash_mismatch_downgrades_allow_to_ask():
    tool = _tool()
    payload = _payload(
        servers={tool.server_key: {"tools": {tool.name: {"state": "allow", "definition_hash": "stale-hash"}}}}
    )

    result = resolve_effective_state(payload, tool)

    assert result.state == "ask"
    assert result.origin == "tool_override"
    assert result.config_changed is True


def test_resolve_effective_state_config_changed_marker_downgrades_despite_matching_hash():
    tool = _tool()
    current_hash = definition_hash(tool.description, tool.input_schema)
    payload = _payload(
        servers={
            tool.server_key: {
                "tools": {
                    tool.name: {
                        "state": "allow",
                        "definition_hash": current_hash,
                        "config_changed": True,
                    }
                }
            }
        }
    )

    result = resolve_effective_state(payload, tool)

    assert result.state == "ask"
    assert result.config_changed is True


def test_resolve_effective_state_config_changed_marker_downgrades_with_mismatched_hash_too():
    tool = _tool()
    payload = _payload(
        servers={
            tool.server_key: {
                "tools": {
                    tool.name: {
                        "state": "allow",
                        "definition_hash": "stale-hash",
                        "config_changed": True,
                    }
                }
            }
        }
    )

    result = resolve_effective_state(payload, tool)

    assert result.state == "ask"
    assert result.config_changed is True


# -- resolve_effective_state: high-risk floor ----------------------------------


def test_resolve_effective_state_floor_applies_to_inherited_allow_via_server_default():
    tool = _tool(tags=("mutates",))
    payload = _payload(servers={tool.server_key: {"default": "allow"}})

    result = resolve_effective_state(payload, tool)

    assert result.state == "ask"
    assert result.origin == "server_default"
    assert result.risk_floored is True


def test_resolve_effective_state_floor_applies_to_inherited_allow_via_global_default():
    tool = _tool(tags=("process",))
    payload = _payload(global_default="allow", servers={})

    result = resolve_effective_state(payload, tool)

    assert result.state == "ask"
    assert result.origin == "global_default"
    assert result.risk_floored is True


def test_resolve_effective_state_floor_does_not_apply_to_explicit_tool_override_allow():
    tool = _tool(tags=("mutates",))
    current_hash = definition_hash(tool.description, tool.input_schema)
    payload = _payload(servers={tool.server_key: {"tools": {tool.name: {"state": "allow", "definition_hash": current_hash}}}})

    result = resolve_effective_state(payload, tool)

    assert result.state == "allow"
    assert result.origin == "tool_override"
    assert result.risk_floored is False


def test_resolve_effective_state_floor_does_not_apply_to_inherited_ask():
    tool = _tool(tags=("mutates",))
    payload = _payload(servers={tool.server_key: {"default": "ask"}})

    result = resolve_effective_state(payload, tool)

    assert result.state == "ask"
    assert result.risk_floored is False


def test_resolve_effective_state_floor_does_not_apply_when_tags_dont_intersect():
    tool = _tool(tags=("readonly",))
    payload = _payload(servers={tool.server_key: {"default": "allow"}})

    result = resolve_effective_state(payload, tool)

    assert result.state == "allow"
    assert result.risk_floored is False


# -- cycle helpers --------------------------------------------------------------


def test_cycle_ui_state_full_loop():
    assert cycle_ui_state(None) == "allow"
    assert cycle_ui_state("allow") == "ask"
    assert cycle_ui_state("ask") == "deny"
    assert cycle_ui_state("deny") is None


def test_cycle_global_full_loop():
    assert cycle_global("allow") == "ask"
    assert cycle_global("ask") == "deny"
    assert cycle_global("deny") == "allow"
