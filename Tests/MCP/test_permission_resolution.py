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
    resolve_effective_state_by_key,
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


def test_ui_label_is_defensive_against_unknown_state():
    """I2's second layer of defense: `resolve_effective_state()` itself now
    never produces an out-of-`STORE_STATES` `state`, but `ui_label` must
    not `KeyError` regardless -- a future direct `EffectiveToolState(...)`
    construction, or a store shape this module hasn't seen yet, must
    render SOMETHING rather than panic whatever render pass called it."""
    assert EffectiveToolState(state="banana", origin="global_default").ui_label == "Banana"
    assert EffectiveToolState(state="", origin="global_default").ui_label == "Ask"


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


def test_resolve_effective_state_invalid_global_default_falls_back_to_ask():
    """I2: a hand-edited `mcp_permissions.json` with an invalid
    `global_default` (e.g. "banana" -- a valid `schema_version`, so
    `load()`'s own corruption check never backs it up/resets it) must
    resolve to "ask", not the raw invalid string -- passing that through
    used to `KeyError` out of `ui_label`/`format_tool_state_label` inside
    `_sync_children`, panicking the app on the very next matrix render."""
    tool = _tool()
    payload = _payload(global_default="banana", servers={})

    result = resolve_effective_state(payload, tool)

    assert result.state == "ask"
    assert result.origin == "global_default"
    assert result.ui_label == "Ask"


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


# -- I1: resolve_effective_state_by_key (hashless, no live HubTool) -----------
#
# Backs `UnifiedMCPControlPlaneService.gate_tool_test_by_key()` -- the Test
# Tool gate's fallback for when the tool has dropped out of the catalog
# snapshot (`_tool_for()` came back empty) but the gate must still resolve
# deny/ask/allow from the store alone, with no `HubTool` to hash-compare a
# rug-pull guard against.


def test_resolve_by_key_deny_tool_override_passes_through():
    payload = _payload(servers={"local:demo": {"tools": {"search": {"state": "deny"}}}})

    result = resolve_effective_state_by_key(payload, "local:demo", "search")

    assert result.state == "deny"
    assert result.origin == "tool_override"
    assert result.config_changed is False


def test_resolve_by_key_ask_tool_override_passes_through():
    payload = _payload(servers={"local:demo": {"tools": {"search": {"state": "ask"}}}})

    result = resolve_effective_state_by_key(payload, "local:demo", "search")

    assert result.state == "ask"
    assert result.origin == "tool_override"


def test_resolve_by_key_explicit_allow_downgrades_to_ask_config_unknown():
    """No live tool to hash-check against -- an explicit tool-level
    ``allow`` can never be confirmed fresh here, so it resolves to "ask"
    rather than silently trusting a possibly-stale allow (this is the
    exact gap the I1 fix closes: the gate must not resolve "allow" for a
    tool it can't verify)."""
    payload = _payload(
        servers={"local:demo": {"tools": {"search": {"state": "allow", "definition_hash": "whatever"}}}}
    )

    result = resolve_effective_state_by_key(payload, "local:demo", "search")

    assert result.state == "ask"
    assert result.origin == "tool_override"
    assert result.config_changed is True


def test_resolve_by_key_inherited_server_default_deny_passes_through():
    payload = _payload(servers={"local:demo": {"default": "deny"}})

    result = resolve_effective_state_by_key(payload, "local:demo", "search")

    assert result.state == "deny"
    assert result.origin == "server_default"


def test_resolve_by_key_inherited_server_default_allow_downgrades_to_ask():
    payload = _payload(servers={"local:demo": {"default": "allow"}})

    result = resolve_effective_state_by_key(payload, "local:demo", "search")

    assert result.state == "ask"
    assert result.origin == "server_default"
    assert result.config_changed is True


def test_resolve_by_key_inherited_global_default_deny_passes_through():
    payload = _payload(global_default="deny", servers={})

    result = resolve_effective_state_by_key(payload, "local:demo", "search")

    assert result.state == "deny"
    assert result.origin == "global_default"


def test_resolve_by_key_inherited_global_default_allow_downgrades_to_ask():
    payload = _payload(global_default="allow", servers={})

    result = resolve_effective_state_by_key(payload, "local:demo", "search")

    assert result.state == "ask"
    assert result.origin == "global_default"
    assert result.config_changed is True


def test_resolve_by_key_invalid_global_default_falls_back_to_ask():
    payload = _payload(global_default="banana", servers={})

    result = resolve_effective_state_by_key(payload, "local:demo", "search")

    assert result.state == "ask"
    assert result.origin == "global_default"


# -- hand-edited store: null/malformed intermediates never crash --------------
#
# A hand-edited `mcp_permissions.json` can pass `load()`'s top-level dict +
# schema_version check yet still carry `null` (or other non-mapping junk)
# for `profiles`, the default profile, `servers`, a server entry's `tools`,
# or an individual tool entry. Both resolvers take a raw payload directly
# (bypassing `load()`'s own normalization) and must never `AttributeError`
# out of a hand-edited file -- they degrade to the same "nothing configured
# here" result as an absent key.


def test_resolve_effective_state_null_profiles_does_not_raise():
    tool = _tool()
    payload = {"schema_version": 1, "kill_switch": False, "profiles": None}

    result = resolve_effective_state(payload, tool)

    assert result.state == "ask"
    assert result.origin == "global_default"


def test_resolve_effective_state_null_profile_does_not_raise():
    tool = _tool()
    payload = {"schema_version": 1, "kill_switch": False, "profiles": {"default": None}}

    result = resolve_effective_state(payload, tool)

    assert result.state == "ask"
    assert result.origin == "global_default"


def test_resolve_effective_state_null_servers_does_not_raise():
    tool = _tool()
    payload = {
        "schema_version": 1,
        "kill_switch": False,
        "profiles": {"default": {"global_default": "ask", "servers": None}},
    }

    result = resolve_effective_state(payload, tool)

    assert result.state == "ask"
    assert result.origin == "global_default"


def test_resolve_effective_state_null_tools_falls_back_to_server_default():
    tool = _tool()
    payload = _payload(servers={tool.server_key: {"default": "deny", "tools": None}})

    result = resolve_effective_state(payload, tool)

    assert result.state == "deny"
    assert result.origin == "server_default"


def test_resolve_effective_state_non_mapping_tool_entry_does_not_raise():
    tool = _tool()
    payload = _payload(servers={tool.server_key: {"tools": {tool.name: "not-a-mapping"}}})

    result = resolve_effective_state(payload, tool)

    assert result.state == "ask"
    assert result.origin == "global_default"


def test_resolve_by_key_null_profiles_does_not_raise():
    payload = {"schema_version": 1, "kill_switch": False, "profiles": None}

    result = resolve_effective_state_by_key(payload, "local:demo", "search")

    assert result.state == "ask"
    assert result.origin == "global_default"


def test_resolve_by_key_null_profile_does_not_raise():
    payload = {"schema_version": 1, "kill_switch": False, "profiles": {"default": None}}

    result = resolve_effective_state_by_key(payload, "local:demo", "search")

    assert result.state == "ask"
    assert result.origin == "global_default"


def test_resolve_by_key_null_servers_does_not_raise():
    payload = {
        "schema_version": 1,
        "kill_switch": False,
        "profiles": {"default": {"global_default": "ask", "servers": None}},
    }

    result = resolve_effective_state_by_key(payload, "local:demo", "search")

    assert result.state == "ask"
    assert result.origin == "global_default"


def test_resolve_by_key_null_tools_falls_back_to_server_default():
    payload = _payload(servers={"local:demo": {"default": "deny", "tools": None}})

    result = resolve_effective_state_by_key(payload, "local:demo", "search")

    assert result.state == "deny"
    assert result.origin == "server_default"


def test_resolve_by_key_non_mapping_tool_entry_does_not_raise():
    payload = _payload(servers={"local:demo": {"tools": {"search": "not-a-mapping"}}})

    result = resolve_effective_state_by_key(payload, "local:demo", "search")

    assert result.state == "ask"
    assert result.origin == "global_default"


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
