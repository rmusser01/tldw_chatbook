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

import pytest

from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import (
    BUILTIN_TOOL_SERVER_KEY,
    BY_KEY_HASH_FREE_SERVER_KEYS,
    GatedToolRef,
    HASH_FREE_SERVER_KEYS,
    HIGH_RISK_TAGS,
    EffectiveToolState,
    cycle_global,
    cycle_ui_state,
    definition_hash,
    profile_lifecycle_disposition,
    resolve_builtin_state,
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


def _named_payload(profile_id: str, profile: dict) -> dict:
    payload = _payload(global_default="allow")
    payload["profiles"][profile_id] = profile
    return payload


def _valid_lifecycle(*, origin: str = "imported") -> dict:
    lifecycle = {
        "schema": "tldw.tool-pack-lifecycle/v1",
        "origin": origin,
        "pack_digest": "a" * 64,
        "imported_at": "2026-08-31T00:00:00Z",
        "first_bind_confirmation_required": origin == "imported",
        "receipt_id": "tp-" + "b" * 32,
        "receipt_digest": "c" * 64,
        "policy_digest": "d" * 64,
        "revision": 1,
    }
    if origin == "imported":
        lifecycle["counts"] = {"matched": 0, "omitted": 0, "pending_deny": 0}
    else:
        lifecycle["removed_at"] = "2026-08-31T01:00:00Z"
    return lifecycle


def _tombstone_payload(*, default_global: str) -> dict:
    payload = _payload(global_default=default_global)
    payload["profiles"]["portable"] = {
        "profile_kind": "tool_pack_tombstone",
        "tool_pack_lifecycle": _valid_lifecycle(origin="tombstone"),
        "servers": {},
    }
    return payload


def _builtin() -> GatedToolRef:
    return GatedToolRef(
        server_key=BUILTIN_TOOL_SERVER_KEY,
        name="calculator",
        description="Calculator",
        input_schema=None,
        tags=(),
    )


@pytest.mark.parametrize(
    ("profile", "origin"),
    [
        ({"profile_kind": "tool_pack_imported", "servers": {}}, "lifecycle_invalid"),
        (
            {
                "tool_pack_lifecycle": {"schema": "tldw.tool-pack-lifecycle/v1"},
                "servers": {},
            },
            "lifecycle_invalid",
        ),
        (
            {"profile_kind": "unknown", "tool_pack_lifecycle": {}, "servers": {}},
            "lifecycle_invalid",
        ),
    ],
)
def test_invalid_lifecycle_resolves_deny(profile, origin):
    """Removing the lifecycle authority check must fail closed, not inherit."""
    payload = _named_payload("portable", profile)

    assert resolve_effective_state(
        payload, _tool(), profile_id="portable"
    ) == EffectiveToolState("deny", origin)


def test_tombstone_short_circuits_named_inheritance():
    """A valid tombstone remains Deny even when default would allow."""
    payload = _tombstone_payload(default_global="allow")

    assert (
        resolve_builtin_state(payload, _builtin(), profile_id="portable").state
        == "deny"
    )


def test_lifecycle_disposition_requires_the_exact_tombstone_variant():
    """A tombstone must not gain imported counts or drop removal provenance."""
    profile = {
        "profile_kind": "tool_pack_tombstone",
        "tool_pack_lifecycle": _valid_lifecycle(origin="tombstone"),
        "servers": {},
    }

    assert profile_lifecycle_disposition(profile) == "tombstone"
    profile["tool_pack_lifecycle"]["counts"] = {
        "matched": 0,
        "omitted": 0,
        "pending_deny": 0,
    }
    assert profile_lifecycle_disposition(profile) == "invalid"


def test_imported_named_global_fallback_protects_an_unseen_server():
    profile = {
        "global_default": "ask",
        "servers": {BUILTIN_TOOL_SERVER_KEY: {"default": "deny"}},
        "profile_kind": "tool_pack_imported",
        "tool_pack_lifecycle": _valid_lifecycle(),
    }
    payload = _named_payload("portable", profile)
    unseen = _tool(server_key="future:server", name="future")

    assert resolve_effective_state(
        payload, unseen, profile_id="portable"
    ) == EffectiveToolState("ask", "global_default")
    assert resolve_effective_state_by_key(
        payload,
        unseen.server_key,
        unseen.name,
        profile_id="portable",
    ) == EffectiveToolState("ask", "global_default")


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
    assert definition_hash("desc", {"a": 1, "b": 2}) == definition_hash(
        "desc", {"b": 2, "a": 1}
    )


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
    assert (
        EffectiveToolState(state="banana", origin="global_default").ui_label == "Banana"
    )
    assert EffectiveToolState(state="", origin="global_default").ui_label == "Ask"


def test_ui_label_reads_unknown_not_off_for_a_gate_error_origin():
    """task-2870: `origin="gate_error"` is the synthesized fail-closed
    verdict -- the permission RESOLVER raised, not a configured Off
    (`MCPWorkbench._resolve_test_gate()`/`_effective_for_display()` pair it
    unconditionally with `state="deny"`). Mapping it to "Off" made every
    `ui_label` renderer (Permissions matrix State cells, Tools-mode State
    column) print a confident configuration claim about a state that could
    not be read -- the same lie PR #1385's round J removed from the
    inspector's permission block one surface at a time. Owning it HERE
    fixes every renderer at once; a genuine deny keeps "Off"."""
    assert EffectiveToolState(state="deny", origin="gate_error").ui_label == "Unknown"
    # Genuine denies -- any non-gate_error origin -- keep the honest "Off".
    assert EffectiveToolState(state="deny", origin="tool_override").ui_label == "Off"
    assert EffectiveToolState(state="deny", origin="global_default").ui_label == "Off"


def test_raw_shell_permission_coerces_allow_and_ask_to_ask() -> None:
    from tldw_chatbook.Agents.raw_shell_tool_provider import resolve_raw_shell_state

    assert (
        resolve_raw_shell_state(
            EffectiveToolState(state="allow", origin="tool_override")
        )
        == "ask"
    )
    assert (
        resolve_raw_shell_state(
            EffectiveToolState(state="ask", origin="global_default")
        )
        == "ask"
    )


def test_raw_shell_permission_keeps_internal_deny_for_user_visible_off() -> None:
    from tldw_chatbook.Agents.raw_shell_tool_provider import resolve_raw_shell_state

    state = EffectiveToolState(state="deny", origin="tool_override")

    assert resolve_raw_shell_state(state) == "deny"
    assert state.ui_label == "Off"


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
                "tools": {
                    tool.name: {"state": "allow", "definition_hash": current_hash}
                },
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
    payload = _payload(
        global_default="deny", servers={tool.server_key: {"default": "ask"}}
    )

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
    payload = _payload(
        servers={tool.server_key: {"tools": {tool.name: {"state": "ask"}}}}
    )

    result = resolve_effective_state(payload, tool)

    assert result.state == "ask"
    assert result.origin == "tool_override"
    assert result.config_changed is False


# -- resolve_effective_state: rug-pull hash guard ------------------------------


def test_resolve_effective_state_matching_hash_does_not_downgrade():
    tool = _tool()
    current_hash = definition_hash(tool.description, tool.input_schema)
    payload = _payload(
        servers={
            tool.server_key: {
                "tools": {
                    tool.name: {"state": "allow", "definition_hash": current_hash}
                }
            }
        }
    )

    result = resolve_effective_state(payload, tool)

    assert result.state == "allow"
    assert result.config_changed is False


def test_resolve_effective_state_hash_mismatch_downgrades_allow_to_ask():
    tool = _tool()
    payload = _payload(
        servers={
            tool.server_key: {
                "tools": {
                    tool.name: {"state": "allow", "definition_hash": "stale-hash"}
                }
            }
        }
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
    payload = _payload(
        servers={
            tool.server_key: {
                "tools": {
                    tool.name: {"state": "allow", "definition_hash": current_hash}
                }
            }
        }
    )

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
        servers={
            "local:demo": {
                "tools": {"search": {"state": "allow", "definition_hash": "whatever"}}
            }
        }
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


# -- BY_KEY_HASH_FREE_SERVER_KEYS exemption at the by-key collapse ------------
#
# `resolve_effective_state()` (the live-`HubTool` resolver) never downgrades
# an explicit `allow` for a server in `HASH_FREE_SERVER_KEYS` -- those keys
# are in-process code (the agent-runtime built-in tools, and the built-in MCP
# server) that changes only via an app update, so the hash-staleness guard
# protects nothing for them and would only produce rug-pull false positives.
# `resolve_effective_state_by_key()`'s unconditional "any allow collapses to
# ask" final step used to ignore that exemption entirely, so the SAME tool
# set to Allow recorded `decision="approved"` (asked-and-approved) when
# resolved by key (the Advanced pane's `tool.execute` route) and
# `decision="allowed"` (no ask needed) when resolved with a live `HubTool`
# (the Test Tool panel) -- a cross-surface split in the audit trail this
# programme exists to close.
#
# Fix Round C, Item 2: the exemption at the by-key seam is narrower than
# `HASH_FREE_SERVER_KEYS` -- `BY_KEY_HASH_FREE_SERVER_KEYS` covers only
# `"builtin:tldw_chatbook"`, NOT `BUILTIN_TOOL_SERVER_KEY`
# (`"agent:builtin"`). Unlike `resolve_effective_state()`, this function has
# no `HubTool.tags` to floor a high-risk inherited allow with, so exempting
# a key here means "an inherited/explicit allow for this key returns with
# NO floor of any kind." That is safe for `builtin:tldw_chatbook` only
# because its `HubTool`s always carry `tags=()` (see
# `test_hub_tool_catalog.py`'s tripwire) -- it would NOT be safe for
# `agent:builtin`, whose real resolver (`resolve_builtin_state`) floors
# `BUILTIN_HIGH_RISK_TAGS`. The three fixtures below pin the narrowed
# exemption directly (no loop over the wider `HASH_FREE_SERVER_KEYS`, which
# would incorrectly pin "allow survives" for `agent:builtin` too), and the
# two control cases pin that the exemption stays narrow at both edges: a
# ordinary key never gets it, and the wider-but-excluded hash-free key
# (`agent:builtin`) does not either, even though it's exempt from the
# hash-staleness guard in `resolve_effective_state()`.


def test_resolve_by_key_hash_free_server_explicit_allow_is_not_downgraded():
    payload = _payload(
        servers={"builtin:tldw_chatbook": {"tools": {"calculator": {"state": "allow"}}}}
    )

    result = resolve_effective_state_by_key(
        payload, "builtin:tldw_chatbook", "calculator"
    )

    assert result.state == "allow"
    assert result.origin == "tool_override"
    assert result.config_changed is False


def test_resolve_by_key_hash_free_server_inherited_allow_is_not_downgraded():
    payload = _payload(global_default="allow", servers={"builtin:tldw_chatbook": {}})

    result = resolve_effective_state_by_key(
        payload, "builtin:tldw_chatbook", "calculator"
    )

    assert result.state == "allow"
    assert result.origin == "global_default"
    assert result.config_changed is False


def test_resolve_by_key_hash_free_server_default_allow_is_not_downgraded():
    payload = _payload(servers={"builtin:tldw_chatbook": {"default": "allow"}})

    result = resolve_effective_state_by_key(
        payload, "builtin:tldw_chatbook", "calculator"
    )

    assert result.state == "allow"
    assert result.origin == "server_default"
    assert result.config_changed is False


def test_resolve_by_key_non_hash_free_server_allow_still_downgrades():
    """Control: a server key NOT in ``BY_KEY_HASH_FREE_SERVER_KEYS`` keeps
    the existing "can't verify without a live tool, so ask" collapse -- the
    exemption above must be narrow, scoped to the pinned in-process key,
    not a general relaxation of the by-key rug-pull-safety collapse."""
    payload = _payload(
        servers={"local:demo": {"tools": {"search": {"state": "allow"}}}}
    )

    result = resolve_effective_state_by_key(payload, "local:demo", "search")

    assert result.state == "ask"
    assert result.origin == "tool_override"
    assert result.config_changed is True


def test_resolve_by_key_agent_builtin_allow_still_downgrades():
    """Control: `BUILTIN_TOOL_SERVER_KEY` (`"agent:builtin"`) IS in the
    wider `HASH_FREE_SERVER_KEYS` (exempt from the hash-staleness guard in
    `resolve_effective_state()`) but must NOT be in the narrower
    `BY_KEY_HASH_FREE_SERVER_KEYS` this resolver checks -- this function has
    no tags to floor an inherited allow with, and the real resolver for
    this key (`resolve_builtin_state`) DOES floor high-risk tags, so
    exempting it here would silently bypass that floor if this seam were
    ever reached for an `agent:builtin` key. Confirms both that
    `agent:builtin` keeps the "ask" collapse here, and that it is genuinely
    absent from the narrower set (not just coincidentally behaving the
    same)."""
    assert BUILTIN_TOOL_SERVER_KEY in HASH_FREE_SERVER_KEYS
    assert BUILTIN_TOOL_SERVER_KEY not in BY_KEY_HASH_FREE_SERVER_KEYS

    payload = _payload(
        servers={BUILTIN_TOOL_SERVER_KEY: {"tools": {"calculator": {"state": "allow"}}}}
    )

    result = resolve_effective_state_by_key(
        payload, BUILTIN_TOOL_SERVER_KEY, "calculator"
    )

    assert result.state == "ask"
    assert result.origin == "tool_override"
    assert result.config_changed is True


def test_resolve_by_key_agent_builtin_server_default_allow_still_downgrades():
    """Fix Round E, Item 3: restores coverage Round C's narrowing dropped.
    The control test above only pins `agent:builtin` at the
    ``tool_override`` origin; this pins the INHERITED ``server_default``
    origin -- exactly the case this function's own docstring argues about
    ("this function has no tags to floor an inherited allow with", and
    unlike ``builtin:tldw_chatbook``, `agent:builtin`'s real resolver
    (``resolve_builtin_state``) DOES floor high-risk tags). This asserts
    the by-key path's own strict behaviour (it downgrades to ``ask`` here);
    it does NOT assert that behaviour matches ``resolve_builtin_state`` --
    the two resolvers diverge on this exact shape (an untagged inherited
    allow), with the by-key path being the stricter of the two."""
    payload = _payload(servers={BUILTIN_TOOL_SERVER_KEY: {"default": "allow"}})

    result = resolve_effective_state_by_key(
        payload, BUILTIN_TOOL_SERVER_KEY, "calculator"
    )

    assert result.state == "ask"
    assert result.origin == "server_default"
    assert result.config_changed is True


def test_resolve_by_key_agent_builtin_global_default_allow_still_downgrades():
    """Fix Round E, Item 3: the other inherited origin the narrowing
    dropped coverage for -- ``global_default``. Same reasoning as the
    ``server_default`` sibling above: this is the by-key path's own strict
    behaviour, not a claim that it matches ``resolve_builtin_state``."""
    payload = _payload(global_default="allow", servers={BUILTIN_TOOL_SERVER_KEY: {}})

    result = resolve_effective_state_by_key(
        payload, BUILTIN_TOOL_SERVER_KEY, "calculator"
    )

    assert result.state == "ask"
    assert result.origin == "global_default"
    assert result.config_changed is True


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
    payload = _payload(
        servers={tool.server_key: {"tools": {tool.name: "not-a-mapping"}}}
    )

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
