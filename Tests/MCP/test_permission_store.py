"""Tests for the schema-versioned MCP permission store (Phase 4, Task 1).

Covers: fresh-default payload shape, corrupt/unknown-version backup-and-reset
policy, kill switch + global default round-trips, server/tool state
set/inherit-prune semantics, the allow-requires-hash guard, the
config_changed clearing/marking contract, and the atomic-write pattern.
"""

from __future__ import annotations

import json

import pytest

from tldw_chatbook.MCP.permission_store import (
    DEFAULT_GLOBAL,
    SCHEMA_VERSION,
    STORE_STATES,
    MCPPermissionStore,
)


def _fresh_payload_shape() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kill_switch": False,
        "profiles": {
            "default": {
                "global_default": DEFAULT_GLOBAL,
                "servers": {},
            }
        },
    }


def test_load_returns_fresh_default_payload_when_file_missing(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")

    payload = store.load()

    assert payload["schema_version"] == 1
    assert payload["kill_switch"] is False
    assert payload["profiles"]["default"]["global_default"] == "ask"
    assert payload["profiles"]["default"]["servers"] == {}
    assert not (tmp_path / "mcp_permissions.json").exists()


def test_load_backs_up_corrupt_json_and_returns_fresh_default(tmp_path):
    path = tmp_path / "mcp_permissions.json"
    path.write_text("{not valid json", encoding="utf-8")

    store = MCPPermissionStore(path)
    payload = store.load()

    assert payload == _fresh_payload_shape()
    backup_path = tmp_path / "mcp_permissions.json.bak"
    assert backup_path.exists()
    assert backup_path.read_text(encoding="utf-8") == "{not valid json"
    assert not path.exists()


def test_load_backs_up_non_dict_json_and_returns_fresh_default(tmp_path):
    path = tmp_path / "mcp_permissions.json"
    path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")

    store = MCPPermissionStore(path)
    payload = store.load()

    assert payload == _fresh_payload_shape()
    assert (tmp_path / "mcp_permissions.json.bak").exists()
    assert not path.exists()


def test_load_backs_up_unknown_schema_version_and_returns_fresh_default(tmp_path):
    path = tmp_path / "mcp_permissions.json"
    path.write_text(
        json.dumps({"schema_version": 2, "kill_switch": True}), encoding="utf-8"
    )

    store = MCPPermissionStore(path)
    payload = store.load()

    assert payload == _fresh_payload_shape()
    backup_path = tmp_path / "mcp_permissions.json.bak"
    assert backup_path.exists()
    assert json.loads(backup_path.read_text(encoding="utf-8"))["schema_version"] == 2
    assert not path.exists()


def test_load_backup_replaces_prior_bak_file(tmp_path):
    path = tmp_path / "mcp_permissions.json"
    backup_path = tmp_path / "mcp_permissions.json.bak"
    backup_path.write_text("stale backup contents", encoding="utf-8")
    path.write_text("still not json", encoding="utf-8")

    store = MCPPermissionStore(path)
    store.load()

    assert backup_path.read_text(encoding="utf-8") == "still not json"


def test_kill_switch_round_trip(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")

    assert store.get_kill_switch() is False

    store.set_kill_switch(True)

    assert store.get_kill_switch() is True
    on_disk = json.loads(
        (tmp_path / "mcp_permissions.json").read_text(encoding="utf-8")
    )
    assert on_disk["kill_switch"] is True


def test_global_default_validates_and_round_trips(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")

    assert store.get_global_default() == "ask"

    store.set_global_default("deny")

    assert store.get_global_default() == "deny"

    with pytest.raises(ValueError):
        store.set_global_default("nonsense")


def test_set_server_default_and_inherit_prunes_entry(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    server_key = "local:demo-server"

    assert store.get_server_entry(server_key) is None

    store.set_server_default(server_key, "allow")

    entry = store.get_server_entry(server_key)
    assert entry is not None
    assert entry["default"] == "allow"

    store.set_server_default(server_key, None)

    assert store.get_server_entry(server_key) is None


def test_set_tool_state_and_inherit_prunes_entry(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    server_key = "local:demo-server"

    assert store.get_tool_entry(server_key, "search") is None

    store.set_tool_state(server_key, "search", "ask")

    tool_entry = store.get_tool_entry(server_key, "search")
    assert tool_entry is not None
    assert tool_entry["state"] == "ask"

    store.set_tool_state(server_key, "search", None)

    assert store.get_tool_entry(server_key, "search") is None
    # Pruning the sole tool must also prune the now-empty server entry.
    assert store.get_server_entry(server_key) is None


def test_set_tool_state_inherit_prunes_tool_but_keeps_server_default(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    server_key = "local:demo-server"

    store.set_server_default(server_key, "ask")
    store.set_tool_state(server_key, "search", "allow", definition_hash="hash-1")

    store.set_tool_state(server_key, "search", None)

    assert store.get_tool_entry(server_key, "search") is None
    entry = store.get_server_entry(server_key)
    assert entry is not None
    assert entry["default"] == "ask"


def test_set_tool_state_allow_without_hash_raises_value_error(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")

    with pytest.raises(ValueError):
        store.set_tool_state("local:demo-server", "search", "allow")


def test_set_tool_state_allow_stores_hash_and_clears_config_changed(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    server_key = "local:demo-server"

    store.set_tool_state(server_key, "search", "ask")
    assert store.mark_config_changed(server_key, "search") is True
    tool_entry = store.get_tool_entry(server_key, "search")
    assert tool_entry.get("config_changed") is True

    store.set_tool_state(server_key, "search", "allow", definition_hash="abc123")

    tool_entry = store.get_tool_entry(server_key, "search")
    assert tool_entry["state"] == "allow"
    assert tool_entry["definition_hash"] == "abc123"
    assert "config_changed" not in tool_entry


def test_mark_config_changed_returns_true_then_false(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    server_key = "local:demo-server"
    store.set_tool_state(server_key, "search", "allow", definition_hash="abc123")

    first = store.mark_config_changed(server_key, "search")
    second = store.mark_config_changed(server_key, "search")

    assert first is True
    assert second is False


def test_mark_config_changed_second_call_does_not_rewrite_file(tmp_path):
    """Perf: the control plane calls `mark_config_changed()` on every
    resolution pass for every tool that already carries the marker (any
    ⚠ tool in the catalog) -- the second-and-later calls must be a no-op
    read, not a load()+save() that rewrites the store file every time."""
    path = tmp_path / "mcp_permissions.json"
    store = MCPPermissionStore(path)
    server_key = "local:demo-server"
    store.set_tool_state(server_key, "search", "allow", definition_hash="abc123")

    assert store.mark_config_changed(server_key, "search") is True
    before_bytes = path.read_bytes()
    before_mtime_ns = path.stat().st_mtime_ns

    assert store.mark_config_changed(server_key, "search") is False

    assert path.read_bytes() == before_bytes
    assert path.stat().st_mtime_ns == before_mtime_ns


def test_save_atomic_write_leaves_no_tmp_behind(tmp_path):
    path = tmp_path / "mcp_permissions.json"
    store = MCPPermissionStore(path)

    store.set_kill_switch(True)

    assert path.exists()
    assert not (tmp_path / "mcp_permissions.json.tmp").exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert "updated_at" in on_disk
    assert on_disk["schema_version"] == 1


def test_store_states_and_default_global_constants():
    assert STORE_STATES == ("allow", "ask", "deny")
    assert DEFAULT_GLOBAL == "ask"
    assert SCHEMA_VERSION == 1


# -- Task 2: GatedToolRef + resolve_builtin_state -----------------------------

from tldw_chatbook.MCP.permission_store import (
    BUILTIN_TOOL_SERVER_KEY,
    GatedToolRef,
    HIGH_RISK_TAGS,
    resolve_builtin_state,
)

BUILTIN_KEY = "agent:builtin"


def _ref(name="calculator", tags=()):
    return GatedToolRef(
        server_key=BUILTIN_KEY,
        name=name,
        description="d",
        input_schema={"type": "object"},
        tags=tuple(tags),
    )


def _payload(*, global_default=None, server_default=None, tool_state=None):
    server: dict = {}
    if server_default is not None:
        server["default"] = server_default
    if tool_state is not None:
        server["tools"] = {"calculator": {"state": tool_state}}
    profile: dict = {"servers": {BUILTIN_KEY: server} if server else {}}
    if global_default is not None:
        profile["global_default"] = global_default
    return {"profiles": {"default": profile}}


def test_builtin_floor_is_allow_not_the_mcp_global_default():
    # MCP's global default is "ask"; built-ins must NOT inherit it, or
    # calculator would prompt on every use.
    eff = resolve_builtin_state(_payload(global_default="ask"), _ref())
    assert eff.state == "allow"
    assert eff.risk_floored is False


def test_empty_payload_resolves_to_allow_floor():
    eff = resolve_builtin_state({}, _ref())
    assert eff.state == "allow"


def test_high_risk_tag_floors_inherited_allow_to_ask():
    eff = resolve_builtin_state({}, _ref(tags=("mutates",)))
    assert eff.state == "ask"
    assert eff.risk_floored is True


def test_explicit_tool_override_allow_is_not_floored():
    eff = resolve_builtin_state(
        _payload(tool_state="allow"), _ref(tags=("mutates",))
    )
    assert eff.state == "allow"
    assert eff.origin == "tool_override"
    assert eff.risk_floored is False


def test_server_default_beats_the_builtin_floor():
    eff = resolve_builtin_state(_payload(server_default="deny"), _ref())
    assert eff.state == "deny"
    assert eff.origin == "server_default"


def test_no_hash_comparison_for_builtins():
    # An allow override with a STALE/absent definition_hash must stay
    # allow -- the rug-pull guard is deliberately not applied to
    # in-process code (it would re-prompt on every app upgrade).
    payload = _payload(tool_state="allow")
    entry = payload["profiles"]["default"]["servers"][BUILTIN_KEY]["tools"]
    entry["calculator"]["definition_hash"] = "stale-and-wrong"
    eff = resolve_builtin_state(payload, _ref())
    assert eff.state == "allow"
    assert eff.config_changed is False


def test_builtin_and_mcp_builtin_server_namespaces_are_disjoint():
    """A decision for the built-in MCP SERVER must not govern the
    agent-runtime tool of the same name, or vice versa."""
    from tldw_chatbook.MCP.readiness import BUILTIN_SERVER_KEY

    assert BUILTIN_SERVER_KEY != BUILTIN_TOOL_SERVER_KEY
    # A deny recorded against the MCP built-in server leaves the
    # agent-runtime tool of the same name on its own floor.
    payload = {
        "profiles": {
            "default": {
                "servers": {
                    BUILTIN_SERVER_KEY: {"tools": {"calculator": {"state": "deny"}}}
                }
            }
        }
    }
    assert resolve_builtin_state(payload, _ref("calculator")).state == "allow"


def test_mcp_resolver_is_unaffected_by_the_builtin_floor():
    # Guard the "do not modify resolve_effective_state" constraint: an
    # MCP tool with no entries still inherits the MCP global default.
    from tldw_chatbook.MCP.permission_store import resolve_effective_state
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    tool = HubTool(
        server_key="local:x", server_label="x", source="local",
        name="t", description="d", input_schema=None, tags=(),
        stale=False, executable=True,
    )
    assert resolve_effective_state({}, tool).state == "ask"


# -- Task 1: HASH_FREE_SERVER_KEYS exemption ----------------------------------

from tldw_chatbook.MCP.permission_store import HASH_FREE_SERVER_KEYS


def test_hash_free_keys_contains_exactly_the_builtin_namespace():
    """Pin the CONTENTS. Adding a remote namespace here would silently
    disable the rug-pull guard for it -- the one way this change could
    become a real weakening.

    RAG-48 part 2: widened to also cover ``builtin:tldw_chatbook`` (the
    built-in MCP *server* namespace, see
    ``test_builtin_and_mcp_builtin_server_namespaces_are_disjoint`` above)
    -- it is, like ``agent:builtin``, in-process code shipped with the app,
    not a remote server, so the same "adding a remote namespace would be
    the real weakening" guarantee still holds with two members."""
    from tldw_chatbook.MCP.readiness import BUILTIN_SERVER_KEY

    assert HASH_FREE_SERVER_KEYS == frozenset({BUILTIN_TOOL_SERVER_KEY, BUILTIN_SERVER_KEY})


def test_builtin_server_is_hash_free():
    from tldw_chatbook.MCP.readiness import BUILTIN_SERVER_KEY

    assert BUILTIN_SERVER_KEY in HASH_FREE_SERVER_KEYS
    assert "builtin:tldw_chatbook" in HASH_FREE_SERVER_KEYS


def test_builtin_allow_survives_schema_change(tmp_path):
    """A schema arriving for a builtin MCP-hub tool that already has a
    stored 'allow' (granted back when input_schema was always None, per
    RAG-48 part 1) must not rug-pull-downgrade it to 'ask' -- that is the
    entire point of the builtin server namespace being hash-free. Mirrors
    test_resolve_effective_state_hash_mismatch_downgrades_allow_to_ask's
    (Tests/MCP/test_permission_resolution.py) arrange/act, flipping the
    expectation, but drives it through the real store (not a hand-built
    payload) to exercise the actual write path too.
    """
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool
    from tldw_chatbook.MCP.permission_store import resolve_effective_state

    server_key = "builtin:tldw_chatbook"
    tool_name = "chat_with_llm"
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    # Hash-free namespace: no definition_hash kwarg needed or stored.
    store.set_tool_state(server_key, tool_name, "allow")
    entry = store.get_tool_entry(server_key, tool_name)
    assert entry["state"] == "allow"
    assert not entry.get("definition_hash")

    def _tool(input_schema):
        return HubTool(
            server_key=server_key,
            server_label="tldw_chatbook",
            source="builtin",
            name=tool_name,
            description="Chat.",
            input_schema=input_schema,
            tags=(),
            stale=False,
            executable=True,
        )

    payload = store.load()

    before = resolve_effective_state(payload, _tool(None))
    assert before.state == "allow"
    assert before.config_changed is False

    # The schema now arrives (RAG-48 part 2's catalog flip) -- resolving
    # the SAME stored allow against a tool that now carries a real
    # inputSchema must not downgrade it.
    after = resolve_effective_state(
        payload,
        _tool({"type": "object", "properties": {"message": {"type": "string"}}}),
    )
    assert after.state == "allow"
    assert after.origin == "tool_override"
    assert after.config_changed is False


def test_builtin_allow_with_legacy_stored_hash_is_not_rug_pulled():
    """Real on-disk data: entries written under the OLD code (before this
    task shipped) carry a REAL definition_hash(description, None) -- prior
    to RAG-48 part 1, builtin tools' input_schema was always None, so that
    is exactly what unified_control_plane_service.py's set_tool_state()
    hashed and stored for every builtin "allow" a user had already granted.

    The hash-free skip in resolve_effective_state() must key off
    tool.server_key (HASH_FREE_SERVER_KEYS membership) -- NOT off whether
    the stored entry happens to carry a hash. An implementation that
    instead skipped the comparison only when `tool_entry.get(
    "definition_hash")` was falsy would pass
    test_builtin_allow_survives_schema_change (which arranges its stored
    entry via the new hash-free set_tool_state(), so definition_hash is
    always None there) while still rug-pulling every REAL upgraded user,
    whose stored entry carries a real legacy digest -- exactly the failure
    the ORDER constraint in the task brief exists to prevent.

    Arrange the payload directly (bypassing set_tool_state entirely) to
    reproduce the actual legacy on-disk shape, not the shape the new
    hash-free write path produces.
    """
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool
    from tldw_chatbook.MCP.permission_store import (
        definition_hash,
        resolve_effective_state,
    )

    server_key = "builtin:tldw_chatbook"
    tool_name = "chat_with_llm"
    description = "Chat."
    legacy_hash = definition_hash(description, None)  # what the OLD code stored
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kill_switch": False,
        "profiles": {
            "default": {
                "global_default": DEFAULT_GLOBAL,
                "servers": {
                    server_key: {
                        "tools": {
                            tool_name: {
                                "state": "allow",
                                "definition_hash": legacy_hash,
                            }
                        }
                    }
                },
            }
        },
    }
    tool = HubTool(
        server_key=server_key,
        server_label="tldw_chatbook",
        source="builtin",
        name=tool_name,
        description=description,
        # The schema has now arrived (RAG-48 part 2's catalog flip) -- this
        # does NOT match the None the legacy hash was computed against.
        input_schema={"type": "object", "properties": {"message": {"type": "string"}}},
        tags=(),
        stale=False,
        executable=True,
    )

    result = resolve_effective_state(payload, tool)

    assert result.state == "allow"
    assert result.origin == "tool_override"
    assert result.config_changed is False


def test_builtin_allow_with_legacy_config_changed_marker_is_ignored():
    """A persisted config_changed=True marker (e.g. set by a pre-fix
    resolve pass, or a hand-edited store) must also be ignored for
    hash-free server keys -- resolve_builtin_state's precedent
    (agent:builtin) never reads any hash/config_changed field at all for
    its namespace; resolve_effective_state's hash-free skip must match
    that for builtin:tldw_chatbook too. Neither the stale-hash branch nor
    the marked-changed branch of the rug-pull guard may fire for this
    namespace."""
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool
    from tldw_chatbook.MCP.permission_store import resolve_effective_state

    server_key = "builtin:tldw_chatbook"
    tool_name = "chat_with_llm"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kill_switch": False,
        "profiles": {
            "default": {
                "global_default": DEFAULT_GLOBAL,
                "servers": {
                    server_key: {
                        "tools": {
                            tool_name: {
                                "state": "allow",
                                "definition_hash": "stale-and-wrong",
                                "config_changed": True,
                            }
                        }
                    }
                },
            }
        },
    }
    tool = HubTool(
        server_key=server_key,
        server_label="tldw_chatbook",
        source="builtin",
        name=tool_name,
        description="Chat.",
        input_schema={"type": "object"},
        tags=(),
        stale=False,
        executable=True,
    )

    result = resolve_effective_state(payload, tool)

    assert result.state == "allow"
    assert result.origin == "tool_override"
    assert result.config_changed is False


def test_allow_without_hash_is_permitted_for_the_builtin_namespace(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")

    store.set_tool_state(BUILTIN_TOOL_SERVER_KEY, "write_thing", "allow")

    entry = store.get_tool_entry(BUILTIN_TOOL_SERVER_KEY, "write_thing")
    assert entry["state"] == "allow"
    assert not entry.get("definition_hash")


def test_allow_without_hash_still_raises_for_an_mcp_server(tmp_path):
    """MCP's guard is unchanged."""
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")

    with pytest.raises(ValueError, match="definition_hash"):
        store.set_tool_state("local:docs", "some_tool", "allow")


def test_deny_and_clear_need_no_hash_for_either_namespace(tmp_path):
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")

    store.set_tool_state(BUILTIN_TOOL_SERVER_KEY, "write_thing", "deny")
    store.set_tool_state("local:docs", "some_tool", "deny")
    store.set_tool_state(BUILTIN_TOOL_SERVER_KEY, "write_thing", None)

    assert store.get_tool_entry(BUILTIN_TOOL_SERVER_KEY, "write_thing") is None


# -- P2 Task 1: built-in-only risk vocabulary ---------------------------------

from tldw_chatbook.MCP.permission_store import BUILTIN_HIGH_RISK_TAGS


def test_builtin_risk_set_is_a_strict_superset_of_the_mcp_set():
    """Built-ins floor on everything MCP does, plus reads and network."""
    assert HIGH_RISK_TAGS < BUILTIN_HIGH_RISK_TAGS
    assert "reads" in BUILTIN_HIGH_RISK_TAGS
    assert "reads" not in HIGH_RISK_TAGS
    assert "network" in BUILTIN_HIGH_RISK_TAGS
    assert "network" not in HIGH_RISK_TAGS


def test_mcp_high_risk_set_is_unchanged():
    """Pin the CONTENTS: widening this set would make remote MCP tools
    carrying the new tag start prompting, which P2 must not cause."""
    assert HIGH_RISK_TAGS == frozenset({"mutates", "process"})


def test_reads_tag_floors_an_inherited_builtin_allow_to_ask():
    eff = resolve_builtin_state({}, _ref(name="read_file", tags=("reads",)))
    assert eff.state == "ask"
    assert eff.risk_floored is True


def test_reads_tag_does_not_floor_an_mcp_tool():
    """The asymmetry is deliberate: only resolve_builtin_state learned the
    new tag. An MCP tool inheriting `allow` keeps it despite the tag."""
    from tldw_chatbook.MCP.permission_store import resolve_effective_state
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    tool = HubTool(
        server_key="local:x", server_label="x", source="local",
        name="t", description="d", input_schema=None, tags=("reads",),
        stale=False, executable=True,
    )
    eff = resolve_effective_state(_payload(global_default="allow"), tool)
    assert eff.state == "allow"
    assert eff.risk_floored is False


def test_mutates_still_floors_an_mcp_tool():
    """Negative control for the test above: MCP's own flooring still works,
    so a passing `test_reads_tag_does_not_floor_an_mcp_tool` proves the tag
    was not added to HIGH_RISK_TAGS -- not merely that flooring is broken."""
    from tldw_chatbook.MCP.permission_store import resolve_effective_state
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    tool = HubTool(
        server_key="local:x", server_label="x", source="local",
        name="t", description="d", input_schema=None, tags=("mutates",),
        stale=False, executable=True,
    )
    eff = resolve_effective_state(_payload(global_default="allow"), tool)
    assert eff.state == "ask"
    assert eff.risk_floored is True


def test_explicit_builtin_tool_override_beats_the_reads_floor():
    """An explicit user choice is still not floored -- same rule the
    `mutates` path already follows."""
    eff = resolve_builtin_state(
        _payload(tool_state="allow"), _ref(tags=("reads",))
    )
    assert eff.state == "allow"
    assert eff.origin == "tool_override"
    assert eff.risk_floored is False


# -- Task 5 (workspace-assistant-defaults): named permission profiles -----------

from tldw_chatbook.MCP.permission_store import resolve_effective_state_by_key

#: Server key for the named-profile resolver tests. Deliberately the one
#: ``BY_KEY_HASH_FREE_SERVER_KEYS`` member ("builtin:tldw_chatbook"):
#: ``resolve_effective_state_by_key`` collapses any other key's "allow" to
#: "ask" (no live HubTool to verify the grant against), which would make
#: the "allow survives in the default profile" assertion below test the
#: collapse instead of the cross-profile inheritance. It is also in
#: ``HASH_FREE_SERVER_KEYS``, so ``set_tool_state(..., "allow")`` needs no
#: definition_hash for it ("local:__local__" is in neither set, which is
#: why the brief directs the adaptation).
PROFILE_TEST_SERVER_KEY = "builtin:tldw_chatbook"


@pytest.fixture()
def store(tmp_path):
    """A fresh permission store on a per-test file, matching this file's
    construct-per-test tmp_path style."""
    return MCPPermissionStore(tmp_path / "mcp_permissions.json")


def test_named_profile_survives_load_and_normalizes(store, tmp_path):
    store.ensure_profile("ws-w-1")
    store.save({**store.load()})
    reloaded = MCPPermissionStore(tmp_path / "mcp_permissions.json").load()
    assert "ws-w-1" in reloaded["profiles"]
    # hand-edited named profile with null servers coerces on load
    payload = reloaded
    payload["profiles"]["ws-w-1"]["servers"] = None
    store.save(payload)
    assert isinstance(store.load()["profiles"]["ws-w-1"]["servers"], dict)


def test_mutators_write_only_the_named_profile(store):
    store.ensure_profile("ws-w-1")
    store.set_tool_state("local:__local__", "fs_write", "deny", profile_id="ws-w-1")
    payload = store.load()
    named = payload["profiles"]["ws-w-1"]["servers"]["local:__local__"]["tools"]["fs_write"]
    assert named["state"] == "deny"
    assert "local:__local__" not in payload["profiles"]["default"]["servers"]


def test_resolver_inherits_level_by_level(store):
    store.set_tool_state(PROFILE_TEST_SERVER_KEY, "fs_read", "allow", definition_hash=None)
    store.set_server_default(PROFILE_TEST_SERVER_KEY, "ask", profile_id="ws-w-1")
    payload = store.load()
    # named server default beats default-profile tool override (per-level chain)
    state = resolve_effective_state_by_key(
        payload, PROFILE_TEST_SERVER_KEY, "fs_read", profile_id="ws-w-1"
    )
    assert state.state == "ask"
    # key absent from named falls through to default-profile tool override
    state = resolve_effective_state_by_key(payload, PROFILE_TEST_SERVER_KEY, "fs_read")
    assert state.state == "allow"


def test_unknown_profile_id_inherits_everything(store):
    payload = store.load()
    state = resolve_effective_state_by_key(
        payload, "local:__local__", "fs_read", profile_id="ws-never-created"
    )
    assert state.state == DEFAULT_GLOBAL  # fresh workspace behaves like today
