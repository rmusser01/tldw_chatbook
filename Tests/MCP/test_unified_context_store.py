from __future__ import annotations

from tldw_chatbook.MCP.unified_context_store import UnifiedMCPContextStore
from tldw_chatbook.MCP.unified_control_models import ServerAccessContext, UnifiedMCPContext


def test_unified_mcp_context_partitions_per_server_state(tmp_path):
    store = UnifiedMCPContextStore(tmp_path / "unified_mcp_context.json")
    context = UnifiedMCPContext(
        selected_source="server",
        selected_active_server_id="server-a",
        selected_scope="personal",
        selected_section="overview",
        per_server_state={
            "server-a": ServerAccessContext(
                server_id="server-a",
                selected_scope="personal",
                selected_section="inventory",
            ),
            "server-b": ServerAccessContext(
                server_id="server-b",
                selected_scope="team",
                selected_section="catalogs",
            ),
        },
    )

    store.save(context)
    restored = store.load()

    assert restored.selected_source == "server"
    assert restored.selected_active_server_id == "server-a"
    assert restored.per_server_state["server-a"].selected_section == "inventory"
    assert restored.per_server_state["server-b"].selected_section == "catalogs"


def test_unified_mcp_context_store_loads_safe_default_on_invalid_json(tmp_path):
    path = tmp_path / "unified_mcp_context.json"
    path.write_text("{not-json", encoding="utf-8")

    restored = UnifiedMCPContextStore(path).load()

    assert restored == UnifiedMCPContext()


def test_unified_mcp_context_store_keeps_destination_local_state_separate_from_runtime_policy(tmp_path):
    store = UnifiedMCPContextStore(tmp_path / "unified_mcp_context.json")
    context = UnifiedMCPContext(
        selected_source="server",
        selected_active_server_id="server-a",
        selected_scope="team",
        selected_section="governance",
        per_server_state={
            "server-a": ServerAccessContext(
                server_id="server-a",
                selected_scope="team",
                selected_section="governance",
            )
        },
    )

    store.save(context)
    raw_payload = (tmp_path / "unified_mcp_context.json").read_text(encoding="utf-8")

    assert "runtime_policy_snapshot" not in raw_payload
    assert "server_reachability" not in raw_payload
