from __future__ import annotations

from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget


def test_update_target_status_preserves_existing_projection_on_invalid_status_update(tmp_path):
    store = ConfiguredServerTargetStore(tmp_path / "server_targets.json")
    target = ConfiguredServerTarget(
        server_id="server-a",
        label="Server A",
        base_url="https://server-a.example/api",
        auth_reference="legacy:tldw_api",
        last_known_server_label="Primary Projection",
        last_known_reachability="reachable",
        last_known_auth_state="authenticated",
    )
    store.save_targets([target])

    updated = store.update_target_status(
        "server-a",
        last_known_server_label="",
        last_known_reachability="BROKEN",
        last_known_auth_state="INVALID",
    )

    assert updated.last_known_server_label == "Primary Projection"
    assert updated.last_known_reachability == "reachable"
    assert updated.last_known_auth_state == "authenticated"

    restored = store.list_targets()[0]
    assert restored.last_known_server_label == "Primary Projection"
    assert restored.last_known_reachability == "reachable"
    assert restored.last_known_auth_state == "authenticated"
