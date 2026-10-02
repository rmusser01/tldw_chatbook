"""R62: real captured references and authenticated recovery, not publication UI."""

import json

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

from Tests.MCP.test_credential_bindings import MemoryCredentialBackend
from tldw_chatbook.MCP.credential_bindings import CredentialBindingService
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalMCPStore


@pytest.fixture
def mapped_stack(plugin_stack, native_package, tmp_path):
    from tldw_chatbook.Plugins.inspection import inspect_package

    stack = plugin_stack
    root = native_package()
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
                "mcpServers": {
                    "remote": {
                        "type": "streamable-http",
                        "url": "https://mcp.example/rpc",
                    }
                },
            }
        )
    )
    credentials = CredentialBindingService(MemoryCredentialBackend())
    binding = credentials.create_opaque(
        endpoint_origin="https://mcp.example",
        headers={"Authorization": "Bearer secret-sentinel"},
        method="bearer",
    )
    local = LocalMCPControlService(
        store=LocalMCPStore(tmp_path / "mcp.json"), credential_service=credentials
    )
    local.save_external_profile(
        {
            "profile_id": "remote",
            "transport": "streamable_http",
            "url": "https://mcp.example/rpc",
            "credential_reference": binding.reference_id,
            "credential_generation": binding.authority_generation,
        }
    )
    review = stack.call(
        lambda: stack.coordinator.review(
            inspect_package(root),
            selection=("mcp:remote", "skill:review"),
            workspace_id=None,
        )
    )
    stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    snapshot = stack.call(stack.authority.verify_current)
    from tldw_chatbook.Plugins.recovery import retained_inspections

    inspection = retained_inspections(snapshot)[
        (review.installation_id, review.inspection.effective_digest)
    ]
    return stack, local, credentials, review, inspection


def capture(case):
    _stack, local, _credentials, review, inspection = case
    return local.capture_connection_mapping(
        installation_id=review.installation_id,
        mapping_id="mcp-connection",
        inspection=inspection,
        component_id="mcp:remote",
        profile_id="remote",
    )


def test_capture_uses_saved_profile_and_exact_retained_component(mapped_stack):
    mapping = capture(mapped_stack)
    assert mapping["kind"] == "connection"
    assert mapping["target_reference"] == "local:remote"
    assert mapping["credential_bindings"][0]["identity_state"] == "opaque_reviewed"
    assert "secret-sentinel" not in json.dumps(mapping)
    mapped_stack[1].validate_connection_mapping(mapping, mapped_stack[4])


def publish_fixture_mapping(case, mapping):
    """Seed existing guarded/MAC protocol; M4 owns public reviewed publication."""
    from tldw_chatbook.Plugins.authority import PluginMarker, snapshot_digest

    stack, _local, _credentials, review, inspection = case

    def publish():
        old = stack.authority.load_marker()
        result = {
            "installation_id": review.installation_id,
            "kind": "trust",
            "revision_digest": inspection.effective_digest,
            "result": "committed",
        }
        operation = stack.authority.issue_operation_id(
            old.generation + 1, "a" * 64, result, "b" * 64
        )
        result["operation_id"] = operation
        with stack.registry.transaction() as cursor:
            cursor.execute(
                "INSERT INTO mappings VALUES (?, ?, ?)",
                (review.installation_id, mapping["mapping_id"], json.dumps(mapping)),
            )
            snapshot = stack.registry.authority_projection(operation_result=result)
            new = PluginMarker(
                generation=old.generation + 1,
                operation_id=operation,
                recovery_snapshot_digest=snapshot_digest(snapshot),
            )
            stack.authority.prepare(snapshot, old, new)
            stack.registry.write_operation(cursor, result, phase="committed")
        stack.authority.certify_commit(old, new)
        stack.authority.advance_marker(old, new)
        return snapshot

    return stack.call(publish)


@pytest.mark.parametrize(
    "change",
    [
        None,
        "missing",
        "replacement",
        "profile",
        "kind",
        "definition",
        "configuration",
        "reference",
        "binding_metadata",
    ],
)
def test_public_recovery_validates_complete_current_reference(mapped_stack, change):
    from Tests.Plugins.test_recovery import lose_registry

    stack, local, credentials, _review, _inspection = mapped_stack
    mapping = capture(mapped_stack)
    if change == "kind":
        mapping["kind"] = "tool"
    elif change == "definition":
        mapping["definition_digest"] = "0" * 64
    elif change == "configuration":
        mapping["configuration_digest"] = "0" * 64
    elif change == "reference":
        mapping["target_reference"] = "local:missing"
    elif change == "binding_metadata":
        mapping["credential_bindings"][0]["principal"] = "forged-claim"
    before = publish_fixture_mapping(mapped_stack, mapping)
    if change == "missing":
        credentials.backend.records.clear()
    elif change == "replacement":
        credentials.set_opaque(
            mapping["credential_bindings"][0]["reference_id"],
            endpoint_origin="https://mcp.example",
            headers={"Authorization": "Bearer secret-replaced"},
            method="bearer",
        )
    elif change == "profile":
        profile = local.store.get_profile("remote").to_input_dict()
        profile["url"] = "https://different.example/rpc"
        local.save_external_profile(profile)
    lose_registry(stack)
    stack.call(lambda: setattr(stack.coordinator, "mcp_mapping_owner", local))
    receipts = stack.call(stack.coordinator.recover)
    assert "secret-sentinel" not in repr(receipts)
    if change is None:
        assert any(receipt.committed for receipt in receipts)
        assert (
            stack.call(stack.coordinator.published_snapshot)["mappings"]
            == before["mappings"]
        )
    else:
        assert receipts[0].phase == "recovery_required"
        assert not stack.call(
            lambda: stack.registry.list_installations(limit=50, offset=0)
        )
        with pytest.raises(PermissionError):
            stack.call(stack.coordinator.published_snapshot)


def test_changed_live_binding_fences_even_matching_registry(mapped_stack):
    stack, local, credentials, _review, _inspection = mapped_stack
    publish_fixture_mapping(mapped_stack, capture(mapped_stack))
    stack.call(lambda: setattr(stack.coordinator, "mcp_mapping_owner", local))
    assert all(
        item.phase != "recovery_required"
        for item in stack.call(stack.coordinator.recover)
    )
    credentials.revoke(local.store.get_profile("remote").credential_reference)
    assert stack.call(stack.coordinator.recover)[0].phase == "recovery_required"


@pytest.mark.parametrize("account_change", [False, True])
def test_pending_review_survives_verified_renewal_only(
    mapped_stack, native_package, account_change
):
    import asyncio

    from tldw_chatbook.MCP.credential_bindings import HostCredential
    from tldw_chatbook.Plugins.inspection import inspect_package

    stack, local, credentials, _review, _inspection = mapped_stack
    state = {"principal": "first", "token": "secret-sentinel-original"}

    async def adapter(reference):
        return HostCredential(
            "bearer",
            "https://mcp.example",
            {"Authorization": "Bearer " + state["token"]},
            issuer="issuer",
            audience="mcp",
            principal=state["principal"],
            scopes=("read",),
        )

    credentials.adapters["verified-host"] = adapter
    binding = asyncio.run(credentials.create("verified-host"))
    profile = local.store.get_profile("remote").to_input_dict()
    profile.update(
        credential_reference=binding.reference_id,
        credential_generation=binding.authority_generation,
    )
    local.save_external_profile(profile)
    snapshot = publish_fixture_mapping(mapped_stack, capture(mapped_stack))
    stack.call(lambda: setattr(stack.coordinator, "mcp_mapping_owner", local))
    stack.call(stack.coordinator.recover)
    package = native_package()
    pending = stack.call(
        lambda: stack.coordinator.review(
            inspect_package(package), selection=("skill:review",), workspace_id=None
        )
    )
    state["token"] = "secret-sentinel-renewed"
    if account_change:
        state["principal"] = "second"
    asyncio.run(credentials.renew(binding.reference_id))
    if account_change:
        with pytest.raises(PermissionError):
            stack.call(lambda: stack.coordinator.commit(pending, pending.operation_id))
    else:
        receipt = stack.call(
            lambda: stack.coordinator.commit(pending, pending.operation_id)
        )
        assert receipt.committed
        assert (
            stack.call(stack.coordinator.published_snapshot)["mappings"]
            == snapshot["mappings"]
        )
        assert "secret-sentinel" not in pending.authority_json
        assert "secret-sentinel" not in repr(receipt)
