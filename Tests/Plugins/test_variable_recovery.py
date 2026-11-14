"""Recovery control-flow tests; native persistence is covered by variable_case.

The credential owner, mapping validation and recovery body are real. Persistence
and retained-package inspection are boundary doubles, not native qualification.
"""

import hashlib
import json
from types import SimpleNamespace

import pytest

from Tests.MCP.test_credential_bindings import MemoryCredentialBackend

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


@pytest.fixture
def recovery_case(tmp_path, monkeypatch):
    from tldw_chatbook.MCP.credential_bindings import CredentialBindingService
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalMCPStore
    from tldw_chatbook.Plugins import recovery
    from tldw_chatbook.Plugins.authority import PluginMarker
    from tldw_chatbook.Plugins.coordinator import PluginCoordinator
    from tldw_chatbook.Plugins.models import ComponentRecord, PackageInspection
    from tldw_chatbook.Plugins.variables import capture_mapping, handle

    backend = MemoryCredentialBackend()
    protected = CredentialBindingService(backend)
    owner = LocalMCPControlService(
        store=LocalMCPStore(tmp_path / "mcp.json"), credential_service=protected
    )
    component = ComponentRecord(
        component_id="hook:guard",
        kind="hook",
        local_id="guard",
        path="hooks.json",
        definition_json=json.dumps({"type": "command", "env": {"TOKEN": "${TOKEN}"}}),
    )
    inspection = PackageInspection(
        source_identity="fixture",
        effective_digest="a" * 64,
        inventory={"hook:guard": component},
        variables_json=json.dumps({"TOKEN": {"type": "string", "secret": True}}),
    )
    coordinator = SimpleNamespace(mcp_mapping_owner=owner, _published=None)
    coordinator.validate_mapping = lambda *args: PluginCoordinator.validate_mapping(
        coordinator, *args
    )
    metadata = protected.create_variable(
        installation_id="installed",
        name="TOKEN",
        value_type="string",
        secret=True,
        value="private sentinel",
    )
    mapping = capture_mapping(
        coordinator, "installed", inspection, "hook:guard", "TOKEN", handle(metadata)
    )
    # Compatibility with the value-free metadata actually issued by the owner,
    # independently of the retained mapping builder used by recovery.
    from tldw_chatbook.Plugins.package_files import canonical_json

    assert (
        mapping["configuration_digest"]
        == hashlib.sha256(canonical_json(metadata).encode()).hexdigest()
    )
    target = {"mappings": [mapping], "operation_result": None}
    marker = PluginMarker(
        generation=0, operation_id="bootstrap", recovery_snapshot_digest="b" * 64
    )
    coordinator.authority = SimpleNamespace(
        load_marker=lambda: marker,
        verify_current=lambda: target,
        list_transitions=lambda **_: [],
        ensure_issued_metadata=lambda _: None,
    )
    coordinator.registry = SimpleNamespace(
        list_operations=lambda **_: [],
        authority_projection=lambda **_: target,
    )
    coordinator.root_usage = SimpleNamespace(initialize=lambda *_, **__: None)
    monkeypatch.setattr(
        recovery,
        "retained_inspections",
        lambda *_, **__: {("installed", inspection.effective_digest): inspection},
    )
    return SimpleNamespace(
        coordinator=coordinator,
        protected=protected,
        backend=backend,
        metadata=metadata,
        mapping=mapping,
        inspection=inspection,
        target=target,
    )


@pytest.mark.parametrize("change", ["active", "revoked", "missing"])
def test_recovery_retains_binding_without_reopening_unavailable_value(
    recovery_case, change
):
    from tldw_chatbook.MCP.credential_bindings import CredentialError
    from tldw_chatbook.Plugins.recovery import recover_coordinator
    from tldw_chatbook.Plugins.variables import capture_mapping, handle, values

    case = recovery_case
    if change == "revoked":
        case.protected.revoke_variable(
            case.metadata["reference_id"], installation_id="installed"
        )
    elif change == "missing":
        del case.backend.records[case.metadata["reference_id"]]
    receipts = recover_coordinator(case.coordinator)
    assert not any(row.phase == "recovery_required" for row in receipts)
    assert case.coordinator._published == case.target
    if change == "active":
        assert values(
            case.coordinator,
            "installed",
            case.inspection,
            "hook:guard",
            [case.mapping],
            resolve=True,
        ) == {"TOKEN": "private sentinel"}
    else:
        with pytest.raises(CredentialError):
            capture_mapping(
                case.coordinator,
                "installed",
                case.inspection,
                "hook:guard",
                "TOKEN",
                handle(case.metadata),
            )
        with pytest.raises(CredentialError):
            case.coordinator.validate_mapping(
                case.mapping, case.inspection, [case.mapping]
            )
        with pytest.raises(CredentialError):
            values(
                case.coordinator,
                "installed",
                case.inspection,
                "hook:guard",
                [case.mapping],
                resolve=True,
            )


@pytest.mark.parametrize(
    "field,value",
    [
        ("definition_digest", "c" * 64),
        ("configuration_digest", "c" * 64),
        ("target_reference", "var1." + "d" * 32 + ".1"),
        ("kind", "tool"),
        ("credential_bindings", [{"reference_id": "foreign"}]),
    ],
)
def test_recovery_refuses_changed_retained_variable_binding(
    recovery_case, field, value
):
    from tldw_chatbook.Plugins.recovery import recover_coordinator

    case = recovery_case
    case.mapping[field] = value
    receipts = recover_coordinator(case.coordinator)
    assert receipts[0].phase == "recovery_required"
    assert case.coordinator._published is None


@pytest.mark.parametrize(
    "change", [{"type": "integer", "secret": True}, {"type": "string", "secret": False}]
)
def test_recovery_refuses_changed_variable_declaration(recovery_case, change):
    from tldw_chatbook.Plugins.variables import _validate_retained_mapping

    case = recovery_case
    changed = case.inspection.model_copy(
        update={"variables_json": json.dumps({"TOKEN": change})}
    )
    with pytest.raises(ValueError, match="plugin_variable_mapping_changed"):
        _validate_retained_mapping(case.mapping, changed)
