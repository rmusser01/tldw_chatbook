"""Authenticated admission using the real owned coordinator and retained bytes."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Plugins.inspection import inspect_package


def install(stack, package):
    def operation():
        inspection = inspect_package(package)
        review = stack.coordinator.review(
            inspection, selection=tuple(inspection.inventory), workspace_id=None
        )
        return review

    review = stack.call(operation)
    stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    return review.installation_id


def trust(stack, installation):
    review = stack.call(lambda: stack.coordinator.review_trust(installation))
    return stack.call(lambda: stack.coordinator.commit(review, review.operation_id))


def activate(stack, installation, workspace, intent):
    review = stack.call(
        lambda: stack.coordinator.review_activation(
            installation, workspace_id=workspace, intent=intent
        )
    )
    return stack.call(lambda: stack.coordinator.commit(review, review.operation_id))


def admission(stack, workspaces=None):
    from tldw_chatbook.Plugins.admission import PluginAdmission

    workspaces = workspaces or {"workspace-a": SimpleNamespace(archived=False)}
    return stack.call(lambda: PluginAdmission(stack.coordinator, workspaces.get))


def test_install_needs_explicit_trust_and_scope_enable(plugin_stack, native_package):
    from tldw_chatbook.Plugins.admission import PluginUnavailable

    stack = plugin_stack
    identity = install(stack, native_package())
    gate = admission(stack)
    with pytest.raises(PluginUnavailable):
        stack.call(lambda: gate.capture(identity, "workspace-a", "root-run"))
    assert trust(stack, identity).committed
    with pytest.raises(PluginUnavailable):
        stack.call(lambda: gate.capture(identity, "workspace-a", "root-run"))
    assert activate(stack, identity, "workspace-a", "enabled").committed
    snapshot = stack.call(lambda: gate.capture(identity, "workspace-a", "root-run"))
    assert snapshot.run_id == "root-run"
    assert snapshot.workspace_id == "workspace-a"
    assert snapshot.selection == ("skill:review",)
    stack.call(lambda: gate.check(snapshot, "skill:review"))
    with pytest.raises(PluginUnavailable):
        stack.call(lambda: gate.capture(identity, None, "root-run"))


def test_current_scope_and_material_revalidation(plugin_stack, native_package):
    from tldw_chatbook.Plugins.admission import PluginUnavailable

    stack = plugin_stack
    identity = install(stack, native_package())
    trust(stack, identity)
    activate(stack, identity, None, "enabled")
    workspaces = {
        "workspace-a": SimpleNamespace(archived=False),
        "workspace-b": SimpleNamespace(archived=False),
    }
    gate = admission(stack, workspaces)
    snapshot = stack.call(lambda: gate.capture(identity, "workspace-a", "root-run"))
    activate(stack, identity, "workspace-b", "disabled")
    stack.call(lambda: gate.check(snapshot, "skill:review"))
    with pytest.raises(PluginUnavailable):
        stack.call(lambda: gate.capture(identity, "missing", "root-run"))
    workspaces["workspace-a"].archived = True
    with pytest.raises(PluginUnavailable):
        stack.call(lambda: gate.check(snapshot, "skill:review"))
    workspaces["workspace-a"].archived = False
    retained = stack.owner.root / "packages" / identity / "skills/review/SKILL.md"
    retained.chmod(0o600)
    retained.write_text(retained.read_text() + "\nTampered instructions")
    with pytest.raises(PluginUnavailable):
        stack.call(lambda: gate.check(snapshot, "skill:review"))


def test_missing_dependency_stays_blocked(plugin_stack, native_package):
    from tldw_chatbook.Plugins.admission import PluginUnavailable

    stack = plugin_stack
    identity = install(
        stack, native_package(requires={"skill:review": ["mcp:missing"]})
    )
    trust(stack, identity)
    activate(stack, identity, None, "enabled")
    gate = admission(stack)
    with pytest.raises(PluginUnavailable):
        stack.call(lambda: gate.capture(identity, None, "run"))


def test_installation_alias_is_authenticated_unique_and_recovered(
    plugin_stack, native_package
):
    stack = plugin_stack
    first = install(stack, native_package())
    second = install(stack, native_package())
    rows = stack.call(lambda: stack.coordinator.published_snapshot()["installations"])
    aliases = {row["installation_id"]: row["alias"] for row in rows}
    assert aliases[first] == "review-helper"
    assert aliases[second] == "review-helper-" + second[:8]
    assert aliases[first] != aliases[second]
    trust(stack, first)
    stack.call(lambda: stack.coordinator.recover())
    assert {
        row["installation_id"]: row["alias"]
        for row in stack.call(lambda: stack.coordinator.published_snapshot())[
            "installations"
        ]
    } == aliases


def test_old_authority_without_alias_keeps_exact_digest():
    from tldw_chatbook.Plugins.authority import (
        canonical_snapshot,
        empty_snapshot,
        snapshot_digest,
    )

    snapshot = empty_snapshot()
    snapshot["installations"] = [
        {"installation_id": "old", "revision_digest": None, "activation_default": False}
    ]
    assert canonical_snapshot(snapshot) == snapshot
    assert snapshot_digest(canonical_snapshot(snapshot)) == snapshot_digest(snapshot)


@pytest.mark.parametrize(
    "workspace,intent", [("workspace-b", "enabled"), ("workspace-a", "disabled")]
)
def test_activation_operation_id_cannot_retarget_review(
    plugin_stack, native_package, workspace, intent
):
    stack = plugin_stack
    identity = install(stack, native_package())
    activate(stack, identity, "workspace-b", "enabled")
    first = stack.call(
        lambda: stack.coordinator.review_activation(
            identity, workspace_id="workspace-a", intent="enabled"
        )
    )
    other = stack.call(
        lambda: stack.coordinator.review_activation(
            identity, workspace_id=workspace, intent=intent
        )
    )
    assert stack.call(
        lambda: stack.coordinator.commit(first, first.operation_id)
    ).committed
    assert stack.call(
        lambda: stack.coordinator.commit(first, first.operation_id)
    ).committed
    with pytest.raises(ValueError, match="different review"):
        stack.call(lambda: stack.coordinator.commit(other, first.operation_id))


@pytest.mark.parametrize("workspace", ["global", "workspace-default", ""])
def test_coordinator_rejects_reserved_activation_scope(
    plugin_stack, native_package, workspace
):
    identity = install(plugin_stack, native_package())
    with pytest.raises(ValueError, match="workspace"):
        plugin_stack.call(
            lambda: plugin_stack.coordinator.review_activation(
                identity, workspace_id=workspace, intent="enabled"
            )
        )


@pytest.mark.parametrize("workspace", ["global", "workspace-default"])
def test_historical_authenticated_sentinel_generations_still_fence_old_runs(
    plugin_stack, native_package, workspace
):
    from tldw_chatbook.Plugins.admission import PluginUnavailable
    from tldw_chatbook.Plugins.authority import PluginMarker, snapshot_digest

    stack = plugin_stack
    identity = install(stack, native_package())
    trust(stack, identity)
    gate = admission(stack)

    def legacy_transition(intent, operation):
        # Reconstruct a signed pre-fix transition, not a new public activation.
        old = stack.authority.load_marker()
        result = {
            "operation_id": operation,
            "installation_id": identity,
            "kind": "activate",
            "revision_digest": stack.coordinator.published_snapshot()["installations"][
                0
            ]["revision_digest"],
            "result": "committed",
        }
        # Historical scope spelling is the fixture subject. Its mutation uses
        # the current issued-ID namespace; legacy-ID wire fixtures live in recovery.
        import hashlib

        operation = stack.authority.issue_operation_id(
            old.generation + 1,
            hashlib.sha256(operation.encode()).hexdigest(),
            {key: value for key, value in result.items() if key != "operation_id"},
            hashlib.sha256((operation + "-nonce").encode()).hexdigest(),
        )
        result["operation_id"] = operation
        with stack.registry.transaction() as cursor:
            cursor.execute(
                "INSERT INTO activation VALUES (?, ?, ?) ON CONFLICT(installation_id, workspace_id) DO UPDATE SET intent=excluded.intent",
                (identity, workspace, intent),
            )
            cursor.execute(
                "INSERT INTO authority_generations VALUES (?, 'workspace', ?, 1, 0) ON CONFLICT(installation_id, scope_kind, workspace_id) DO UPDATE SET generation=generation+1",
                (identity, workspace),
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
        return stack.coordinator.recover()

    stack.call(lambda: legacy_transition("enabled", "legacy-initial"))
    snapshot = stack.call(lambda: gate.capture(identity, workspace, "old-run"))
    stack.call(lambda: gate.check(snapshot, "skill:review"))
    stack.call(lambda: legacy_transition("disabled", "legacy-disable"))
    with pytest.raises(PluginUnavailable):
        stack.call(lambda: gate.check(snapshot, "skill:review"))
    stack.call(lambda: legacy_transition("enabled", "legacy-reenable"))
    with pytest.raises(PluginUnavailable, match="changed"):
        stack.call(lambda: gate.check(snapshot, "skill:review"))


def test_recorded_eligibility_is_shared_but_execution_still_verifies_bytes(
    plugin_stack, native_package, monkeypatch
):
    from pathlib import Path

    from tldw_chatbook.Plugins import admission as module

    stack = plugin_stack
    identity = install(stack, native_package())
    trust(stack, identity)
    activate(stack, identity, "workspace-a", "enabled")
    gate = admission(stack)
    runtime = stack.call(lambda: gate.capture(identity, "workspace-a", "control"))
    authority, records = stack.call(
        lambda: stack.coordinator.recorded_inspections(
            ((identity, runtime.revision_digest),)
        )
    )
    recorded = records[(identity, runtime.revision_digest)]
    assert not recorded.inventory["skill:review"].evidence
    assert not recorded.candidates and not recorded.diagnostics
    assert dict(recorded.dependency_edges) == {
        key: item.dependencies for key, item in recorded.inventory.items()
    }

    def display(workspace="workspace-a"):
        return stack.call(
            lambda: gate.recorded_eligibility(
                identity, workspace, authority=authority, inspection=recorded
            )
        )

    with monkeypatch.context() as patch:
        patch.setattr(
            module,
            "retained_inspections",
            lambda *a, **k: pytest.fail("passive body read"),
        )
        patch.setattr(
            module,
            "RunPluginSnapshot",
            lambda *a, **k: pytest.fail("display execution snapshot"),
        )
        assert display() == runtime.selection
        with pytest.raises(module.PluginUnavailable):
            display("missing-workspace")
        gate.fences.seal(identity, "workspace-a")
        with pytest.raises(module.PluginUnavailable):
            display()
        gate.fences._sealed.remove((identity, "workspace-a"))
    # Recorded eligibility is explicitly not a retained-byte integrity claim.
    member = Path(recorded.materialized_identity) / "skills/review/SKILL.md"
    member.chmod(0o600)
    member.write_text("tampered")
    assert display() == runtime.selection
    with pytest.raises(module.PluginUnavailable):
        stack.call(lambda: gate.capture(identity, "workspace-a", "tampered-runtime"))


def test_recorded_metadata_strips_unprotected_observations_and_rejects_authority_tamper(
    plugin_stack, native_package
):
    import json

    stack = plugin_stack
    identity = install(stack, native_package())
    snapshot = stack.call(stack.coordinator.published_snapshot)
    revision = snapshot["installations"][0]["revision_digest"]

    def rewrite(*, protected):
        with stack.registry.transaction() as cursor:
            # Deliberate corruption of this private fixture, beyond SQL guards.
            cursor.execute("DROP TRIGGER IF EXISTS immutable_revisions")
            cursor.execute("DROP TRIGGER IF EXISTS immutable_components")
            row = cursor.execute(
                "SELECT inspection_json FROM revisions WHERE installation_id=?",
                (identity,),
            ).fetchone()
            raw = json.loads(row[0])
            if protected:
                raw["source_identity"] = "forged-origin"
            else:
                raw["inventory"]["skill:review"]["evidence"] = []
                cursor.execute(
                    "UPDATE components SET definition_json=? WHERE installation_id=? AND component_id=?",
                    (
                        json.dumps(raw["inventory"]["skill:review"]),
                        identity,
                        "skill:review",
                    ),
                )
            cursor.execute(
                "UPDATE revisions SET inspection_json=? WHERE installation_id=?",
                (json.dumps(raw), identity),
            )

    stack.call(lambda: rewrite(protected=False))
    _, records = stack.call(
        lambda: stack.coordinator.recorded_inspections(((identity, revision),))
    )
    assert not records[(identity, revision)].inventory["skill:review"].evidence
    with pytest.raises(ValueError):
        stack.call(
            lambda: stack.coordinator.recorded_inspections(((identity, "0" * 64),))
        )
    stack.call(lambda: rewrite(protected=True))
    with pytest.raises(PermissionError):
        stack.call(
            lambda: stack.coordinator.recorded_inspections(((identity, revision),))
        )


def test_recorded_and_runtime_eligibility_share_mapping_and_dependency_gates(
    plugin_stack, native_package
):
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.Plugins.admission import PluginUnavailable

    stack = plugin_stack
    models = {"llama_cpp": ["m"]}
    owner = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: models,
        settings_loader=lambda: {
            "api_settings": {
                "llama_cpp": {"api_url": "http://127.0.0.1:8080/v1/chat/completions"}
            }
        },
        environ={},
    )
    stack.call(lambda: setattr(stack.coordinator, "model_mapping_owner", owner))
    package = native_package(extension={"version": 1, "agents": ["agent.md"]})
    (package / "agent.md").write_text(
        "---\nname: analyst\ndescription: Analyze\ntools: inherit\nmodel: preferred\n---\nAnalyze."
    )
    identity = install(stack, package)
    trust(stack, identity)
    activate(stack, identity, "workspace-a", "enabled")
    gate = admission(stack)

    def selections():
        authority = stack.coordinator.published_snapshot()
        revision = authority["installations"][0]["revision_digest"]
        authority, records = stack.coordinator.recorded_inspections(
            ((identity, revision),)
        )
        recorded = gate.recorded_eligibility(
            identity,
            "workspace-a",
            authority=authority,
            inspection=records[(identity, revision)],
        )
        runtime = gate.capture(identity, "workspace-a", "mapping-control").selection
        assert recorded == runtime
        return recorded

    assert stack.call(selections) == ("skill:review",)
    review = stack.call(
        lambda: stack.coordinator.review_configuration(
            identity, models={"agent:analyst": "llama_cpp/m"}
        )
    )
    stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    assert "agent:analyst" in stack.call(selections)
    models["llama_cpp"] = ["changed"]
    assert stack.call(selections) == ("skill:review",)
    # Restore the fixture mapping before issuing an unrelated durable mutation.
    models["llama_cpp"] = ["m"]
    # Both paths also reject disabled current scope, not merely its display label.
    activate(stack, identity, "workspace-a", "disabled")
    with pytest.raises(PluginUnavailable):
        stack.call(selections)
    with pytest.raises(PluginUnavailable):
        stack.call(lambda: gate.capture(identity, "workspace-a", "disabled-control"))


@pytest.mark.bootstrap_profile
@pytest.mark.requires_cleanup
@pytest.mark.parametrize("aliases", ["builtin", "canonical", "mixed"])
def test_recorded_mapped_eligibility_keeps_actual_owners_without_package_reads(
    plugin_stack, native_package, tmp_path, monkeypatch, aliases
):
    """Passive current-reference checks use recorded definitions, never recovery."""
    import json
    from copy import deepcopy
    from dataclasses import replace

    from Tests.MCP.test_credential_bindings import MemoryCredentialBackend
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider
    from tldw_chatbook.MCP.credential_bindings import CredentialBindingService
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalMCPStore
    from tldw_chatbook.Plugins import admission as admission_module
    from tldw_chatbook.Plugins import package_files, recovery
    from tldw_chatbook.Plugins.host_references import (
        capture_reference,
        mapping_id,
        tool_constraint_id,
    )

    stack = plugin_stack
    credentials = CredentialBindingService(MemoryCredentialBackend())
    binding = credentials.create_opaque(
        endpoint_origin="https://mcp.example",
        headers={"Authorization": "Bearer fixture-only"},
        method="bearer",
    )
    local = LocalMCPControlService(
        store=LocalMCPStore(tmp_path / "recorded-mcp.json"),
        credential_service=credentials,
    )
    local.save_external_profile({
        "profile_id": "remote",
        "transport": "streamable_http",
        "url": "https://mcp.example/rpc",
        "credential_reference": binding.reference_id,
        "credential_generation": binding.authority_generation,
    })
    local.store.save_discovery_snapshot("remote", {
        "tools": [{"name": "echo", "description": "Echo", "inputSchema": {"type": "object"}}],
    })
    stack.call(lambda: setattr(stack.coordinator, "mcp_mapping_owner", local))
    names = {
        "builtin": ["arithmetic"],
        "canonical": ["search"],
        "mixed": ["search", "arithmetic"],
    }[aliases]
    root = native_package(extension={
        "version": 1,
        "agents": ["agent.md"],
        "requires": {"agent:reader": ["mcp:remote"]},
    })
    (root / "agent.md").write_text(
        "---\nname: reader\ndescription: Read\ntools: " + json.dumps(names) + "\n---\nRead."
    )
    (root / "mcp.json").write_text(json.dumps({
        "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
        "mcpServers": {"remote": {"type": "streamable-http", "url": "https://mcp.example/rpc"}},
    }))
    identity = install(stack, root)
    trust(stack, identity)
    activate(stack, identity, "workspace-a", "enabled")
    configured = stack.call(lambda: stack.coordinator.review_configuration(
        identity,
        connections={"mcp:remote": "remote"},
        tools={"mcp:remote": ("echo",)},
        tool_references=(
            {"agent:reader": {"arithmetic": "builtin:calculator"}}
            if "arithmetic" in names else None
        ),
        component_tools=(
            {"agent:reader": {"search": "mcp:remote::echo"}}
            if "search" in names else None
        ),
    ))
    stack.call(lambda: stack.coordinator.commit(configured, configured.operation_id))
    gate = admission(stack)
    runtime = stack.call(lambda: gate.capture(identity, "workspace-a", "mapped-control"))
    assert "agent:reader" in runtime.selection

    def forbidden(*args, **kwargs):
        pytest.fail("recorded eligibility reached a package reader or execution snapshot")

    with monkeypatch.context() as patch:
        patch.setattr(admission_module, "retained_inspections", forbidden)
        patch.setattr(admission_module, "RunPluginSnapshot", forbidden)
        patch.setattr(recovery, "retained_inspections", forbidden)
        patch.setattr(package_files, "capture_package", forbidden)
        patch.setattr(package_files, "_directory_fd", forbidden)
        patch.setattr(package_files, "_member_fd", forbidden)
        authority, records = stack.call(lambda: stack.coordinator.recorded_inspections(
            ((identity, runtime.revision_digest),)
        ))
        recorded = records[(identity, runtime.revision_digest)]

        def display(projection=authority):
            return stack.call(lambda: gate.recorded_eligibility(
                identity, "workspace-a", authority=projection, inspection=recorded
            ))

        assert display() == runtime.selection
        rows = authority["mappings"]
        if "search" in names:
            canonical = next(row for row in rows if row["mapping_id"] == tool_constraint_id("agent:reader", "search"))
            target = next(row for row in rows if row["mapping_id"] == canonical["target_reference"])
            legacy = capture_reference(identity, recorded, "agent:reader", "tool",
                                       target["target_reference"], label="search", mappings=rows)
            competing = deepcopy(authority)
            competing["mappings"].append(legacy)
            assert "agent:reader" not in display(competing)
            for field in ("definition_digest", "configuration_digest", "target_reference"):
                changed = deepcopy(authority)
                row = next(row for row in changed["mappings"] if row["mapping_id"] == canonical["mapping_id"])
                row[field] = "tampered"
                assert "agent:reader" not in display(changed)
            discovery = local.store.get_discovery_snapshot("remote")
            changed = deepcopy(discovery)
            changed["tools"][0]["description"] = "Changed schema"
            local.store.save_discovery_snapshot("remote", changed)
            assert "agent:reader" not in display()
            local.store.save_discovery_snapshot("remote", discovery)
        if "arithmetic" in names:
            builtin = next(row for row in rows if row["mapping_id"] == mapping_id("agent:reader", "arithmetic"))
            assert builtin["kind"] == "tool"
            original = BuiltinToolProvider.load_schema
            with monkeypatch.context() as schema_patch:
                schema_patch.setattr(BuiltinToolProvider, "load_schema", lambda self, key: replace(
                    original(self, key), description="Changed builtin schema"
                ))
                assert "agent:reader" not in display()
        assert display() == runtime.selection
        credentials.revoke(binding.reference_id)
        assert "agent:reader" not in display()
        assert "mcp:remote" not in display()
        assert "skill:review" in display()
