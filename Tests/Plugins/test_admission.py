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
