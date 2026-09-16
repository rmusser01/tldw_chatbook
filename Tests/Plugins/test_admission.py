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
    stack.call(lambda: stack.coordinator.commit(review, "install-" + review.token))
    return review.installation_id


def trust(stack, installation):
    review = stack.call(lambda: stack.coordinator.review_trust(installation))
    return stack.call(lambda: stack.coordinator.commit(review, "trust-" + review.token))


def activate(stack, installation, workspace, intent):
    review = stack.call(
        lambda: stack.coordinator.review_activation(
            installation, workspace_id=workspace, intent=intent
        )
    )
    return stack.call(
        lambda: stack.coordinator.commit(review, "enable-" + review.token)
    )


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
        lambda: stack.coordinator.commit(first, "activation-once")
    ).committed
    assert stack.call(
        lambda: stack.coordinator.commit(first, "activation-once")
    ).committed
    with pytest.raises(ValueError, match="different review"):
        stack.call(lambda: stack.coordinator.commit(other, "activation-once"))
