"""Cursor component paths replace defaults and incomplete constraints stay visible."""

import json
from pathlib import Path

import pytest

from tldw_chatbook.Plugins.inspection import inspect_package

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


def test_cursor_inventory_matches_independent_expected_fixture(interop_package):
    root = interop_package("cursor")
    expected = json.loads(
        (Path(__file__).parent / "fixtures/interop/expected.json").read_text()
    )["cursor"]
    result = inspect_package(root)
    assert not result.rejected and not result.activation_blockers
    assert sorted(result.inventory) == expected["components"]
    assert result.inventory["rule:conditional"].support == "unsupported"
    assert (
        "vendor_rule_activation_unsupported"
        in result.inventory["rule:conditional"].activation_blockers
    )
    assert json.loads(result.inventory["agent:empty"].definition_json)["tools"] == []
    assert "skill:ignored" not in result.inventory
    assert all(row.evidence[0].level == "parsed" for row in result.inventory.values())


def test_cursor_explicit_empty_skills_preserves_deliberate_exclusion(interop_package):
    root = interop_package("cursor")
    path = root / ".cursor-plugin/plugin.json"
    manifest = json.loads(path.read_text())
    manifest["skills"] = []
    path.write_text(json.dumps(manifest))
    result = inspect_package(root)
    assert not result.rejected
    assert all(row.kind != "skill" for row in result.inventory.values())
    assert "command:note" in result.inventory


def test_root_skill_exception_and_agent_limits_do_not_inherit(tmp_path):
    root = tmp_path / "single"
    (root / ".cursor-plugin").mkdir(parents=True)
    (root / ".cursor-plugin/plugin.json").write_text(json.dumps({"name": "single"}))
    (root / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review text.\n---\nROOT_BODY"
    )
    result = inspect_package(root)
    assert set(result.inventory) == {"skill:review"}
    (root / "agents").mkdir()
    (root / "agents/reviewer.md").write_text(
        "---\nname: reviewer\ndescription: Review text.\nreadonly: true\n---\nLIMITED_AGENT_BODY"
    )
    result = inspect_package(root)
    assert result.inventory["agent:reviewer"].support == "unsupported"
    assert (
        "vendor_agent_limits_unsupported"
        in result.inventory["agent:reviewer"].activation_blockers
    )
    assert not result.inventory["skill:review"].activation_blockers


def test_retained_catalog_overlay_participates_in_review_and_executable_digest(
    interop_package,
):
    from tldw_chatbook.Plugins.adapters.cursor import inspect_cursor
    from tldw_chatbook.Plugins.package_files import materialize_package
    from tldw_chatbook.Plugins.review import reinspect

    root = interop_package("cursor-catalog")
    result = inspect_cursor(root)
    assert set(result.inventory) == {"command:catalog-note"}
    material = root.parent / "material"
    frozen = materialize_package(root, material, dialect="cursor")
    assert reinspect(frozen, material).effective_digest == result.effective_digest
    member = root / "catalog-note.md"
    member.chmod(0o700)
    assert inspect_cursor(root).effective_digest != result.effective_digest
    external = inspect_cursor(root, {"commands": []})
    assert "catalog_overlay_not_retained" in external.activation_blockers
    path = root / ".cursor-plugin/plugin.json"
    manifest = json.loads(path.read_text())
    manifest["commands"] = []
    path.write_text(json.dumps(manifest))
    assert not inspect_cursor(root).inventory


@pytest.mark.asyncio
async def test_cursor_command_and_always_rule_use_actual_reviewed_console_user_lane(
    native_console, interop_package
):
    rig = native_console
    root = interop_package("cursor")
    review = await rig.service.review_install(
        root,
        selection=("command:note", "rule:always", "agent:empty"),
        workspace_id="workspace-a",
    )
    await rig.service.commit(review, review.operation_id)
    trust = await rig.service.review_trust(review.installation_id)
    await rig.service.commit(trust, trust.operation_id)
    enabled = await rig.service.review_activation(
        review.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    await rig.service.commit(enabled, enabled.operation_id)
    result = await rig.controller.submit_draft(
        "$cursor-demo:command:note", session_id=rig.session.id
    )
    assert result.accepted
    rows = rig.gateway.payloads[-1]
    assert any(
        row["role"] == "user" and "CURSOR_MANUAL_NOTE_BODY" in str(row["content"])
        for row in rows
    )
    assert any(
        row["role"] == "user" and "CURSOR_ALWAYS_RULE_BODY" in str(row["content"])
        for row in rows
    )
    assert all(
        "CURSOR_" not in str(row["content"]) for row in rows if row["role"] == "system"
    )
    assert "CURSOR_CONDITIONAL_RULE_BODY" not in str(rows)


def test_explicit_missing_mcp_file_is_not_an_empty_definition(interop_package):
    root = interop_package("cursor")
    path = root / ".cursor-plugin/plugin.json"
    manifest = json.loads(path.read_text())
    manifest["mcpServers"] = "missing.json"
    path.write_text(json.dumps(manifest))
    result = inspect_package(root)
    assert result.rejected
    assert "vendor_mcp_path_unavailable" in result.activation_blockers


@pytest.mark.parametrize(
    "fixture,dialect,selection",
    [
        ("portable-openai", "openai", ("skill:review",)),
        ("cursor-catalog", "cursor", ("command:catalog-note",)),
    ],
)
def test_reviewed_dialect_and_catalog_survive_source_and_registry_loss(
    plugin_stack, interop_package, fixture, dialect, selection
):
    import shutil

    from Tests.Plugins.test_recovery import lose_registry

    stack = plugin_stack
    root = interop_package(fixture)
    if dialect == "cursor":
        (root / "plugin.json").write_text(
            json.dumps(
                {
                    "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
                    "name": "portable-view",
                }
            )
        )
        assert "dialect_choice_required" in inspect_package(root).activation_blockers
    inspection = inspect_package(root, dialect=dialect)
    review = stack.call(
        lambda: stack.coordinator.review(
            inspection, selection=selection, workspace_id=None
        )
    )
    assert stack.call(
        lambda: stack.coordinator.commit(review, review.operation_id)
    ).committed
    shutil.rmtree(root)
    lose_registry(stack)
    assert stack.call(stack.coordinator.recover)[0].committed
    state = stack.call(stack.authority.verify_current)
    revision = state["revisions"][0]
    assert revision["dialect"] == dialect
    assert revision["revision_digest"] == inspection.effective_digest
    assert {
        row["component_id"] for row in state["selections"] if row["selected"]
    } == set(selection)
