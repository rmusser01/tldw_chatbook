"""Cursor paths, constraints and catalog interpretation."""

import json

import pytest

from Tests.Plugins.test_codex_adapter import UPSTREAM, write_json, write_skill
from tldw_chatbook.Plugins.inspection import inspect_package


def test_selected_cursor_fixture_independent_inventory():
    result = inspect_package(UPSTREAM / "cursor-teaching-f5bdd682", dialect="cursor")
    assert not result.rejected
    assert set(result.inventory) == {
        "skill:create-learning-path",
        "skill:run-learning-retrospective",
    }
    assert all(not row.activation_blockers for row in result.inventory.values())
    assert result.adapter_version == "chatbook-cursor/1"


@pytest.mark.parametrize(
    "paths, expected", [([], set()), (["./other"], {"skill:other"})]
)
def test_explicit_paths_replace_default_discovery(tmp_path, paths, expected):
    write_skill(tmp_path)
    write_skill(tmp_path, "other/other/SKILL.md", name="other")
    write_json(
        tmp_path, ".cursor-plugin/plugin.json", {"name": "test", "skills": paths}
    )
    result = inspect_package(tmp_path, dialect="cursor")
    assert set(result.inventory) == expected
    assert not result.activation_blockers


def test_manifest_replaces_catalog_paths_and_overlay_changes_review_identity(tmp_path):
    from tldw_chatbook.Plugins.adapters.cursor import inspect_cursor
    from tldw_chatbook.Plugins.review import inspection_identity

    write_skill(tmp_path)
    write_json(tmp_path, ".cursor-plugin/plugin.json", {"name": "test", "skills": []})
    result = inspect_cursor(tmp_path, {"skills": "./skills"})
    assert not result.inventory
    write_json(tmp_path, ".cursor-plugin/plugin.json", {"name": "test"})
    included = inspect_cursor(tmp_path, {"skills": "./skills"})
    excluded = inspect_cursor(tmp_path, {"skills": []})
    assert set(included.inventory) == {"skill:example"}
    assert not excluded.inventory
    assert included.content_digest == excluded.content_digest
    assert included.effective_digest != excluded.effective_digest
    assert inspection_identity(included) != inspection_identity(excluded)


def test_foreign_manual_only_and_fork_metadata_use_native_summary(tmp_path):
    from tldw_chatbook.Plugins.skill_provider import skill_summary

    write_json(tmp_path, ".cursor-plugin/plugin.json", {"name": "test"})
    write_skill(tmp_path, **{"disable-model-invocation": True, "context": "fork"})
    result = inspect_package(tmp_path, dialect="cursor")
    summary = skill_summary("installed", result, "skill:example", "alias")
    assert summary["disable_model_invocation"] is True
    assert summary["context"] == "fork"
    assert summary["user_invocable"] is True


def test_conflicting_manual_only_declarations_remain_blocked(tmp_path):
    write_json(tmp_path, ".cursor-plugin/plugin.json", {"name": "test"})
    write_skill(
        tmp_path,
        **{
            "disable-model-invocation": True,
            "metadata": {"disable_model_invocation": "false"},
        },
    )
    result = inspect_package(tmp_path, dialect="cursor")
    assert result.inventory["skill:example"].activation_blockers


def test_catalog_review_commit_and_authenticated_recovery(plugin_stack, tmp_path):
    from dataclasses import replace

    from Tests.Plugins.test_recovery import lose_registry
    from tldw_chatbook.Plugins.adapters.cursor import inspect_cursor
    from tldw_chatbook.Plugins.recovery import retained_inspections

    root = tmp_path / "package"
    root.mkdir()
    write_skill(root, "alternate/example/SKILL.md")
    write_json(root, ".cursor-plugin/plugin.json", {"name": "catalog-example"})
    inspection = inspect_cursor(root, {"skills": "./alternate"})
    stack = plugin_stack
    review = stack.call(
        lambda: stack.coordinator.review(
            inspection, selection=("skill:example",), workspace_id=None
        )
    )
    changed = inspect_cursor(root, {"skills": []})
    with pytest.raises(ValueError):
        stack.call(
            lambda: stack.coordinator.commit(
                replace(review, inspection=changed), review.operation_id
            )
        )
    receipt = stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    assert receipt.committed
    snapshot = stack.call(stack.coordinator.published_snapshot)
    recovered = retained_inspections(snapshot)
    retained = recovered[(review.installation_id, inspection.effective_digest)]
    assert set(retained.inventory) == {"skill:example"}
    assert retained.content_digest == inspection.content_digest
    assert retained.overlay_identities == inspection.overlay_identities
    lose_registry(stack)
    stack.call(stack.coordinator.recover)
    assert (
        stack.call(stack.coordinator.published_snapshot)["revisions"]
        == snapshot["revisions"]
    )


def test_catalog_review_rejects_changed_source(plugin_stack, tmp_path):
    from tldw_chatbook.Plugins.adapters.cursor import inspect_cursor

    root = tmp_path / "package"
    root.mkdir()
    write_skill(root)
    write_json(root, ".cursor-plugin/plugin.json", {"name": "catalog-example"})
    stack = plugin_stack
    review = stack.call(
        lambda: stack.coordinator.review(
            inspect_cursor(root, {"skills": "./skills"}),
            selection=("skill:example",),
            workspace_id=None,
        )
    )
    (root / "skills/example/SKILL.md").write_text("mutated")
    with pytest.raises(ValueError):
        stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    assert not stack.call(stack.coordinator.published_snapshot)["installations"]


@pytest.mark.parametrize(
    "identities",
    [
        ("catalog-v1:{}", "catalog-v1:{}"),
        ("catalog-v2:{}",),
        ('catalog-v1:{"skills":[], "skills":[]}',),
        ('catalog-v1:{ "skills": [] }',),
        ('catalog-v1:{"password":"no"}',),
        ('catalog-v1:{"name":"' + "é" * 9000 + '"}',),
    ],
)
def test_catalog_carrier_refuses_ambiguous_or_unsupported_input(identities):
    from tldw_chatbook.Plugins.inspection import catalog_overlay_from

    with pytest.raises(ValueError):
        catalog_overlay_from(identities)
    assert catalog_overlay_from(('catalog-v1:{"skills":[]}',)) == {"skills": []}


def test_rules_are_manual_and_agent_constraints_are_preserved(tmp_path):
    import yaml

    write_json(tmp_path, ".cursor-plugin/plugin.json", {"name": "test"})
    for path, metadata in [
        (
            "rules/python.mdc",
            {"description": "Python rules", "globs": "**/*.py", "alwaysApply": False},
        ),
        (
            "agents/limited.md",
            {
                "name": "limited",
                "description": "Restricted",
                "tools": [],
                "model": "foreign-model",
            },
        ),
        ("agents/unknown.md", {"name": "unknown", "description": "Unknown"}),
    ]:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("---\n" + yaml.safe_dump(metadata) + "---\nInstruction body")
    result = inspect_package(tmp_path, dialect="cursor")
    assert (
        json.loads(result.inventory["rule:python"].definition_json)["mode"] == "manual"
    )
    assert json.loads(result.inventory["agent:limited"].definition_json)["tools"] == []
    assert (
        "model_mapping_required"
        in result.inventory["agent:limited"].activation_blockers
    )
    assert (
        "agent_tools_unspecified"
        in result.inventory["agent:unknown"].activation_blockers
    )


@pytest.mark.parametrize(
    "variable, blocked",
    [
        ({"type": "string", "default": "safe"}, False),
        ({"type": "string", "pattern": "safe"}, True),
    ],
)
def test_variable_schema_constraints_never_disappear(tmp_path, variable, blocked):
    write_skill(tmp_path)
    write_json(
        tmp_path,
        ".cursor-plugin/plugin.json",
        {
            "name": "test",
            "variables": {
                "type": "object",
                "properties": {"EXAMPLE": variable},
                "required": ["EXAMPLE"],
            },
        },
    )
    result = inspect_package(tmp_path, dialect="cursor")
    assert bool(result.inventory["skill:example"].activation_blockers) is blocked
    if not blocked:
        assert json.loads(result.variables_json)["EXAMPLE"]["default"] == "safe"


def test_explicit_null_paths_do_not_enable_default_components(tmp_path):
    write_skill(tmp_path)
    write_json(tmp_path, ".cursor-plugin/plugin.json", {"name": "test", "skills": None})
    result = inspect_package(tmp_path, dialect="cursor")
    assert not result.inventory
    assert result.activation_blockers


def test_cursor_mcp_variables_use_native_one_pass_forms(tmp_path):
    (tmp_path / "peer.py").write_text("# inert parser fixture\n")
    write_json(
        tmp_path,
        ".cursor-plugin/plugin.json",
        {
            "name": "test",
            "mcpServers": {
                "mcpServers": {
                    "peer": {
                        "command": "${CURSOR_PLUGIN_ROOT}/peer.py",
                        "args": ["${CLAUDE_PLUGIN_ROOT}/input"],
                        "env": {"EXAMPLE": "${CURSOR_PLUGIN_ROOT}"},
                        "cwd": "${CURSOR_PLUGIN_ROOT}",
                    }
                }
            },
        },
    )
    result = inspect_package(tmp_path, dialect="cursor")
    server = result.inventory["mcp:peer"]
    assert not server.activation_blockers
    assert json.loads(server.definition_json) == {
        "type": "stdio",
        "command": "./peer.py",
        "args": ["${PLUGIN_ROOT}/input"],
        "env": {"EXAMPLE": "${PLUGIN_ROOT}"},
        "cwd": "${PLUGIN_ROOT}",
    }


def test_unsupported_guard_blocks_native_extension_siblings(native_package):

    root = native_package(extension={"version": 1, "commands": ["command.md"]})
    (root / "command.md").write_text(
        "---\nname: native\ndescription: Native sibling\n---\nBody"
    )
    write_json(
        root,
        ".cursor-plugin/plugin.json",
        {
            "name": "test",
            "hooks": {
                "version": 1,
                "hooks": {
                    "beforeShellExecution": [
                        {"command": "python guard.py", "matcher": "danger.*"}
                    ]
                },
            },
        },
    )
    result = inspect_package(root, dialect="cursor")
    unsupported = next(key for key in result.inventory if key.startswith("hook:"))
    assert result.inventory[unsupported].support == "unsupported"
    for key in ("skill:review", "command:native"):
        assert (
            f"guard_scope_unknown:{unsupported}"
            in result.inventory[key].activation_blockers
        )


def test_public_catalog_variable_and_literal_authorization_are_not_credentials(
    tmp_path,
):
    from tldw_chatbook.Plugins.adapters.cursor import inspect_cursor
    from tldw_chatbook.Plugins.inspection import catalog_overlay_from

    write_json(tmp_path, ".cursor-plugin/plugin.json", {"name": "public-catalog"})
    overlay = {
        "variables": {
            "type": "object",
            "properties": {
                "API_KEY": {"type": "string", "default": "public-placeholder"}
            },
        },
        "mcpServers": {
            "mcpServers": {
                "public": {
                    "type": "http",
                    "url": "https://example.invalid/mcp",
                    "headers": {"Authorization": "public-literal"},
                }
            }
        },
    }
    result = inspect_cursor(tmp_path, overlay)
    assert catalog_overlay_from(result.overlay_identities) == overlay
    assert not result.inventory["mcp:public"].activation_blockers
    assert json.loads(result.inventory["mcp:public"].definition_json)["headers"] == {
        "Authorization": "public-literal"
    }
    assert result.inventory["mcp:public"].availability == "disabled"
    with pytest.raises(ValueError):
        inspect_cursor(tmp_path, {"credential_reference": "private-host-reference"})


def test_unknown_vendor_constraints_remain_visible(tmp_path):
    write_skill(tmp_path)
    write_json(
        tmp_path,
        ".cursor-plugin/plugin.json",
        {"name": "test", "requiredPolicy": {"guard": "unknown"}},
    )
    result = inspect_package(tmp_path, dialect="cursor")
    assert (
        "vendor_fields_unsupported"
        in result.inventory["skill:example"].activation_blockers
    )
