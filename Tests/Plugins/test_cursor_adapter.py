"""Cursor paths, constraints and catalog interpretation."""

import json

import pytest

from Tests.Plugins import platform_support
from Tests.Plugins.test_codex_adapter import UPSTREAM, write_json, write_skill
from tldw_chatbook.Plugins.inspection import inspect_package


@platform_support.package_capture
def test_selected_cursor_fixture_independent_inventory():
    result = inspect_package(UPSTREAM / "cursor-teaching-f5bdd682", dialect="cursor")
    assert not result.rejected
    assert set(result.inventory) == {
        "skill:create-learning-path",
        "skill:run-learning-retrospective",
    }
    assert all(not row.activation_blockers for row in result.inventory.values())
    assert result.adapter_version == "chatbook-cursor/1"


@platform_support.package_capture
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


@platform_support.package_capture
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


@platform_support.package_capture
def test_foreign_manual_only_and_fork_metadata_use_native_summary(tmp_path):
    from tldw_chatbook.Plugins.skill_provider import skill_summary

    write_json(tmp_path, ".cursor-plugin/plugin.json", {"name": "test"})
    write_skill(tmp_path, **{"disable-model-invocation": True, "context": "fork"})
    result = inspect_package(tmp_path, dialect="cursor")
    summary = skill_summary("installed", result, "skill:example", "alias")
    assert summary["disable_model_invocation"] is True
    assert summary["context"] == "fork"
    assert summary["user_invocable"] is True


@platform_support.package_capture
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


@platform_support.package_capture
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


@platform_support.package_capture
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


@platform_support.package_capture
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


@platform_support.package_capture
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


@platform_support.package_capture
def test_explicit_null_paths_do_not_enable_default_components(tmp_path):
    write_skill(tmp_path)
    write_json(tmp_path, ".cursor-plugin/plugin.json", {"name": "test", "skills": None})
    result = inspect_package(tmp_path, dialect="cursor")
    assert not result.inventory
    assert result.activation_blockers


@platform_support.package_capture
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


@platform_support.package_capture
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


@platform_support.package_capture
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


@platform_support.package_capture
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


@platform_support.package_capture
@pytest.mark.parametrize("dialect", ["cursor", "openai"])
@pytest.mark.parametrize(
    "name, raw",
    [
        ("bad", {}),
        ("bad", {"type": "stdio"}),
        ("bad", {"command": None}),
        ("bad", {"command": ""}),
        ("", None),
        ("", False),
        ("", 7),
        ("", "invalid"),
        ("", []),
        ("", {"name": "good", "command": "python"}),
        ("bad:name", {"command": "python"}),
        ("bad name", {"command": "python"}),
    ],
)
def test_malformed_mcp_entry_preserves_valid_inspection_sibling(
    tmp_path, dialect, name, raw
):
    manifest = (
        ".cursor-plugin/plugin.json"
        if dialect == "cursor"
        else ".codex-plugin/plugin.json"
    )
    write_json(
        tmp_path,
        manifest,
        {
            "name": "test",
            "mcpServers": {"mcpServers": {name: raw, "good": {"command": "python"}}},
        },
    )
    result = inspect_package(tmp_path, dialect=dialect)
    assert not result.rejected and not result.activation_blockers
    assert "mcp:good" in result.inventory and len(result.inventory) == 2
    bad = next(row for key, row in result.inventory.items() if key != "mcp:good")
    if name == "bad":
        assert bad.component_id == "mcp:bad"
    else:
        assert bad.local_id.startswith("invalid:")
        assert any(
            d.code == "mcp_name_invalid" and d.component_id == bad.component_id
            for d in result.diagnostics
        )
    assert bad.support == "unsupported"
    assert bad.activation_blockers == ("definition_unsupported",)
    assert json.loads(bad.definition_json) == raw
    good = result.inventory["mcp:good"]
    assert good.support == "adapted" and not good.activation_blockers
    assert json.loads(good.definition_json) == {"type": "stdio", "command": "python"}


@platform_support.package_capture
@pytest.mark.parametrize("dialect", ["cursor", "openai"])
def test_valid_mcp_entries_share_public_inspection_control(tmp_path, dialect):
    manifest = (
        ".cursor-plugin/plugin.json"
        if dialect == "cursor"
        else ".codex-plugin/plugin.json"
    )
    write_json(
        tmp_path,
        manifest,
        {
            "name": "test",
            "mcpServers": {
                "mcpServers": {
                    "first": {"command": "python"},
                    "good": {"command": "python"},
                }
            },
        },
    )
    result = inspect_package(tmp_path, dialect=dialect)
    assert not result.rejected and not result.activation_blockers
    assert set(result.inventory) == {"mcp:first", "mcp:good"}
    assert all(
        row.support == "adapted" and not row.activation_blockers
        for row in result.inventory.values()
    )


@platform_support.package_capture
@pytest.mark.parametrize(
    "name, raw",
    [
        ("", None),
        ("", {"name": "good"}),
        ("bad name", []),
        ("x" * 129, {}),
        ("\x00", False),
        ("valid", {"command": "python"}),
    ],
)
@pytest.mark.parametrize("catalog", [False, True])
def test_valid_mcp_sibling_can_commit_and_recover_with_invalid_name(
    plugin_stack, tmp_path, name, raw, catalog
):
    from Tests.Plugins.test_recovery import lose_registry
    from tldw_chatbook.Plugins.recovery import retained_inspections

    root = tmp_path / "package"
    root.mkdir()
    overlay = {"mcpServers": {"mcpServers": {name: raw, "good": {"command": "python"}}}}
    write_json(
        root,
        ".cursor-plugin/plugin.json",
        {"name": "malformed-sibling", **({} if catalog else overlay)},
    )
    inspection = inspect_package(
        root, dialect="cursor", catalog_overlay=overlay if catalog else None
    )
    stack = plugin_stack
    review = stack.call(
        lambda: stack.coordinator.review(
            inspection, selection=("mcp:good",), workspace_id=None
        )
    )
    receipt = stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    assert receipt.committed
    snapshot = stack.call(stack.coordinator.published_snapshot)
    retained = retained_inspections(snapshot)[
        (review.installation_id, inspection.effective_digest)
    ]
    assert set(retained.inventory) == set(inspection.inventory)
    for key, row in retained.inventory.items():
        original = inspection.inventory[key]
        assert (row.definition_json, row.support, row.activation_blockers) == (
            original.definition_json,
            original.support,
            original.activation_blockers,
        )
    assert not retained.inventory["mcp:good"].activation_blockers
    assert len(retained.inventory) == 2
    lose_registry(stack)
    stack.call(stack.coordinator.recover)
    recovered = stack.call(stack.coordinator.published_snapshot)
    assert recovered["revisions"] == snapshot["revisions"]
    assert recovered["components"] == snapshot["components"]


@platform_support.package_capture
def test_invalid_mcp_names_have_distinct_stable_diagnostic_ids(tmp_path):
    write_json(tmp_path, ".cursor-plugin/plugin.json", {"name": "test"})
    servers = {
        "": None,
        "invalid:0:0": False,
        "bad name": [],
        "invalid-0": {"command": "python"},
        "good": {"command": "python"},
    }

    def inspect(values):
        return inspect_package(
            tmp_path,
            dialect="cursor",
            catalog_overlay={"mcpServers": {"mcpServers": values}},
        )

    result = inspect(servers)
    reordered = inspect(dict(reversed(list(servers.items()))))
    assert not result.rejected and not result.activation_blockers
    assert result.effective_digest == reordered.effective_digest
    assert len(result.inventory) == len(servers)
    assert all(
        not result.inventory[key].activation_blockers
        for key in ("mcp:good", "mcp:invalid-0")
    )
    bad = [row for row in result.inventory.values() if row.support == "unsupported"]
    assert len(bad) == 3
    assert len({row.component_id for row in bad}) == 3
    assert all(
        row.local_id.startswith("invalid:") and len(row.component_id) < 256
        for row in bad
    )
    assert {row.definition_json for row in bad} == {"null", "false", "[]"}
    assert {
        d.component_id for d in result.diagnostics if d.code == "mcp_name_invalid"
    } == {row.component_id for row in bad}


@platform_support.package_capture
def test_invalid_mcp_placeholder_cannot_be_admitted_when_selected(
    plugin_stack, tmp_path
):
    from Tests.Plugins.test_admission import activate, admission, trust
    from tldw_chatbook.Plugins.admission import PluginUnavailable

    root = tmp_path / "package"
    root.mkdir()
    write_skill(root)
    write_json(
        root,
        ".cursor-plugin/plugin.json",
        {"name": "diagnostic-only", "mcpServers": {"mcpServers": {"": None}}},
    )
    inspection = inspect_package(root, dialect="cursor")
    invalid = next(
        key for key, row in inspection.inventory.items() if row.support == "unsupported"
    )
    stack = plugin_stack
    review = stack.call(
        lambda: stack.coordinator.review(
            inspection, selection=tuple(inspection.inventory), workspace_id=None
        )
    )
    assert stack.call(
        lambda: stack.coordinator.commit(review, review.operation_id)
    ).committed
    assert trust(stack, review.installation_id).committed
    assert activate(stack, review.installation_id, "workspace-a", "enabled").committed
    gate = admission(stack)
    snapshot = stack.call(
        lambda: gate.capture(review.installation_id, "workspace-a", "run")
    )
    assert snapshot.selection == ("skill:example",)
    stack.call(lambda: gate.check(snapshot, "skill:example"))
    with pytest.raises(PluginUnavailable):
        stack.call(lambda: gate.check(snapshot, invalid))
