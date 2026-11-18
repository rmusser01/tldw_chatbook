"""Independent OpenAI precedence and retained interpretation controls."""

from Tests.Plugins import platform_support
from tldw_chatbook.Plugins.inspection import inspect_package


def test_inline_overlay_replaces_compatibility_wholesale():
    from tldw_chatbook.Plugins.adapters.codex import overlay_for_openai

    assert overlay_for_openai(
        {"skills": ["new"]}, {"skills": ["old"], "hooks": "legacy"}
    ) == {"skills": ["new"]}
    assert overlay_for_openai(None, {"skills": ["old"]}) == {"skills": ["old"]}


@platform_support.package_capture
def test_standalone_codex_selected_skill_is_qualified_metadata(tmp_path):
    import json

    root = tmp_path / "codex"
    (root / ".codex-plugin").mkdir(parents=True)
    (root / ".codex-plugin/plugin.json").write_text(
        json.dumps({"name": "codex-demo", "skills": "./custom/"})
    )
    (root / "custom/review").mkdir(parents=True)
    (root / "custom/review/SKILL.md").write_text(
        "---\nname: review\ndescription: Review changes.\n---\nCODEX_REVIEW_BODY"
    )
    result = inspect_package(root)
    assert result.dialect == "openai"
    assert not result.activation_blockers
    assert set(result.inventory) == {"skill:review"}
    assert result.inventory["skill:review"].support == "adapted"


@platform_support.package_capture
def test_portable_inline_overlay_removes_compatibility_guard_and_keeps_locations(
    interop_package,
):
    import json

    root = interop_package("portable-openai")
    result = inspect_package(root, dialect="openai")
    assert not result.rejected and not result.activation_blockers
    assert set(result.inventory) == {"skill:review"}
    assert result.overlay_identities == ("plugin.json#/extensions/com.openai",)
    path = root / "plugin.json"
    manifest = json.loads(path.read_text())
    manifest["extensions"]["com.openai"]["skills"] = []
    path.write_text(json.dumps(manifest))
    assert set(inspect_package(root, dialect="openai").inventory) == {"skill:review"}


@platform_support.package_capture
@platform_support.vendor_command
def test_codex_compatibility_guard_is_retained_without_inline_overlay(interop_package):
    import json

    root = interop_package("portable-openai")
    path = root / "plugin.json"
    manifest = json.loads(path.read_text())
    manifest["extensions"].pop("com.openai")
    path.write_text(json.dumps(manifest))
    inherited = inspect_package(root, dialect="openai")
    assert inherited.adapter_version == "chatbook-codex/2"
    assert not inherited.activation_blockers
    guards = {
        key
        for key, component in inherited.inventory.items()
        if component.kind == "hook" and component.support == "adapted"
    }
    assert guards
    assert guards <= set(inherited.inventory["skill:review"].dependencies)


@platform_support.package_capture
def test_codex_manual_only_policy_is_preserved_and_conflicts_refuse(interop_package):
    import json

    root = interop_package("codex")
    result = inspect_package(root)
    assert set(result.inventory) == {"skill:review", "command:note"}
    definition = json.loads(result.inventory["skill:review"].definition_json)
    assert definition["metadata"]["disable_model_invocation"] == "true"
    skill = root / "custom/review/SKILL.md"
    skill.write_text(
        skill.read_text().replace(
            "description: Review changes.\n",
            "description: Review changes.\ndisable-model-invocation: true\n",
        )
    )
    (root / "custom/review/agents/openai.yaml").write_text(
        "policy:\n  allow_implicit_invocation: true\n"
    )
    result = inspect_package(root)
    assert (
        "vendor_manual_flags_conflict"
        in result.inventory["skill:review"].activation_blockers
    )
    assert result.inventory["skill:review"].support == "unsupported"
    assert not result.inventory["command:note"].activation_blockers


@platform_support.package_capture
def test_unconfigured_vendor_variables_never_become_inherited_defaults(interop_package):
    import json

    root = interop_package("codex")
    path = root / ".codex-plugin/plugin.json"
    manifest = json.loads(path.read_text())
    manifest["variables"] = {
        "type": "object",
        "properties": {"TOKEN": {"type": "string"}},
    }
    path.write_text(json.dumps(manifest))
    from tldw_chatbook.Plugins.inspection import _inspect_capture
    from tldw_chatbook.Plugins.package_files import capture_package

    fresh = inspect_package(root)
    assert json.loads(fresh.variables_json)["TOKEN"]["required"] is False
    assert fresh.inventory["skill:review"].availability == "disabled"
    result = _inspect_capture(capture_package(root), recipe="dev_20261001")
    assert "vendor_variables_unconfigured" in result.activation_blockers
    assert result.inventory["skill:review"].activation_blockers


# Qualified /2 controls use the same current host owners.
"""Codex package adaptation through immutable package inspection."""


@platform_support.package_capture
def test_portable_inventory_positive_control(native_package):
    from tldw_chatbook.Plugins.inspection import inspect_package

    result = inspect_package(native_package())
    assert not result.rejected
    assert set(result.inventory) == {"skill:review"}
    assert not result.inventory["skill:review"].activation_blockers


import json
from pathlib import Path

import pytest

from tldw_chatbook.Plugins.schemas import PLUGIN_SCHEMA

UPSTREAM = Path(__file__).parent / "fixtures/interop/upstream"


def write_json(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value))


def write_skill(root, path="skills/example/SKILL.md", **metadata):
    import yaml

    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "---\n"
        + yaml.safe_dump(
            {"name": "example", "description": "Controlled example", **metadata}
        )
        + "---\nUntrusted example body.\n"
    )


@platform_support.package_capture
def test_openai_root_locations_identity_and_wholesale_overlay(tmp_path):
    write_json(
        tmp_path,
        "plugin.json",
        {
            "$schema": PLUGIN_SCHEMA,
            "name": "portable-name",
            "extensions": {"com.openai": {}},
        },
    )
    write_json(
        tmp_path,
        ".codex-plugin/plugin.json",
        {
            "name": "legacy-name",
            "skills": [],
            "hooks": "./missing-hooks.json",
            "apps": "./missing-apps.json",
        },
    )
    write_skill(tmp_path)
    result = inspect_package(tmp_path, dialect="openai")
    assert not result.rejected
    assert set(result.inventory) == {"skill:example"}
    assert not result.activation_blockers
    assert result.root_manifest == "plugin.json"
    assert result.adapter_version == "chatbook-codex/2"
    assert result.overlay_identities == ("plugin.json#/extensions/com.openai",)
    manifest = json.loads((tmp_path / "plugin.json").read_text())
    del manifest["extensions"]["com.openai"]
    write_json(tmp_path, "plugin.json", manifest)
    fallback = inspect_package(tmp_path, dialect="openai")
    assert "skill:example" in fallback.inventory
    assert fallback.activation_blockers
    assert fallback.effective_digest != result.effective_digest


@platform_support.package_capture
@pytest.mark.parametrize("extension", [{"version": 999}, []])
def test_vendor_view_preserves_malformed_native_constraints(native_package, extension):
    root = native_package(extension=extension)
    write_json(root, ".codex-plugin/plugin.json", {"name": "legacy"})
    result = inspect_package(root, dialect="openai")
    assert "skill:review" in result.inventory
    assert "constraints_unknown" in result.inventory["skill:review"].activation_blockers


@platform_support.package_capture
def test_invalid_root_schema_does_not_fall_back_to_vendor(tmp_path):
    write_json(tmp_path, "plugin.json", {"$schema": "invalid", "name": "root"})
    write_json(tmp_path, ".codex-plugin/plugin.json", {"name": "legacy"})
    result = inspect_package(tmp_path, dialect="openai")
    assert result.rejected


@platform_support.package_capture
def test_selected_codex_metadata_inventory_has_no_oauth_or_app_grant():
    from tldw_chatbook.Plugins.adapters.codex import inspect_codex

    result = inspect_codex(UPSTREAM / "codex-linear-1dc19589")
    assert not result.rejected
    assert set(result.inventory) == {"mcp:linear"}
    server = result.inventory["mcp:linear"]
    assert server.availability == "disabled"
    definition = json.loads(server.definition_json)
    assert definition["type"] == "streamable-http"
    assert definition["url"] == "https://mcp.linear.app/mcp"
    assert "oauth_resource" not in definition
    assert {d.code for d in result.diagnostics} >= {
        "foreign_apps_unavailable",
        "foreign_oauth_metadata_no_grant",
    }


@platform_support.package_capture
def test_selected_portable_fixture_does_not_follow_missing_extension():
    result = inspect_package(UPSTREAM / "portable-arch-7568a482")
    assert not result.rejected
    assert not result.inventory
    assert "extension_ignored" in {d.code for d in result.diagnostics}
