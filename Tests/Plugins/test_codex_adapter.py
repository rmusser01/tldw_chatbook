"""Codex package adaptation through immutable package inspection."""

from Tests.Plugins import platform_support


def test_inline_overlay_replaces_compatibility_wholesale():
    from tldw_chatbook.Plugins.adapters.codex import overlay_for_openai

    assert overlay_for_openai(
        {"skills": ["new"]}, {"skills": ["old"], "hooks": "legacy"}
    ) == {"skills": ["new"]}
    assert overlay_for_openai(None, {"skills": ["old"]}) == {"skills": ["old"]}


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

from tldw_chatbook.Plugins.inspection import inspect_package
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
    assert result.adapter_version == "chatbook-codex/1"
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
