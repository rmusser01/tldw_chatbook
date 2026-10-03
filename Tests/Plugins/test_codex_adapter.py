"""Independent OpenAI precedence and retained interpretation controls."""

from tldw_chatbook.Plugins.inspection import inspect_package


def test_inline_overlay_replaces_compatibility_wholesale():
    from tldw_chatbook.Plugins.adapters.codex import overlay_for_openai

    assert overlay_for_openai(
        {"skills": ["new"]}, {"skills": ["old"], "hooks": "legacy"}
    ) == {"skills": ["new"]}
    assert overlay_for_openai(None, {"skills": ["old"]}) == {"skills": ["old"]}


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
    manifest["extensions"].pop("com.openai")
    path.write_text(json.dumps(manifest))
    assert (
        "vendor_guard_scope_unknown"
        in inspect_package(root, dialect="openai").activation_blockers
    )


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
    assert (
        "vendor_codex_component_contract_unqualified"
        in result.inventory["command:note"].activation_blockers
    )


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
    result = inspect_package(root)
    assert "vendor_variables_unconfigured" in result.activation_blockers
    assert result.inventory["skill:review"].activation_blockers
