"""Native inspection must retain constraints without executing package code."""


def test_missing_required_hook_stays_a_blocker(native_package):
    from tldw_chatbook.Plugins.inspection import inspect_package

    package = native_package(requires={"skill:review": ["hook:missing"]})
    result = inspect_package(package)
    assert "skill:review" in result.inventory
    assert result.inventory["skill:review"].activation_blockers
    control = inspect_package(native_package())
    assert not control.inventory["skill:review"].activation_blockers


import json

import pytest

from tldw_chatbook.Plugins.inspection import inspect_package

NS = "io.github.rmusser01.chatbook"


def edit_manifest(root, **changes):
    path = root / "plugin.json"
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


def write_component(root, name, metadata, body="Untrusted instruction text."):
    import yaml

    path = root / NS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\n" + yaml.safe_dump(metadata) + "---\n" + body)
    return f"./{NS}/{name}"


@pytest.mark.parametrize(
    "extension",
    [
        [],
        {"version": 2},
        {"version": True},
        {"version": 1, "grants": []},
        {"version": 1, "requires": []},
        {"version": 1, "requires": {"unknown:x": ["skill:review"]}},
    ],
)
def test_malformed_extension_preserves_portable_inventory_but_blocks(
    extension, native_package
):
    result = inspect_package(native_package(extension=extension))
    assert not result.rejected
    assert set(result.inventory) == {"skill:review"}
    assert "constraints_unknown" in result.inventory["skill:review"].activation_blockers
    assert (
        not inspect_package(native_package())
        .inventory["skill:review"]
        .activation_blockers
    )


def test_unknown_portable_fields_and_namespaces_are_nonfatal(native_package):
    root = native_package()
    edit_manifest(root, unexpected="ignored", extensions={"org.example.other": 42})
    result = inspect_package(root)
    assert not result.rejected
    assert not result.inventory["skill:review"].activation_blockers
    assert "manifest_unknown_field" in {d.code for d in result.diagnostics}
    assert "extension_ignored" in {d.code for d in result.diagnostics}


def test_nonobject_extensions_preserve_inventory_without_erasing_constraints(
    native_package,
):
    root = native_package()
    edit_manifest(root, extensions=[])
    result = inspect_package(root)
    assert not result.rejected
    assert "constraints_unknown" in result.inventory["skill:review"].activation_blockers


def test_unknown_requires_source_blocks_all(native_package):
    result = inspect_package(native_package(requires={"skill:absent": ["hook:guard"]}))
    assert result.inventory["skill:review"].activation_blockers
    assert result.dependency_edges["skill:absent"] == ("hook:guard",)


def test_cycles_and_dependents_block_while_unrelated_skill_survives(native_package):
    root = native_package(
        requires={"skill:review": ["skill:other"], "skill:other": ["skill:review"]}
    )
    for name in ("other", "independent"):
        path = root / "skills" / name
        path.mkdir()
        (path / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: A useful skill.\n---\nBody"
        )
    result = inspect_package(root)
    assert result.inventory["skill:review"].activation_blockers
    assert result.inventory["skill:other"].activation_blockers
    assert not result.inventory["skill:independent"].activation_blockers


def test_complete_native_inventory_is_parsed_but_disabled(native_package):
    root = native_package()
    command = write_component(
        root,
        "commands/review.md",
        {"name": "review", "description": "Review", "arguments": ["target"]},
    )
    rule = write_component(root, "rules/care.md", {"name": "care", "mode": "always"})
    agent = write_component(
        root,
        "agents/reviewer.md",
        {"name": "reviewer", "description": "Review", "tools": []},
    )
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
                "mcpServers": {
                    "repository": {
                        "type": "stdio",
                        "command": "python3",
                        "args": ["--version"],
                    }
                },
            }
        )
    )
    (root / NS / "hooks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "protect-writes",
                        "event": "PreToolUse",
                        "type": "command",
                        "argv": ["python3", "--version"],
                        "effects": ["deny"],
                    }
                ],
            }
        )
    )
    edit_manifest(
        root,
        extensions={
            NS: {
                "version": 1,
                "commands": [command],
                "rules": [rule],
                "agents": [agent],
                "hooks": f"./{NS}/hooks.json",
                "requires": {"skill:review": ["hook:protect-writes", "mcp:repository"]},
            }
        },
    )
    result = inspect_package(root)
    assert set(result.inventory) == {
        "skill:review",
        "command:review",
        "rule:care",
        "agent:reviewer",
        "hook:protect-writes",
        "mcp:repository",
    }
    assert not result.inventory["skill:review"].activation_blockers
    assert all(
        c.availability == "disabled" and c.evidence[0].level == "parsed"
        for c in result.inventory.values()
    )
    assert json.loads(result.inventory["agent:reviewer"].definition_json)["tools"] == []
    with pytest.raises(TypeError):
        result.inventory["skill:review"] = result.inventory["skill:review"]


@pytest.mark.parametrize(
    "declaration",
    [
        {"type": "integer", "required": True, "secret": False, "default": True},
        {"type": "string", "required": True, "secret": True, "default": "bad"},
        {"type": "string", "required": True, "secret": False, "pattern": ".*"},
    ],
)
def test_invalid_variables_cannot_erase_constraints(native_package, declaration):
    result = inspect_package(
        native_package(extension={"version": 1, "variables": {"TOKEN": declaration}})
    )
    assert "constraints_unknown" in result.inventory["skill:review"].activation_blockers


def test_unset_required_variable_and_reserved_name(native_package):
    declaration = {"type": "string", "required": True, "secret": True}
    result = inspect_package(
        native_package(extension={"version": 1, "variables": {"TOKEN": declaration}})
    )
    assert (
        "variable_unset:TOKEN" in result.inventory["skill:review"].activation_blockers
    )
    result = inspect_package(
        native_package(
            extension={"version": 1, "variables": {"PLUGIN_ROOT": declaration}}
        )
    )
    assert "constraints_unknown" in result.inventory["skill:review"].activation_blockers
    result = inspect_package(
        native_package(
            extension={
                "version": 1,
                "variables": {
                    "COUNT": {
                        "type": "integer",
                        "required": True,
                        "secret": False,
                        "default": 0,
                    }
                },
            }
        )
    )
    assert not result.inventory["skill:review"].activation_blockers


def test_duplicate_ids_do_not_silently_overwrite(native_package):
    root = native_package()
    first = write_component(
        root, "commands/one.md", {"name": "same", "description": "First"}
    )
    second = write_component(
        root, "commands/two.md", {"name": "same", "description": "Second"}
    )
    edit_manifest(root, extensions={NS: {"version": 1, "commands": [first, second]}})
    result = inspect_package(root)
    assert result.inventory["command:same"].support == "invalid"
    assert "duplicate_id" in result.inventory["command:same"].activation_blockers


def test_ambiguous_vendor_candidates_are_not_unioned(native_package):
    root = native_package()
    (root / "plugin.json").unlink()
    for folder in (".cursor-plugin", ".codex-plugin"):
        (root / folder).mkdir()
        (root / folder / "plugin.json").write_text('{"name":"vendor"}')
    result = inspect_package(root)
    assert {c.dialect for c in result.candidates} == {"cursor", "openai"}
    assert result.dialect is None
    assert "dialect_choice_required" in result.activation_blockers
    selected = inspect_package(root, dialect="cursor")
    assert selected.dialect == "cursor"
    assert not selected.inventory
    assert "adapter_unqualified" in selected.activation_blockers


def test_inline_openai_overlay_replaces_compatibility_file(native_package):
    root = native_package()
    (root / ".codex-plugin").mkdir()
    (root / ".codex-plugin" / "plugin.json").write_text('{"name":"overlay"}')
    edit_manifest(root, extensions={NS: {"version": 1}, "com.openai": {"apps": []}})
    result = inspect_package(root)
    assert result.dialect == "portable"
    selected = inspect_package(root, dialect="openai")
    assert selected.overlay_identities == ("plugin.json#/extensions/com.openai",)
    assert selected.root_manifest == "plugin.json"
    assert "adapter_unqualified" in selected.activation_blockers


def test_content_and_effective_identity_are_distinct(native_package):
    root = native_package()
    before = inspect_package(root)
    edit_manifest(root, ignored=True)
    after = inspect_package(root)
    assert before.content_digest != after.content_digest
    assert before.effective_digest == after.effective_digest
    edit_manifest(
        root,
        extensions={NS: {"version": 1, "requires": {"skill:review": ["hook:missing"]}}},
    )
    assert inspect_package(root).effective_digest != after.effective_digest


def test_bad_skill_and_mcp_entry_do_not_disable_siblings(native_package):
    root = native_package()
    bad = root / "skills" / "bad"
    bad.mkdir()
    (bad / "SKILL.md").write_text("---\nname: mismatch\ndescription: Bad\n---\n")
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
                "mcpServers": {
                    "good": {"type": "stdio", "command": "python3"},
                    "bad": {"type": "stdio", "command": "../escape"},
                },
            }
        )
    )
    result = inspect_package(root)
    assert result.inventory["mcp:bad"].support == "invalid"
    assert result.inventory["mcp:good"].support == "supported"
    assert not result.inventory["skill:review"].activation_blockers
    assert "skill:bad" not in result.inventory


@pytest.mark.parametrize(
    "invalid",
    [
        {"headers": {"X-Trace": "a", "x-trace": "b"}},
        {"headers": {"X-Trace": "a\x00b"}},
        {"headers": {"X-Trace": "a\x1fb"}},
        {"headers": {"X-Trace": "a\x7fb"}},
        {"headers": {"X-Trace": " a"}},
        {"headers": {"X-Trace": "a "}},
        {"headers": {"X-Trace": "\ta"}},
        {"headers": {"X-Trace": "a\t"}},
        {"url": "https://@example.com/mcp"},
        {"url": "https://:@example.com/mcp"},
        {"url": "https://example.com/mcp#"},
    ],
)
def test_invalid_remote_mcp_syntax_isolated_from_valid_siblings(
    native_package, invalid
):
    root = native_package()
    bad = {"type": "streamable-http", "url": "https://example.com/mcp"}
    bad.update(invalid)
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
                "mcpServers": {
                    "bad": bad,
                    "good": {
                        "type": "streamable-http",
                        "url": "https://example.com/mcp",
                        "headers": {"X-Empty": "", "X-Interior": "a b\tc"},
                    },
                },
            }
        )
    )
    result = inspect_package(root)
    assert not result.rejected
    assert result.inventory["mcp:bad"].support == "invalid"
    assert result.inventory["mcp:good"].support == "supported"
    assert not result.inventory["skill:review"].activation_blockers
    assert json.loads(result.inventory["mcp:good"].definition_json)["headers"] == {
        "X-Empty": "",
        "X-Interior": "a b\tc",
    }


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/path@name",
        "https://example.com/path%23name?query=%23value",
    ],
)
def test_remote_mcp_url_delimiter_controls_remain_valid(native_package, url):
    root = native_package()
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
                "mcpServers": {"good": {"type": "streamable-http", "url": url}},
            }
        )
    )
    result = inspect_package(root)
    assert result.inventory["mcp:good"].support == "supported"
    assert not result.inventory["skill:review"].activation_blockers


def test_document_and_inventory_limits(native_package, monkeypatch):
    from tldw_chatbook.Plugins import inspection, package_files

    root = native_package()
    monkeypatch.setattr(package_files, "MAX_DOCUMENT_BYTES", 64)
    assert inspect_package(root).rejected
    monkeypatch.setattr(package_files, "MAX_DOCUMENT_BYTES", 256 * 1024)
    edit_manifest(root, ignored=[[[[[]]]]])
    monkeypatch.setattr(package_files, "MAX_JSON_DEPTH", 4)
    assert inspect_package(root).rejected
    monkeypatch.setattr(package_files, "MAX_JSON_DEPTH", 32)
    monkeypatch.setattr(inspection, "MAX_COMPONENTS", 0)
    result = inspect_package(root)
    assert result.rejected and not result.inventory


@pytest.mark.parametrize(
    "raw", [b'{"name":"one","name":"two"}', b'{"name":NaN}', b"[]", b"\xff"]
)
def test_bad_manifest_documents_reject_without_exceptions(native_package, raw):
    root = native_package()
    (root / "plugin.json").write_bytes(raw)
    assert inspect_package(root).rejected


def test_component_limit_precedes_dependency_recursion(native_package, monkeypatch):
    from tldw_chatbook.Plugins import inspection

    root = native_package(requires={"skill:review": ["skill:extra"]})
    child = root / "skills" / "extra"
    child.mkdir()
    (child / "SKILL.md").write_text("---\nname: extra\ndescription: Extra\n---\n")
    monkeypatch.setattr(inspection, "MAX_COMPONENTS", 1)
    result = inspect_package(root)
    assert result.rejected
    assert "component_count_limit" in result.activation_blockers


@pytest.mark.parametrize(
    "field,value", [("tools", None), ("tools", ["shell exec"]), ("permission", "allow")]
)
def test_agent_constraints_fail_closed(field, value, native_package):
    root = native_package()
    fields = {"name": "reviewer", "description": "Review", "tools": []}
    fields[field] = value
    path = write_component(root, "agents/reviewer.md", fields)
    edit_manifest(root, extensions={NS: {"version": 1, "agents": [path]}})
    result = inspect_package(root)
    assert result.inventory["agent:reviewer"].activation_blockers
    assert not result.inventory["skill:review"].activation_blockers


@pytest.mark.parametrize(
    "mutation",
    [
        {"version": 3},
        {"extra": True},
        {
            "hooks": [
                {
                    "id": "guard",
                    "event": "PreToolUse",
                    "type": "command",
                    "argv": ["python3"],
                    "effects": ["allow"],
                    "required": True,
                }
            ]
        },
    ],
)
def test_malformed_required_hooks_never_vanish(mutation, native_package):
    root = native_package()
    folder = root / NS
    folder.mkdir()
    document = {"version": 2, "hooks": []}
    document.update(mutation)
    (folder / "hooks.json").write_text(json.dumps(document))
    edit_manifest(root, extensions={NS: {"version": 1, "hooks": f"./{NS}/hooks.json"}})
    result = inspect_package(root)
    assert result.inventory["skill:review"].activation_blockers


def test_hook_dependencies_preserve_transformer_versus_guard_and_mcp_edge(
    native_package,
):
    root = native_package()
    folder = root / NS
    folder.mkdir()
    (folder / "hooks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "transform",
                        "event": "PreToolUse",
                        "type": "mcp_tool",
                        "server": "repository",
                        "tool": "transform",
                        "effects": ["updated_input", "deny"],
                    }
                ],
            }
        )
    )
    edit_manifest(
        root,
        extensions={
            NS: {
                "version": 1,
                "hooks": f"./{NS}/hooks.json",
                "requires": {"skill:review": ["hook:transform"]},
            }
        },
    )
    result = inspect_package(root)
    assert result.dependency_edges["hook:transform"] == ("mcp:repository",)
    assert result.inventory["skill:review"].activation_blockers
    assert json.loads(result.inventory["hook:transform"].definition_json)[
        "effects"
    ] == ["updated_input", "deny"]


def test_undeclared_native_variables_block_affected_components(native_package):
    root = native_package()
    folder = root / NS
    folder.mkdir()
    (folder / "hooks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "guard",
                        "event": "PreToolUse",
                        "type": "command",
                        "argv": ["python3"],
                        "env": {"AUTH": "${TOKEN}"},
                        "effects": ["deny"],
                    }
                ],
            }
        )
    )
    edit_manifest(
        root,
        extensions={
            NS: {
                "version": 1,
                "hooks": f"./{NS}/hooks.json",
                "requires": {"skill:review": ["hook:guard"]},
            }
        },
    )
    result = inspect_package(root)
    assert (
        "variable_undeclared:TOKEN"
        in result.inventory["hook:guard"].activation_blockers
    )
    assert result.inventory["skill:review"].activation_blockers


@pytest.mark.parametrize("value", [None, [], 17, "bad"])
def test_malformed_optional_hook_fields_are_inspection_results(native_package, value):
    root = native_package()
    folder = root / NS
    folder.mkdir()
    (folder / "hooks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "bad",
                        "event": "PreToolUse",
                        "type": "command",
                        "argv": ["python3"],
                        "env": value,
                        "effects": [],
                    }
                ],
            }
        )
    )
    edit_manifest(root, extensions={NS: {"version": 1, "hooks": f"./{NS}/hooks.json"}})
    result = inspect_package(root)
    assert result.inventory["hook:bad"].support == "invalid"
    assert not result.inventory["skill:review"].activation_blockers


def test_inventory_limit_cannot_be_swallowed_by_mcp_failure_isolation(
    native_package, monkeypatch
):
    from tldw_chatbook.Plugins import inspection

    root = native_package()
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
                "mcpServers": {"one": {"type": "stdio", "command": "python3"}},
            }
        )
    )
    monkeypatch.setattr(inspection, "MAX_COMPONENTS", 1)
    result = inspect_package(root)
    assert result.rejected
    assert "component_count_limit" in result.activation_blockers


def test_duplicate_hook_ids_block_dependents(native_package):
    root = native_package()
    folder = root / NS
    folder.mkdir()
    hook = {
        "id": "guard",
        "event": "PreToolUse",
        "type": "command",
        "argv": ["python3"],
        "effects": ["deny"],
    }
    (folder / "hooks.json").write_text(
        json.dumps({"version": 2, "hooks": [hook, hook]})
    )
    edit_manifest(
        root,
        extensions={
            NS: {
                "version": 1,
                "hooks": f"./{NS}/hooks.json",
                "requires": {"skill:review": ["hook:guard"]},
            }
        },
    )
    result = inspect_package(root)
    assert "duplicate_id" in result.inventory["hook:guard"].activation_blockers
    assert result.inventory["skill:review"].activation_blockers


@pytest.mark.parametrize(
    "field,value",
    [("event", []), ("effects", {}), ("argv", None), ("timeout_seconds", True)],
)
def test_malformed_dependency_hooks_return_blockers(native_package, field, value):
    root = native_package()
    folder = root / NS
    folder.mkdir()
    hook = {
        "id": "guard",
        "event": "PreToolUse",
        "type": "command",
        "argv": ["python3"],
        "effects": ["deny"],
    }
    hook[field] = value
    (folder / "hooks.json").write_text(json.dumps({"version": 2, "hooks": [hook]}))
    edit_manifest(
        root,
        extensions={
            NS: {
                "version": 1,
                "hooks": f"./{NS}/hooks.json",
                "requires": {"skill:review": ["hook:guard"]},
            }
        },
    )
    assert inspect_package(root).inventory["skill:review"].activation_blockers


def test_mcp_root_variable_cannot_prefix_a_different_directory(native_package):
    root = native_package()
    (root / "evil").mkdir()
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
                "mcpServers": {
                    "bad": {
                        "type": "stdio",
                        "command": "python3",
                        "cwd": "${PLUGIN_ROOT}evil",
                    },
                    "good": {
                        "type": "stdio",
                        "command": "python3",
                        "cwd": "${PLUGIN_ROOT}/evil",
                    },
                },
            }
        )
    )
    result = inspect_package(root)
    assert result.inventory["mcp:bad"].support == "invalid"
    assert result.inventory["mcp:good"].support == "supported"


def test_native_hook_relative_executable_cannot_escape(native_package):
    root = native_package()
    folder = root / NS
    folder.mkdir()
    (folder / "hooks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "guard",
                        "event": "PreToolUse",
                        "type": "command",
                        "argv": ["./../outside"],
                        "effects": ["deny"],
                    }
                ],
            }
        )
    )
    edit_manifest(
        root,
        extensions={
            NS: {
                "version": 1,
                "hooks": f"./{NS}/hooks.json",
                "requires": {"skill:review": ["hook:guard"]},
            }
        },
    )
    assert inspect_package(root).inventory["skill:review"].activation_blockers


@pytest.mark.parametrize("description", [chr(0xD800), chr(0xDFFF), float("inf")])
def test_unrepresentable_manifest_values_return_diagnostics(
    native_package, description
):
    control = native_package()
    edit_manifest(control, description="Review \U0001f50d")
    assert not inspect_package(control).rejected
    root = native_package()
    edit_manifest(root, description=description)
    # Exercise a legal JSON exponent that overflows float conversion, not only
    # the explicit Infinity spelling already rejected by the JSON decoder.
    if description == float("inf"):
        path = root / "plugin.json"
        path.write_text(path.read_text().replace("Infinity", "1e999"))
    result = inspect_package(root)
    assert result.rejected
    assert {d.code for d in result.diagnostics} & {
        "json_unicode_invalid",
        "json_nonfinite",
    }


def test_unpaired_surrogate_json_key_returns_diagnostic(native_package):
    root = native_package()
    edit_manifest(root, extensions={"org.example.client": {"\ud800": "value"}})
    result = inspect_package(root)
    assert result.rejected
    assert "json_unicode_invalid" in {d.code for d in result.diagnostics}
    assert not inspect_package(native_package()).rejected


@pytest.mark.parametrize("timeout_json", [str(10**400), "1e999", "-1e999"])
def test_unrepresentable_hook_timeout_is_blocked_without_crashing(
    native_package, timeout_json
):
    root = native_package()
    folder = root / NS
    folder.mkdir()
    path = folder / "hooks.json"
    hook = {
        "id": "guard",
        "event": "PreToolUse",
        "type": "command",
        "argv": ["python3"],
        "effects": ["deny"],
        "timeout_seconds": 10,
    }
    path.write_text(json.dumps({"version": 2, "hooks": [hook]}))
    edit_manifest(
        root,
        extensions={
            NS: {
                "version": 1,
                "hooks": f"./{NS}/hooks.json",
                "requires": {"skill:review": ["hook:guard"]},
            }
        },
    )
    control = inspect_package(root)
    assert not control.inventory["skill:review"].activation_blockers
    path.write_text(
        path.read_text().replace(
            '"timeout_seconds": 10', f'"timeout_seconds": {timeout_json}'
        )
    )
    result = inspect_package(root)
    assert result.inventory["skill:review"].activation_blockers
    assert {d.code for d in result.diagnostics} & {"hook_invalid", "json_nonfinite"}


@pytest.mark.parametrize(
    "shape", ["native", "rejected", "unavailable", "ambiguous", "vendor"]
)
def test_every_inspection_result_has_immutable_mapping_defaults(native_package, shape):
    root = native_package()
    assert "skill:review" in inspect_package(root).inventory
    if shape == "rejected":
        (root / "plugin.json").write_text("{}")
    elif shape == "unavailable":
        root = root / "missing"
    elif shape in {"ambiguous", "vendor"}:
        (root / "plugin.json").unlink()
        folders = (
            (".cursor-plugin", ".codex-plugin")
            if shape == "ambiguous"
            else (".cursor-plugin",)
        )
        for folder in folders:
            (root / folder).mkdir()
            (root / folder / "plugin.json").write_text('{"name":"vendor"}')
    result = inspect_package(root)
    before = result.model_dump()
    for field in ("inventory", "dependency_edges", "link_targets"):
        with pytest.raises(TypeError):
            getattr(result, field)["injected"] = "invalid and mutable"
    assert result.model_dump() == before
