# Tests/Agents/test_skill_tool_provider.py
"""SkillToolProvider (catalog/schema; invoke raises) + intersect_skill_tools.

Pure unit tests: no Skills_Interop import, no DB, no network. The provider
is built from a plain per-run snapshot of skill summaries (dicts), matching
the module's existing import discipline (tool_catalog.py stays importable
without Skills_Interop at module scope).
"""

import pytest

from tldw_chatbook.Agents.tool_catalog import (
    SkillToolProvider,
    intersect_skill_tools,
)


def test_intersect_none_is_all_builtins():
    assert intersect_skill_tools(None, ["calculator", "get_current_datetime"]) == (
        "calculator",
        "get_current_datetime",
    )


def test_intersect_narrows_never_grants():
    assert intersect_skill_tools(
        ["calculator", "nonexistent"], ["calculator", "get_current_datetime"]
    ) == ("calculator",)


def test_intersect_preserves_builtin_order_not_skill_order():
    assert intersect_skill_tools(
        ["get_current_datetime", "calculator"], ["calculator", "get_current_datetime"]
    ) == ("calculator", "get_current_datetime")


def test_intersect_empty_list_yields_empty_tuple():
    assert intersect_skill_tools([], ["calculator", "get_current_datetime"]) == ()


def test_provider_catalog_and_schema():
    prov = SkillToolProvider(
        [
            {
                "name": "code-review",
                "description": "Review code",
                "argument_hint": "[path]",
            }
        ]
    )
    entry = prov.list_catalog()[0]
    assert (entry.id, entry.name, entry.source) == (
        "skill:code-review",
        "code-review",
        "skill",
    )
    schema = prov.load_schema("skill:code-review")
    assert schema.name == "code-review"
    assert schema.parameters["properties"]["args"]["type"] == "string"
    assert schema.parameters["properties"]["args"]["description"] == "[path]"


def test_provider_schema_falls_back_to_description_when_no_hint():
    prov = SkillToolProvider(
        [{"name": "code-review", "description": "Review code", "argument_hint": None}]
    )
    schema = prov.load_schema("skill:code-review")
    assert schema.parameters["properties"]["args"]["description"] == "Review code"
    assert schema.parameters["required"] == []


def test_provider_catalog_empty_when_no_entries():
    assert SkillToolProvider([]).list_catalog() == []


def test_invoke_raises_by_design():
    prov = SkillToolProvider([{"name": "x", "description": "d", "argument_hint": None}])
    with pytest.raises(RuntimeError):
        prov.invoke("skill:x", {"args": "y"})


def test_plugin_provider_preserves_manual_only_and_existing_spawn_owner():
    from tldw_chatbook.Plugins.skill_provider import PluginSkillProvider

    provider = PluginSkillProvider(
        [
            {
                "name": "package:review",
                "tool_name": "plugin_review",
                "description": "d",
            },
            {
                "name": "package:manual",
                "tool_name": "plugin_manual",
                "description": "d",
                "disable_model_invocation": True,
            },
        ]
    )
    assert [row.name for row in provider.list_catalog()] == ["plugin_review"]
    with pytest.raises(RuntimeError, match="spawn executor"):
        provider.invoke("skill:plugin_review", {"args": "x"})


def test_plugin_model_name_collision_fails_closed_at_console_composition():
    from tldw_chatbook.Chat.console_agent_bridge import _non_colliding_skill_entries

    entries = [
        {
            "name": "package:one",
            "tool_name": "plugin_collision",
            "description": "one",
            "plugin_owned": True,
        },
        {
            "name": "other:two",
            "tool_name": "plugin_collision",
            "description": "two",
            "plugin_owned": True,
        },
    ]
    assert not _non_colliding_skill_entries({"available_skills": entries}, ())
