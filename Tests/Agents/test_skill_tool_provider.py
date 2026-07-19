# Tests/Agents/test_skill_tool_provider.py
"""SkillToolProvider (catalog/schema; invoke raises) + intersect_skill_tools.

Pure unit tests: no Skills_Interop import, no DB, no network. The provider
is built from a plain per-run snapshot of skill summaries (dicts), matching
the module's existing import discipline (tool_catalog.py stays importable
without Skills_Interop at module scope).
"""
import pytest

from tldw_chatbook.Agents.tool_catalog import (
    SkillToolProvider, intersect_skill_tools,
)


def test_intersect_none_is_all_builtins():
    assert intersect_skill_tools(None, ["calculator", "get_current_datetime"]) == (
        "calculator", "get_current_datetime")


def test_intersect_narrows_never_grants():
    assert intersect_skill_tools(["calculator", "nonexistent"],
                                 ["calculator", "get_current_datetime"]) == ("calculator",)


def test_intersect_preserves_builtin_order_not_skill_order():
    assert intersect_skill_tools(["get_current_datetime", "calculator"],
                                 ["calculator", "get_current_datetime"]) == (
        "calculator", "get_current_datetime")


def test_intersect_empty_list_yields_empty_tuple():
    assert intersect_skill_tools([], ["calculator", "get_current_datetime"]) == ()


def test_provider_catalog_and_schema():
    prov = SkillToolProvider([
        {"name": "code-review", "description": "Review code", "argument_hint": "[path]"}])
    entry = prov.list_catalog()[0]
    assert (entry.id, entry.name, entry.source) == ("skill:code-review", "code-review", "skill")
    schema = prov.load_schema("skill:code-review")
    assert schema.name == "code-review"
    assert schema.parameters["properties"]["args"]["type"] == "string"
    assert schema.parameters["properties"]["args"]["description"] == "[path]"


def test_provider_schema_falls_back_to_description_when_no_hint():
    prov = SkillToolProvider([
        {"name": "code-review", "description": "Review code", "argument_hint": None}])
    schema = prov.load_schema("skill:code-review")
    assert schema.parameters["properties"]["args"]["description"] == "Review code"
    assert schema.parameters["required"] == []


def test_provider_catalog_empty_when_no_entries():
    assert SkillToolProvider([]).list_catalog() == []


def test_invoke_raises_by_design():
    prov = SkillToolProvider([{"name": "x", "description": "d", "argument_hint": None}])
    with pytest.raises(RuntimeError):
        prov.invoke("skill:x", {"args": "y"})
