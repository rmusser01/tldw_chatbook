import pytest

from tldw_chatbook.Agents.builtin_services import BuiltinToolServices


def test_files_pack_declares_its_tools_and_no_optional_deps():
    from tldw_chatbook.Agents.builtin_packs import files

    assert files.REQUIRES == ()
    assert {c.__name__ for c in files.TOOLS} == {"ReadFile", "ListDirectory"}


def test_every_pack_tool_constructs_with_services_none():
    """The metadata contract: enumeration never has live services."""
    from tldw_chatbook.Agents.builtin_packs import PACKS

    for pack in PACKS.values():
        for cls in pack.TOOLS:
            tool = cls(services=None)
            assert isinstance(tool.name, str) and tool.name
            assert isinstance(tool.description, str) and tool.description
            assert isinstance(tool.parameters, dict)
            assert isinstance(tool.risk_tags, tuple)


def test_pack_tool_classes_returns_only_enabled_packs():
    from tldw_chatbook.Agents.builtin_packs import pack_tool_classes

    assert pack_tool_classes(frozenset()) == ()
    assert len(pack_tool_classes(frozenset({"files"}))) == 2
    assert pack_tool_classes(frozenset({"nope"})) == ()


def test_services_are_accepted_but_unused_by_file_tools():
    from tldw_chatbook.Agents.builtin_packs import files

    tool = files.ReadFile(services=BuiltinToolServices())
    assert tool.name == "read_file"
