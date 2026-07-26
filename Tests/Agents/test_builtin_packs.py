from pathlib import Path

import pytest

from tldw_chatbook.Agents.builtin_services import BuiltinToolServices


def test_files_pack_declares_its_tools_and_no_optional_deps():
    from tldw_chatbook.Agents.builtin_packs import files

    assert files.REQUIRES == ()
    assert {c.__name__ for c in files.TOOLS} == {
        "ReadFile",
        "ListDirectory",
        "GlobFiles",
        "GrepFiles",
    }


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
    assert len(pack_tool_classes(frozenset({"files"}))) == 4
    assert pack_tool_classes(frozenset({"nope"})) == ()


def test_services_are_accepted_but_unused_by_file_tools():
    from tldw_chatbook.Agents.builtin_packs import files

    tool = files.ReadFile(services=BuiltinToolServices())
    assert tool.name == "read_file"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point the file-tool sandbox root at a temp dir.

    Patches ``_resolve_sandbox_config`` rather than ``_tool_sandbox_root``.
    Most of ``Tests/Tools/test_file_tool_sandbox.py`` patches
    ``_tool_sandbox_root`` directly, which works there because those tools
    (``ReadFileTool``/``ListDirectoryTool``) call it as a same-module global
    inside ``file_operation_tools.py``. ``GlobFiles``/``GrepFiles`` instead
    do ``from ...file_operation_tools import _tool_sandbox_root`` -- a
    separate name binding in ``builtin_packs.files``'s own namespace, so
    reassigning the attribute on the ``file_operation_tools`` module would
    not reach it. ``_tool_sandbox_root()`` itself, wherever it is called
    from, still looks up ``_resolve_sandbox_config`` as a global inside
    ``file_operation_tools`` -- patching that name is what actually reaches
    every caller.
    """
    import tldw_chatbook.Tools.file_operation_tools as fot

    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: str(tmp_path))
    (tmp_path / "a.py").write_text("import os\nDEBUG = True\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("DEBUG = False\n")
    (tmp_path / "notes.md").write_text("nothing here\n")
    return tmp_path


@pytest.mark.asyncio
async def test_glob_files_matches_recursively_within_the_sandbox(sandbox):
    from tldw_chatbook.Agents.builtin_packs.files import GlobFiles

    result = await GlobFiles().execute(pattern="**/*.py")

    assert sorted(Path(p).name for p in result["matches"]) == ["a.py", "b.py"]


@pytest.mark.asyncio
async def test_grep_files_reports_matching_lines(sandbox):
    from tldw_chatbook.Agents.builtin_packs.files import GrepFiles

    result = await GrepFiles().execute(pattern="DEBUG", glob="**/*.py")

    assert len(result["matches"]) == 2
    assert all("DEBUG" in m["line"] for m in result["matches"])
    assert all(m["line_number"] >= 1 for m in result["matches"])


@pytest.mark.asyncio
async def test_grep_files_rejects_a_bad_regex_without_raising(sandbox):
    from tldw_chatbook.Agents.builtin_packs.files import GrepFiles

    result = await GrepFiles().execute(pattern="([", glob="**/*.py")

    assert "error" in result


@pytest.mark.asyncio
async def test_glob_files_refuses_parent_traversal(sandbox):
    """`Path.glob('../**/*')` does not raise -- it yields ~1.4M paths.

    Filtering by containment afterwards still walks all of them, so the
    pattern is refused up front.
    """
    from tldw_chatbook.Agents.builtin_packs.files import GlobFiles

    result = await GlobFiles().execute(pattern="../**/*")

    assert "error" in result
    assert "matches" not in result


@pytest.mark.asyncio
async def test_glob_files_refuses_absolute_patterns(sandbox):
    from tldw_chatbook.Agents.builtin_packs.files import GlobFiles

    result = await GlobFiles().execute(pattern="/etc/*")

    assert "error" in result


@pytest.mark.asyncio
async def test_grep_files_refuses_parent_traversal(sandbox):
    from tldw_chatbook.Agents.builtin_packs.files import GrepFiles

    result = await GrepFiles().execute(pattern="DEBUG", glob="../**/*.py")

    assert "error" in result
