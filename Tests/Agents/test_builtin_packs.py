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


@pytest.mark.asyncio
async def test_glob_files_reports_a_syntactically_invalid_pattern(sandbox):
    """`Path.glob()` validates lazily -- the invalid `**` here doesn't raise
    at construction, only on the first `next()` inside iteration. This is
    not a traversal pattern, so it must reach that iteration to reproduce
    the bug; a naive fix that only wraps the construction call still lets
    this raise uncaught.
    """
    from tldw_chatbook.Agents.builtin_packs.files import GlobFiles

    result = await GlobFiles().execute(pattern="**foo/*")

    assert "error" in result
    assert "matches" not in result


@pytest.mark.asyncio
async def test_grep_files_reports_a_syntactically_invalid_glob(sandbox):
    """Same lazy-validation trap as above, via the `glob` narrowing param."""
    from tldw_chatbook.Agents.builtin_packs.files import GrepFiles

    result = await GrepFiles().execute(pattern="DEBUG", glob="**foo/*")

    assert "error" in result
    assert "matches" not in result


@pytest.mark.asyncio
async def test_glob_files_bounds_examined_candidates(sandbox, monkeypatch):
    """Exercise `_MAX_CANDIDATES` for real, rather than short-circuiting on
    the up-front traversal refusal like every other bound-adjacent test.

    Shrinks the module's `_MAX_CANDIDATES` and builds a tree with more
    entries than that, then asserts the walk is actually cut off -- not
    just that the (much larger) `_MAX_MATCHES` cap alone would explain the
    result.
    """
    import tldw_chatbook.Agents.builtin_packs.files as files_mod

    monkeypatch.setattr(files_mod, "_MAX_CANDIDATES", 5)
    monkeypatch.setattr(files_mod, "_MAX_MATCHES", 1_000)
    for i in range(20):
        (sandbox / f"extra{i}.py").write_text("x = 1\n")

    result = await files_mod.GlobFiles().execute(pattern="**/*.py")

    assert len(result["matches"]) <= 5


@pytest.mark.asyncio
async def test_grep_files_bounds_examined_candidates(sandbox, monkeypatch):
    """Same as above for `GrepFiles`, whose loop also enforces
    `_MAX_CANDIDATES` on files *examined*, independent of `_MAX_MATCHES`
    (matched lines).
    """
    import tldw_chatbook.Agents.builtin_packs.files as files_mod

    monkeypatch.setattr(files_mod, "_MAX_CANDIDATES", 5)
    monkeypatch.setattr(files_mod, "_MAX_MATCHES", 1_000)
    for i in range(20):
        (sandbox / f"extra{i}.py").write_text("DEBUG = True\n")

    result = await files_mod.GrepFiles().execute(pattern="DEBUG", glob="**/*.py")

    assert len(result["matches"]) <= 5


# ---------------------------------------------------------------------------
# Finding 3 (pre-merge review): _tool_sandbox_root() runs outside any try in
# both GlobFiles.execute and GrepFiles.execute. It calls Path.mkdir(parents=
# True), so an unusable configured root (verified for real with the
# "/dev/null/nope" case below -- /dev/null is a file, so mkdir under it raises
# NotADirectoryError) previously raised straight out of execute(), violating
# the "tools never raise, they return an error dict" contract every sibling
# tool (read_file, write_file, list_directory) already honours.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_glob_files_returns_error_dict_when_sandbox_root_is_unusable(monkeypatch):
    import tldw_chatbook.Tools.file_operation_tools as fot
    from tldw_chatbook.Agents.builtin_packs.files import GlobFiles

    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: "/dev/null/nope")

    result = await GlobFiles().execute(pattern="**/*.py")

    assert "error" in result
    assert "matches" not in result


@pytest.mark.asyncio
async def test_grep_files_returns_error_dict_when_sandbox_root_is_unusable(monkeypatch):
    import tldw_chatbook.Tools.file_operation_tools as fot
    from tldw_chatbook.Agents.builtin_packs.files import GrepFiles

    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: "/dev/null/nope")

    result = await GrepFiles().execute(pattern="DEBUG")

    assert "error" in result
    assert "matches" not in result


# ---------------------------------------------------------------------------
# Finding 2 (pre-merge review): grep_files/glob_files called is_within() ->
# is_sensitive_path() once per CANDIDATE, and that helper is deliberately
# uncached -- it resolves 11 config accessors every time so it cannot go
# stale across the test suite's TLDW_CONFIG_PATH switches. Over a 1,530-file
# sandbox that measured ~4.6s (~1.9ms/candidate), ~37s at the 20k candidate
# bound. The fix resolves the sensitive-path set ONCE per tool call and
# reuses it across every candidate. These regressions pin the call count
# structurally; Tests/Agents/test_builtin_packs.py's benchmark script (see
# the final-review-fixes-report) pins the wall-clock improvement.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grep_files_resolves_sensitive_context_once_per_call(sandbox, monkeypatch):
    import tldw_chatbook.Agents.builtin_packs.files as files_mod
    from tldw_chatbook.Utils import sensitive_paths

    real_resolve = sensitive_paths.resolve_sensitive_context
    calls: list[None] = []

    def counting_resolve():
        calls.append(None)
        return real_resolve()

    monkeypatch.setattr(files_mod, "resolve_sensitive_context", counting_resolve)
    for i in range(50):
        (sandbox / f"extra{i}.py").write_text("DEBUG = True\n")

    result = await files_mod.GrepFiles().execute(pattern="DEBUG", glob="**/*.py")

    assert len(calls) == 1, "sensitive-path set must be resolved once per call, not per candidate"
    assert len(result["matches"]) >= 50


@pytest.mark.asyncio
async def test_glob_files_resolves_sensitive_context_once_per_call(sandbox, monkeypatch):
    import tldw_chatbook.Agents.builtin_packs.files as files_mod
    from tldw_chatbook.Utils import sensitive_paths

    real_resolve = sensitive_paths.resolve_sensitive_context
    calls: list[None] = []

    def counting_resolve():
        calls.append(None)
        return real_resolve()

    monkeypatch.setattr(files_mod, "resolve_sensitive_context", counting_resolve)
    for i in range(50):
        (sandbox / f"extra{i}.py").write_text("x = 1\n")

    result = await files_mod.GlobFiles().execute(pattern="**/*.py")

    assert len(calls) == 1, "sensitive-path set must be resolved once per call, not per candidate"
    assert len(result["matches"]) >= 50
