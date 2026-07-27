"""`glob_files`/`grep_files` -- Tools/file_operation_tools.py.

Ported from the wt-builtin-tool-packs reference branch's
``Tests/Agents/test_builtin_packs.py`` (glob/grep sections), adapted to
dev's structure: ``GlobFiles``/``GrepFiles`` live directly in
``Tools/file_operation_tools.py`` (no separate `builtin_packs` package), so
there is no `_resolve_sandbox_config`-vs-`_tool_sandbox_root` name-binding
indirection to work around -- patching `fot._resolve_sandbox_config`
reaches every caller directly.
"""

from pathlib import Path

import pytest

import tldw_chatbook.Tools.file_operation_tools as fot
from tldw_chatbook.Tools.file_operation_tools import (
    GlobFiles,
    GrepFiles,
    _rejects_traversal,
)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: str(tmp_path))
    (tmp_path / "a.py").write_text("import os\nDEBUG = True\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("DEBUG = False\n")
    (tmp_path / "notes.md").write_text("nothing here\n")
    return tmp_path


@pytest.mark.asyncio
async def test_glob_files_matches_recursively_within_the_sandbox(sandbox):
    result = await GlobFiles().execute(pattern="**/*.py")

    assert sorted(Path(p).name for p in result["matches"]) == ["a.py", "b.py"]


@pytest.mark.asyncio
async def test_grep_files_reports_matching_lines(sandbox):
    result = await GrepFiles().execute(pattern="DEBUG", glob="**/*.py")

    assert len(result["matches"]) == 2
    assert all("DEBUG" in m["line"] for m in result["matches"])
    assert all(m["line_number"] >= 1 for m in result["matches"])


@pytest.mark.asyncio
async def test_grep_files_rejects_a_bad_regex_without_raising(sandbox):
    result = await GrepFiles().execute(pattern="([", glob="**/*.py")

    assert "error" in result


@pytest.mark.asyncio
async def test_glob_files_refuses_parent_traversal(sandbox):
    """`Path.glob('../**/*')` does not raise -- it yields ~1.4M paths.

    Filtering by containment afterwards still walks all of them, so the
    pattern is refused up front.
    """
    result = await GlobFiles().execute(pattern="../**/*")

    assert "error" in result
    assert "matches" not in result


@pytest.mark.asyncio
async def test_glob_files_refuses_absolute_patterns(sandbox):
    result = await GlobFiles().execute(pattern="/etc/*")

    assert "error" in result


@pytest.mark.asyncio
async def test_grep_files_refuses_parent_traversal(sandbox):
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
    result = await GlobFiles().execute(pattern="**foo/*")

    assert "error" in result
    assert "matches" not in result


@pytest.mark.asyncio
async def test_grep_files_reports_a_syntactically_invalid_glob(sandbox):
    """Same lazy-validation trap as above, via the `glob` narrowing param."""
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
    monkeypatch.setattr(fot, "_MAX_CANDIDATES", 5)
    monkeypatch.setattr(fot, "_MAX_MATCHES", 1_000)
    for i in range(20):
        (sandbox / f"extra{i}.py").write_text("x = 1\n")

    result = await fot.GlobFiles().execute(pattern="**/*.py")

    assert len(result["matches"]) <= 5


@pytest.mark.asyncio
async def test_grep_files_bounds_examined_candidates(sandbox, monkeypatch):
    """Same as above for `GrepFiles`, whose loop also enforces
    `_MAX_CANDIDATES` on files *examined*, independent of `_MAX_MATCHES`
    (matched lines).
    """
    monkeypatch.setattr(fot, "_MAX_CANDIDATES", 5)
    monkeypatch.setattr(fot, "_MAX_MATCHES", 1_000)
    for i in range(20):
        (sandbox / f"extra{i}.py").write_text("DEBUG = True\n")

    result = await fot.GrepFiles().execute(pattern="DEBUG", glob="**/*.py")

    assert len(result["matches"]) <= 5


# ---------------------------------------------------------------------------
# _tool_sandbox_root() runs outside any try in both GlobFiles.execute and
# GrepFiles.execute. It calls Path.mkdir(parents=True), so an unusable
# configured root (verified for real with the "/dev/null/nope" case below --
# /dev/null is a file, so mkdir under it raises NotADirectoryError) must not
# raise straight out of execute(): every sibling tool (read_file, write_file,
# list_directory) returns an error dict instead of raising.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_glob_files_returns_error_dict_when_sandbox_root_is_unusable(monkeypatch):
    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: "/dev/null/nope")

    result = await GlobFiles().execute(pattern="**/*.py")

    assert "error" in result
    assert "matches" not in result


@pytest.mark.asyncio
async def test_grep_files_returns_error_dict_when_sandbox_root_is_unusable(monkeypatch):
    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: "/dev/null/nope")

    result = await GrepFiles().execute(pattern="DEBUG")

    assert "error" in result
    assert "matches" not in result


# ---------------------------------------------------------------------------
# grep_files/glob_files call is_within() -> is_sensitive_path() once per
# CANDIDATE if not careful, and that helper is deliberately uncached -- it
# resolves 11 config accessors every time so it cannot go stale across the
# test suite's TLDW_CONFIG_PATH switches. The fix resolves the sensitive-path
# set ONCE per tool call and reuses it across every candidate; these
# regressions pin the call count structurally.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grep_files_resolves_sensitive_context_once_per_call(sandbox, monkeypatch):
    from tldw_chatbook.Utils import sensitive_paths

    real_resolve = sensitive_paths.resolve_sensitive_context
    calls: list[None] = []

    def counting_resolve():
        calls.append(None)
        return real_resolve()

    monkeypatch.setattr(fot, "resolve_sensitive_context", counting_resolve)
    for i in range(50):
        (sandbox / f"extra{i}.py").write_text("DEBUG = True\n")

    result = await fot.GrepFiles().execute(pattern="DEBUG", glob="**/*.py")

    assert len(calls) == 1, "sensitive-path set must be resolved once per call, not per candidate"
    assert len(result["matches"]) >= 50


@pytest.mark.asyncio
async def test_glob_files_resolves_sensitive_context_once_per_call(sandbox, monkeypatch):
    from tldw_chatbook.Utils import sensitive_paths

    real_resolve = sensitive_paths.resolve_sensitive_context
    calls: list[None] = []

    def counting_resolve():
        calls.append(None)
        return real_resolve()

    monkeypatch.setattr(fot, "resolve_sensitive_context", counting_resolve)
    for i in range(50):
        (sandbox / f"extra{i}.py").write_text("x = 1\n")

    result = await fot.GlobFiles().execute(pattern="**/*.py")

    assert len(calls) == 1, "sensitive-path set must be resolved once per call, not per candidate"
    assert len(result["matches"]) >= 50


# ---------------------------------------------------------------------------
# `glob_files`/`grep_files` filtered candidates with `is_within()` only,
# which applies the credential/app-state denylist but NOT the
# hidden-component rule `Utils.path_validation.validate_path` enforces for
# `read_file`/`write_file`. Live repro pre-fix: `read_file('.env')` was
# refused ("Access to hidden files/directories is not allowed") while
# `grep_files('API_KEY', glob='**/.env')` returned the secret line -- an
# exploitable inconsistency even though `.env` is not on the
# `Utils/sensitive_paths.py` denylist (nor does it need to be; the
# hidden-component rule alone is what `read_file` relies on). These pin the
# fix: a dotfile/dotdir inside the sandbox must be invisible to
# `glob_files` and unreadable by `grep_files`, mirroring `read_file`'s
# refusal.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_glob_files_hides_a_dotfile_in_the_sandbox(sandbox):
    (sandbox / ".env").write_text("API_KEY=supersecret123\n")

    result = await GlobFiles().execute(pattern="**/*")

    assert ".env" not in {Path(p).name for p in result["matches"]}


@pytest.mark.asyncio
async def test_glob_files_hides_a_file_under_a_dotted_directory(sandbox):
    """The hidden-component rule applies to any dotted ancestor, not just a
    dotted leaf name -- e.g. a secret sitting inside `.git/`.
    """
    (sandbox / ".git").mkdir()
    (sandbox / ".git" / "config").write_text("[core]\n")

    result = await GlobFiles().execute(pattern="**/*")

    assert "config" not in {Path(p).name for p in result["matches"]}


@pytest.mark.asyncio
async def test_grep_files_cannot_read_a_dotfile_in_the_sandbox(sandbox):
    """Reproduces the exact live finding: `grep_files('API_KEY',
    glob='**/.env')` must no longer surface the secret line.
    """
    (sandbox / ".env").write_text("API_KEY=supersecret123\n")

    result = await GrepFiles().execute(pattern="API_KEY", glob="**/.env")

    assert result["matches"] == []


@pytest.mark.asyncio
async def test_grep_files_cannot_read_a_dotfile_via_a_broad_glob(sandbox):
    """Same as above, but via the tool's own default glob (`**/*`) rather
    than a glob that names the dotfile explicitly -- the broader, more
    realistic case an LLM would actually issue.

    Uses a unique token rather than ``API_KEY``: the isolated test HOME's
    own generated ``config.toml`` legitimately contains the substring
    ``API_KEY`` (as ``OPENAI_API_KEY``) in a real, non-hidden file the
    sandbox also happens to contain, which would make a broad-glob search
    for ``API_KEY`` alone match regardless of this fix.
    """
    (sandbox / ".env").write_text("SUPER_UNIQUE_SECRET_TOKEN_4f8a1c\n")

    result = await GrepFiles().execute(pattern="SUPER_UNIQUE_SECRET_TOKEN_4f8a1c")

    assert result["matches"] == []


# ---------------------------------------------------------------------------
# `_rejects_traversal()` must recognize Windows drive-letter (`C:\...`) and
# UNC (`\\server\share\...`) absolute forms too, not just a leading `/` --
# an OS-dependent gap (`is_within` still guards every candidate regardless,
# so this was a cost/consistency issue, not an escape).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_glob_files_refuses_windows_drive_letter_pattern(sandbox):
    result = await GlobFiles().execute(pattern="C:\\Windows\\System32\\*")

    assert "error" in result
    assert "matches" not in result


@pytest.mark.asyncio
async def test_glob_files_refuses_windows_unc_pattern(sandbox):
    result = await GlobFiles().execute(pattern="\\\\server\\share\\*")

    assert "error" in result
    assert "matches" not in result


@pytest.mark.asyncio
async def test_grep_files_refuses_windows_drive_letter_glob(sandbox):
    result = await GrepFiles().execute(pattern="DEBUG", glob="C:\\Windows\\System32\\*")

    assert "error" in result
    assert "matches" not in result


@pytest.mark.asyncio
async def test_grep_files_refuses_windows_unc_glob(sandbox):
    result = await GrepFiles().execute(pattern="DEBUG", glob="\\\\server\\share\\*")

    assert "error" in result
    assert "matches" not in result


def test_rejects_traversal_recognizes_windows_absolute_forms():
    """Direct unit test of the helper itself, independent of the tools."""
    assert _rejects_traversal("C:\\Users\\x\\file.txt") is True
    assert _rejects_traversal("\\\\server\\share\\file.txt") is True
    assert _rejects_traversal("relative/path.txt") is False


# ---------------------------------------------------------------------------
# `grep_files` must stream rather than `read_text()` a whole file. Now
# streamed line by line, with a per-file byte cap (`_MAX_GREP_FILE_BYTES`)
# bounding the worst case for a single pathological file with no newline
# characters.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grep_files_skips_a_file_over_the_size_cap(sandbox, monkeypatch):
    monkeypatch.setattr(fot, "_MAX_GREP_FILE_BYTES", 10)
    (sandbox / "big.py").write_text("DEBUG = True\n" * 5)

    result = await fot.GrepFiles().execute(pattern="DEBUG", glob="big.py")

    assert result["matches"] == []


@pytest.mark.asyncio
async def test_grep_files_still_matches_within_the_size_cap(sandbox, monkeypatch):
    monkeypatch.setattr(fot, "_MAX_GREP_FILE_BYTES", 10_000)
    (sandbox / "small.py").write_text("DEBUG = True\n")

    result = await fot.GrepFiles().execute(pattern="DEBUG", glob="small.py")

    assert len(result["matches"]) == 1
    assert result["matches"][0]["line"] == "DEBUG = True"


# ---------------------------------------------------------------------------
# Tool-level integration: sensitive-path denial reachable through
# glob_files/grep_files too, with the sandbox root configured to CONTAIN the
# denied path -- the one configuration in which the bug is observable (see
# the matching comment block in test_file_tool_sandbox.py).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_glob_files_hides_this_apps_own_sqlite_db(monkeypatch):
    from tldw_chatbook import config as app_config

    db_path = app_config.get_chachanotes_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text("marker")

    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: db_path.parent.resolve())

    result = await GlobFiles().execute(pattern="**/*")

    assert db_path.name not in {Path(p).name for p in result["matches"]}


@pytest.mark.asyncio
async def test_grep_files_cannot_read_this_apps_own_sqlite_db_wal_sidecar(monkeypatch):
    from tldw_chatbook import config as app_config

    db_path = app_config.get_chachanotes_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    wal_path = db_path.with_name(db_path.name + "-wal")
    wal_path.write_text("recent-uncommitted-row-marker")

    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: db_path.parent.resolve())

    result = await GrepFiles().execute(pattern="marker")

    assert result["matches"] == []
