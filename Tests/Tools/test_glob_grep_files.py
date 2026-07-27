"""`glob_files`/`grep_files` -- Tools/file_operation_tools.py.

Ported from the wt-builtin-tool-packs reference branch's
``Tests/Agents/test_builtin_packs.py`` (glob/grep sections), adapted to
dev's structure: ``GlobFiles``/``GrepFiles`` live directly in
``Tools/file_operation_tools.py`` (no separate `builtin_packs` package), so
there is no `_resolve_sandbox_config`-vs-`_tool_sandbox_root` name-binding
indirection to work around -- patching `fot._resolve_sandbox_config`
reaches every caller directly.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import psutil
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


# ---------------------------------------------------------------------------
# Finding 4 (substrate review): a dotted `[tools] file_sandbox_root` (e.g.
# `~/.tldw_sandbox`) inverts the hidden-file protection. `read_file`/
# `write_file`/`list_directory` all route through `validate_path_multi` ->
# `validate_path`, which refuses EVERY candidate once the root's own final
# component is dotted (`path_validation.py`'s "hidden base directory"
# check) -- an over-broad refusal, but the safe direction. `glob_files`/
# `grep_files` instead glob `_tool_sandbox_root()` directly and never passed
# through that check at all, so a PLAIN, non-hidden file sitting directly in
# a dotted root was enumerated/read normally -- live-reproduced pre-fix:
# `grep_files` returned "API_KEY=sk-live-abc123" from inside `.tldw_sandbox`
# while `read_file` refused the identical path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_glob_files_refuses_a_dotted_sandbox_root(tmp_path, monkeypatch):
    dotted_root = tmp_path / ".tldw_sandbox"
    dotted_root.mkdir()
    (dotted_root / "secrets.txt").write_text("API_KEY=sk-live-abc123\n")
    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: str(dotted_root))

    result = await GlobFiles().execute(pattern="**/*")

    assert "error" in result
    assert "matches" not in result


@pytest.mark.asyncio
async def test_grep_files_refuses_a_dotted_sandbox_root(tmp_path, monkeypatch):
    """The exact live finding: `grep_files('API_KEY')` against a dotted
    sandbox root must no longer surface the secret line.
    """
    dotted_root = tmp_path / ".tldw_sandbox"
    dotted_root.mkdir()
    (dotted_root / "secrets.txt").write_text("API_KEY=sk-live-abc123\n")
    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: str(dotted_root))

    result = await GrepFiles().execute(pattern="API_KEY")

    assert "error" in result
    assert "matches" not in result
    assert "sk-live-abc123" not in str(result)


@pytest.mark.asyncio
async def test_glob_files_consistent_with_read_file_on_a_dotted_root(tmp_path, monkeypatch):
    """Both must refuse -- neither leaking (glob_files) nor over-refusing
    silently different from its sibling (read_file). Pins the two tools to
    the SAME observable behavior on the identical misconfiguration.
    """
    dotted_root = tmp_path / ".tldw_sandbox"
    dotted_root.mkdir()
    (dotted_root / "note.txt").write_text("hello\n")
    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: str(dotted_root))

    glob_result = await GlobFiles().execute(pattern="**/*")
    read_result = await fot.ReadFileTool().execute(file_path="note.txt")

    assert "error" in glob_result
    assert "error" in read_result


# ---------------------------------------------------------------------------
# Finding 1 (PR #953 review): `re.compile(pattern).search(line)` ran against
# the FULL line, and Python's `re` has no match timeout. A
# catastrophic-backtracking pattern (e.g. `(a+)+$`) burns CPU superlinearly
# in input length, and since a timed-out tool call ABANDONS its worker
# thread rather than killing it (`Agents/agent_service.py`'s
# `_call_with_timeout` -- Python cannot kill a thread), that CPU burn
# outlives the agent's own timeout report. `_MAX_GREP_LINE_SEARCH_CHARS`
# bounds what `regex.search` actually sees; `_MAX_GREP_LINES_SCANNED` bounds
# the total lines read across the whole call; `GrepFiles.timeout_seconds`
# gives the run loop a much tighter ceiling than the run's own default.
#
# Locally reproduced (see the grep-dos-fix report for the full numbers):
# `re.compile(r"(a+)+$").search("a" * 28 + "X\n")` -- the FULL, uncapped
# line used below -- took ~11.7s and still found no match; capped to the
# first 10 characters (all still `a`, the `X` falls outside the slice) it
# matches in under a millisecond. Both cap tests below were verified by
# mutation: temporarily undoing the corresponding cap in
# `file_operation_tools.py` and re-running each test in isolation makes it
# fail (the line-cap test exceeds its time budget; the total-scan test
# returns far more than the capped count) -- not shipped in this suite,
# since a genuinely uncapped run of the line-cap scenario would hang the
# test for the better part of a minute.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grep_files_search_input_is_bounded_to_a_capped_line_length(sandbox, monkeypatch):
    """A pattern that is pathological against the FULL line completes
    quickly once the regex only ever sees a length-capped slice of it.
    """
    monkeypatch.setattr(fot, "_MAX_GREP_LINE_SEARCH_CHARS", 10)
    (sandbox / "danger.txt").write_text("a" * 28 + "X\n")

    start = time.perf_counter()
    result = await fot.GrepFiles().execute(pattern=r"(a+)+$", glob="danger.txt")
    elapsed = time.perf_counter() - start

    # The capped slice ("a" * 10, the "X" falls outside it) matches
    # trivially and fast. Without the cap, the full line -- 28 a's then a
    # mismatching "X" -- forces exhaustive backtracking that (a) takes
    # over ten seconds and (b) still finds no match at all, since no
    # position in the full line is immediately followed by end-of-string.
    # Either symptom alone would catch a removed cap; asserting both pins
    # the fix precisely rather than one that could pass by coincidence.
    assert elapsed < 2.0, f"regex.search took {elapsed:.2f}s -- is the line-length cap applied?"
    assert len(result["matches"]) == 1


@pytest.mark.asyncio
async def test_grep_files_bounds_total_lines_scanned_across_the_whole_call(sandbox, monkeypatch):
    """`_MAX_GREP_LINES_SCANNED` bounds AGGREGATE lines read across every
    file in one invocation. `_MAX_MATCHES` and `_MAX_CANDIDATES` are both
    set generously high here so neither of those -- not this new cap --
    can explain a bounded result.
    """
    monkeypatch.setattr(fot, "_MAX_GREP_LINES_SCANNED", 30)
    monkeypatch.setattr(fot, "_MAX_MATCHES", 10_000)
    monkeypatch.setattr(fot, "_MAX_CANDIDATES", 10_000)
    for i in range(5):
        (sandbox / f"lines{i}.txt").write_text("x\n" * 20)

    result = await fot.GrepFiles().execute(pattern="x", glob="lines*.txt")

    # 5 files * 20 matching lines = 100 lines available; the scan must
    # stop at exactly the 30-line cap rather than returning all of them.
    assert len(result["matches"]) == 30


def test_grep_files_declares_a_nonzero_timeout():
    """`grep_files` must override the `Tool.timeout_seconds` 0.0 default --
    a call that never times out never triggers `_call_with_timeout`'s
    abandon-the-thread path in the first place, but nor does it ever hand
    back control to the run loop.
    """
    assert GrepFiles().timeout_seconds > 0.0


def test_grep_files_timeout_resolves_through_the_tool_catalog_registry():
    """The per-tool override seam this PR adds
    (`Tool.timeout_seconds` -> `BuiltinToolProvider.timeout_for` ->
    `ToolCatalogRegistry.timeout_for`) must actually carry `GrepFiles`'s
    value end to end -- not just be readable on the class directly.
    """
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry

    provider = BuiltinToolProvider()
    provider._tools["grep_files"] = GrepFiles()
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)

    assert registry.timeout_for("grep_files") == GrepFiles().timeout_seconds
    assert registry.timeout_for("grep_files") == 20.0


# ---------------------------------------------------------------------------
# TASK-843: the regex search itself now runs in a killable child process
# (`_run_grep_subprocess` / `_grep_worker.py`) rather than in-process, so a
# catastrophic-backtracking pattern's CPU burn is bounded even AFTER the
# tool call has returned -- unlike the run loop's own thread-based
# `_call_with_timeout` (`Agents/agent_service.py`), which abandons a hung
# worker thread rather than killing it, because Python cannot forcibly kill
# a thread. These pin that guarantee directly, at the process level.
# ---------------------------------------------------------------------------


def test_grep_subprocess_kills_the_worker_process_on_timeout(tmp_path, monkeypatch):
    """The core TASK-843 guarantee: once the internal subprocess timeout
    elapses, the child is actually killed -- not merely abandoned. Proven
    two ways: (a) the call returns close to the timeout, not the many
    seconds the pathological pattern would otherwise cost, and (b) the
    captured child pid no longer exists immediately after return.
    """
    real_popen = subprocess.Popen
    captured: dict = {}

    def spying_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        captured["pid"] = proc.pid
        return proc

    monkeypatch.setattr(fot.subprocess, "Popen", spying_popen)

    danger = tmp_path / "danger.txt"
    danger.write_text("a" * 28 + "X\n")

    start = time.perf_counter()
    result = fot._run_grep_subprocess(
        pattern=r"(a+)+$",
        file_paths=[str(danger)],
        max_matches=fot._MAX_MATCHES,
        # Deliberately large so the FULL pathological line reaches
        # regex.search inside the worker, unclamped by the line-length
        # cap -- isolates what the subprocess boundary alone buys,
        # independent of _MAX_GREP_LINE_SEARCH_CHARS.
        max_line_search_chars=10_000,
        max_lines_scanned=fot._MAX_GREP_LINES_SCANNED,
        max_file_bytes=fot._MAX_GREP_FILE_BYTES,
        timeout_seconds=1.5,
    )
    elapsed = time.perf_counter() - start

    assert "error" in result
    assert "timed out" in result["error"]
    # Bounded near the 1.5s ceiling -- NOT the ~11.7s this exact pattern
    # takes to complete uncapped (see the module-level comment on
    # `_MAX_GREP_LINE_SEARCH_CHARS` for that measurement).
    assert elapsed < 4.0, f"took {elapsed:.2f}s -- is the subprocess actually being killed?"
    assert "pid" in captured
    assert not psutil.pid_exists(captured["pid"]), (
        "child process must be killed on timeout, not left running"
    )


@pytest.mark.asyncio
async def test_grep_files_execute_survives_a_pathological_pattern_without_raising(
    sandbox, monkeypatch
):
    """End-to-end through the async `GrepFiles.execute()` -- never raises,
    reports a timeout error, and returns well within a bounded wall-clock
    budget even for a pattern that would otherwise run for many seconds.
    """
    monkeypatch.setattr(fot, "_GREP_SUBPROCESS_TIMEOUT_SECONDS", 1.5)
    (sandbox / "danger.txt").write_text("a" * 28 + "X\n")

    start = time.perf_counter()
    result = await fot.GrepFiles().execute(pattern=r"(a+)+$", glob="danger.txt")
    elapsed = time.perf_counter() - start

    assert "error" in result
    assert elapsed < 4.0


def test_run_grep_subprocess_returns_error_dict_when_popen_cannot_start(monkeypatch):
    """Never raises, even if the OS refuses to spawn the worker at all."""

    def boom(*args, **kwargs):
        raise OSError("no more processes")

    monkeypatch.setattr(fot.subprocess, "Popen", boom)

    result = fot._run_grep_subprocess(
        pattern="x",
        file_paths=["/tmp/does-not-matter.txt"],
        max_matches=200,
        max_line_search_chars=500,
        max_lines_scanned=200_000,
        max_file_bytes=5_000_000,
        timeout_seconds=5.0,
    )

    assert "error" in result


def test_run_grep_subprocess_reports_a_nonzero_worker_exit(monkeypatch):
    """A worker that exits nonzero (crash, not a normal error-JSON path) is
    reported as a plain error dict, never raised.
    """

    class _FakeProc:
        pid = 999999
        returncode = 1

        def communicate(self, input=None, timeout=None):
            return "", "boom: unhandled exception in worker"

        def kill(self):
            pass

    monkeypatch.setattr(fot.subprocess, "Popen", lambda *a, **k: _FakeProc())

    result = fot._run_grep_subprocess(
        pattern="x",
        file_paths=[],
        max_matches=200,
        max_line_search_chars=500,
        max_lines_scanned=200_000,
        max_file_bytes=5_000_000,
        timeout_seconds=5.0,
    )

    assert "error" in result
    assert "boom" in result["error"]


def test_run_grep_subprocess_reports_malformed_worker_output(monkeypatch):
    """Non-JSON (or non-dict-JSON) stdout from the worker is a plain error,
    never an uncaught exception.
    """

    class _FakeProc:
        pid = 999998
        returncode = 0

        def communicate(self, input=None, timeout=None):
            return "not valid json {{{", ""

        def kill(self):
            pass

    monkeypatch.setattr(fot.subprocess, "Popen", lambda *a, **k: _FakeProc())

    result = fot._run_grep_subprocess(
        pattern="x",
        file_paths=[],
        max_matches=200,
        max_line_search_chars=500,
        max_lines_scanned=200_000,
        max_file_bytes=5_000_000,
        timeout_seconds=5.0,
    )

    assert "error" in result


@pytest.mark.asyncio
async def test_grep_files_ordinary_search_still_delegates_to_a_subprocess(
    sandbox, monkeypatch
):
    """Sanity check that the normal (non-pathological) path still routes
    through `_run_grep_subprocess` -- i.e. this isn't accidentally bypassed
    for the common case, which would silently regress the TASK-843 fix.
    """
    calls: list[str] = []
    real = fot._run_grep_subprocess

    def spy(pattern, file_paths, **kwargs):
        calls.append(pattern)
        return real(pattern, file_paths, **kwargs)

    monkeypatch.setattr(fot, "_run_grep_subprocess", spy)

    result = await fot.GrepFiles().execute(pattern="DEBUG", glob="**/*.py")

    assert calls == ["DEBUG"]
    assert len(result["matches"]) == 2


# ---------------------------------------------------------------------------
# `_grep_worker.py` itself: a standalone script, deliberately with no import
# of `tldw_chatbook`. Exercised directly (both as a plain function and as a
# real subprocess) so its own error handling is pinned independently of
# `_run_grep_subprocess`.
# ---------------------------------------------------------------------------


def _load_worker_module():
    import importlib.util

    worker_path = Path(fot.__file__).with_name("_grep_worker.py")
    spec = importlib.util.spec_from_file_location("_grep_worker_under_test", worker_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grep_worker_run_search_matches_ordinary_content(tmp_path):
    worker = _load_worker_module()
    target = tmp_path / "a.txt"
    target.write_text("hello DEBUG world\n")

    result = worker.run_search(
        {
            "pattern": "DEBUG",
            "file_paths": [str(target)],
            "max_matches": 200,
            "max_line_search_chars": 500,
            "max_lines_scanned": 200_000,
            "max_file_bytes": 5_000_000,
        }
    )

    assert len(result["matches"]) == 1
    assert result["matches"][0]["line"] == "hello DEBUG world"


def test_grep_worker_run_search_rejects_invalid_regex():
    worker = _load_worker_module()
    result = worker.run_search(
        {
            "pattern": "([",
            "file_paths": [],
            "max_matches": 200,
            "max_line_search_chars": 500,
            "max_lines_scanned": 200_000,
            "max_file_bytes": 5_000_000,
        }
    )
    assert "error" in result


def test_grep_worker_run_search_handles_a_malformed_request():
    worker = _load_worker_module()
    result = worker.run_search({"pattern": "x"})  # missing required keys
    assert "error" in result


def test_grep_worker_script_reports_malformed_json_on_stdin_without_crashing():
    """Exercises the worker as a REAL subprocess -- confirms it always
    exits 0 and always writes valid JSON, even fed garbage.
    """
    import sys

    worker_path = Path(fot.__file__).with_name("_grep_worker.py")
    proc = subprocess.run(
        [sys.executable, "-S", str(worker_path)],
        input="not json {{{",
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert proc.returncode == 0
    parsed = json.loads(proc.stdout)
    assert "error" in parsed


def test_grep_worker_script_end_to_end_via_real_subprocess(tmp_path):
    """The worker script invoked exactly the way `_run_grep_subprocess`
    invokes it in production -- proves the protocol works, not just the
    in-process `run_search` function.
    """
    import sys

    target = tmp_path / "notes.txt"
    target.write_text("first line\nDEBUG marker line\nlast line\n")
    worker_path = Path(fot.__file__).with_name("_grep_worker.py")
    request = json.dumps(
        {
            "pattern": "DEBUG",
            "file_paths": [str(target)],
            "max_matches": 200,
            "max_line_search_chars": 500,
            "max_lines_scanned": 200_000,
            "max_file_bytes": 5_000_000,
        }
    )
    proc = subprocess.run(
        [sys.executable, "-S", str(worker_path)],
        input=request,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert proc.returncode == 0
    parsed = json.loads(proc.stdout)
    assert len(parsed["matches"]) == 1
    assert parsed["matches"][0]["line"] == "DEBUG marker line"


# ---------------------------------------------------------------------------
# Follow-up hardening review (post TASK-843/TASK-850), Finding 1: draining
# `_iter_candidates_across_roots` all the way to `_MAX_CANDIDATES` before
# ever spawning the search subprocess undid the pre-subprocess early-break
# (`len(matches) >= _MAX_MATCHES` / `lines_scanned >=
# _MAX_GREP_LINES_SCANNED`, checked DURING enumeration). Reviewer-measured:
# ~0.32ms/candidate means a 5,000-file tree with a pattern matching every
# file went from ~0.1s (old early-break, ~200 files examined) to ~1.58s (all
# 5,000 examined first). `_run_grep_search` restores the early exit by
# streaming candidate discovery and the subprocess search together in
# growing batches, and starts its wall-clock deadline before the first
# candidate is even pulled -- these pin both halves of that fix.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grep_files_high_hit_rate_pattern_does_not_over_enumerate(
    tmp_path, monkeypatch
):
    """Reproduces the reviewer's exact regression scenario: a 5,000-file
    tree where the pattern matches every file. Pins BOTH symptoms of the
    fix rather than either alone:

    - Structural (the real proof, immune to machine speed): the number of
      candidates actually PULLED from `_iter_candidates_across_roots`
      before the match budget (`_MAX_MATCHES` = 200) is satisfied must be
      a small fraction of the 5,000-file tree, not the whole thing.
    - Wall-clock (a generous sanity check): the call completes well
      within the ballpark the fix targets, not the ~1s+ the drained-
      enumeration bug cost (measured directly on this branch: median
      ~1.0-1.3s before this fix, ~0.07-0.36s after).
    """
    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: str(tmp_path))
    for i in range(5_000):
        (tmp_path / f"f{i}.py").write_text("DEBUG = True\n")

    pulled = 0
    real_iter = fot._iter_candidates_across_roots

    def counting_iter(*args, **kwargs):
        nonlocal pulled
        for path in real_iter(*args, **kwargs):
            pulled += 1
            yield path

    monkeypatch.setattr(fot, "_iter_candidates_across_roots", counting_iter)

    start = time.perf_counter()
    result = await fot.GrepFiles().execute(pattern="DEBUG", glob="**/*.py")
    elapsed = time.perf_counter() - start

    assert len(result["matches"]) == fot._MAX_MATCHES
    assert pulled < 2_500, (
        f"enumerated {pulled} of 5,000 candidates before the match budget "
        "was satisfied -- expected a small fraction, not most/all of the "
        "tree (is discovery being drained before the subprocess ever "
        "starts searching?)"
    )
    assert elapsed < 2.0, (
        f"grep_files took {elapsed:.2f}s for a high-hit-rate pattern over "
        "a large tree -- is candidate discovery running away before the "
        "search subprocess is ever spawned?"
    )


@pytest.mark.asyncio
async def test_grep_files_aggregates_matches_across_multiple_batches(
    sandbox, monkeypatch
):
    """`_run_grep_search` must correctly accumulate matches (and the
    lines-scanned budget) across MULTIPLE batches/subprocess calls, not
    just the common single-batch case every other test exercises. Forces
    several small batches by shrinking the batch-size constants, and
    confirms every match survives into the final, merged result -- and
    that more than one subprocess call actually happened.
    """
    monkeypatch.setattr(fot, "_GREP_INITIAL_CANDIDATE_BATCH_SIZE", 2)
    monkeypatch.setattr(fot, "_GREP_MAX_CANDIDATE_BATCH_SIZE", 2)
    monkeypatch.setattr(fot, "_MAX_MATCHES", 1_000)
    for i in range(10):
        (sandbox / f"extra{i}.py").write_text(f"DEBUG_MARKER_{i}\n")

    calls: list[list[str]] = []
    real = fot._run_grep_subprocess

    def spy(pattern, file_paths, **kwargs):
        calls.append(list(file_paths))
        return real(pattern, file_paths, **kwargs)

    monkeypatch.setattr(fot, "_run_grep_subprocess", spy)

    result = await fot.GrepFiles().execute(
        pattern=r"DEBUG_MARKER_\d+", glob="extra*.py"
    )

    assert len(result["matches"]) == 10
    assert len(calls) > 1, (
        "expected multiple batched subprocess calls with a batch size of 2 "
        "and 10 candidates"
    )
    # Every candidate handed to any one call stays within that call's batch
    # size -- proves batching, not one call quietly given everything anyway.
    assert all(len(batch) <= 2 for batch in calls)


@pytest.mark.asyncio
async def test_grep_files_search_deadline_starts_before_the_first_candidate_is_pulled(
    sandbox, monkeypatch
):
    """Finding 1's second half: the wall-clock deadline must start
    counting BEFORE candidate discovery begins, not after it finishes --
    otherwise a slow discovery phase could push the real wall-clock past
    `GrepFiles.timeout_seconds` without this deadline ever catching it
    (exactly the false invariant the pre-fix `_GREP_SUBPROCESS_TIMEOUT_
    SECONDS` comment claimed held). Proven by making discovery itself
    slow and setting the deadline shorter than that delay: the call must
    report a timeout rather than silently proceeding to search anyway.
    """
    monkeypatch.setattr(fot, "_GREP_SUBPROCESS_TIMEOUT_SECONDS", 0.05)
    (sandbox / "slow.py").write_text("DEBUG\n")

    real_iter = fot._iter_candidates_across_roots

    def slow_iter(*args, **kwargs):
        time.sleep(0.2)
        yield from real_iter(*args, **kwargs)

    monkeypatch.setattr(fot, "_iter_candidates_across_roots", slow_iter)

    result = await fot.GrepFiles().execute(pattern="DEBUG", glob="slow.py")

    assert "error" in result
    assert "timed out" in result["error"]


# ---------------------------------------------------------------------------
# Finding 3 (follow-up hardening review): the search subprocess previously
# inherited the parent's full environment, actual cwd, and (via the
# script's own directory) `sys.path[0]` -- a probe confirmed a parent-only
# secret env var visible in the child, and `sys.path[0]` pointing at this
# project's own `Tools/` source directory. Not an escalation on its own
# (planting a shadow module there already requires source-tree write
# access), but removing the surface costs nothing. These pin the fix:
# explicit `-P`, `cwd=`, and a minimal `env=`.
# ---------------------------------------------------------------------------


def test_run_grep_subprocess_isolates_worker_cwd_and_env(tmp_path, monkeypatch):
    """The worker must not inherit the parent's real cwd or its full
    environment -- it only ever needs the absolute paths handed to it on
    stdin.
    """
    monkeypatch.setenv("MY_FAKE_SECRET_9f21", "sk-should-not-leak")
    captured: dict = {}
    real_popen = subprocess.Popen

    def spying_popen(args, **kwargs):
        captured["args"] = args
        captured["cwd"] = kwargs.get("cwd")
        captured["env"] = kwargs.get("env")
        return real_popen(args, **kwargs)

    monkeypatch.setattr(fot.subprocess, "Popen", spying_popen)

    target = tmp_path / "a.txt"
    target.write_text("DEBUG\n")

    result = fot._run_grep_subprocess(
        pattern="DEBUG",
        file_paths=[str(target)],
        max_matches=200,
        max_line_search_chars=500,
        max_lines_scanned=200_000,
        max_file_bytes=5_000_000,
        timeout_seconds=5.0,
    )

    assert len(result["matches"]) == 1
    assert "-P" in captured["args"]
    assert captured["cwd"] is not None
    assert os.path.realpath(captured["cwd"]) != os.path.realpath(os.getcwd())
    assert captured["env"] is not None
    assert "MY_FAKE_SECRET_9f21" not in captured["env"]


def test_grep_worker_env_omits_arbitrary_parent_variables(monkeypatch):
    """Direct unit test of the helper itself: only the small, named
    allowlist survives, never an arbitrary parent-set variable.
    """
    monkeypatch.setenv("MY_FAKE_API_KEY_TEST_71a2", "sk-LEAKED")

    env = fot._grep_worker_env()

    assert "MY_FAKE_API_KEY_TEST_71a2" not in env


# ---------------------------------------------------------------------------
# Finding 4 (follow-up hardening review): `_run_grep_subprocess` previously
# returned the worker's parsed JSON verbatim once it was confirmed to be a
# dict -- never confirming `matches` was actually a list of well-formed
# entries, or that `lines_scanned` was actually an int. A worker emitting
# `{"matches": "not-a-list"}` would propagate that shape straight through.
# These pin the added validation.
# ---------------------------------------------------------------------------


def test_run_grep_subprocess_rejects_a_matches_field_that_is_not_a_list(monkeypatch):
    class _FakeProc:
        pid = 999_994
        returncode = 0

        def communicate(self, input=None, timeout=None):
            return '{"matches": "not-a-list", "lines_scanned": 1}', ""

        def kill(self):
            pass

    monkeypatch.setattr(fot.subprocess, "Popen", lambda *a, **k: _FakeProc())

    result = fot._run_grep_subprocess(
        pattern="x",
        file_paths=[],
        max_matches=200,
        max_line_search_chars=500,
        max_lines_scanned=200_000,
        max_file_bytes=5_000_000,
        timeout_seconds=5.0,
    )

    assert "error" in result
    assert "matches" not in result


def test_run_grep_subprocess_rejects_a_malformed_match_entry(monkeypatch):
    """Each match entry must be `{"path": str, "line_number": int, "line":
    str}` -- a worker omitting `line_number` (or any other required key)
    must not slip through.
    """

    class _FakeProc:
        pid = 999_993
        returncode = 0

        def communicate(self, input=None, timeout=None):
            return (
                '{"matches": [{"path": "x.txt", "line": "hi"}], '
                '"lines_scanned": 1}',
                "",
            )

        def kill(self):
            pass

    monkeypatch.setattr(fot.subprocess, "Popen", lambda *a, **k: _FakeProc())

    result = fot._run_grep_subprocess(
        pattern="x",
        file_paths=[],
        max_matches=200,
        max_line_search_chars=500,
        max_lines_scanned=200_000,
        max_file_bytes=5_000_000,
        timeout_seconds=5.0,
    )

    assert "error" in result


def test_run_grep_subprocess_rejects_a_non_int_lines_scanned(monkeypatch):
    class _FakeProc:
        pid = 999_992
        returncode = 0

        def communicate(self, input=None, timeout=None):
            return '{"matches": [], "lines_scanned": "many"}', ""

        def kill(self):
            pass

    monkeypatch.setattr(fot.subprocess, "Popen", lambda *a, **k: _FakeProc())

    result = fot._run_grep_subprocess(
        pattern="x",
        file_paths=[],
        max_matches=200,
        max_line_search_chars=500,
        max_lines_scanned=200_000,
        max_file_bytes=5_000_000,
        timeout_seconds=5.0,
    )

    assert "error" in result


def test_run_grep_subprocess_still_accepts_a_well_formed_payload(monkeypatch):
    """Sanity check alongside the malformed-payload tests above: a
    genuinely well-formed worker payload must still pass validation
    unchanged, not just be rejected less often by accident.
    """

    class _FakeProc:
        pid = 999_991
        returncode = 0

        def communicate(self, input=None, timeout=None):
            return (
                '{"matches": [{"path": "x.txt", "line_number": 3, '
                '"line": "hi"}], "lines_scanned": 3}',
                "",
            )

        def kill(self):
            pass

    monkeypatch.setattr(fot.subprocess, "Popen", lambda *a, **k: _FakeProc())

    result = fot._run_grep_subprocess(
        pattern="x",
        file_paths=[],
        max_matches=200,
        max_line_search_chars=500,
        max_lines_scanned=200_000,
        max_file_bytes=5_000_000,
        timeout_seconds=5.0,
    )

    assert result == {
        "matches": [{"path": "x.txt", "line_number": 3, "line": "hi"}],
        "lines_scanned": 3,
    }
