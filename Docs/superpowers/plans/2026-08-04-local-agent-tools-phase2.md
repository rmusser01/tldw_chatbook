# Local Agent Tools — Phase 2 (File Tools) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Console agent the full claude-code-style file tool set — `fs_read`, `fs_write`, `fs_edit`, `fs_glob`, `fs_grep` — built on the phase-1 plumbing, with the legacy `Tools/file_operation_tools.py` implementations migrated onto the same sync cores.

**Architecture:** New sync cores in `Tools/local_tool_impls.py` (phase-1 file); new `LocalToolSpec` entries in `Agents/local_tool_provider.py:_default_specs` (write tools carry the `mutates` risk tag); legacy `ReadFileTool`/`WriteFileTool`/`ListDirectoryTool` wrappers refactored to delegate to the shared cores (behavior-preserving for the legacy executor path). No changes to the runtime loop, hooks, bridge, or config — phase 1 froze those seams.

**Tech Stack:** Python ≥3.11, pytest, Hypothesis (already a project dependency).

**Spec:** `Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md` (§2 inventory, §6 testing)
**ADR:** `backlog/decisions/032-local-agent-tool-permission-boundary.md` (binding: confinement, fail-closed, naming, 32 KiB result fitting)

**Carryovers from phase 1 (binding):**
- `fs_list` already landed in phase 1 — it is NOT in this phase's scope.
- Audit recording (`record_tool_decision`) for local deny/timeout outcomes is in scope (Task 7) — phase-1 docstring flagged it.
- AGENTS.md "Tool Calling" update is in scope (Task 8).
- With 6 local tools + 2 builtins, the catalog exceeds `DIRECT_DISCLOSE_THRESHOLD = 8`; an integration test must prove tools remain reachable via `find_tools`/`load_tools` (Task 6).

---

## Verified facts from phase 1 (do not re-derive)

- `LocalToolSpec{name, description, parameters, handler, tags}`; `handler: Callable[[dict], str]` raising `LocalToolError` on model-actionable failures; the provider byte-fits results to 32 KiB and converts ALL exceptions to `ToolResult(ok=False, error=...)` (`Agents/local_tool_provider.py`).
- `resolve_workspace_path(path, workspace_root) -> Path` confines + allows hidden under root, raises `LocalToolError("...outside the workspace root...")` (`Tools/local_tool_impls.py:26`).
- `_default_specs(workspace_root)` in `Agents/local_tool_provider.py` is where new specs register; catalog ids are `local:<name>`; `tags=("mutates",)` forces risk-floored asks in permission resolution.
- Legacy tools live in `tldw_chatbook/Tools/file_operation_tools.py` (~379 lines): `ReadFileTool` (:20), `ListDirectoryTool` (:110), `WriteFileTool` (:266) — async `execute` wrapping fully synchronous I/O, returning `{"error": ...}` dicts on failure. Their `validate_path(file_path, "file")` calls are broken confinement (confine to a nonexistent `<cwd>/file` root); the legacy wrappers keep their existing validation behavior unless a minimal swap to the workspace root is clean — see Task 5's note.
- Provider tests: `Tests/Agents/test_local_tool_provider.py`; impl tests: `Tests/Tools/test_local_tool_impls.py` (note: listing-style assertions must use a `ws = tmp_path/"ws"` subdir — the autouse `isolate_test_environment` fixture creates `tmp_path/"test_data"`).
- Integration harness: `Tests/Agents/test_local_tools_integration.py` (ScriptedChat + fence helper + real AgentService.run_turn).
- Hypothesis is configured in the repo (`.hypothesis/` exists); see existing property tests for style (`grep -rln "from hypothesis" Tests/ | head`).
- Run tests from the worktree with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <args>`. Known pre-existing failures to ignore/deselect: `Tests/Chat/test_anthropic_native_tools.py::test_anthropic_shaped_tools_pass_through_untouched`, `Tests/Utils/test_github_api_client.py::TestGitHubAPIClient::test_client_property_without_token`.

---

## Task 0: Backlog task

**Files:**
- Create: backlog task via CLI (fallback: markdown)

- [ ] **Step 1: Create the task**

```bash
backlog task create "Local agent tools phase 2: file tools (fs_read/fs_write/fs_edit/fs_glob/fs_grep)" \
  -d "Spec: Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md (phase 2). Plan: Docs/superpowers/plans/2026-08-04-local-agent-tools-phase2.md. ADR: backlog/decisions/032. Builds on task-1338 (phase 1, fs_list pilot). NOTE: fs_list already landed in phase 1 and is out of scope here." \
  --ac "fs_read pages line-numbered output with offset/limit and refuses binary files,fs_write creates/overwrites files confined to workspace root with mutates risk tag,fs_edit performs unique-match replacement with ambiguity errors and replace_all,fs_glob and fs_grep search the workspace with result caps,Legacy ReadFileTool/WriteFileTool delegate to the shared cores with unchanged legacy behavior,Local deny/timeout outcomes are audit-recorded,Tools remain reachable via find_tools/load_tools past the direct-disclosure threshold,All new tests pass" \
  --plan "See Docs/superpowers/plans/2026-08-04-local-agent-tools-phase2.md"
```

- [ ] **Step 2: Commit** — `git add backlog/ && git commit -m "docs: phase-2 backlog task for local agent file tools"`

---

## Task 1: `fs_read` core + spec

**Files:**
- Modify: `tldw_chatbook/Tools/local_tool_impls.py`
- Modify: `tldw_chatbook/Agents/local_tool_provider.py` (`_default_specs`)
- Test: `Tests/Tools/test_local_tool_impls.py`, `Tests/Agents/test_local_tool_provider.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_fs_read_line_numbered(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "a.txt").write_text("one\ntwo\nthree\n")
    out = read_file("a.txt", workspace_root=ws)
    assert out.splitlines() == ["1\tone", "2\ttwo", "3\tthree"]


def test_fs_read_offset_limit(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "a.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)))
    out = read_file("a.txt", workspace_root=ws, offset=3, limit=2)
    assert out.splitlines() == ["3\tline3", "4\tline4"]


def test_fs_read_offset_past_eof_returns_notice(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "a.txt").write_text("only\n")
    assert "past end of file" in read_file("a.txt", workspace_root=ws, offset=99)


def test_fs_read_refuses_binary(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    with pytest.raises(LocalToolError, match="binary"):
        read_file("img.png", workspace_root=ws)


def test_fs_read_missing_file(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    with pytest.raises(LocalToolError, match="not found"):
        read_file("nope.txt", workspace_root=ws)
```

Provider: catalog now lists `local:fs_read` with schema requiring `path` (offset/limit optional ints).

- [ ] **Step 2: Verify failure** — `ImportError: cannot import name 'read_file'`

- [ ] **Step 3: Implement** in `local_tool_impls.py`:

```python
MAX_READ_CHARS = 32 * 1024  # provider byte-fits too; core caps content meaningfully


def read_file(
    path: str,
    *,
    workspace_root: Path,
    offset: int = 1,
    limit: int | None = None,
) -> str:
    """Read ``path`` with 1-based line numbers, ``offset``/``limit`` paging.

    Lines are numbered from 1 (matching claude-code's Read). ``offset`` is
    the 1-based first line to return; ``limit`` caps the line count.
    Binary files (NUL byte in the first 8 KiB) and missing files raise
    LocalToolError with model-actionable messages.
    """
    root = resolve_workspace_path(path, workspace_root)
    if not root.is_file():
        raise LocalToolError(f"file not found: {path}")
    with open(root, "rb") as fh:
        sniff = fh.read(8192)
    if b"\x00" in sniff:
        raise LocalToolError(f"'{path}' appears to be binary; fs_read only reads text files")
    text = root.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    start = max(offset, 1) - 1
    if start >= len(lines) and lines:
        return f"(offset {offset} is past end of file; {len(lines)} lines total)"
    window = lines[start:] if limit is None else lines[start:start + max(limit, 0)]
    numbered = "\n".join(f"{i}\t{line}" for i, line in enumerate(window, start=start + 1))
    if len(numbered) > MAX_READ_CHARS:
        numbered = numbered[:MAX_READ_CHARS] + "\n… [truncated]"
    return numbered
```

Add the `fs_read` `LocalToolSpec` (no tags — read-only) to `_default_specs`: params `path` (required string), `offset` (int, default 1), `limit` (int, optional).

- [ ] **Step 4: Run tests** — `pytest Tests/Tools/test_local_tool_impls.py Tests/Agents/test_local_tool_provider.py -q` PASS
- [ ] **Step 5: Commit** — `git commit -m "feat: fs_read core with paging and binary refusal"`

---

## Task 2: `fs_write` core + spec (mutates)

**Files:**
- Modify: `tldw_chatbook/Tools/local_tool_impls.py`
- Modify: `tldw_chatbook/Agents/local_tool_provider.py` (`_default_specs`)
- Test: `Tests/Tools/test_local_tool_impls.py`, `Tests/Agents/test_local_tool_provider.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_fs_write_creates_file(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    out = write_file("new.txt", "hello\n", workspace_root=ws)
    assert (ws / "new.txt").read_text() == "hello\n"
    assert "wrote" in out and "new.txt" in out


def test_fs_write_overwrites(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "f.txt").write_text("old")
    write_file("f.txt", "new", workspace_root=ws)
    assert (ws / "f.txt").read_text() == "new"


def test_fs_write_requires_existing_parent(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    with pytest.raises(LocalToolError, match="parent directory"):
        write_file("no/such/dir/f.txt", "x", workspace_root=ws)


def test_fs_write_confined(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    with pytest.raises(LocalToolError, match="outside the workspace root"):
        write_file("../evil.txt", "x", workspace_root=ws)
```

Provider: `fs_write` spec carries `tags=("mutates",)` — assert `hub_tool_for("fs_write").tags == ("mutates",)`.

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Implement**

```python
def write_file(path: str, content: str, *, workspace_root: Path) -> str:
    """Create or overwrite ``path`` with ``content`` (full-file write).

    The parent directory must already exist (deliberate divergence from
    claude-code's Write, to catch model path typos early — spec §2).
    """
    root = resolve_workspace_path(path, workspace_root)
    if not root.parent.is_dir():
        raise LocalToolError(f"parent directory does not exist for: {path}")
    root.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} characters to {path}"
```

Spec: params `path` + `content` (both required), `tags=("mutates",)`.

- [ ] **Step 4: Run tests** — PASS
- [ ] **Step 5: Commit** — `git commit -m "feat: fs_write core with mutates risk tag"`

---

## Task 3: `fs_edit` core + spec (mutates) + Hypothesis

**Files:**
- Modify: `tldw_chatbook/Tools/local_tool_impls.py`
- Modify: `tldw_chatbook/Agents/local_tool_provider.py` (`_default_specs`)
- Test: `Tests/Tools/test_local_tool_impls.py`, `Tests/Tools/test_local_tool_impls_properties.py` (new)

- [ ] **Step 1: Write the failing tests**

```python
def test_fs_edit_unique_match(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "f.txt").write_text("alpha beta gamma")
    out = edit_file("f.txt", "beta", "BETA", workspace_root=ws)
    assert (ws / "f.txt").read_text() == "alpha BETA gamma"
    assert "1 replacement" in out


def test_fs_edit_requires_match(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "f.txt").write_text("alpha")
    with pytest.raises(LocalToolError, match="not found"):
        edit_file("f.txt", "zzz", "q", workspace_root=ws)


def test_fs_edit_ambiguous_match_reports_count(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "f.txt").write_text("dup dup dup")
    with pytest.raises(LocalToolError, match="3 times"):
        edit_file("f.txt", "dup", "x", workspace_root=ws)


def test_fs_edit_replace_all(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "f.txt").write_text("dup dup dup")
    out = edit_file("f.txt", "dup", "x", workspace_root=ws, replace_all=True)
    assert (ws / "f.txt").read_text() == "x x x"
    assert "3 replacements" in out
```

Property tests (`test_local_tool_impls_properties.py`):

```python
from hypothesis import given, strategies as st

from tldw_chatbook.Tools.local_tool_impls import LocalToolError, edit_file


@given(
    prefix=st.text(max_size=50), needle=st.text(min_size=1, max_size=10),
    suffix=st.text(max_size=50), replacement=st.text(max_size=20),
)
def test_edit_replaces_exactly_one_occurrence(tmp_path, prefix, needle, suffix, replacement):
    ws = tmp_path / "ws"; ws.mkdir(exist_ok=True)
    content = prefix + needle + suffix
    if content.count(needle) != 1:
        return  # only unique-match inputs are in scope for this property
    (ws / "f.txt").write_text(content)
    edit_file("f.txt", needle, replacement, workspace_root=ws)
    assert (ws / "f.txt").read_text() == prefix + replacement + suffix


@given(path=st.text(min_size=1))
def test_workspace_confinement_never_escapes(tmp_path, path):
    ws = tmp_path / "ws"; ws.mkdir(exist_ok=True)
    from tldw_chatbook.Tools.local_tool_impls import resolve_workspace_path
    try:
        resolved = resolve_workspace_path(path, ws)
    except (LocalToolError, ValueError):
        return  # refusal is fine
    assert str(resolved).startswith(str(ws.resolve()))
```

NOTE: Hypothesis + tmp_path interaction — use `@settings(max_examples=...)` modestly (tmp_path is function-scoped; reuse one ws dir per test as shown) and match the repo's existing Hypothesis test style. If tmp_path-per-example is awkward, use `tempfile.TemporaryDirectory()` inside the test instead.

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Implement**

```python
def edit_file(
    path: str,
    old_string: str,
    new_string: str,
    *,
    workspace_root: Path,
    replace_all: bool = False,
) -> str:
    """Replace exact ``old_string`` with ``new_string`` in ``path``.

    Fails unless the match is unique (or ``replace_all=True``); ambiguity
    errors include the match count so the model can self-correct. Exact
    semantics per spec §2 (claude-code Edit parity).
    """
    if not old_string:
        raise LocalToolError("old_string must not be empty")
    root = resolve_workspace_path(path, workspace_root)
    if not root.is_file():
        raise LocalToolError(f"file not found: {path}")
    content = root.read_text(encoding="utf-8")
    count = content.count(old_string)
    if count == 0:
        raise LocalToolError(f"old_string not found in {path}")
    if count > 1 and not replace_all:
        raise LocalToolError(
            f"old_string appears {count} times in {path}; "
            "provide more context to make it unique, or set replace_all=true"
        )
    updated = content.replace(old_string, new_string)
    root.write_text(updated, encoding="utf-8")
    n = count if replace_all else 1
    return f"made {n} replacement{'s' if n != 1 else ''} in {path}"
```

Spec: params `path`, `old_string`, `new_string` (required), `replace_all` (bool, default false); `tags=("mutates",)`.

- [ ] **Step 4: Run tests** — PASS (unit + property)
- [ ] **Step 5: Commit** — `git commit -m "feat: fs_edit exact-replacement core with property tests"`

---

## Task 4: `fs_glob` + `fs_grep` cores + specs

**Files:**
- Modify: `tldw_chatbook/Tools/local_tool_impls.py`
- Modify: `tldw_chatbook/Agents/local_tool_provider.py` (`_default_specs`)
- Test: `Tests/Tools/test_local_tool_impls.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_fs_glob_matches_and_sorts_by_mtime(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    import os, time
    old = ws / "old.py"; old.write_text("x")
    new = ws / "new.py"; new.write_text("x")
    (ws / "skip.txt").write_text("x")
    past = time.time() - 100
    os.utime(old, (past, past))
    out = glob_files("*.py", workspace_root=ws)
    assert out.splitlines() == ["new.py", "old.py"]  # newest first


def test_fs_glob_recursive_and_cap(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "sub").mkdir()
    (ws / "sub" / "deep.py").write_text("x")
    assert "sub/deep.py" in glob_files("**/*.py", workspace_root=ws)


def test_fs_grep_line_numbers(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "a.py").write_text("def foo():\n    return 1\n")
    out = grep_files("def foo", workspace_root=ws)
    assert "a.py:1:def foo():" in out


def test_fs_grep_files_with_matches_and_count(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "a.py").write_text("hit\nhit\n")
    (ws / "b.py").write_text("hit\n")
    assert set(grep_files("hit", workspace_root=ws, mode="files").splitlines()) == {"a.py", "b.py"}
    assert "a.py:2" in grep_files("hit", workspace_root=ws, mode="count")


def test_fs_grep_caps_output(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "big.py").write_text("hit\n" * 500)
    out = grep_files("hit", workspace_root=ws, max_results=10)
    assert "10" in out and "more" in out
```

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Implement** — pure-Python, no ripgrep dependency (spec §2):

```python
MAX_GLOB_RESULTS = 100
MAX_GREP_RESULTS = 100
_MAX_GREP_FILE_BYTES = 2 * 1024 * 1024  # skip huge files


def glob_files(pattern: str, *, workspace_root: Path, max_results: int = MAX_GLOB_RESULTS) -> str:
    """Match ``pattern`` under the workspace, newest-mtime first, capped.

    Paths in the result are workspace-relative. Hidden files/dirs under the
    root ARE matched (workspace policy, ADR-032).
    """
    root = workspace_root.resolve()
    matches = [p for p in root.glob(pattern) if p.is_file()]
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    lines = [str(p.relative_to(root)) for p in matches[:max_results]]
    if len(matches) > max_results:
        lines.append(f"… ({len(matches) - max_results} more, truncated)")
    return "\n".join(lines) if lines else f"(no files matching {pattern!r})"


def grep_files(
    pattern: str,
    *,
    workspace_root: Path,
    mode: str = "content",  # content | files | count
    max_results: int = MAX_GREP_RESULTS,
) -> str:
    """Regex search under the workspace.

    Modes: ``content`` -> ``relpath:lineno:line``; ``files`` -> one relpath
    per matching file; ``count`` -> ``relpath:N``. Binary and >2 MiB files
    are skipped. Invalid regex raises LocalToolError.
    """
    import re

    try:
        rx = re.compile(pattern)
    except re.error as exc:
        raise LocalToolError(f"invalid regex: {exc}") from exc
    if mode not in ("content", "files", "count"):
        raise LocalToolError(f"unknown mode: {mode}")
    root = workspace_root.resolve()
    content_hits: list[str] = []
    file_hits: list[str] = []
    count_hits: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.stat().st_size > _MAX_GREP_FILE_BYTES:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary/unreadable — skip
        rel = str(p.relative_to(root))
        lines = [f"{i}:{line}" for i, line in enumerate(text.splitlines(), 1) if rx.search(line)]
        if not lines:
            continue
        file_hits.append(rel)
        count_hits.append(f"{rel}:{len(lines)}")
        content_hits.extend(f"{rel}:{hit}" for hit in lines)
    if mode == "files":
        out, total = file_hits, len(file_hits)
    elif mode == "count":
        out, total = count_hits, len(count_hits)
    else:
        out, total = content_hits, len(content_hits)
    shown = out[:max_results]
    if total > max_results:
        shown.append(f"… ({total - max_results} more, truncated)")
    return "\n".join(shown) if shown else f"(no matches for {pattern!r})"
```

Specs (both read-only, no tags):
- `fs_glob`: params `pattern` (required), `max_results` optional.
- `fs_grep`: params `pattern` (required), `mode` (enum content/files/count, default content), `max_results` optional.

- [ ] **Step 4: Run tests** — PASS
- [ ] **Step 5: Commit** — `git commit -m "feat: fs_glob and fs_grep cores"`

---

## Task 5: Legacy wrapper delegation

**Files:**
- Modify: `tldw_chatbook/Tools/file_operation_tools.py`
- Test: run existing legacy tests — find with `grep -rln "ReadFileTool\|WriteFileTool\|ListDirectoryTool" Tests/`

- [ ] **Step 1: Characterize first.** Run the existing legacy file-tool tests and record green baseline. READ the three legacy implementations carefully.

- [ ] **Step 2: Delegate the bodies.** Refactor `ReadFileTool.execute`, `WriteFileTool.execute`, and `ListDirectoryTool.execute` so their core logic lives in/uses `Tools/local_tool_impls.py` (`read_file`/`write_file`/`list_directory`), with the legacy wrappers translating their own arg shapes, output formats, and error dicts. CONSTRAINTS:
  - Legacy behavior must stay byte-compatible from its callers' perspective: same return dict shapes, same `{"error": ...}` semantics, same parameter names.
  - The legacy tools' BROKEN confinement (`validate_path(path, "file")`) is NOT preserved — that call rejects virtually everything; if existing legacy tests pass against it, they must be stubbing. Replace legacy validation with the workspace-root core (root = the tool's configured base or cwd — check how the legacy executor instantiates these tools in `tool_executor.py:648-788` and thread a sensible root through). Document any behavior delta in the commit message.
  - If a legacy wrapper's output format differs from the core's (e.g. no line numbers), keep the wrapper's format — share the I/O and validation, not necessarily the rendering. Minimal diff wins: if a wrapper's body is already thin and correct, leave it and note why.

- [ ] **Step 3: Run tests** — legacy file-tool tests green + `Tests/Tools/` green.
- [ ] **Step 4: Commit** — `git commit -m "refactor: legacy file tools delegate to shared local cores"`

---

## Task 6: Disclosure-threshold integration + e2e update

**Files:**
- Test: `Tests/Agents/test_local_tools_integration.py` (extend)

- [ ] **Step 1: Write the tests**

1. **find/load path**: build the production-composed registry (all 6 local tools + builtins = 8 entries, past `DIRECT_DISCLOSE_THRESHOLD`); ScriptedChat turn 1 calls `find_tools` with query "edit", turn 2 `load_tools` for `local:fs_edit`, turn 3 calls `fs_edit` (approval: approve_once), turn 4 final text. Assert the edit happened on disk and exactly one approval round trip occurred.
2. **allow-state e2e** (phase-1 review follow-up): `resolve_state` → allow; scripted `fs_read` call; assert ZERO approval round trips and the file content reached the model's second-turn payload.
3. **deny-path feedback symmetry** (phase-1 review follow-up): extend the existing deny test — assert the `Tool result for fs_list: ERROR: …` line appears in the second-turn messages payload.

- [ ] **Step 2: Run** — new + existing integration tests PASS.
- [ ] **Step 3: Commit** — `git commit -m "test: find/load disclosure path and allow-state e2e coverage"`

---

## Task 7: Audit recording for local decisions

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`_compose_local_provider` wiring)
- Modify: `tldw_chatbook/Agents/local_tool_provider.py` (add `record_decision` seam)
- Test: `Tests/Agents/test_local_tool_provider.py`, `Tests/Chat/test_console_local_review_hook.py`

- [ ] **Step 1: Study the MCP path.** Read how `MCPToolProvider` records decisions (`grep -n record_tool_decision tldw_chatbook/Agents/mcp_tool_provider.py tldw_chatbook/MCP/unified_control_plane_service.py`) — what payload, which outcomes (deny/timeout/allow?), exact call sites.

- [ ] **Step 2: Write the failing tests.** Provider: when a `record_decision: Callable[[HubTool, str], None]` seam is injected, deny and timeout invocations call it with the hub tool and the decision string; allow executions call it per MCP parity (match whatever MCP records — if MCP only records refusals, match that). Seam is optional (None = no recording) and never-raise guarded.

- [ ] **Step 3: Implement** the seam in the provider (call it in `invoke()`'s refusal paths and wherever MCP parity requires) and wire it in `_compose_local_provider` to the same `record_tool_decision` path the MCP provider uses, with server key `local:__local__`. Update the `_compose_local_provider` docstring (remove the "not wired" note).

- [ ] **Step 4: Run tests** — PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: audit-record local tool decisions (MCP parity)"`

---

## Task 8: Docs + close-out

**Files:**
- Modify: `AGENTS.md`
- Modify: backlog phase-2 task

- [ ] **Step 1: AGENTS.md.** Update "Special Systems → Tool Calling" — it currently says "Detection works, execution pending", which is stale for both MCP and local tools. Describe: the `Agents/tool_catalog.py` provider seam, builtin/local/skill/MCP providers, approval flow via the MCP permission store, `[console] local_tools_enabled`/`workspace_root`, and the fs_* tool set. Also fix the stale `Coding_Window.py` reference (that window is retired; the Console screen is `UI/Screens/chat_screen.py`) if still present. Keep it brief and in the file's existing style.

- [ ] **Step 2: Backlog close-out.** Check all ACs, add `## Implementation Notes` (approach, files, deviations — including any legacy-delegation behavior deltas from Task 5), set status Done.

- [ ] **Step 3: Final full-suite run** from the worktree: `pytest Tests/Agents Tests/Tools Tests/Chat Tests/Utils Tests/test_config_console_defaults.py -q` (deselect the two known pre-existing failures).

- [ ] **Step 4: Commit** — `git commit -m "docs: AGENTS.md tool-calling update + close phase-2 task"`

---

## Final step (controller, not a delegated task)

Dispatch the final whole-implementation reviewer (spec + plan + ACs, full diff, test run), then invoke superpowers:finishing-a-development-branch.
