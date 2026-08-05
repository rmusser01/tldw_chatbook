# Local Agent Tools — Phase 3b-ii (Git Tools) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the read-only git tool set — `git_status`, `git_diff`, `git_log`, `git_blame`, `git_branches` — ported from tldw_server's `git_module.py` and adapted to the sync-core shape, under ADR-033's process-execution boundary.

**Architecture:** New core module `Tools/git_tool_impls.py`: a sync `run_git` subprocess wrapper (fixed argv, subcommand/global-option allowlist, sanitized env, timeout, 1 MB output cap) + per-tool sync cores adapted from the reference's `_execute_*` functions. Five new `LocalToolSpec`s with NO risk tags (ADR-033: the `process` tag is deliberately not applied to this read-only allowlisted set; tripwire documented).

**Spec:** `Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md` §2.5 · **ADR:** 033 (binding boundary), 032
**Reference source:** tldw_server @ `5605b9d9906322c2e6b5342b48c391ae674d315e`, `tldw_Server_API/app/core/MCP_unified/modules/implementations/git_module.py` (2,130 lines, async). Clone may exist at `/tmp/tldw_server_mcp/tldw_server`. **Attribution header binding (re-plan §5).**

## Verified facts from the reference (do not re-derive)

- `_ALLOWED_GIT_SUBCOMMANDS = {"--version", "blame", "branch", "diff", "log", "ls-files", "rev-parse", "status"}` (`git_module.py:51`).
- Argv validation (`:270-297`): argv[0] must be `git`; global options limited to `-C <path>` and `--no-pager` (`--version` must be alone); anything else starting with `-` before the subcommand is rejected; subcommand must be allowlisted.
- Sanitized env (`:247-268`): PATH (+SYSTEMROOT/WINDIR on Windows) plus `GIT_TERMINAL_PROMPT=0`, `GIT_OPTIONAL_LOCKS=0`, `GIT_PAGER=cat`, `GIT_EXTERNAL_DIFF=""`, `GIT_CONFIG_COUNT/KEY_0/VALUE_0` disabling fsmonitor. stdin is DEVNULL.
- Output: bounded stream reads, decoded utf-8/replace, `truncated` flag; default cap 1 MB (`:50`). Timeout kills the process and reports `timed_out`.
- Tool implementations (`:570-1010`): status (porcelain v2 + branch header), branches (verbose list + ahead/behind), diff (THREE scopes only — `unstaged`/`staged`/`working_tree` (`_DIFF_SCOPES`, `:46`) via `_run_diff_command` (`:1049-1080`, which also sets `--no-ext-diff/--no-textconv/--no-color` — port those flags); optional path filter; NO commit-range mode and NO --stat in the reference — see Task 2's disclosed deviations), log (count cap, max 100 — an absent limit falls back to the max, there is no default-20 in the reference; optional path filter), blame (line-range optional, porcelain parse), conflicts.list/read (deferred this phase). The reference diff also has its own 120 KB `max_bytes` cap (`:2092`) below the 1 MB command cap — omitted here given the provider's 32 KiB byte-fit; noted so nobody thinks they missed it.
- `_prepare_repository` (`:1134+`): repo root discovered via `git rev-parse --show-toplevel`, must exist and be a repo.
- Chatbook side: cores are SYNC, `LocalToolError` on model-actionable failures, confinement via `resolve_workspace_path` / workspace root, provider byte-fits results to 32 KiB (git's own 1 MB cap stays the inner bound; per-call `max_bytes` arg optional).
- Tests use tmp git repos (git IS available on this machine); add an availability skip (`shutil.which("git")` / pytest.importorskip pattern) per re-plan §2.5. `ws = tmp_path/"ws"` fixture pattern.
- Run tests with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest` from the worktree. Known pre-existing failures to deselect: anthropic native-tools, github-api-client.

---

## Task 0: Backlog task

- [ ] Create "Local agent tools phase 3b-ii: git tools (read-only port)". ACs:
  1. git subprocess wrapper enforces fixed-argv allowlist (subcommands + -C/--no-pager only), sanitized env, timeout, 1 MB output cap
  2. git_status/git_diff/git_log/git_blame/git_branches work against workspace-confined repos
  3. Non-repo paths, git-unavailable, and disallowed invocations return model-actionable errors (no raw exceptions)
  4. Injection attempts (flag smuggling via args, path escapes) are refused
  5. All new tests pass
  Commit: `docs: create phase-3b-ii backlog task`

---

## Task 1: `run_git` wrapper + repo preparation

**Files:**
- Create: `tldw_chatbook/Tools/git_tool_impls.py` (attribution header)
- Test: `Tests/Tools/test_git_tool_impls.py`

- [ ] **Step 1: Failing tests**

```python
def test_run_git_version(): ...            # allowlisted --version works
def test_run_git_rejects_disallowed_subcommand(): ...  # ["git", "push"] -> LocalToolError "not allowlisted"
def test_run_git_rejects_global_option_smuggling(): ...  # ["git", "--exec-path=/tmp", "status"] refused; ["git", "-c", "x=y", "status"] refused
def test_run_git_timeout(monkeypatch): ...  # long-running command killed -> "timed out" (simulate or use a tiny timeout on `git log` in a big repo; monkeypatch subprocess if cleaner)
def test_run_git_output_cap(): ...         # output > cap -> truncated flag/marker
def test_run_git_env_sanitized(): ...      # captured env lacks HOME/credentials, has GIT_TERMINAL_PROMPT=0
def test_prepare_repository_finds_root(tmp_git_repo): ...  # nested cwd resolves repo root
def test_prepare_repository_rejects_non_repo(tmp_path): ...  # "not a git repository"
def test_git_unavailable(monkeypatch): ...  # shutil.which -> None: graceful "git is not available" (not an exception)
```

- [ ] **Step 2: Implement** in `git_tool_impls.py`:

```python
GIT_TIMEOUT_SECONDS = 30.0
GIT_MAX_OUTPUT_BYTES = 1_000_000
_ALLOWED_GIT_SUBCOMMANDS = frozenset({"--version", "blame", "branch", "diff", "log", "ls-files", "rev-parse", "status"})

def run_git(argv: list[str], *, timeout: float = GIT_TIMEOUT_SECONDS,
            max_output_bytes: int = GIT_MAX_OUTPUT_BYTES) -> GitCommandResult: ...
    # fixed argv, sanitized env (reference :247-267), stdin=DEVNULL,
    # _validate_argv ported from reference (:270-297) near-verbatim,
    # timeout -> kill + LocalToolError "git command timed out",
    # output capped with truncated marker.
    # Memory-bound: subprocess.run(capture_output=True, timeout=...) buffers ALL
    # output before the cap applies; the reference kills AT the cap via bounded
    # stream reads (:166-223). Prefer Popen with a bounded read loop (kill on
    # exceed) — the reference-faithful option; capture-then-truncate only if
    # documented as the trade-off.

def prepare_repository(workspace_root: Path, path: str = ".") -> Path:
    """Resolve the repo root for ``path`` (confined to workspace_root).

    Refuses when git is unavailable, path escapes the workspace, or no repo
    is found (via `git -C <resolved> rev-parse --show-toplevel`). The repo
    root itself must also be inside the workspace root (a repo whose root is
    the workspace root or below it; a repo ABOVE the workspace root is
    refused — the model shouldn't read repo state outside the confinement;
    matches the reference's _path_inside rule, :2062-2063). A test must pin
    this direction: ws nested INSIDE a tmp repo -> refused.
    """
```

Also port the diff argv flags `--no-ext-diff/--no-textconv/--no-color` from `_run_diff_command` (`:1049-1080`) — they keep output machine-safe.

- [ ] **Step 3:** tests pass — `pytest Tests/Tools/test_git_tool_impls.py -q`
- [ ] **Step 4:** `git commit -m "feat: sync run_git wrapper with ADR-033 allowlist boundary"`

---

## Task 2: Tool cores

**Files:**
- Modify: `tldw_chatbook/Tools/git_tool_impls.py`
- Test: `Tests/Tools/test_git_tool_impls.py`

- [ ] **Step 1: Failing tests** (tmp git repo fixture: init, config user, commits, branch, modification):

```python
def test_git_status_porcelain(tmp_git_repo): ...      # branch line + modified/untracked entries
def test_git_status_not_repo(tmp_path): ...           # graceful error
def test_git_branches(tmp_git_repo): ...              # lists branches, marks current
def test_git_log(tmp_git_repo): ...                   # count cap; format; path filter
def test_git_diff_worktree_and_staged(tmp_git_repo): ...  # both modes; path filter; stat mode
def test_git_diff_commit_range(tmp_git_repo): ...     # HEAD~1..HEAD
def test_git_blame(tmp_git_repo): ...                 # author lines; line range; missing file error
def test_path_filter_confined(tmp_git_repo): ...      # path="../x" refused
```

- [ ] **Step 2: Implement** — adapt the reference's `_execute_status/_execute_branches/_execute_log/_execute_diff/_execute_blame` (`git_module.py:570-955`) to sync: read each reference function and port its argv construction, formatting, and caps (`log` count cap default 20 max 100; diff stat option; blame optional line range). Path filters resolve via `resolve_workspace_path` and are passed to git as repo-relative paths. All failures → `LocalToolError` with the git stderr gist (trimmed). Function signatures:

```python
def git_status(workspace_root: Path, path: str = ".") -> str: ...
def git_branches(workspace_root: Path) -> str: ...
def git_log(workspace_root: Path, *, count: int = 20, path: str | None = None) -> str: ...
def git_diff(workspace_root: Path, *, staged: bool = False, commit_range: str | None = None,
             path: str | None = None, stat: bool = False) -> str: ...
def git_blame(workspace_root: Path, path: str, *, start_line: int | None = None,
              end_line: int | None = None) -> str: ...
```

  **Disclosed deviations from the reference (deliberate, record them in the module header):** (a) `git_diff` adds `commit_range` and `stat` modes the reference does not have (the reference has a third `working_tree` scope instead — map `staged=False`→unstaged, `staged=True`→staged; working_tree omitted for simplicity); (b) `git_log` defaults `count=20` (reference falls back to its max-100 with no default); both remain fixed-argv, read-only, allowlisted — the ADR-033 no-tag rationale is unaffected.
  NOTE on `commit_range`: it goes into argv after validation — it must match `^[A-Za-z0-9._/~^-]+$` (no spaces/semicolons) to keep the fixed-argv guarantee meaningful; test an injection attempt (`"HEAD; rm -rf"` → refused).

- [ ] **Step 3:** tests pass
- [ ] **Step 4:** `git commit -m "feat: git_status/diff/log/blame/branches cores"`

---

## Task 3: Specs + catalog tests

**Files:**
- Modify: `tldw_chatbook/Agents/local_tool_provider.py`
- Test: `Tests/Agents/test_local_tool_provider.py`, `Tests/Agents/test_local_tools_integration.py`

- [ ] **Step 1: Failing tests** — catalog includes all five `local:git_*` ids; all `tags == ()` (ADR-033 binding — NO process tag; the test should pin this so the tripwire stays visible); schemas have correct required params (`git_blame` requires `path`; others optional-only); one handler smoke test per tool against a tmp repo (or one combined).
- [ ] **Step 2: Implement** — five specs in `_default_specs` (order: after `fs_grep`, before `web_fetch`; keep the exact-id test in sync). Descriptions: read-only emphasis ("read-only; cannot modify the repository"), param docs; `git_diff` documents staged/commit_range/stat modes.
- [ ] **Step 3:** update the integration harness count pins (`LOCAL_TOOL_NAMES` + full-catalog count, following the fs_patch precedent); ALSO extend the find/load e2e per re-plan §4 (binding): a scripted run where the model reaches one of the new git tools through `find_tools`/`load_tools` (the harness's find/load test pattern from phase 2/3a) proving new tools stay reachable past the disclosure threshold; run `pytest Tests/Agents/ -q`
- [ ] **Step 4:** `git commit -m "feat: git_* read-only tool specs (no process tag per ADR-033)"`

---

## Task 4: Close-out (controller-led)

- [ ] Backlog task: ACs checked, Implementation Notes, Done.
- [ ] Final review subagent (diff + ACs + ADR-033 boundary verification + test run).
- [ ] superpowers:finishing-a-development-branch.
