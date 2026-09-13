# TASK-19637 Atomic Local-Tool Workspace Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind every structured local filesystem, patch, read-only Git, and equivalent Virtual CLI call to the exact run-admitted workspace identity with a one-shot, fail-closed worker.

**Architecture:** The provider captures a canonical directory identity and validates model arguments before launching a fixed Python helper. The parent admits that helper to the repository's existing process-tree containment owner, sends one bounded JSON request on stdin, and the helper pins the root before dispatching one relative filesystem or Git operation. Local Tool, Virtual CLI, and external MCP use the same executor; legacy sync cores remain only as non-production compatibility adapters.

**Tech Stack:** Python 3.11+, `subprocess`, POSIX directory descriptors/`fchdir`, Windows directory handles and `SetCurrentDirectoryW`, existing `ExecutorProcessTree`, pytest, GitHub Actions.

**Spec:** `Docs/superpowers/specs/2026-08-28-task-19637-atomic-local-tool-workspace-execution-design.md`

## Global Constraints

- ADR required: yes.
- ADR path: `backlog/decisions/101-one-shot-pinned-workspace-tool-execution.md`.
- One helper process is created per operation and always exits; no daemon, pool, retained workspace process, or cross-call mutable worker state.
- Requests use bounded stdin; workspace locators, file content, patches, and model arguments never appear in argv, environment, or generic diagnostics.
- New executor/worker diagnostics use fixed metadata only: no root, relative path, request or response payload, exception text, traceback, or child stderr is emitted to production logs.
- The helper accepts only `fs_list`, `fs_read`, `fs_write`, `fs_edit`, `fs_patch`, `fs_glob`, `fs_grep`, `stat_path`, `git_status`, `git_diff`, `git_log`, `git_blame`, and `git_branches`.
- `shell=False`; no shell text, arbitrary Python, caller-selected module, caller-selected executable, or unsafe `preexec_fn` is accepted.
- The helper is admitted to a retained POSIX process group or Windows kill-on-close Job Object before it receives workspace authority.
- POSIX retains a verified root descriptor and performs I/O through relative names; Windows retains a verified root handle and helper-only current directory.
- Git receives no `-C <workspace>`, launches no new POSIX session inside the helper, and remains in the helper's process tree.
- Existing permission identities, sensitive-path rules, result text, in-root symlink behavior, escaping-symlink name-only glob behavior, configured roots, selected bindings, and linked worktrees remain compatible when no root drift occurs.
- Root retargeting is in scope; a general OS sandbox, raw CLI confinement, mutating Git, transactional multi-file patching, and every descendant-content race are not.
- Exact protocol ceilings: 16 MiB request, 15 MiB individual string, 16 KiB path, `PATCH_MAX_BYTES` patch, `GIT_MAX_OUTPUT_BYTES + 64 KiB` response, 8 KiB diagnostic stderr, and 300-second outer helper timeout. Existing smaller caps still win.
- Development verification is targeted. Do not run the full suite without explicit user approval.

## File Map

- Create `tldw_chatbook/Utils/filesystem_identity.py`: shared canonical directory/ancestor identity capture and reparse detection.
- Modify `tldw_chatbook/Agents/project_instruction_resolver.py`: consume shared identity helpers without changing public behavior.
- Create `tldw_chatbook/Tools/workspace_tool_protocol.py`: closed operations and strict bounded request/response serialization.
- Create `tldw_chatbook/Tools/workspace_root_pin.py`: POSIX and Windows root-pin context managers.
- Create `tldw_chatbook/Tools/workspace_tool_executor.py`: parent one-shot launch, containment, protocol, timeout, redaction, cleanup.
- Create `tldw_chatbook/Tools/workspace_tool_worker.py`: fixed module entry; one request, one operation, one terminal response.
- Create `tldw_chatbook/Tools/workspace_tool_dispatch.py`: normalized relative-operation dispatch inside the pinned helper.
- Modify `tldw_chatbook/Tools/local_tool_impls.py` and `patch_tool_impls.py`: extract relative operation bodies while retaining public sync adapters.
- Modify `tldw_chatbook/Tools/git_tool_impls.py`: explicit cwd, absolute Git executable, inherited helper containment, no `-C`.
- Modify `tldw_chatbook/Agents/local_tool_provider.py`, `Tools/virtual_cli_impls.py`, and `Agents/virtual_cli_provider.py`: route every production frontend through the executor.
- Modify `Docs/security/production-diagnostic-inventory.json` only after reviewing any branch-added diagnostic rows with the repository scanner.
- Test `Tests/Architecture/test_diagnostic_path_privacy.py` and `Tests/Architecture/test_persistent_diagnostic_inventory.py`: enforce the latest `origin/dev` diagnostic privacy and inventory contracts.
- Use `scripts/check_persistent_diagnostic_inventory.py`: review statement-level drift before regenerating the committed inventory.
- Add focused protocol, pinning, race, operation, provider, platform, and performance tests.

---

### Task 0: Synchronize and record the untouched baseline

**Files:** Verify only.

- [ ] **Step 1: Fetch and rebase**

```bash
git fetch origin dev
git rebase origin/dev
git status --short --branch
```

Expected: clean worktree. If upstream changed a File Map path, inspect the semantic delta and revise this plan before code.

- [ ] **Step 2: Establish focused behavior baseline**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Tools/test_local_tool_impls.py Tests/Tools/test_local_tool_impls_properties.py \
  Tests/Tools/test_patch_tool_impls.py Tests/Tools/test_git_tool_impls.py \
  Tests/Tools/test_git_tool_sensitive_paths.py Tests/Tools/test_virtual_cli_impls.py \
  Tests/Agents/test_local_tool_provider.py Tests/Agents/test_virtual_cli_provider.py \
  Tests/Agents/test_project_instruction_runtime.py Tests/STT/test_executor_process_tree.py \
  -q --tb=short
```

Expected: PASS. Record exact count and warnings.

- [ ] **Step 3: Record touched-file lint baseline**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/Agents/project_instruction_resolver.py \
  tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/Agents/virtual_cli_provider.py \
  tldw_chatbook/Tools/local_tool_impls.py tldw_chatbook/Tools/patch_tool_impls.py \
  tldw_chatbook/Tools/git_tool_impls.py tldw_chatbook/Tools/virtual_cli_impls.py
```

Expected: PASS or a recorded exact pre-existing failure set that the branch must not enlarge.

### Task 1: Share directory identity and lock the protocol contract

**Files:**

- Create: `tldw_chatbook/Utils/filesystem_identity.py`
- Create: `tldw_chatbook/Tools/workspace_tool_protocol.py`
- Create: `Tests/Utils/test_filesystem_identity.py`
- Create: `Tests/Tools/test_workspace_tool_protocol.py`
- Modify: `tldw_chatbook/Agents/project_instruction_resolver.py:105-116,409-435,717-761`
- Test: `Tests/Agents/test_project_instruction_resolver.py`
- Test: `Tests/Agents/test_project_instruction_runtime.py`

**Interfaces:**

- `DirectoryIdentity(device: int, inode: int, mode: int, reparse: bool)`.
- `DirectoryChain(canonical_root: Path, identities: tuple[DirectoryIdentity, ...])`, root first.
- `capture_directory_chain(root: Path) -> DirectoryChain`.
- `WorkspaceToolRequest.from_bytes(raw: bytes)`, `.to_bytes()`, and equivalent `WorkspaceToolResponse` methods.
- `WorkspaceOperation` and `WorkspaceIntent` literal aliases consumed by all later tasks.

- [ ] **Step 1: Write RED identity tests**

Cover canonical alias capture, root-first ancestors, symlink/reparse rejection at the canonical locator, missing Windows attributes, and stable equality:

```python
def test_capture_directory_chain_canonicalizes_stable_alias(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    chain = capture_directory_chain(alias)
    assert chain.canonical_root == root.resolve()
    assert chain.identities[0] == directory_identity_from_stat(os.stat(root))
```

- [ ] **Step 2: Verify identity tests fail**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Utils/test_filesystem_identity.py -q --tb=short
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement shared identity values and adopt them in the resolver**

```python
@dataclass(frozen=True, slots=True)
class DirectoryIdentity:
    device: int
    inode: int
    mode: int
    reparse: bool

@dataclass(frozen=True, slots=True)
class DirectoryChain:
    canonical_root: Path = field(repr=False)
    identities: tuple[DirectoryIdentity, ...] = field(repr=False)
```

Preserve `BindingRootIdentity` as the resolver-facing type, built from this helper. On Windows, absent `st_file_attributes` fails closed.

- [ ] **Step 4: Write RED protocol tests**

Cover exact keys, version, every operation/intent, identity encoding, duplicate JSON keys, non-finite values, NUL, wrong types, unknown keys, each ceiling, wrong operation ID, malformed frames, and redacted repr/errors.

```python
def test_duplicate_json_keys_fail_before_request_construction() -> None:
    with pytest.raises(WorkspaceProtocolError, match="duplicate key"):
        WorkspaceToolRequest.from_bytes(b'{"version":1,"version":1}')
```

- [ ] **Step 5: Verify protocol tests fail**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_workspace_tool_protocol.py -q --tb=short
```

Expected: FAIL because the protocol module does not exist.

- [ ] **Step 6: Implement strict JSON framing**

Use `object_pairs_hook` for duplicate rejection, `parse_constant` rejection, `allow_nan=False`, exact type checks where bool/int ambiguity matters, and UTF-8 byte counts. Add no serialization dependency.

- [ ] **Step 7: Verify and commit**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Utils/test_filesystem_identity.py Tests/Tools/test_workspace_tool_protocol.py \
  Tests/Agents/test_project_instruction_resolver.py Tests/Agents/test_project_instruction_runtime.py \
  -q --tb=short
git add tldw_chatbook/Utils/filesystem_identity.py tldw_chatbook/Tools/workspace_tool_protocol.py \
  tldw_chatbook/Agents/project_instruction_resolver.py Tests/Utils/test_filesystem_identity.py \
  Tests/Tools/test_workspace_tool_protocol.py
git commit -m "feat(security): define workspace execution protocol"
```

Expected: PASS, then one focused commit.

### Task 2: Deliver one contained, pinned `stat_path` operation

**Files:**

- Create: `tldw_chatbook/Tools/workspace_root_pin.py`
- Create: `tldw_chatbook/Tools/workspace_tool_executor.py`
- Create: `tldw_chatbook/Tools/workspace_tool_worker.py`
- Create: `tldw_chatbook/Tools/workspace_tool_dispatch.py`
- Create: `Tests/Tools/test_workspace_root_pin.py`
- Create: `Tests/Tools/test_workspace_tool_executor.py`
- Modify: `tldw_chatbook/Tools/local_tool_impls.py`
- Test: `Tests/Architecture/test_diagnostic_path_privacy.py`
- Reference: `tldw_chatbook/STT/executor_process_tree.py`

**Interfaces:**

- `PinnedWorkspaceRoot` context with `canonical_locator`, `identity`, POSIX `root_fd`, and `relative_path(value: str) -> Path`.
- `WorkspaceToolExecutor(workspace_root: Path).execute(operation, arguments, *, intent) -> str`.
- `execute_pinned_operation(request, root) -> str`; initially `stat_path` succeeds and other closed operations return `unsupported_operation`.

- [ ] **Step 1: Write deterministic RED root-race tests**

Use a dedicated child process and `multiprocessing.Event`, never sleeps. Before pin, replace A with B and expect identity mismatch. After pin, attempt replacement and read `sentinel.txt` relative to the retained pin; expect A or documented Windows sharing refusal, never B.

```python
def _post_pin_child(locator: str, chain: DirectoryChain, ready, resume, output) -> None:
    with pin_workspace_root(Path(locator), chain) as pinned:
        ready.set()
        if not resume.wait(5):
            raise RuntimeError("test barrier timed out")
        output.put(pinned.relative_path("sentinel.txt").read_text(encoding="utf-8"))
```

- [ ] **Step 2: Verify root-pin tests fail**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_workspace_root_pin.py -q --tb=short
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement POSIX and Windows pins**

POSIX opens no-follow/directory/close-on-exec, compares `fstat`, calls `fchdir`, verifies `.`, and retains the descriptor. Windows opens without following reparse points, compares volume/file identity, calls helper-only `SetCurrentDirectoryW`, verifies `.`, and retains the handle. Unsupported capability raises `WorkspaceRootPinError`; no fallback.

- [ ] **Step 4: Write RED executor lifecycle/privacy tests**

Cover fixed argv, `shell=False`, allowlisted environment, stdin-only request, containment before request write, one terminal frame, timeout, crash, malformed/oversized response, bounded stderr, cleanup-unproven refusal, and no in-process fallback. Assert marker root/content/exception/stderr strings are absent from argv, env, and production logs, and that only fixed operation/status metadata may be logged.

- [ ] **Step 5: Verify executor tests fail**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_workspace_tool_executor.py -q --tb=short
```

Expected: FAIL because executor/worker modules do not exist.

- [ ] **Step 6: Implement the executor and fixed worker**

Launch only:

```python
argv = [sys.executable, "-I", "-m", "tldw_chatbook.Tools.workspace_tool_worker"]
process = subprocess.Popen(
    argv,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=workspace_worker_environment(),
    shell=False,
    start_new_session=(os.name == "posix"),
)
```

Wrap Popen with the minimal `is_alive/join/terminate/kill` adapter expected by `ExecutorProcessTree`; do not copy Job Object code. Admit before writing stdin. Helper parses one request, pins once, dispatches once, emits one terminal response, and exits.

- [ ] **Step 7: Extract relative `stat_path`, verify, and commit**

The worker's private stat body accepts only an already validated relative `Path` and never resolves it to an absolute pathname before I/O. Keep public `stat_path(path, workspace_root=...)` unchanged.

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Tools/test_workspace_root_pin.py Tests/Tools/test_workspace_tool_executor.py \
  Tests/Tools/test_local_tool_impls.py::test_stat_path_returns_only_allowlisted_workspace_metadata \
  Tests/STT/test_executor_process_tree.py Tests/Architecture/test_diagnostic_path_privacy.py \
  -q --tb=short
git add tldw_chatbook/Tools/workspace_root_pin.py tldw_chatbook/Tools/workspace_tool_executor.py \
  tldw_chatbook/Tools/workspace_tool_worker.py tldw_chatbook/Tools/workspace_tool_dispatch.py \
  tldw_chatbook/Tools/local_tool_impls.py Tests/Tools/test_workspace_root_pin.py \
  Tests/Tools/test_workspace_tool_executor.py
git commit -m "feat(security): run pinned workspace stat operations"
```

Expected: PASS, then one vertical-slice commit.

### Task 3: Move every read-only filesystem operation behind the pin

**Files:**

- Modify: `tldw_chatbook/Tools/workspace_tool_dispatch.py`
- Modify: `tldw_chatbook/Tools/local_tool_impls.py:196-329,405-513`
- Modify: `Tests/Tools/test_workspace_tool_executor.py`
- Modify: `Tests/Tools/test_local_tool_impls.py`
- Modify: `Tests/Tools/test_local_tool_impls_properties.py`
- Test: `Tests/Tools/test_local_tool_sensitive_paths.py`

**Interfaces:** Pinned handlers for `fs_list`, `fs_read`, `fs_glob`, and `fs_grep`; parent normalization converts admitted targets to relative strings and computes bounded relative sensitive exclusions.

- [ ] **Step 1: Write RED parameterized A/B tests**

For each operation, create A/B roots with distinct sentinels, replace before pin, and assert output is A/refusal and never B. Add in-root symlink, escaping read/grep symlink, and name-only glob cases.

```python
READ_CASES = (
    ("fs_list", {"path": "."}, "A_ONLY"),
    ("fs_read", {"path": "sentinel.txt", "offset": 1, "limit": None}, "A_ONLY"),
    ("fs_glob", {"pattern": "**/*.txt", "max_results": 100}, "sentinel.txt"),
    ("fs_grep", {"pattern": "A_ONLY", "mode": "content", "max_results": 100}, "A_ONLY"),
)
```

- [ ] **Step 2: Verify RED failures**

Run the new parameterized nodes. Expected: stable `unsupported_operation` failures, not timeouts.

- [ ] **Step 3: Extract relative operation bodies**

Parent uses current confinement/sensitive validation while A is admitted. Worker uses normalized relative paths only. Enumeration receives bounded root-relative sensitive exclusions, preserves scan/work caps and ordering, and never calls `.resolve()` then opens that absolute result.

- [ ] **Step 4: Verify and commit**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Tools/test_workspace_tool_executor.py Tests/Tools/test_local_tool_impls.py \
  Tests/Tools/test_local_tool_impls_properties.py Tests/Tools/test_local_tool_sensitive_paths.py \
  -q --tb=short
git add tldw_chatbook/Tools/workspace_tool_dispatch.py tldw_chatbook/Tools/local_tool_impls.py \
  Tests/Tools/test_workspace_tool_executor.py Tests/Tools/test_local_tool_impls.py \
  Tests/Tools/test_local_tool_impls_properties.py
git commit -m "feat(security): pin local filesystem reads"
```

Expected: PASS, including current ordering, truncation, binary refusal, line numbering, and symlink behavior.

### Task 4: Move filesystem mutations and patch behind one pin

**Files:**

- Modify: `tldw_chatbook/Tools/workspace_tool_dispatch.py`
- Modify: `tldw_chatbook/Tools/local_tool_impls.py:332-402`
- Modify: `tldw_chatbook/Tools/patch_tool_impls.py:381-449`
- Modify: `Tests/Tools/test_workspace_tool_executor.py`
- Modify: `Tests/Tools/test_local_tool_impls.py`
- Modify: `Tests/Tools/test_patch_tool_impls.py`
- Test: `Tests/Tools/test_local_tool_sensitive_paths.py`

**Interfaces:**

- Pinned handlers for `fs_write`, `fs_edit`, and `fs_patch`.
- `patch_validated_files(plans: tuple[PatchFile, ...], *, root: PinnedWorkspaceRoot, dry_run: bool = False) -> str`.

- [ ] **Step 1: Write RED A/B mutation tests**

For write, edit, and a two-file patch, pause before pin, replace A with B, resume, and assert every B file plus an external sentinel remain byte-exact. For post-pin, attempt replacement and assert changes land only in A or refuse. Assert one patch request emits one `admitted` frame and retains one root identity across both files.

```python
MUTATION_CASES = (
    ("fs_write", {"path": "note.txt", "content": "changed"}),
    ("fs_edit", {"path": "note.txt", "old_string": "before", "new_string": "after", "replace_all": False}),
)
```

- [ ] **Step 2: Verify RED mutation failures**

Run the exact new nodes. Expected: `unsupported_operation`; byte-exact B assertions pass.

- [ ] **Step 3: Extract relative mutation/patch bodies**

Encode before opening, preserve newline and partial multi-file semantics, and use relative I/O only. Parent parses and validates all patch targets before spawn. Worker reparses bounded patch text and rejects any target set differing from the normalized request target list.

- [ ] **Step 4: Verify and commit**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Tools/test_workspace_tool_executor.py Tests/Tools/test_local_tool_impls.py \
  Tests/Tools/test_patch_tool_impls.py Tests/Tools/test_local_tool_sensitive_paths.py \
  -q --tb=short
git add tldw_chatbook/Tools/workspace_tool_dispatch.py tldw_chatbook/Tools/local_tool_impls.py \
  tldw_chatbook/Tools/patch_tool_impls.py Tests/Tools/test_workspace_tool_executor.py \
  Tests/Tools/test_local_tool_impls.py Tests/Tools/test_patch_tool_impls.py
git commit -m "feat(security): pin local filesystem mutations"
```

Expected: PASS, then one mutation commit.

### Task 5: Run read-only Git inside the helper's containment owner

**Files:**

- Modify: `tldw_chatbook/Tools/git_tool_impls.py:132-317,436-602,602-1035`
- Modify: `tldw_chatbook/Tools/workspace_tool_dispatch.py`
- Modify: `Tests/Tools/test_git_tool_impls.py`
- Modify: `Tests/Tools/test_git_tool_sensitive_paths.py`
- Modify: `Tests/Tools/test_workspace_tool_executor.py`
- Test: `Tests/Chunking/test_sync_script.py::test_validated_source_accepts_linked_git_worktree`

**Interfaces:**

Extend `run_git` to the exact signature `run_git(argv: list[str], *, cwd:
Path | None = None, executable: Path | None = None, own_process_group: bool =
True, timeout: float = GIT_TIMEOUT_SECONDS, max_output_bytes: int =
GIT_MAX_OUTPUT_BYTES) -> GitCommandResult`.

Worker dispatch supplies relative repository cwd, absolute `shutil.which("git")`, and `own_process_group=False`. Direct compatibility adapters may use absolute cwd and `own_process_group=True`.

- [ ] **Step 1: Write RED argv/tree/race tests**

Assert all five operations omit `-C`, Popen argv zero is absolute, worker Git uses relative cwd and `start_new_session=False`, outer helper owns timeout cleanup, and A/B replacement never returns B status/diff/log/blame/branch content.

```python
def test_worker_git_uses_relative_cwd_without_new_session(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr(subprocess, "Popen", recording_popen(seen))
    run_git(["git", "status"], cwd=Path("repo"), executable=Path("/usr/bin/git"), own_process_group=False)
    assert seen["argv"][0] == "/usr/bin/git"
    assert "-C" not in seen["argv"]
    assert seen["cwd"] == Path("repo")
    assert seen["start_new_session"] is False
```

- [ ] **Step 2: Verify RED Git failures**

Run exact new nodes. Expected: current `run_git` rejects new kwargs or records `-C`/new session.

- [ ] **Step 3: Implement cwd-based launch and pinned dispatch**

Logical argv still begins `git` for validation/result compatibility; Popen receives `[str(executable), *argv[1:]]`. Remove `-C` from discovery and operations. Preserve subcommands, machine-safe flags, sensitive pathspecs, output caps, and timeouts. When `own_process_group=False`, direct kill does not signal a new group; outer retained tree proves descendant cleanup.

- [ ] **Step 4: Prove linked-worktree compatibility**

Create a real linked worktree whose `.git` file points outside its working root. Run helper Git operations successfully while a normal local filesystem read of that external administrative path remains unauthorized.

- [ ] **Step 5: Verify and commit**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Tools/test_git_tool_impls.py Tests/Tools/test_git_tool_sensitive_paths.py \
  Tests/Tools/test_workspace_tool_executor.py \
  Tests/Chunking/test_sync_script.py::test_validated_source_accepts_linked_git_worktree \
  -q --tb=short
git add tldw_chatbook/Tools/git_tool_impls.py tldw_chatbook/Tools/workspace_tool_dispatch.py \
  Tests/Tools/test_git_tool_impls.py Tests/Tools/test_git_tool_sensitive_paths.py \
  Tests/Tools/test_workspace_tool_executor.py
git commit -m "feat(security): pin read-only Git execution"
```

Expected: PASS, then one Git commit.

### Task 6: Route every production frontend through the executor

**Files:**

- Modify: `tldw_chatbook/Agents/local_tool_provider.py:228-327,748-806,1126-1425`
- Modify: `tldw_chatbook/Tools/virtual_cli_impls.py:175-230`
- Modify: `tldw_chatbook/Agents/virtual_cli_provider.py:95-127,273-327`
- Modify: `Tests/Agents/test_local_tool_provider.py`
- Modify: `Tests/Agents/test_virtual_cli_provider.py`
- Modify: `Tests/Tools/test_virtual_cli_impls.py`
- Modify: `Tests/Agents/test_local_tools_integration.py`
- Test: `Tests/Chat/test_console_agent_project_instructions.py`
- Test: `Tests/Chat/test_console_agent_bridge_local.py`
- Test: `Tests/MCP/test_control_plane_permissions.py`
- Test: `Tests/MCP/test_local_server_tools.py`

**Interfaces:**

- Add keyword `workspace_executor: WorkspaceToolExecutor | None = None` to
  `LocalToolProvider.__init__`; it constructs the real executor when omitted,
  while tests may inject a recording fake.
- Add required keyword `workspace_executor: WorkspaceToolExecutor` to
  `_default_specs`; it binds only path-authority handlers to `.execute`.
- `VirtualCliRegistry(workspace_root, *, workspace_executor=None)` retains direct cores only for explicitly unleased compatibility tests.
- Add keyword `workspace_executor: WorkspaceToolExecutor | None = None` to
  `VirtualCliProvider.__init__`; it constructs/injects a real executor by
  default.

- [ ] **Step 1: Write RED no-bypass tests**

Inject a recording executor; assert each Local Tool and Virtual CLI path command creates exactly one operation/intent/args call. Monkeypatch every direct local/Git core to raise if reached. Assert web/todo/Watchlists do not launch helpers. Construct external MCP local tools and prove `fs_read` reaches the executor.

- [ ] **Step 2: Verify provider tests fail**

Expected: direct handlers and Virtual CLI registry reach the fail-fast monkeypatches.

- [ ] **Step 3: Wire exact intent and refusal mapping**

| Operations | Intent |
| --- | --- |
| `fs_write`, `fs_edit`, `fs_patch` | `write` |
| every other workspace/Git operation | `read` |

Keep root guard, authority scope, permission verdict, and kill switch before executor admission. Root mismatch maps to `LOCAL_ROOT_CHANGED_REFUSAL`; containment/protocol/platform inability maps to `LOCAL_AUTHORITY_UNAVAILABLE_REFUSAL`; domain failures retain bounded current text.

- [ ] **Step 4: Prove schemas/permissions unchanged**

Assert catalog order, every schema, risk tags, definition hashes, approval copy, `allow_write=False`, path-target preflight, and Virtual CLI independent permission identities match baseline.

- [ ] **Step 5: Verify and commit**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Agents/test_local_tool_provider.py Tests/Agents/test_virtual_cli_provider.py \
  Tests/Agents/test_local_tools_integration.py Tests/Tools/test_virtual_cli_impls.py \
  Tests/Chat/test_console_agent_project_instructions.py Tests/Chat/test_console_agent_bridge_local.py \
  Tests/MCP/test_control_plane_permissions.py Tests/MCP/test_local_server_tools.py \
  -q --tb=short
git add tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/Agents/virtual_cli_provider.py \
  tldw_chatbook/Tools/virtual_cli_impls.py Tests/Agents/test_local_tool_provider.py \
  Tests/Agents/test_virtual_cli_provider.py Tests/Agents/test_local_tools_integration.py \
  Tests/Tools/test_virtual_cli_impls.py
git commit -m "feat(security): lease all local workspace frontends"
```

Expected: PASS, then one routing commit.

### Task 7: Add bounded platform and performance evidence

**Files:**

- Create: `.github/workflows/task-19637-platform-evidence.yml`
- Create: `Tests/CI/test_task19637_platform_evidence.py`
- Create: `Tests/Performance/run_workspace_tool_executor_profile.py`
- Create: `Tests/Performance/test_workspace_tool_executor_profile.py`
- Modify: `Tests/Tools/test_workspace_root_pin.py`
- Modify: `Tests/Tools/test_workspace_tool_executor.py`

**Interfaces:**

- Workflow label `task-19637-platform-evidence`; `ubuntu-24.04`, `windows-2022`, `macos-15-intel`; Python 3.12; exact PR head; read-only permissions; 30-minute timeout.
- Profile CLI: `python Tests/Performance/run_workspace_tool_executor_profile.py --samples 30 --output <path>`.
- JSON keys: `schema_version`, `head_commit`, `platform`, `python`, `samples`, and `operations`; operation metrics contain `direct_ms`, `one_shot_ms`, and `startup_overhead_ms`, each with `median` and nearest-rank `p95`.

- [ ] **Step 1: Write RED workflow contract tests**

Assert only labeled-PR/manual triggers, exact matrix/head checkout, no secrets/write permissions, bounded test nodes, JUnit upload, and failure propagation.

- [ ] **Step 2: Add three-OS workflow**

Run root-pin races, executor lifecycle/privacy, one representative read/write/patch/Git, linked-worktree, and provider-routing test. Do not run the full suite or install unrelated extras.

- [ ] **Step 3: Write RED profile tests and implement runner**

Test nearest-rank p95, JSON finiteness, exact operations (`stat`, `read`, `write`, `list`, `git_status`, `git_diff`), content-free metadata, and fake clock/sample injection. Add no timing threshold.

- [ ] **Step 4: Run profile and evidence tests**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python Tests/Performance/run_workspace_tool_executor_profile.py \
  --samples 30 --output /tmp/task-19637-workspace-tool-profile.json
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Performance/test_workspace_tool_executor_profile.py \
  Tests/CI/test_task19637_platform_evidence.py -q --tb=short
```

Expected: finite median/p95 for all operations and PASS. Record profile hash and summary; do not commit host-specific output.

- [ ] **Step 5: Run full targeted feature shard**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Utils/test_filesystem_identity.py Tests/Tools/test_workspace_tool_protocol.py \
  Tests/Tools/test_workspace_root_pin.py Tests/Tools/test_workspace_tool_executor.py \
  Tests/Tools/test_local_tool_impls.py Tests/Tools/test_local_tool_impls_properties.py \
  Tests/Tools/test_local_tool_sensitive_paths.py Tests/Tools/test_patch_tool_impls.py \
  Tests/Tools/test_git_tool_impls.py Tests/Tools/test_git_tool_sensitive_paths.py \
  Tests/Tools/test_virtual_cli_impls.py Tests/Agents/test_local_tool_provider.py \
  Tests/Agents/test_virtual_cli_provider.py Tests/Agents/test_local_tools_integration.py \
  Tests/Agents/test_project_instruction_resolver.py Tests/Agents/test_project_instruction_runtime.py \
  Tests/STT/test_executor_process_tree.py Tests/CI/test_task19637_platform_evidence.py \
  Tests/Performance/test_workspace_tool_executor_profile.py -q --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit evidence infrastructure**

```bash
git add .github/workflows/task-19637-platform-evidence.yml \
  Tests/CI/test_task19637_platform_evidence.py \
  Tests/Performance/run_workspace_tool_executor_profile.py \
  Tests/Performance/test_workspace_tool_executor_profile.py \
  Tests/Tools/test_workspace_root_pin.py Tests/Tools/test_workspace_tool_executor.py
git commit -m "test(security): prove pinned workspace execution"
```

### Task 8: Final review, documentation, and merge readiness

**Files:**

- Modify: `backlog/decisions/101-one-shot-pinned-workspace-tool-execution.md`
- Modify: `backlog/tasks/task-19637 - Atomically-pin-local-tool-workspace-execution.md`
- Modify: `Docs/security/production-diagnostic-inventory.json` only when reviewed branch-added diagnostics require it.
- Modify: `backlog/docs/lessons-testing-evidence.md` only for a real reusable incident.

- [ ] **Step 1: Run branch static gates**

```bash
git diff --check origin/dev...HEAD
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/Utils/filesystem_identity.py tldw_chatbook/Tools/workspace_tool_protocol.py \
  tldw_chatbook/Tools/workspace_root_pin.py tldw_chatbook/Tools/workspace_tool_executor.py \
  tldw_chatbook/Tools/workspace_tool_worker.py tldw_chatbook/Tools/workspace_tool_dispatch.py \
  tldw_chatbook/Tools/local_tool_impls.py tldw_chatbook/Tools/patch_tool_impls.py \
  tldw_chatbook/Tools/git_tool_impls.py tldw_chatbook/Tools/virtual_cli_impls.py \
  tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/Agents/virtual_cli_provider.py \
  tldw_chatbook/Agents/project_instruction_resolver.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q \
  tldw_chatbook/Utils/filesystem_identity.py tldw_chatbook/Tools/workspace_tool_protocol.py \
  tldw_chatbook/Tools/workspace_root_pin.py tldw_chatbook/Tools/workspace_tool_executor.py \
  tldw_chatbook/Tools/workspace_tool_worker.py tldw_chatbook/Tools/workspace_tool_dispatch.py
```

Expected: PASS or no branch-added lint delta against Task 0.

- [ ] **Step 2: Review diagnostic privacy and inventory drift**

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Architecture/test_diagnostic_path_privacy.py \
  Tests/Architecture/test_persistent_diagnostic_inventory.py -q --tb=short
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  scripts/check_persistent_diagnostic_inventory.py --diff
```

Expected: privacy tests pass. If the inventory reports branch-added rows, inspect the statement-level report and confirm every new diagnostic contains fixed metadata only. Then, and only then, run `python scripts/check_persistent_diagnostic_inventory.py --write`, inspect the JSON diff, rerun both architecture tests and `--diff`, and include the inventory in the relevant implementation commit. Never regenerate merely to silence drift.

- [ ] **Step 3: Review boundary tripwires**

```bash
rg -n 'git.*-C|start_new_session=True|preexec_fn|shell=True' \
  tldw_chatbook/Tools/workspace_* tldw_chatbook/Tools/git_tool_impls.py
rg -n 'LocalToolProvider\(|VirtualCliProvider\(|VirtualCliRegistry\(' tldw_chatbook
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD
```

Expected: only the fixed helper owns a new POSIX session; Git does not. All production frontends are leased, raw CLI is unchanged, no pool exists, and no workspace locator enters argv/env/logs.

- [ ] **Step 4: Request full-suite approval**

Ask the user whether to run the repository-wide suite. If approved, run the documented full command and record exact output. If not, final evidence is the targeted shard plus three-OS workflow.

- [ ] **Step 5: Rebase and repeat affected verification**

Fetch/rebase `origin/dev`. Rerun all targeted tests if affected files changed; otherwise rerun protocol, races, provider routing, Git, static, and diff checks. Recheck ADR-101 against `origin/dev` and open PR file lists; renumber all references if collision appears.

- [ ] **Step 6: Complete task and ADR**

Set ADR-101 to `Accepted`; check all six acceptance criteria; add concise Implementation Notes with approach, files, trade-offs, exact targeted/cross-platform/performance evidence, full-suite decision, and ADR; set task status `Done`. Add a lesson only when a concrete incident generalizes.

- [ ] **Step 7: Commit closeout docs and prove clean tree**

```bash
git add backlog/decisions/101-one-shot-pinned-workspace-tool-execution.md \
  "backlog/tasks/task-19637 - Atomically-pin-local-tool-workspace-execution.md" \
  Docs/security/production-diagnostic-inventory.json
git commit -m "docs(security): close pinned workspace execution task"
git status --short --branch
git log --oneline --decorate origin/dev..HEAD
git diff --check origin/dev...HEAD
```

Expected: intentional TASK-19637 commits only, no modified/untracked files, no whitespace errors.
