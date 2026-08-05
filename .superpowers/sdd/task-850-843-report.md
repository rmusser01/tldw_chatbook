# TASK-850 / TASK-843 — implementation report

**Worktree:** `/Users/macbook-dev/Documents/GitHub/wt-path-hardening` (branch `feat/agent-path-hardening`)
**Interpreter:** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`, run from this worktree (the venv's editable install resolves `tldw_chatbook` to the ORIGINAL checkout at `/Users/macbook-dev/Documents/GitHub/tldw_chatbook` unless cwd-relative `sys.path[0]` resolution picks up this worktree first — true for `python -m pytest`/`python -c` invoked from this directory, NOT true for a standalone script file, which uses its own directory instead. Every ad-hoc verification below was run via `-c` from this worktree, or with an explicit `sys.path.insert(0, ...)`, after one false start caught by this exact trap).

Commits: `7fd033f1b` (TASK-850), `8d66becf9` (TASK-843), both on `feat/agent-path-hardening`.

---

## TASK-850 — scope `glob_files`/`grep_files` to workspace folder roots

**Gap.** `read_file`/`write_file`/`list_directory` all resolve their root set via `allowed_file_roots(write=…, sandbox_root=…)` (the tool sandbox plus every workspace folder bound to the run). `glob_files`/`grep_files` globbed `_tool_sandbox_root()` directly — strictly narrower, so safe, but inconsistent: an agent could `read_file` a path inside a bound workspace folder that `glob_files`/`grep_files` could never surface.

### Root-set resolution

Both tools now call the exact same accessor the other three use:
```python
roots = allowed_file_roots(write=False, sandbox_root=_tool_sandbox_root())
```

### Merging across roots without multiplying the worst case

A new shared generator, `_iter_candidates_across_roots(pattern, roots, sensitive_ctx)`, globs each usable root in turn and yields validated candidates. Every existing guard applies to every candidate from every root:

- **Containment** (`is_within`, which also applies the sensitive-path denylist) — checked per candidate against the SPECIFIC root that produced it (a candidate from `root.glob()` is checked against that same `root`, not "any root").
- **Hidden-component rule** (`_is_hidden_within`) — same per-root pairing.
- **`_MAX_CANDIDATES`** — a single `examined` counter shared across ALL roots (declared once, outside the per-root loop), so N configured roots cannot multiply the worst-case walk by N. Verified by `test_grep_candidate_bound_is_shared_across_roots_not_multiplied` (5 sandbox files + 5 bound-folder files, cap set to 3, matches ≤ 3).
- **Deduplication** — a `seen_resolved: set[Path]` shared across roots, so a path reachable through more than one root (e.g. an accidental overlap between the sandbox and a bound folder) is yielded once, not once per root that can see it.

### Dotted-root rule, decided and documented

`_sandbox_root_is_hidden(root)` previously ran once against the single sandbox root; a dotted root refused the WHOLE call (mirroring `validate_path`'s hidden-base-directory rejection). With a root SET, the decision made and documented in both `_sandbox_root_is_hidden`'s and `_iter_candidates_across_roots`'s docstrings:

> Each root is checked independently. A dotted root is **excluded from the search**, not fatal to every other, still-valid root's results. The call is refused outright only when **zero roots survive** that filter — which is exactly the pre-existing single-root (sandbox-only, dotted) case, so that behavior is byte-for-byte unchanged.

```python
usable_roots = tuple(root for root in roots if not _sandbox_root_is_hidden(root))
if not usable_roots:
    return {"error": "Access to hidden files/directories is not allowed"}
```

Pinned by `test_dotted_workspace_root_is_skipped_not_fatal_to_other_roots`: a dotted bound folder and an ordinary bound folder both configured; the dotted one's content never appears, the ordinary one's does, and the call succeeds (no error) — as opposed to the single-root case (`test_glob_files_refuses_a_dotted_sandbox_root`, pre-existing, still passing unmodified) where the identical dotted-root misconfiguration DOES refuse the whole call because it's the only root there is.

### `SensitivePathContext` resolved once per call

Unchanged in shape from before this task (`resolve_sensitive_context()` called once in `execute()`), now threaded through every root's candidates via `_iter_candidates_across_roots` rather than just the sandbox's.

### Proof: no reach beyond `read_file`

`test_glob_grep_and_the_read_family_all_refuse_a_path_outside_every_root` exercises all five tools against a path outside every configured root (sandbox + bound workspace folder):

- `read_file`/`write_file`/`list_directory` — refused directly, against the literal outside path (each takes a target-path argument).
- `glob_files`/`grep_files` — take no target-path argument, only a relative glob pattern, and `_rejects_traversal` already refuses any absolute form or `..` component outright, so the meaningful equivalent is a **symlink planted inside an allowed root** pointing at the outside directory (the one real avenue a search tool could otherwise reach it through). `is_within`'s resolved-ancestry containment check refuses the symlinked target exactly like it refuses everything else outside every root — neither tool's `matches` ever contains the outside file's name or content.

### Tests

`Tests/Tools/test_file_tools_workspace_roots.py` (+165 lines): `test_glob_finds_file_in_bound_workspace_folder`, `test_grep_finds_content_in_bound_workspace_folder`, `test_glob_merges_matches_across_sandbox_and_bound_folder`, `test_grep_candidate_bound_is_shared_across_roots_not_multiplied`, `test_dotted_workspace_root_is_skipped_not_fatal_to_other_roots`, `test_glob_grep_and_the_read_family_all_refuse_a_path_outside_every_root`.

Every pre-existing test in `Tests/Tools/test_glob_grep_files.py` (34 tests, sandbox-only configuration) still passes unmodified — including `test_default_sandbox_configuration_still_works_end_to_end` in `test_file_tool_sandbox.py`, which exercises the real, unmocked default configuration through all five tools.

**Files:** `tldw_chatbook/Tools/file_operation_tools.py` (`GlobFiles.execute`, `GrepFiles.execute`, new `_iter_candidates_across_roots`, `_sandbox_root_is_hidden` docstring generalized), `Tests/Tools/test_file_tools_workspace_roots.py`.

---

## TASK-843 — complete the `grep_files` catastrophic-backtracking mitigation

**Gap.** `_MAX_GREP_LINE_SEARCH_CHARS`/`_MAX_GREP_LINES_SCANNED`/`GrepFiles.timeout_seconds` made the worst case finite and small, but didn't stop it: Python's `re` has no match timeout, and `Agents/agent_service.py`'s `_call_with_timeout` runs each tool call on a daemon thread and, on timeout, **abandons** that thread rather than killing it (Python cannot forcibly kill a thread). A pathological pattern kept burning real CPU after the agent was told the call failed, and repeated calls would accumulate abandoned, still-running threads.

### Options weighed

1. **Killable subprocess** (chosen). Only approach that actually bounds CPU past return, since a process — unlike a thread — genuinely can be killed (`SIGKILL`/`TerminateProcess`).
2. **Third-party `regex` module** (supports `timeout=`). Checked whether it's already a dependency before proposing it, per the task's instruction: it **is** importable in this dev venv, but only **transitively**, via `nltk`/`transformers`/`dateparser` (all pulled in by the optional `embeddings_rag`/similar extras — confirmed via `pip show regex` → `Required-by: dateparser, nltk, transformers`, and `pyproject.toml` has no `regex` entry anywhere, direct or under `[project.optional-dependencies]`). `grep_files` is a core built-in, reachable with **zero** extras installed. Depending on `regex` here would silently turn a currently-optional, transitive package into a hard requirement of the base install — a real cost this project's own optional-dependency convention (`Utils/optional_deps.py`, lazy-checked extras) exists specifically to avoid. Ruled out for that reason, not for any deficiency in `regex` itself.
3. Nothing else considered materially better: `signal.alarm`-based timeouts don't work across threads and are POSIX-only; `multiprocessing` with the default `'spawn'` start method risks re-executing whatever the ORIGINAL entry-point script's module-level code is (a real hazard for a long-lived TUI app or under pytest) — a plain `subprocess.Popen` running a dependency-free worker script sidesteps that entirely and gives the same guarantee.

### What changed

The actual line-by-line regex search moved into a new standalone worker, `tldw_chatbook/Tools/_grep_worker.py`: stdlib-only (`json`/`re`/`sys`/`pathlib`, optionally `resource`), **no import of `tldw_chatbook` at all** — deliberately, so the child carries none of the host process's live state (open DB connections, loaded config, the running Textual app) and starts fast. Protocol: one JSON request on stdin (`pattern`, `file_paths`, the four existing bound constants), one JSON response on stdout (`matches`/`lines_scanned` or `error`); always exits 0.

`GrepFiles.execute` still does candidate discovery in-process (containment, sensitive-path denylist, hidden-component rule, `_MAX_CANDIDATES`, via the same `_iter_candidates_across_roots` TASK-850 added) — none of that runs the user-supplied regex, so none of it needs the subprocess boundary. Only the resulting file list crosses it, via `_run_grep_subprocess`, run off the event loop with `asyncio.to_thread`:

```python
proc = subprocess.Popen([sys.executable, "-S", _GREP_WORKER_SCRIPT], ...)
try:
    stdout, stderr = proc.communicate(input=request, timeout=timeout_seconds)
except subprocess.TimeoutExpired:
    proc.kill()          # SIGKILL — the line that actually stops the CPU burn
    proc.communicate(timeout=5.0)
    return {"error": f"grep search timed out after {timeout_seconds:g}s and was terminated"}
```

`_GREP_SUBPROCESS_TIMEOUT_SECONDS = 18.0`, deliberately shorter than `GrepFiles.timeout_seconds = 20.0` so the kill fires (and CPU consumption actually stops) at or before the point the run loop's own outer timeout gives up and reports failure — not sometime after.

The worker also self-limits via POSIX `RLIMIT_CPU` (`_apply_cpu_limit`, best-effort, `resource` unavailable on Windows): defense in depth for the case where the **parent itself** dies before ever reaching the `kill()` call — an orphaned worker otherwise has nothing left to stop it.

Every existing bound (`_MAX_GREP_LINE_SEARCH_CHARS`, `_MAX_GREP_LINES_SCANNED`, `_MAX_GREP_FILE_BYTES`) stays in place, read from the module's own globals at call time and passed into the subprocess as arguments — so pre-existing tests that monkeypatch these (`fot._MAX_GREP_LINE_SEARCH_CHARS = 10`, etc.) still take effect correctly through the subprocess boundary.

### Verification that CPU actually stops (not just wall-clock proxy)

`test_grep_subprocess_kills_the_worker_process_on_timeout` spies on `subprocess.Popen` to capture the worker's real pid, runs `_run_grep_subprocess` directly against `(a+)+$` over `"a"*28 + "X"` (line-cap deliberately widened to 10,000 so the full pathological line reaches the worker, isolating what the subprocess boundary alone buys) with a 1.5s timeout, and asserts:

- the call returns in `< 4.0s` (bounded near 1.5s, not the ~11.7s this exact pattern costs uncapped), and
- `psutil.pid_exists(pid)` is **`False`** immediately after return.

Manually re-verified outside pytest (see below) with a clean, single-pid check (an earlier attempt using a broad `psutil.process_iter` substring scan for `"_grep_worker.py"` produced a false positive — the scanning script's OWN command-line text, visible to `process_iter`, contained that literal substring; re-checking by the captured pid specifically eliminated the false positive).

### Benchmark

Realistic tree: this project's own `Tools/`, `Utils/`, `Chat/`, `Agents/` source (221 `.py` files), pattern `"def execute"`, median of 5 runs each, apples-to-apples (both the "before" reimplementation and "after" real implementation pay the identical `is_within`/`resolve_sensitive_context` candidate-discovery cost — an earlier draft of this benchmark omitted that from the "before" side and produced a misleadingly large gap):

| | median |
|---|---|
| BEFORE (in-process, this task's TASK-850-only checkpoint) | **78.5ms** |
| AFTER (subprocess-based) | **97.7ms** |

Added overhead ≈ **19ms** (~24% relative), matching an isolated pure round-trip measurement of the subprocess call with zero files to search (~18ms, `python -S` startup dominates). Comfortably negligible against the 20s tool-call ceiling and against `grep_files`' own `"reads"`-risk-tag `ask` permission floor (a human approves every individual call regardless).

Pathological case (`(a+)+$` against a 28–30 character adversarial line, line-cap disabled to isolate this task's fix):

- **Before** this task (documented in the pre-existing `_MAX_GREP_LINE_SEARCH_CHARS` comment, reproduced again here): an uncapped search of this exact pattern takes ~11.7s (28 chars) to ~47s (30 chars) and keeps running past whatever timeout the agent is told about, since the thread is abandoned, not killed — growing worse per additional character with nothing to stop it.
- **After**: bounded to the production ceiling, measured directly: `elapsed=18.00s`, `result={'error': 'grep search timed out after 18s and was terminated'}`, child pid confirmed dead (`pid_exists` → `False`) immediately on return.

### Residual exposure — stated precisely, not claimed closed

- The child can still burn real CPU for up to `_GREP_SUBPROCESS_TIMEOUT_SECONDS` (18s) before it is killed. Down from unbounded, **not zero**.
- Every `grep_files` call now pays a small, fixed process-spawn/teardown cost (~15–20ms) whether or not the pattern is pathological.
- `RLIMIT_CPU` is POSIX-only (no-op on Windows) and is explicitly documented as additive, not the primary guarantee; `communicate(timeout=)` + `kill()` is what holds on every platform.
- A pathological file-path string containing bytes that cannot round-trip through UTF-8 text-mode JSON (an unusual, not project-specific edge case — e.g. non-UTF-8 filenames on some POSIX filesystems) is a known, pre-existing class of risk not specifically hardened by this task; out of scope of both ACs.

This is documented in the same terms in `_run_grep_subprocess`'s and `_grep_worker.py`'s own docstrings, not only here.

### Tests

`Tests/Tools/test_glob_grep_files.py` (+297 lines): subprocess-kill proof (`test_grep_subprocess_kills_the_worker_process_on_timeout`), end-to-end never-raises (`test_grep_files_execute_survives_a_pathological_pattern_without_raising`), `_run_grep_subprocess` error handling (`Popen` spawn failure, nonzero worker exit, malformed worker output), a regression guard that the ordinary path still delegates to the subprocess (`test_grep_files_ordinary_search_still_delegates_to_a_subprocess`), and the worker script exercised directly (`run_search` unit tests) and as a real subprocess (malformed stdin, end-to-end match).

**Files:** `tldw_chatbook/Tools/_grep_worker.py` (new), `tldw_chatbook/Tools/file_operation_tools.py` (`GrepFiles.execute`, new `_run_grep_subprocess`/`_GREP_SUBPROCESS_TIMEOUT_SECONDS`/`_GREP_WORKER_SCRIPT`, updated docstrings on `_MAX_GREP_LINE_SEARCH_CHARS`/`GrepFiles.timeout_seconds`/`GrepFiles.execute`), `Tests/Tools/test_glob_grep_files.py`.

---

## Full test run

```
$ .venv/bin/python -m pytest Tests/Utils/ Tests/Tools/ Tests/Agents/ -q
910 passed, 12 warnings in 237.27s (0:03:57)
```

No regressions. Pre-existing warnings (RequestsDependencyWarning, a couple of loguru/pydub DeprecationWarnings, two unrelated MCP-provider `RuntimeWarning: coroutine … was never awaited`) are unrelated to this change and present on the pre-TASK-850 baseline too.

**Hard constraints re-checked:**
- Both tools' gates (`glob_files_enabled`/`grep_files_enabled` in `Agents/tool_catalog.py`) untouched — still default `False` (`get_cli_setting(..., False)`), still gateable built-ins. Nothing in this change reads or writes those keys.
- Matching stays resolved-ancestry (`is_within`/`is_sensitive_path`, both `Path.resolve()`-based) throughout; no string-prefix comparison was introduced.
- Default configuration proven end to end: `test_default_sandbox_configuration_still_works_end_to_end` (no monkeypatching of the sandbox root/config at all) covers `write_file`/`read_file`/`list_directory`/`glob_files`/`grep_files` back to back, and now genuinely exercises the real subprocess spawn for `grep_files` since nothing in that test forces a bypass.
