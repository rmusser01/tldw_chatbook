# TASK-843/TASK-850 follow-up hardening review — fixes

**Worktree:** `/Users/macbook-dev/Documents/GitHub/wt-path-hardening` (branch `feat/agent-path-hardening`)
**Interpreter:** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`, run from this worktree.

This addresses five findings from a follow-up review of `7fd033f1b` (TASK-850, scope `glob_files`/`grep_files` to workspace roots) and `8d66becf9` (TASK-843, killable-subprocess regex search) — both already `Done` on this branch. All work is in `tldw_chatbook/Tools/file_operation_tools.py`, `tldw_chatbook/Utils/sensitive_paths.py`, and `Tests/Tools/test_glob_grep_files.py`.

---

## Finding 1 (Important) — enumeration ran away before the subprocess ever started

**Gap.** TASK-843 split `grep_files` into two sequential phases: candidate discovery (`_iter_candidates_across_roots`, in-process, bounded by `_MAX_CANDIDATES` = 20,000) then one subprocess call over the *entire* resulting list. The pre-subprocess implementation had checked `len(matches) >= _MAX_MATCHES` / `lines_scanned >= _MAX_GREP_LINES_SCANNED` *during* enumeration and broke out immediately; the split lost that early exit; `GrepFiles.execute` drained discovery to completion regardless of how quickly the match budget would have been satisfied.

Consequences, both real:
- A high-hit-rate pattern over a large tree paid for candidates the match budget never needed.
- `_GREP_SUBPROCESS_TIMEOUT_SECONDS`'s comment claimed its ~2s headroom below `GrepFiles.timeout_seconds` (20.0s) meant the kill fires "at or before" the agent is told the call failed. With unbounded discovery time added on top of the subprocess's own 18s, that was false.

**Fix.** Discovery and search are now streamed together by a new `_run_grep_search` (`tldw_chatbook/Tools/file_operation_tools.py`), called from `GrepFiles.execute` in place of the old "collect everything, then one subprocess call":

- Candidates are pulled from `_iter_candidates_across_roots` in **growing batches** — `_GREP_INITIAL_CANDIDATE_BATCH_SIZE = 256`, doubling up to `_GREP_MAX_CANDIDATE_BATCH_SIZE = 4096` — each batch searched by its **own** `_run_grep_subprocess` call with whatever `max_matches`/`max_lines_scanned` budget remains.
- The loop stops **without pulling another candidate** the moment the remaining budget is `0` (i.e. `_MAX_MATCHES` matches found or `_MAX_GREP_LINES_SCANNED` lines scanned) — this is the restored early exit.
- A small initial batch keeps a high-hit-rate search's pre-first-call enumeration cost close to what the old in-process early-break paid; growing the batch on later rounds keeps a rare/zero-hit search (which must still examine up to `_MAX_CANDIDATES`) from paying for dozens of small, separately-spawned subprocesses to get there (doubling from 256 reaches 20,000 in ~8 rounds).
- **Killability is unaffected** — every batch is still searched by its own fully killable child process; only how many candidates reach it, and in how many calls, changed.
- Every existing guard (containment, sensitive-path denylist, hidden-component rule, the global `_MAX_CANDIDATES` counter shared across all roots) still runs inside `_iter_candidates_across_roots` exactly as before — `_run_grep_search` performs none of that itself and trusts every candidate completely, same contract as before.

**The comment is now true, restated precisely.** The wall-clock deadline (`deadline = time.monotonic() + deadline_seconds`) now starts **before the first candidate is even pulled**, not after discovery finishes, and is re-derived before pulling each batch and again before spawning each batch's subprocess call. Discovery time and every batch's subprocess wait are drawn from the *same* 18.0s window, so however that time is spent — a slow disk during discovery, one long subprocess wait, or several short ones — the aggregate cannot exceed it. `_GREP_SUBPROCESS_TIMEOUT_SECONDS`'s docstring and `GrepFiles.timeout_seconds`'s docstring were rewritten to say exactly this rather than the old claim (which implicitly assumed discovery was already finished when the clock started). What now actually holds: the kill (or a timeout-error return before ever starting another batch) fires at or before `GrepFiles.timeout_seconds`'s 20.0s ceiling for the *aggregate* of however many batches one `grep_files` call needed — not, as the old comment implied, for one subprocess call alone with discovery assumed free.

**Benchmark test added** (`test_grep_files_high_hit_rate_pattern_does_not_over_enumerate`, `Tests/Tools/test_glob_grep_files.py`): reproduces the reviewer's exact scenario — a 5,000-file tree where the pattern matches every file. Pins two things: (a) a structural, machine-speed-independent proof that the number of candidates actually *pulled* from discovery before the match budget is satisfied is far below the tree size (`< 2,500`, actual measured value: 256 — exactly one initial batch), and (b) a generous wall-clock sanity bound (`< 2.0s`). **Verified this test fails against the pre-fix code**: stashed only the two source files (kept the new test), re-ran it, and got `enumerated 5000 of 5,000 candidates ... assert 5000 < 2500` — confirming it would have caught the shipped regression. Restored the fix afterward; full targeted suite still green.

Two more tests pin the two halves of the mechanism directly:
- `test_grep_files_aggregates_matches_across_multiple_batches` — forces several small batches (batch size 2, 10 files) and confirms every match survives the multi-batch merge, with more than one subprocess call actually happening.
- `test_grep_files_search_deadline_starts_before_the_first_candidate_is_pulled` — makes discovery itself slow (a monkeypatched sleep inside a wrapped `_iter_candidates_across_roots`) with a tiny deadline, and confirms the call reports a timeout rather than silently proceeding to search — proving the deadline really does start before discovery, not after.

### Measured before/after timing (the exact scenario given: 5,000 files, pattern matching in every file)

Measured directly on this branch (`git stash push -- <the two source files>` to get the pre-fix behavior, `git stash pop` to restore; same 5,000-file `tmp_path` sandbox, `pattern="DEBUG"`, `glob="**/*.py"`, `loguru.logger.remove()` to strip logging overhead from the timing):

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| **BEFORE** (this review's fix reverted, TASK-843/850 code as shipped) | 1.2892s | 0.9636s | 0.9409s |
| **AFTER** (this fix) | 0.3609s (cold) | 0.0752s | 0.0773s |

Candidates actually pulled from discovery, after the fix: **256** (one initial batch) — vs. all **5,000** before. Consistent with the reviewer's own estimate (~0.32ms/candidate × 5,000 ≈ 1.6s before; ×256 ≈ 82μs... negligible after) and with the old in-process early-break's ~0.1s ballpark this restores.

---

## Finding 3 (Minor) — grep worker subprocess isolation

**Gap.** The worker subprocess inherited the parent's full environment, actual cwd, and (via the script's own directory being prepended to `sys.path`) exposed `sys.path[0] == .../Tools/`. Confirmed directly: a parent-only env var was visible in a probe child process, and `sys.path[0]` pointed at this project's own source directory when the worker was spawned without `-P`.

**Fix.**
- Added `-P` to the interpreter flags (`[sys.executable, "-S", "-P", _GREP_WORKER_SCRIPT]`) — available since Python 3.11, this project's floor. Verified directly: without `-P`, `sys.path[0]` inside a spawned worker-shaped script is its own script directory; with `-P`, it becomes the interpreter's own zip path, never the source tree.
- Added an explicit `cwd=_GREP_WORKER_CWD` (`tempfile.gettempdir()`) — the worker only ever touches absolute paths handed to it on stdin, so it has no legitimate use for the parent's actual working directory.
- Added an explicit, minimal `env=_grep_worker_env()`: just `PATH` (needed for the interpreter's own startup), plus `SystemRoot` on Windows only (an empty `env={}` can make `subprocess.Popen` fail outright on Windows). Verified: a fake secret (`MY_FAKE_SECRET_9f21`) set in the parent's environment is absent from `_grep_worker_env()`'s output and from the captured `Popen` kwargs in a real call.

Tests: `test_run_grep_subprocess_isolates_worker_cwd_and_env` (spies on `subprocess.Popen`, asserts `-P` present, `cwd` differs from the real cwd, and a parent-only secret env var is absent from the passed `env`), `test_grep_worker_env_omits_arbitrary_parent_variables` (direct unit test of the helper).

This is not an escalation on its own (planting a module to exploit a leaked `sys.path[0]` already requires source-tree write access — full compromise) — it removes a needless surface for free, as the finding states.

---

## Finding 4 (Minor) — worker payload validated before returning

**Gap.** `_run_grep_subprocess` returned `json.loads(stdout)` verbatim once confirmed to be *a dict* — never confirming `matches` was actually a list, or that each entry / `lines_scanned` had the documented shape. A worker emitting `{"matches": "not-a-list"}` would have propagated that shape straight to the agent.

**Fix.** New `_validated_grep_worker_payload(parsed)` checks, on the worker's claimed-success path: `matches` is a list; every entry is a dict with `path: str`, `line_number: int`, `line: str`; `lines_scanned` is an int. Any violation returns `{"error": "grep worker produced malformed output"}` — consistent with the tool's never-raise contract and reusing the exact error string already used for other malformed-output cases. The `"error"` path itself is also now checked (a non-string `error` value is likewise normalized to that same message).

Tests: `test_run_grep_subprocess_rejects_a_matches_field_that_is_not_a_list`, `test_run_grep_subprocess_rejects_a_malformed_match_entry`, `test_run_grep_subprocess_rejects_a_non_int_lines_scanned`, and a sanity check that a genuinely well-formed payload still passes through unchanged (`test_run_grep_subprocess_still_accepts_a_well_formed_payload`).

---

## Finding 2 (Minor, documented not changed) — `refuses_new_directory_chain` over-refuses new containers

Added a paragraph to `refuses_new_directory_chain`'s docstring (`tldw_chatbook/Utils/sensitive_paths.py`) naming the consequence: because a not-yet-existing name always fails `is_sensitive_path`'s `is_dir()` gate, this refuses creating **any** brand-new subdirectory directly inside a protected container (e.g. the ChromaDB persist directory), not only one that collides with a specific state-file name — reproduced: `write_file("chromadb/newcoll/x.txt", create_directories=True)` is refused while `write_file("chromadb/coll1/new.txt", ...)` succeeds once `coll1` already exists. Documented as deliberate (distinguishing "legitimate new container" from "shadow directory" by name alone would require the enumeration this design avoids) and only reachable under a widened sandbox root. No behavior changed.

---

## Finding 5 (Minor, documented not changed) — `_MAX_CANDIDATES` consumed root-by-root, starving later roots

Added a paragraph to `_iter_candidates_across_roots`'s docstring naming the consequence: the global `_MAX_CANDIDATES` budget is consumed in root order, so a first root holding `_MAX_CANDIDATES` or more matches starves every later root entirely, indistinguishable from those roots genuinely having nothing. Documented as correct (the bound must stay global; there is no principled way to split it fairly across an arbitrary number of roots) with a suggested workaround (narrow the search with a root-scoped glob). No behavior changed.

---

## Hard constraints re-checked

- Every agent tool still returns a result dict and never raises — `_run_grep_search`'s only non-dict exit is `next(candidates)`'s pre-existing `ValueError`/`NotImplementedError` for a syntactically invalid glob, caught in `GrepFiles.execute` exactly as before this change.
- Killability unaffected: every batch is still searched by its own `Popen`-spawned, `communicate(timeout=)`+`kill()`-bounded child process; `test_grep_subprocess_kills_the_worker_process_on_timeout` (pre-existing) still passes unmodified.
- Every guard (containment, sensitive-path denylist, hidden-component rule, dotted-root rule) still applies to every candidate from every root — unchanged, since `_iter_candidates_across_roots` itself was not modified beyond its docstring.
- `SensitivePathContext` still resolved exactly once per call (`resolve_sensitive_context()` in `GrepFiles.execute`, threaded through to the shared generator).
- No tool's default-OFF posture changed; the sandbox root and its default are untouched.

## Test run

```
$ .venv/bin/python -m pytest Tests/Tools/test_glob_grep_files.py Tests/Tools/test_file_tools_workspace_roots.py Tests/Utils/test_sensitive_paths.py -q
97 passed, 1 warning in 10.89s
```

Full required command:

```
$ .venv/bin/python -m pytest Tests/Utils/ Tests/Tools/ Tests/Agents/ -q
919 passed, 12 warnings in 240.64s (0:04:00)
```

919 = the pre-existing 910-pass baseline (per the prior TASK-843/850 report) + the 9 new tests added here. No failures. All warnings match the known, pre-existing set (RequestsDependencyWarning, pydub/audioop and SWIG DeprecationWarnings, two unrelated MCP-provider "coroutine was never awaited" RuntimeWarnings) — none introduced by this change.

## Files

- `tldw_chatbook/Tools/file_operation_tools.py` — `_run_grep_search` (new), `_grep_worker_env`/`_GREP_WORKER_CWD`/`_validated_grep_worker_payload` (new), `_GREP_INITIAL_CANDIDATE_BATCH_SIZE`/`_GREP_MAX_CANDIDATE_BATCH_SIZE` (new constants), `_run_grep_subprocess` (Popen call + return-path validation), `GrepFiles.execute`/`GrepFiles.timeout_seconds` (docstrings + body), `_GREP_SUBPROCESS_TIMEOUT_SECONDS`/`_iter_candidates_across_roots` (docstrings).
- `tldw_chatbook/Utils/sensitive_paths.py` — `refuses_new_directory_chain` docstring.
- `Tests/Tools/test_glob_grep_files.py` — 9 new tests (benchmark/batching/deadline for Finding 1, isolation for Finding 3, payload validation for Finding 4).
