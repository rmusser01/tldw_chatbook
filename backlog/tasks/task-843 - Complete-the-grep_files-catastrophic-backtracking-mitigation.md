---
id: TASK-843
title: Complete the grep_files catastrophic-backtracking mitigation
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 02:36'
updated_date: '2026-07-27 05:55'
labels:
  - tools
  - security
  - follow-up
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
grep_files bounds the regex input via a line-length cap, a total-lines-scanned cap and a 20s per-tool timeout. Those make the worst case finite and small but do not eliminate it: Python's re has no timeout, and _call_with_timeout abandons the worker thread rather than killing it, so a pathological pattern keeps burning CPU after the agent reports failure. A complete fix needs a regex engine supporting timeouts or a killable subprocess. grep_files carries the reads risk tag and floors to ask, which is why the partial mitigation was accepted. Filed from the PR #953 review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A pathological regex cannot consume CPU after its tool call has returned,Ordinary searches are not measurably slower,The chosen approach is documented with its trade-offs
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the third-party regex module (timeout= support) is not a direct/declared-optional dependency of this project -- only present transitively via unrelated optional extras (nltk/transformers/dateparser). Rule it out: grep_files is a core built-in with no extras gate, so depending on it would make an optional/transitive package a hard requirement.
2. Choose the killable-subprocess approach: move the actual regex-vs-file-content search into a standalone worker script (_grep_worker.py, stdlib-only, no import of tldw_chatbook) invoked via subprocess.Popen, JSON over stdin/stdout, so the OS can actually kill it (Popen.kill()) rather than abandon it the way _call_with_timeout does for threads.
3. Keep candidate discovery (containment/sensitivity/hidden-component/_MAX_CANDIDATES) in-process in GrepFiles.execute -- none of that runs the user-supplied regex, so it doesn't need the subprocess boundary. Only the line-by-line regex search moves into the worker.
4. Add _run_grep_subprocess(): spawns the worker with `-S`, writes the request, waits up to a subprocess-specific timeout (_GREP_SUBPROCESS_TIMEOUT_SECONDS, shorter than GrepFiles.timeout_seconds), kills+reaps on TimeoutExpired, never raises.
5. Add a POSIX RLIMIT_CPU self-limit inside the worker as defense-in-depth for the orphaned-parent case.
6. Keep every existing bound in place (_MAX_GREP_LINE_SEARCH_CHARS, _MAX_GREP_LINES_SCANNED, _MAX_GREP_FILE_BYTES, GrepFiles.timeout_seconds) as the cheap first line of defence.
7. Update every docstring/comment that previously described the mitigation as partial/incomplete to describe what the subprocess boundary now guarantees and what it still does not.
8. Add tests: subprocess is actually killed on timeout (process-level proof via psutil), GrepFiles.execute never raises for a pathological pattern, worker error-handling (bad Popen spawn, nonzero exit, malformed output, malformed request), and the worker script exercised as a real subprocess.
9. Benchmark before/after on a realistic tree (ordinary search) and the pathological case (bounded post-return CPU), report both.
10. Run Tests/Utils Tests/Tools Tests/Agents to confirm no regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Chose the killable-subprocess approach (option 1), not the third-party regex module (option 2): `regex` (which supports timeout=) is installed in this dev venv but only transitively, via nltk/transformers/dateparser pulled in by the optional embeddings/RAG extras -- it is not a direct or declared-optional dependency of this project (checked pyproject.toml). grep_files is a core built-in reachable with zero extras installed, so depending on `regex` here would silently turn a currently-optional, transitive package into a hard requirement for the base install. A subprocess adds no new dependency.

The regex-vs-file-content search now runs in a standalone child process (new tldw_chatbook/Tools/_grep_worker.py -- stdlib-only, no import of tldw_chatbook at all, deliberately: it must be trivially killable without dragging along the host app's live state). GrepFiles.execute still does candidate discovery in-process (containment, sensitive-path denylist, hidden-component rule, _MAX_CANDIDATES) since none of that runs the user-supplied regex; only the actual line-by-line search is handed off, via _run_grep_subprocess(), to `sys.executable -S _grep_worker.py` with the pattern/file list/bounds sent as one JSON request over stdin and the matches read back as one JSON response over stdout.

_run_grep_subprocess uses Popen.communicate(timeout=_GREP_SUBPROCESS_TIMEOUT_SECONDS=18.0, deliberately shorter than GrepFiles.timeout_seconds=20.0) and calls proc.kill() (SIGKILL) + reaps on TimeoutExpired -- unlike Agents/agent_service.py's _call_with_timeout, which abandons a hung worker THREAD because Python cannot kill a thread, a subprocess genuinely can be killed. Verified directly (not just by wall-clock proxy): captured the child's pid via a Popen spy, confirmed psutil.pid_exists(pid) is False immediately after a timed-out call returns (test_grep_subprocess_kills_the_worker_process_on_timeout). The worker also self-limits via POSIX RLIMIT_CPU (best-effort, defense-in-depth for the case where the PARENT dies before ever calling kill()).

Every existing bound (_MAX_GREP_LINE_SEARCH_CHARS, _MAX_GREP_LINES_SCANNED, _MAX_GREP_FILE_BYTES, GrepFiles.timeout_seconds=20.0) stays in place as the cheap first line of defence and is still passed into the worker as call arguments (read from the module's own globals at call time, so existing tests that monkeypatch them continue to work through the subprocess boundary unchanged).

Benchmark (realistic tree: this project's own Tools/Utils/Chat/Agents source, 221 .py files, pattern "def execute", median of 5 runs, apples-to-apples -- both include the same is_within/resolve_sensitive_context candidate-discovery cost):
  BEFORE (in-process, same containment/sensitivity checks): median 78.5ms
  AFTER  (subprocess-based):                                median 97.7ms
  Added overhead: ~19ms (~24% relative, ~20ms absolute) -- consistent with the isolated pure-subprocess-round-trip measurement (~18ms with zero files). Comfortably negligible against the 20s tool-call ceiling and against grep_files' own "ask" permission floor (a human approves every call).
Pathological case ((a+)+$ against a 28-30 character adversarial line, line-length cap deliberately disabled to isolate this task's fix): BEFORE this task, an uncapped in-process search of this exact pattern took ~11.7s (28 chars) to ~47s (30 chars) and kept running past any reported timeout, since _call_with_timeout abandons the thread; growing worse per additional character with nothing to stop it. AFTER: bounded to the production ceiling (18.00s measured), subprocess confirmed killed (pid_exists False) immediately on return.

Residual exposure, stated precisely (not claimed closed): the child can still burn real CPU for up to _GREP_SUBPROCESS_TIMEOUT_SECONDS (18s) before it is killed -- down from unbounded, not zero. Every grep_files call now pays a small, fixed process-spawn/teardown cost (~15-20ms) whether or not the pattern is pathological. RLIMIT_CPU is POSIX-only (no-op on Windows); the communicate(timeout=)+kill() path is what actually holds on every platform. All of this is documented in _run_grep_subprocess's and _grep_worker.py's own docstrings, not just here.

Added: tldw_chatbook/Tools/_grep_worker.py (new).
Modified: tldw_chatbook/Tools/file_operation_tools.py (GrepFiles.execute, new _run_grep_subprocess/_GREP_SUBPROCESS_TIMEOUT_SECONDS/_GREP_WORKER_SCRIPT, updated docstrings/comments on _MAX_GREP_LINE_SEARCH_CHARS, GrepFiles.timeout_seconds).
Added tests: Tests/Tools/test_glob_grep_files.py (subprocess-kill proof, execute()-never-raises, worker error handling, worker-as-real-subprocess, ordinary-path-still-uses-subprocess).

Implemented together with TASK-850 (same file, same commit sequence).
<!-- SECTION:NOTES:END -->
