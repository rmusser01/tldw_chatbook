# Agent-runtime durability hardening (TASK-327)

**Date:** 2026-07-24
**Backlog:** TASK-327 (agent runtime durability and robustness hardening, LOW, agents/tech-debt). The **final** task of the LLM-harness review 2026-07 (all of 320-326, 328-334 shipped).
**Branch:** `feat/agent-runtime-durability` (worktree off `origin/dev` @ `f32ac64fc`).

Five bounded robustness gaps in the agent runtime, one PR. Four code fixes + one documentation AC. All verified against origin/dev; **re-verify every line number at implementation time** — the AC citations are already ~20 lines stale and dev is active. No other branch edits the five target files ahead of dev (clean rebase runway).

Architecture recap: the runtime is layered — `agent_models` (pure dataclasses/constants), `agent_runtime` (pure control loop, "No Textual, app, DB, or I/O imports"), `agent_stream` (pure), `agent_service` (the one impure seam wiring the loop to providers/DB via `deps`), `tool_catalog` (`ToolProvider`), `native_tools`, `mcp_tool_provider` (worker-thread→main-loop bridge). The runtime already bounds steps/model-turns/wall-clock/tokens/subagents; depth is hard-capped at 1; cancellation is cooperative; errors → `RUN_ERROR`, never a crash.

## AC#1 — Cyclic (non-consecutive) loop detection

**Gap (verified):** `agent_runtime.py` tracks one-slot `last_key`/`repeat_count`; the check `key=(call.name, json.dumps(call.args, sort_keys=True)); repeat_count = repeat_count+1 if key==last_key else 1; if repeat_count >= LOOP_DETECTION_N(=3): RUN_STUCK`. It catches only N identical **consecutive** calls. Traced A→B→A→B: `repeat_count` never exceeds 1 → never trips. `max_steps=8`/`max_wall_seconds=240` are the blunt, delayed backstop.

**Fix (runtime stays pure):** Replace the one-slot state with a bounded `deque(maxlen=LOOP_DETECTION_N * MAX_LOOP_PERIOD)` of call-keys `(call.name, json.dumps(call.args, sort_keys=True))`, **initialized loop-locally** (fresh per `run_agent_loop` call, exactly where `last_key`/`repeat_count` live today). On each dispatched call — at the **same position** as today's check (top of the per-call handling, **before** the `STEP_TOOL_CALL` is added and the tool invoked; on trip it adds a `STEP_ERROR` and returns, so the looping tool is never dispatched again) — append the key, then check for a repeating cycle: for period `p` in `1 .. MAX_LOOP_PERIOD` (smallest first), let `t = LOOP_DETECTION_N` if `p == 1` else `2`; if `len(history) >= t*p` and the last `t*p` keys equal `t` consecutive copies of the trailing `p`-block, trip `RUN_STUCK` with `STEP_ERROR` summary `f"loop detected: {p}-cycle repeated {t}x"` and `return _outcome(RUN_STUCK)`. New constant `MAX_LOOP_PERIOD = 4` in `agent_models.py` next to `LOOP_DETECTION_N`.

Threshold rationale (user-approved): period-1 keeps `LOOP_DETECTION_N=3` (backward-compatible; avoids false-positive on a legitimate identical double-retry). Periods ≥2 trip at 2 full repeats, so the canonical A→B→A→B fires at 4 calls — before the default step budget — with a clear message. Smallest-period-first + exact-repeat means a longer cycle can't cross-trip a shorter period; the args-inclusive key means a real varied-arg workflow (search(q1)→read(u1)→search(q2)→…) never forms an identical cycle.

**Tests** (`Tests/Agents/test_agent_runtime.py`): existing `test_identical_consecutive_calls_trip_loop_detection` and `test_same_tool_different_args_is_not_stuck` still pass; add A→B→A→B trips (`RUN_STUCK`, last step error) at the 4th call; A→B→C→A→B→C (period-3, 2 repeats) trips; a non-cyclic A→B→C→D→E does NOT trip (runs to a normal outcome). Drive with the existing deterministic `ModelTurn`-list fakes.

## AC#2 — Reconcile orphaned `running` rows on open

**Gap (verified):** `create_run` INSERTs `status='running'` at run start; `_persist` (`agent_service.py`, `append_steps` + `set_status`) runs exactly once at run **end**. A hard crash between skips `_persist` → the row stays `running`, `steps='[]'` forever. No reconciliation exists. `result` column is plain `TEXT` (`set_status` writes the raw final-answer string via `COALESCE(?, result)`), and `RUN_ERROR="error"` is already in `TERMINAL_RUN_STATUSES` with an existing `console_chat_controller` UI branch — so no schema/status/UI change.

**Fix (mirrors the shipped `Library_Ingest_Jobs` "Interrupted by app restart" precedent):** Add `AgentRunsDB.reconcile_orphaned_runs() -> int` — inside the existing `transaction()`/`BEGIN IMMEDIATE`, run `UPDATE agent_runs SET status='error', result=COALESCE(result, 'Interrupted by app restart'), updated_at=? WHERE status='running'` (`updated_at` = `_now_iso()`); return `cursor.rowcount`. Call it from `AgentRunsDB.__init__` **immediately after `super().__init__(...)`** — verified `base_db.__init__` runs `_initialize_schema()` (creating `agent_runs`) and sets `is_memory_db` before returning, so the table exists and the memory flag is ready. Reconcile **only for file-backed DBs** (skip `:memory:` — orphans only exist in persisted files, and each in-memory DB is a distinct database), guarded by a class-level `_swept_paths: set[str]` so it fires exactly once per DB file per process regardless of how many times `ChatScreen` lazily reconstructs the bridge. Register the path in `_swept_paths` even when 0 rows are flipped (once-per-process is the contract, not once-per-orphan).

**Single-instance assumption (documented):** a `running` row present at first open is treated as orphaned. If two app instances shared the same data dir, instance B's open would flip instance A's *actively-running* run to `error` — an accepted edge case, identical to the shipped `Library_Ingest_Jobs` behavior. Note it in the method docstring.

**Tests** (`Tests/DB/test_agent_runs_db.py`): seed a file-backed DB with two `running` rows + one `done` row, then construct a fresh `AgentRunsDB` on that path → the two become `error` with `result='Interrupted by app restart'`, `done` untouched; a `running` row that already has a non-null `result` keeps its result (COALESCE); a second construction on the same path is a no-op (idempotent via `_swept_paths`); a `:memory:` DB construction does not error and does not register a swept path.

## AC#3 — WAL + busy_timeout on AgentRunsDB

**Gap (verified):** `base_db.py`'s `_get_connection` sets neither `PRAGMA journal_mode=WAL` nor an explicit `busy_timeout`; `AgentRunsDB._get_connection` calls `super()._get_connection()` and adds only `foreign_keys=ON`. It's the **only** DB class in the codebase with neither (every sibling — ChaChaNotes, Client_Media, Evals, Prompts, Library_Ingest_Jobs, Kanban — sets WAL). Its own `transaction()` docstring already flags the concurrent-write hazard (primary run + sub-agent runs; plus a 0.2s-polled main-thread read).

**Fix (scoped to this class, no base_db change, mirrors ChaChaNotes):** In `AgentRunsDB._get_connection`, after `super()._get_connection()`, add `conn.execute("PRAGMA journal_mode=WAL;")` **guarded by the existing `is_memory_db` check** (WAL is invalid for `:memory:`), `conn.execute("PRAGMA busy_timeout=5000;")`, and keep `foreign_keys=ON`.

**Test:** a file-backed `AgentRunsDB` connection reports `PRAGMA journal_mode` == `wal` and `PRAGMA busy_timeout` == `5000`; a `:memory:` connection does not attempt WAL and does not error.

## AC#4 — Per-tool-call timeout

**Gap (verified):** the four budget checks run once per outer loop iteration (before `deps.call_model`); the inner `for call in calls:` dispatch only re-polls `should_cancel()` — no wall-clock recheck between tool calls. `BuiltinToolProvider.invoke` does a bare `asyncio.run(tool.execute(**args))` with no timeout; a custom/blocking `ToolProvider.invoke()` can therefore wedge the synchronous run past its wall-clock deadline. (MCP's provider already self-times-out via `future.result(timeout=…)` and "NEVER hangs unbounded"; skill calls route around `invoke_tool` into a budget-clamped nested loop.)

**Fix (impure seam only — runtime stays byte-identical and pure):**
- New `RunBudget.max_tool_call_seconds: float = 120.0` (agent_models.py), with `0` = unlimited sentinel (mirrors `max_total_tokens`). **Add it to `clamp_child_budget`'s explicit `RunBudget(...)` construction** (`max_tool_call_seconds=child.max_tool_call_seconds`) so sub-agents inherit it rather than silently dropping to the default.
- In `agent_service`'s `invoke_tool` closure, wrap **only** the builtin/custom `registry.invoke_by_name(call.name, call.args)` call (i.e. *after* the skill-routing branch — skill runs keep their own clamped budget). When `max_tool_call_seconds > 0`, run the sync call via a **module-level** helper `_call_with_timeout(fn, seconds) -> ToolResult` (module-level so it's directly unit-testable, not a nested closure):
  - **Mechanism (correctness-critical):** a **per-call daemon thread** with a result/exception box + `thread.join(seconds)`. NOT `with ThreadPoolExecutor()` (its `shutdown(wait=True)` blocks on the hung thread and defeats the timeout / hangs exit) and NOT a shared bounded pool (one hung tool would wedge every later call). The helper **always returns a `ToolResult`**: success → the tool's result; the worker raised → `ToolResult(ok=False, error=str(exc))` (defensive — a well-behaved provider already returns a `ToolResult`, but a custom one might raise); `join` timed out → `ToolResult(ok=False, error=f"tool call timed out after {seconds}s: {call.name}")`, leaving the daemon thread to finish/die with the process.
  - When `max_tool_call_seconds == 0`, call `invoke_by_name` directly (byte-identical to today).
  - **MCP compatibility:** an MCP tool invoked through the daemon thread still works — `MCPToolProvider.invoke` uses `run_coroutine_threadsafe(coro, self._main_loop)`, which schedules onto the main loop regardless of the calling thread, so running `invoke_by_name` on a fresh daemon thread does not disturb it.
- **Documented residual:** a timeout preempts the loop (the run is not wedged, cancellation/budget can proceed) but cannot kill the underlying thread — a truly-hung tool's daemon thread leaks until process exit (same limitation as MCP's future path). Interaction: an MCP tool sees both its own (usually smaller) timeout and this outer cap; the inner fires first for any reasonable config; the outer is a hard ceiling.

**Tests:** prefer **direct unit tests of `_call_with_timeout`** (deterministic, no AgentService wiring): a fake that blocks on a `threading.Event` past a tiny timeout → returns the timeout `ToolResult`; a fast fn → returns its real `ToolResult`; a fn that raises → `ToolResult(ok=False, error=…)`. Release the event after the timeout assertion so the daemon thread exits cleanly (no leaked-thread warnings in the suite). Plus: `max_tool_call_seconds=0` bypasses the helper (direct call); and a unit test that `clamp_child_budget` carries `max_tool_call_seconds` through to the child.

## AC#5 — Document the bridge's serialization contract (documentation-only)

**Verdict (verified):** `ConsoleAgentBridge.run_reply` has no internal single-active-run guard (it unconditionally writes `self._live[conversation_id]`), but `ConsoleChatController._active_run_rejection` / `run_state.is_send_allowed` serializes **all** runs controller-wide (strictly stronger than one-per-conversation) before every `bridge.run_reply` call, and this invariant is tested by `Tests/UI/test_console_run_gate.py` (added for TASK-232). The AC explicitly permits documenting the controller as the sole serialization point.

**Fix:** Add a docstring note on `ConsoleAgentBridge.run_reply` (and near `_live`/`_historical_cache`): `run_reply` is not internally concurrency-safe; single-active-run serialization is enforced externally by `ConsoleChatController._active_run_rejection` / `run_state.is_send_allowed` (see `Tests/UI/test_console_run_gate.py`); a future direct caller (e.g. a batch/background-run feature) must go through that gate or add its own guard. No code/test change.

## Cross-cutting

One PR; ~5 commits: (1) AC#1 loop detection; (2) AC#2 reconcile; (3) AC#3 PRAGMAs; (4) AC#4 tool timeout; (5) AC#5 bridge doc + backlog close. `agent_models.py` is touched by AC#1 (`MAX_LOOP_PERIOD`) and AC#4 (`max_tool_call_seconds` field + `clamp_child_budget`) — group those edits per their task. Re-verify all line numbers before editing. Fresh worktree; `git add` only each task's files (never `-A`); tests via the main-checkout `.venv`. Backlog: TASK-327 → Done (all 5 ACs) — the last task in the harness-review stream.

## Out of scope / residual (explicit)
- Killing a truly-hung tool thread (Python can't; daemon-leak-until-exit is the accepted bound — same as MCP).
- Multi-instance-shared-datadir reconcile races (accepted, matches the ingest-jobs precedent; documented).
- Any change to `base_db.py` (AC#3 is scoped to `AgentRunsDB` only).
- Adding a real guard inside the bridge (AC#5 is doc-only per the controller's stronger invariant).
