# Agent-Runtime Durability Hardening Implementation Plan (TASK-327)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Five bounded robustness fixes to the agent runtime — generalized cyclic loop detection, orphaned-run reconciliation on open, WAL+busy_timeout, a per-tool-call timeout, and a serialization-contract doc note.

**Architecture:** The runtime is layered — `agent_models` (pure dataclasses/constants), `agent_runtime` (pure control loop, no I/O), `agent_service` (the impure seam wiring the loop via `deps`), `AgentRuns_DB` (persistence), `console_agent_bridge` (UI bridge). Four fixes are code (loop detection stays in the pure runtime; the tool-timeout lives entirely in the impure seam so the runtime stays pure); AC#5 is documentation.

**Tech Stack:** Python 3.11, sqlite3, pytest, threading. No new dependencies.

**Spec:** `Docs/superpowers/specs/2026-07-24-agent-runtime-durability-design.md` (committed `38726cc98`). Read it for rationale; THIS plan carries the exact code.

## Global Constraints

1. **The pure runtime stays pure.** `agent_runtime.run_agent_loop` is a plain `def` with "No Textual, app, DB, or I/O imports." AC#1's fix uses only `collections.deque` + pure comparisons; the AC#4 timeout lives ENTIRELY in `agent_service` (the impure seam) — `run_agent_loop`'s signature and body outside the loop-detection block stay byte-identical, and `LoopDeps.invoke_tool: Callable[..., ToolResult]` is unchanged.
2. **Loop-detection thresholds:** period-1 trips at `LOOP_DETECTION_N (=3)` identical consecutive calls (backward-compatible); periods `2..MAX_LOOP_PERIOD (=4)` trip at 2 full repeats. Smallest period first. Call-key = `(call.name, json.dumps(call.args, sort_keys=True))`.
3. **Reconcile:** `running`→`error` with `result=COALESCE(result,'Interrupted by app restart')`, file-backed DBs only, once per file per process (`_swept_paths`), called right after `super().__init__()`.
4. **PRAGMAs:** `journal_mode=WAL` (guarded by `is_memory_db`) + `busy_timeout=5000` on `AgentRunsDB._get_connection` only (no `base_db` change).
5. **Tool timeout:** `RunBudget.max_tool_call_seconds` (0 = unlimited); enforced via a module-level `_call_with_timeout` using a **per-call daemon thread + join(timeout)** (NOT a ThreadPoolExecutor `with`, NOT a shared pool); wraps ONLY the builtin/custom `registry.invoke_by_name` (skill calls route around it); MUST be added to `clamp_child_budget`.
   **CORRECTED post-review (2026-07-25):** the default is **`300.0`**, not the `120.0` written in Task 4 below, and MCP tools are **NOT** exempt from the wrapper — `MCPToolProvider` shares the same per-run registry, so MCP calls go through it. 120.0 collided with MCP's own ~186s worst case (121s approval wait + 65s execution), which would report a timeout for a call that later really executes. Task 4's code blocks below retain the original `120.0`; the shipped value is `300.0`. See the spec's CORRECTION block for the full reasoning.
6. **Line numbers below are as-of origin/dev `f32ac64fc` — re-verify with `grep -n` before editing (they've already drifted ~20 lines from the AC citations). The target TEXT is authoritative.**
7. Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-agent-durability` (branch `feat/agent-runtime-durability`); tests via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest` FROM the worktree. Never touch the main checkout. `git add` only each task's listed files, never `-A`.

**Baseline:** before Task 1, run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/test_agent_runtime.py Tests/DB/test_agent_runs_db.py -q` and note any pre-existing failures — report, don't fix.

---

### Task 1: Generalized cyclic loop detection (AC#1)

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py` (add `MAX_LOOP_PERIOD` near `LOOP_DETECTION_N` ~L47)
- Modify: `tldw_chatbook/Agents/agent_runtime.py` (state ~L322-323, check ~L409-419; add a module-level `_detect_cycle` helper)
- Test: `Tests/Agents/test_agent_runtime.py`

**Interfaces:**
- Produces: `MAX_LOOP_PERIOD = 4` (agent_models); module-level `_detect_cycle(recent) -> tuple[int,int] | None` (agent_runtime).

- [ ] **Step 1: Write the failing tests** — append to `Tests/Agents/test_agent_runtime.py` (reuse its existing `run`/`ModelTurn`/`fence` helpers; check the top of the file for their exact names and import `_detect_cycle` + `MAX_LOOP_PERIOD`):

```python
from collections import deque
from tldw_chatbook.Agents.agent_runtime import _detect_cycle
from tldw_chatbook.Agents.agent_models import MAX_LOOP_PERIOD


def _keys(*names):
    # cycle-detection keys are (name, args-json); args identical here
    return deque([(n, "{}") for n in names], maxlen=LOOP_DETECTION_N * MAX_LOOP_PERIOD)


def test_detect_cycle_period1_needs_three():
    assert _detect_cycle(_keys("A", "A")) is None          # 2 identical: not yet
    assert _detect_cycle(_keys("A", "A", "A")) == (1, 3)    # 3 identical: trip


def test_detect_cycle_period2_trips_at_two_repeats():
    assert _detect_cycle(_keys("A", "B", "A")) is None      # incomplete
    assert _detect_cycle(_keys("A", "B", "A", "B")) == (2, 2)


def test_detect_cycle_period3_trips_at_two_repeats():
    assert _detect_cycle(_keys("A", "B", "C", "A", "B", "C")) == (3, 2)


def test_detect_cycle_non_cyclic_is_none():
    assert _detect_cycle(_keys("A", "B", "C", "D", "E")) is None


def test_alternating_calls_trip_loop_detection():
    # A->B->A->B with IDENTICAL args must trip RUN_STUCK (was a gap).
    a = ModelTurn(text=fence("calculator", {"expression": "6*7"}))
    b = ModelTurn(text=fence("get_current_datetime", {}))
    out = run([a, b, a, b, ModelTurn(text="x")], tools=["calculator", "get_current_datetime"])
    assert out.status == RUN_STUCK
    assert out.steps[-1].kind == "error"
    assert "loop detected" in out.steps[-1].summary


def test_alternating_different_args_not_stuck():
    # search(q1)->read(u1)->search(q2)->read(u2): distinct keys, no cycle.
    turns = [
        ModelTurn(text=fence("calculator", {"expression": "1+1"})),
        ModelTurn(text=fence("calculator", {"expression": "2+2"})),
        ModelTurn(text=fence("calculator", {"expression": "3+3"})),
        ModelTurn(text=fence("calculator", {"expression": "4+4"})),
        ModelTurn(text="done"),
    ]
    out = run(turns, tools=["calculator"])
    assert out.status != RUN_STUCK
```

(Adapt `run(...)`/`fence(...)`/tool names to the file's real helpers — read the existing `test_identical_consecutive_calls_trip_loop_detection` to copy its exact `run` call shape and tool set. The load-bearing assertions: `_detect_cycle` returns the right `(period, repeats)`; A→B→A→B trips RUN_STUCK; varied-args does not.)

- [ ] **Step 2: Run — verify fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/test_agent_runtime.py -q -k "detect_cycle or alternating"`
Expected: ImportError (`_detect_cycle`/`MAX_LOOP_PERIOD` don't exist), and the alternating test fails (never trips today).

- [ ] **Step 3a: Add `MAX_LOOP_PERIOD`** to `agent_models.py` right after `LOOP_DETECTION_N = 3`:

```python
LOOP_DETECTION_N = 3
# Longest tool-call cycle period the runtime detects (A->B->A->B is period 2).
MAX_LOOP_PERIOD = 4
```

- [ ] **Step 3b: Add the `_detect_cycle` helper** at module scope in `agent_runtime.py` (near the other module-level helpers; import `MAX_LOOP_PERIOD` alongside the existing `LOOP_DETECTION_N` import ~L18, and ensure `from collections import deque` is imported at the top):

```python
def _detect_cycle(recent) -> "tuple[int, int] | None":
    """Detect a repeating tool-call cycle in the tail of ``recent``.

    Returns ``(period, repeats)`` when the last ``repeats*period`` call-keys
    are ``repeats`` consecutive copies of the trailing ``period``-block, else
    ``None``. Threshold: ``LOOP_DETECTION_N`` (3) repeats for period 1
    (backward-compatible with the prior identical-consecutive check), 2 for
    periods >= 2. Smallest period first, so a longer cycle is never
    mis-attributed to a shorter period. Pure (no I/O).
    """
    seq = list(recent)
    n = len(seq)
    for period in range(1, MAX_LOOP_PERIOD + 1):
        repeats = LOOP_DETECTION_N if period == 1 else 2
        need = repeats * period
        if n < need:
            continue
        tail = seq[-need:]
        block = tail[-period:]
        if all(tail[i] == block[i % period] for i in range(need)):
            return (period, repeats)
    return None
```

- [ ] **Step 3c: Replace the loop-detection state + check** in `run_agent_loop`. Change the state init (currently `last_key: tuple | None = None` / `repeat_count = 0`, ~L322-323) to:

```python
    recent_calls: deque = deque(maxlen=LOOP_DETECTION_N * MAX_LOOP_PERIOD)
```

Replace the per-call check block (currently ~L409-419: the `key = (...)` / `repeat_count = ...` / `if repeat_count >= LOOP_DETECTION_N:` block) — keep its exact position (right after the `should_cancel`/`RUN_CANCELLED` check, before the `STEP_TOOL_CALL` dispatch) — with:

```python
            recent_calls.append((call.name, json.dumps(call.args, sort_keys=True)))
            cycle = _detect_cycle(recent_calls)
            if cycle is not None:
                period, repeats = cycle
                add(
                    STEP_ERROR,
                    summary=f"loop detected: {period}-cycle repeated {repeats}x",
                )
                return _outcome(RUN_STUCK)
```

- [ ] **Step 4: Run — verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/test_agent_runtime.py -q`
Expected: all pass — the new tests AND the pre-existing `test_identical_consecutive_calls_trip_loop_detection` (3 identical → period-1 trip) and `test_same_tool_different_args_is_not_stuck`.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_models.py tldw_chatbook/Agents/agent_runtime.py Tests/Agents/test_agent_runtime.py
git commit -m "fix(agents): detect cyclic (non-consecutive) tool-call loops, not only consecutive repeats [TASK-327]"
```

---

### Task 2: Reconcile orphaned `running` runs on open (AC#2)

**Files:**
- Modify: `tldw_chatbook/DB/AgentRuns_DB.py` (`__init__` ~L29-30; add `reconcile_orphaned_runs`; class-level `_swept_paths`)
- Test: `Tests/DB/test_agent_runs_db.py`

**Interfaces:**
- Produces: `AgentRunsDB.reconcile_orphaned_runs() -> int` (row count); fires once per file path per process.

- [ ] **Step 1: Write the failing test** — append to `Tests/DB/test_agent_runs_db.py` (read the file's existing helpers for constructing an `AgentRunsDB` on a `tmp_path` file and for `create_run`/`set_status`):

```python
def test_orphaned_running_runs_reconciled_on_open(tmp_path):
    db_path = tmp_path / "agent_runs.db"
    db1 = AgentRunsDB(db_path)
    r_run1 = db1.create_run(conversation_id="c1", agent_kind="console")
    r_run2 = db1.create_run(conversation_id="c2", agent_kind="console")
    r_done = db1.create_run(conversation_id="c3", agent_kind="console")
    db1.set_status(r_done, "done", result="the answer")

    # Simulate a fresh process opening the same file (clear the once-guard).
    AgentRunsDB._swept_paths.clear()
    db2 = AgentRunsDB(db_path)

    rows = {row["id"]: row for row in db2.list_runs()}  # or the file's list accessor
    assert rows[r_run1]["status"] == "error"
    assert rows[r_run1]["result"] == "Interrupted by app restart"
    assert rows[r_run2]["status"] == "error"
    assert rows[r_done]["status"] == "done"           # terminal row untouched
    assert rows[r_done]["result"] == "the answer"


def test_reconcile_preserves_existing_result(tmp_path):
    db_path = tmp_path / "agent_runs.db"
    db1 = AgentRunsDB(db_path)
    rid = db1.create_run(conversation_id="c", agent_kind="console")
    db1.set_status(rid, "running", result="partial output")  # running WITH a result
    AgentRunsDB._swept_paths.clear()
    db2 = AgentRunsDB(db_path)
    row = {r["id"]: r for r in db2.list_runs()}[rid]
    assert row["status"] == "error"
    assert row["result"] == "partial output"          # COALESCE keeps it


def test_reconcile_idempotent_same_process(tmp_path):
    db_path = tmp_path / "agent_runs.db"
    db1 = AgentRunsDB(db_path)
    db1.create_run(conversation_id="c", agent_kind="console")
    # second open in the SAME process (guard set) is a no-op
    assert AgentRunsDB(db_path).reconcile_orphaned_runs() == 0


def test_reconcile_skips_memory_db():
    # :memory: must not error and must not register a swept path
    AgentRunsDB._swept_paths.discard(":memory:")
    AgentRunsDB(":memory:")  # must not raise
    assert ":memory:" not in AgentRunsDB._swept_paths
```

(Adapt `list_runs()`/row-access to the file's real read accessor — read an existing test for the exact method name and whether it returns dicts or Rows. Load-bearing: two `running`→`error` with the interrupted message, `done` untouched, existing result preserved, idempotent, `:memory:` skipped.)

- [ ] **Step 2: Run — verify fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/DB/test_agent_runs_db.py -q -k reconcile or orphaned`
Expected: FAIL — `_swept_paths`/`reconcile_orphaned_runs` don't exist; running rows stay `running`.

- [ ] **Step 3: Implement.** In `AgentRuns_DB.py`, add a class-level attribute and reconcile method, and call it from `__init__`. Add `from tldw_chatbook.Agents.agent_models import RUN_ERROR` OR just use the literal `'error'` (the schema uses plain text; `RUN_ERROR == "error"`). Add near the top of the class body:

```python
    _swept_paths: set[str] = set()  # DB files already reconciled this process
```

Change `__init__` (currently just `super().__init__(db_path, client_id)`) to:

```python
    def __init__(self, db_path: Union[str, Path], client_id: str = "default") -> None:
        super().__init__(db_path, client_id)
        # After super().__init__: the agent_runs table exists (base_db ran
        # _initialize_schema) and self.is_memory_db is set. Reconcile once per
        # file per process so a crash mid-run doesn't leave a 'running' row
        # orphaned forever. File-backed only.
        if not self.is_memory_db and self.db_path_str not in self._swept_paths:
            self._swept_paths.add(self.db_path_str)
            try:
                self.reconcile_orphaned_runs()
            except Exception as exc:  # noqa: BLE001 — reconcile is best-effort
                logger.warning(f"AgentRunsDB reconcile skipped: {exc}")
```

Add the method (place near `set_status`):

```python
    def reconcile_orphaned_runs(self) -> int:
        """Mark runs left ``running`` by a crashed process as ``error``.

        A hard crash between run start (``create_run`` -> ``running``) and run
        end (``set_status`` at finalize) leaves a row stuck ``running``
        forever. On open, flip all such rows to ``error`` with a default
        ``result`` (preserving any partial result via COALESCE). Assumes a
        single app instance per data dir: a second instance sharing the file
        would flip the first's actively-running run — an accepted edge case,
        matching Library_Ingest_Jobs' "Interrupted by app restart" behavior.

        Returns:
            The number of rows reconciled.
        """
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE agent_runs "
                "SET status = 'error', "
                "    result = COALESCE(result, 'Interrupted by app restart'), "
                "    updated_at = ? "
                "WHERE status = 'running'",
                (_now_iso(),),
            )
            return cur.rowcount
```

(Confirm `logger` is imported in this module; if not, use the module's existing logging import or drop the try/except's log to a bare `pass` with a comment.)

- [ ] **Step 4: Run — verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/DB/test_agent_runs_db.py -q`
Expected: all pass (new + existing).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/AgentRuns_DB.py Tests/DB/test_agent_runs_db.py
git commit -m "fix(agents): reconcile orphaned 'running' agent runs to 'error' on DB open [TASK-327]"
```

---

### Task 3: WAL + busy_timeout on AgentRunsDB connections (AC#3)

**Files:**
- Modify: `tldw_chatbook/DB/AgentRuns_DB.py` (`_get_connection`)
- Test: `Tests/DB/test_agent_runs_db.py`

**Interfaces:** Consumes nothing new. No new public surface — behavioral change only (PRAGMAs applied on every connection).

- [ ] **Step 1: Write the failing test** — append to `Tests/DB/test_agent_runs_db.py`:

```python
def test_file_db_uses_wal_and_busy_timeout(tmp_path):
    db = AgentRunsDB(tmp_path / "agent_runs.db")
    with db.connection() as conn:  # or db._get_connection(); use the file's read accessor
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_memory_db_skips_wal(tmp_path):
    # :memory: cannot use WAL; must not raise and must stay 'memory'
    db = AgentRunsDB(":memory:")
    with db.connection() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "memory"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
```

(Use whatever connection accessor the existing tests use — `connection()` or `_get_connection()`. Load-bearing: file DB → `wal`; `:memory:` → not `wal`, no error; both → busy_timeout 5000.)

- [ ] **Step 2: Run — verify fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/DB/test_agent_runs_db.py -q -k "wal or busy_timeout"`
Expected: FAIL — `journal_mode` is `delete` (default), `busy_timeout` is `0`.

- [ ] **Step 3: Implement.** Replace `AgentRunsDB._get_connection` with:

```python
    def _get_connection(self) -> sqlite3.Connection:
        conn = super()._get_connection()
        conn.execute("PRAGMA foreign_keys = ON")
        # WAL lets a reader and the single writer proceed concurrently; a
        # busy_timeout makes a contended write wait up to 5s instead of
        # raising 'database is locked' immediately. WAL is unavailable for
        # in-memory DBs, so guard on is_memory_db (busy_timeout is harmless
        # there and kept for uniformity).
        if not self.is_memory_db:
            conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.row_factory = sqlite3.Row
        return conn
```

- [ ] **Step 4: Run — verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/DB/test_agent_runs_db.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/AgentRuns_DB.py Tests/DB/test_agent_runs_db.py
git commit -m "perf(agents): enable WAL + busy_timeout on AgentRunsDB connections [TASK-327]"
```

---

### Task 4: Per-tool-call timeout in the impure seam (AC#4)

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py` (`RunBudget` field + `clamp_child_budget`)
- Modify: `tldw_chatbook/Agents/agent_service.py` (module-level `_call_with_timeout`; wire into `_make_invoke_tool`)
- Test: `Tests/Agents/test_agent_service.py` (or the file that tests `agent_service`/`RunBudget`; check which exists) and `Tests/Agents/test_agent_models.py` for the clamp

**Interfaces:**
- Consumes: `ToolResult` (already imported in agent_service).
- Produces: `RunBudget.max_tool_call_seconds: float = 120.0` (0 = unlimited); module-level `_call_with_timeout(fn: Callable[[], ToolResult], seconds: float, tool_name: str) -> ToolResult` in agent_service.

- [ ] **Step 1: Write the failing tests.**

(a) In the agent_models test file, extend the `clamp_child_budget` test to assert the new field propagates:

```python
def test_clamp_child_budget_propagates_tool_call_seconds():
    parent = RunBudget(max_tool_call_seconds=45.0)
    child = clamp_child_budget(parent, 30.0)
    assert child.max_tool_call_seconds == 45.0   # taken from the child arg (== parent here)
    assert child.max_subagents == 0              # existing invariant still holds
```

(b) In the agent_service test file, unit-test the helper directly:

```python
import time
from tldw_chatbook.Agents.agent_service import _call_with_timeout
from tldw_chatbook.Agents.agent_models import ToolResult


def test_call_with_timeout_returns_result_when_fast():
    out = _call_with_timeout(lambda: ToolResult(ok=True, content="hi"), 5.0, "fast_tool")
    assert out.ok and out.content == "hi"


def test_call_with_timeout_trips_on_slow_call():
    def slow():
        time.sleep(2.0)
        return ToolResult(ok=True, content="late")
    out = _call_with_timeout(slow, 0.2, "slow_tool")
    assert out.ok is False
    assert "timed out" in out.error and "slow_tool" in out.error


def test_call_with_timeout_wraps_exception():
    def boom():
        raise ValueError("kaboom")
    out = _call_with_timeout(boom, 5.0, "bad_tool")
    assert out.ok is False and "kaboom" in out.error
```

(c) In the agent_service test file, add ONE closure-level test proving `max_tool_call_seconds=0` bypasses the timeout wrapper entirely (build the `invoke_tool` via `service._make_invoke_tool(config, disclosed)` with a `config.budget` whose `max_tool_call_seconds=0`, register a fast tool, assert it returns normally). If wiring a full `AgentService` in a unit test is heavy, assert the branch directly instead: with a `RunBudget(max_tool_call_seconds=0)`, `timeout and timeout > 0` is falsy — a targeted test on that expression is acceptable. Keep it a real assertion, not a name-only stub.

(Adapt `ToolResult(...)` construction to its real fields — check `agent_models.py` for whether it's `output=`/`content=`/`data=`. Load-bearing: fast → result passes through; slow → `ok=False` with "timed out" + tool name; raising → `ok=False` with the message.)

- [ ] **Step 2: Run — verify fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/test_agent_service.py Tests/Agents/test_agent_models.py -q -k "timeout or tool_call_seconds"`
Expected: FAIL — `_call_with_timeout` / `max_tool_call_seconds` don't exist.

- [ ] **Step 3a: Add the RunBudget field.** In `agent_models.py`, after `max_total_tokens: int = 0` (the last field, ~L141):

```python
    # task-327: per-tool-call wall-clock ceiling. A single custom/blocking
    # tool provider must not be able to wedge a cooperative-cancel run
    # forever. 0 = unlimited (opt-out). Enforced in agent_service's impure
    # seam (the pure runtime stays timeout-free); MCP tools self-time-out
    # via run_coroutine_threadsafe and are unaffected.
    max_tool_call_seconds: float = 120.0
```

- [ ] **Step 3b: Propagate through `clamp_child_budget`.** Add the field to the explicit `RunBudget(...)` it returns (so a child inherits the parent-configured value rather than silently resetting to the default):

```python
        max_total_tokens=child.max_total_tokens,
        max_tool_call_seconds=child.max_tool_call_seconds,
    )
```

- [ ] **Step 3c: Add the module-level helper** to `agent_service.py` (top level, near the other module-scope helpers; ensure `import threading` and `from typing import Callable` are present):

```python
def _call_with_timeout(
    fn: "Callable[[], ToolResult]", seconds: float, tool_name: str
) -> ToolResult:
    """Run ``fn`` on a daemon thread, bounded by ``seconds`` wall-clock.

    Always returns a ToolResult: ``fn``'s value on success, ``ok=False`` with
    the message on a raised exception, or an ``ok=False`` timeout result if
    ``fn`` does not finish in time. A per-call daemon thread (NOT a
    ThreadPoolExecutor ``with`` block, whose __exit__ would join the hung
    worker and defeat the timeout; NOT a shared pool, which a single hung
    tool would saturate) is used; on timeout the worker is abandoned to die
    with the process — Python cannot forcibly kill a thread, but ``daemon``
    means it never blocks interpreter shutdown.
    """
    box: dict = {}

    def _runner() -> None:
        try:
            box["result"] = fn()
        except Exception as exc:  # noqa: BLE001 — surfaced as a failed ToolResult
            box["error"] = str(exc)

    worker = threading.Thread(target=_runner, name=f"tool-{tool_name}", daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        return ToolResult(
            ok=False, error=f"tool call timed out after {seconds:g}s: {tool_name}"
        )
    if "error" in box:
        return ToolResult(ok=False, error=box["error"])
    return box["result"]
```

- [ ] **Step 3d: Wire it into `_make_invoke_tool`** (the builtin/custom closure — skill calls route around it via the outer `invoke_tool`, and MCP self-times-out, so wrapping here bounds exactly the custom/blocking registry tools). Replace the closure body:

```python
    def _make_invoke_tool(self, config: AgentConfig, disclosed_names: set):
        def invoke_tool(call: ToolCall) -> ToolResult:
            if (
                call.name not in config.allowed_tools
                or call.name not in disclosed_names
            ):
                return ToolResult(ok=False, error=f"Tool not permitted: {call.name}")
            timeout = config.budget.max_tool_call_seconds
            if timeout and timeout > 0:
                return _call_with_timeout(
                    lambda: self.registry.invoke_by_name(call.name, call.args),
                    timeout,
                    call.name,
                )
            return self.registry.invoke_by_name(call.name, call.args)

        return invoke_tool
```

- [ ] **Step 4: Run — verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/test_agent_service.py Tests/Agents/test_agent_models.py -q`
Expected: all pass. Then run the full agents suite to confirm no regression in the run wiring:
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/ -q`

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_models.py tldw_chatbook/Agents/agent_service.py Tests/Agents/test_agent_service.py Tests/Agents/test_agent_models.py
git commit -m "fix(agents): bound each custom tool call with a per-call timeout (max_tool_call_seconds) [TASK-327]"
```

(If `Tests/Agents/test_agent_service.py` / `test_agent_models.py` don't exist under those names, put the tests in whatever file already tests those modules — confirm with `ls Tests/Agents/` first and `git add` the real paths.)

---

### Task 5: Serialization-contract doc note + close TASK-327 (AC#5 + backlog)

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (`run_reply` docstring)
- Modify: `backlog/tasks/task-327 - *.md`

**Interfaces:** Documentation only. No code behavior change; no test.

- [ ] **Step 1: Add the doc note.** In `console_agent_bridge.py`, append a paragraph to the existing `run_reply` docstring (after the "Returns:" block's content, or as a new paragraph before "Returns:"). This records that per-conversation run serialization is owned by the controller, NOT by the bridge's `_live`/`_historical_cache` dicts (which are display state, not a concurrency guard):

```python
        Concurrency: this bridge does NOT serialize runs. The
        ``_live``/``_historical_cache`` dicts are per-conversation DISPLAY
        snapshots, not a mutual-exclusion guard. The single active-run-per-
        conversation invariant is enforced upstream by
        ``ConsoleChatController`` (its ``_active_run_rejection`` /
        ``run_state.is_send_allowed`` gate — covered by
        ``Tests/UI/test_console_run_gate.py``): a second send while a run is
        live is rejected there before ``run_reply`` is ever called. Do not
        add a competing guard here.
```

- [ ] **Step 2: Verify the referenced contract still exists** (so the doc note doesn't cite vanished symbols):

Run:
```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook-agent-durability
grep -rn "_active_run_rejection\|is_send_allowed" tldw_chatbook/Chat/console_chat_controller.py
ls Tests/UI/test_console_run_gate.py
```
Expected: both grep hits present and the test file exists. If a name has drifted, update the docstring to match the real symbol (the CONTRACT — controller is the sole serialization point — is what must be documented; keep the cited names accurate).

- [ ] **Step 3: Commit the doc note**

```bash
git add tldw_chatbook/Chat/console_agent_bridge.py
git commit -m "docs(agents): document ConsoleChatController as the sole per-conversation run-serialization point [TASK-327]"
```

- [ ] **Step 4: Close the backlog task.** Mark all 5 ACs `- [x]`, add an Implementation Notes section, set status Done:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook-agent-durability
backlog task edit 327 -s Done --notes "Five durability fixes to the agent runtime. AC#1: replaced the one-slot consecutive-repeat loop guard with a deque of the last LOOP_DETECTION_N*MAX_LOOP_PERIOD call-keys + a pure _detect_cycle() that trips RUN_STUCK on any period-1..4 cycle (period-1 keeps the 3-repeat threshold; periods>=2 trip at 2 full repeats), so A->B->A->B is now caught. AC#2: AgentRunsDB.reconcile_orphaned_runs() flips crash-orphaned 'running' rows to 'error' (COALESCE-preserving any partial result) once per file per process on open. AC#3: WAL + busy_timeout=5000 on AgentRunsDB connections (WAL guarded for :memory:). AC#4: RunBudget.max_tool_call_seconds (default 120s, 0=unlimited), propagated through clamp_child_budget, enforced by a module-level _call_with_timeout daemon-thread helper wrapping only the builtin/custom registry.invoke_by_name (skills route around it; MCP self-times-out) so a blocking provider cannot wedge a cooperative-cancel run. AC#5: doc-only — documented ConsoleChatController (_active_run_rejection/is_send_allowed, Tests/UI/test_console_run_gate.py) as the sole per-conversation run-serialization point; the bridge's _live/_historical_cache are display-only. The pure runtime stays pure (timeout lives entirely in agent_service). Files: agent_models.py, agent_runtime.py, AgentRuns_DB.py, agent_service.py, console_agent_bridge.py + tests."
```

(Confirm the exact task filename first with `ls "backlog/tasks/" | grep 327`. If `backlog` CLI edits the file in the worktree, `git add` it and commit: `git commit -m "chore(backlog): close TASK-327 [Done]"`.)

---

## Post-Implementation

After all 5 tasks: run the full agents + DB suites once more from the worktree —
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/ Tests/DB/test_agent_runs_db.py -q` —
then hand off to superpowers:subagent-driven-development's final whole-branch review (opus) and superpowers:finishing-a-development-branch. TASK-327 closes the LLM-harness-review stream (320-334).
