# Supervisor Fleet PR 2a — Concurrency Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sub-agents run concurrently on their own threads under a `FleetCoordinator`, with `wait_agents`/`check_agents` runtime tools, while every shared seam the old one-child-at-a-time design relied on (both permission gates, the tool registry cache, run-status writes, step attribution) is made concurrency-safe.

**Architecture:** A new pure-ish `FleetCoordinator` (`Agents/fleet_coordinator.py`) owns child handles, the live-children cap, and the outbound event queue; `agent_service`'s spawn closure registers + launches instead of running inline. Phase 2 keeps children **turn-scoped** (the turn does not end until every child is finished or cancelled), but the coordinator is built for the cross-turn lifetime PR 3a will enable — no throwaway model.

**Tech Stack:** Python ≥3.11 (`threading`, `concurrent.futures` not required), Textual 8.x, SQLite WAL with per-thread held connections, pytest.

**Spec:** `Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-design.md` §5 (phase 2 rows of the corrections table are mandatory, not optional) + §3 invariants. Read both before Task 1.

## Global Constraints

- Worktree `.worktrees/fleet-pr2a`, branch `feat/fleet-concurrency-runtime` (already cut from merged dev). NEVER run git outside it. NEVER use `git stash` (shared across 100+ worktrees). Push after every task.
- pytest is the ONLY python entry point. A bare `python -c` importing `tldw_chatbook.config` triggers the app's config-rewrite and has touched the user's LIVE config — never do it. Never read/write `~/.config/tldw_cli` or `~/.local/share/tldw_cli`.
- `agent_models.py` and `agent_runtime.py` stay pure (stdlib only; no Textual/app/DB/I/O). `fleet_coordinator.py` may import `threading` and stdlib only — no DB, no Textual. The impure wiring lives in `agent_service.py`.
- **Byte-identical behavior AC — amended 2026-08-09 after Task 6 (the original was unsatisfiable).**
  The original constraint required all three of: (i) byte-identical at `max_live_subagents=1`
  *or one spawn*, (ii) the existing spawn suites pass unmodified, (iii) default 3 (Task 8).
  These cannot all hold: those suites drive one ordered reply queue and index
  `chat.calls[1]`, and they assert the *spawn* tool_result carries the child's capped text —
  all three encode inline **semantics**, not just inline ordering. Under a live fleet the
  result arrives via `wait_agents` and the parent emits an extra tool call, so "one spawn"
  is not byte-identical either. Flipping the default measures at **23 failures across 11
  files**. The AC is therefore re-pinned as:
  - **`max_live_subagents=1` means no coordinator is built and the spawn closure takes the
    verbatim pre-PR inline branch** — byte-identical, guarded by the three suites passing
    unmodified. This is the *only* byte-identical claim.
  - **At `max_live_subagents>1` behavior deliberately changes** (results via `wait_agents`);
    the affected suites are converted to addressed replies in **Task 6.5**, which also flips
    the default. No task may claim the byte-identical AC while the fleet is on.
- Depth-1 stays structural: children never spawn (`clamp_child_budget` zeroes `max_subagents`); fleet tools are primary-only, pinned like `install_skill`.
- Intersection-never-union and the identity-contract prompt append (PR 1) are untouched.
- `clamp_child_budget` stays byte-identical in this PR (turn-scoped children must still die with the parent). Containment changes land in PR 3a.
- Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

### Verified seam map (current dev, `f24f8c692`) — cite these, do not re-derive

| Seam | Location |
|---|---|
| spawn closure signature | `agent_service.py:691-696` (`spawn_task`, kw-only `allowed_tools`, `agent`) |
| `sub_agent_spawns` counter | `agent_service.py:689` — a local of `_run_one` |
| budget gate + increment | `agent_service.py:737-739` |
| inline child call | `agent_service.py:814-830`, inside `with scope:` |
| child result handling | `agent_service.py:831-841` |
| `_run_one` signature | `agent_service.py:526-539` |
| `on_step` invocation | `agent_service.py:1379-1383` — `lambda s: self._on_step(s, agent_kind)`, **no run_id** |
| `set_status` UPDATE | `AgentRuns_DB.py:659-664` — unconditional |
| registry caches + race comment | `tool_catalog.py:893-907`, `_ensure_catalog_cache:976-1006`, `invoke_by_name:1032-1063` |
| MCP gate state | `mcp_tool_provider.py:214` `_stamped_decisions` (keyed by llm_name), `stamp_scope:406-434`, `apply_batch_decisions:367-386` |
| builtin gate state | `builtin_tool_gate.py:65-67` `_stamps` (keyed by tool_name), `begin_turn:68-79`, `stamp_scope:79-117` |
| `begin_turn` call site | `console_chat_controller.py:518` (only production caller) |
| `stamp_scope` composition | `console_agent_bridge.py:2021-2036` via `_combine_state_scopes:188` |
| approval round key | `console_chat_controller.py:2752` — `round_id = str(uuid4())` — **already globally unique; no re-keying needed** |
| approval round registry | `console_chat_controller.py:2763-2768`, `add_pending_round:1498`, `discard_pending_round:1537` |
| approval timeout enforcement | `console_chat_controller.py:2770-2771`, `2840`, `2867-2870` |
| `AgentService` lifetime | constructed **per turn** at `console_agent_bridge.py:2063`; the `ToolCatalogRegistry` is **long-lived on the bridge** |

---

### Task 1: `FleetCoordinator` + `FleetHandle` (pure, no I/O)

**Files:**
- Create: `tldw_chatbook/Agents/fleet_coordinator.py`
- Test: Create `Tests/Agents/test_fleet_coordinator.py`

**Interfaces:**
- Consumes: `AgentStep`, run-status constants from `agent_models`.
- Produces (imported by Tasks 2-5):
  - `FleetHandle` dataclass: `handle_id: str`, `run_id: str | None`, `agent: str | None`, `task: str`, `status: str`, `result: str = ""`, `error: str = ""`, `started_at: float`, `finished_at: float | None`
  - `FleetEvent` frozen dataclass: `kind: str` (`FLEET_STARTED`/`FLEET_FINISHED`), `handle_id: str`, `run_id: str | None`, `agent: str | None`, `status: str`
  - `FleetCoordinator(max_live: int, clock: Callable[[], float])` with methods: `reserve(task, agent) -> FleetHandle | None` (None when at cap), `attach_run(handle_id, run_id)`, `finish(handle_id, status, result="", error="")`, `get(handle_id) -> FleetHandle | None`, `snapshot() -> list[FleetHandle]`, `live_count() -> int`, `drain_events() -> list[FleetEvent]`, `all_finished() -> bool`
  - Constants `FLEET_STARTED`, `FLEET_FINISHED`
- Thread-safety: every public method takes one `threading.RLock`. `snapshot()` returns copies, never internal objects.

- [ ] **Step 1: Write the failing tests**

```python
"""FleetCoordinator: pure handle/state machine for concurrent children."""

import threading

import pytest

from tldw_chatbook.Agents.agent_models import RUN_DONE, RUN_ERROR
from tldw_chatbook.Agents.fleet_coordinator import (
    FLEET_FINISHED,
    FLEET_STARTED,
    FleetCoordinator,
)


def _coord(max_live=3):
    ticks = iter(range(1000))
    return FleetCoordinator(max_live=max_live, clock=lambda: float(next(ticks)))


def test_reserve_returns_handle_and_emits_started():
    c = _coord()
    h = c.reserve(task="do x", agent="researcher")
    assert h is not None and h.task == "do x" and h.agent == "researcher"
    assert h.status == "running" and h.finished_at is None
    events = c.drain_events()
    assert [e.kind for e in events] == [FLEET_STARTED]
    assert events[0].handle_id == h.handle_id
    assert c.drain_events() == []  # drain is destructive


def test_reserve_refuses_past_live_cap():
    c = _coord(max_live=2)
    assert c.reserve(task="a", agent=None) is not None
    assert c.reserve(task="b", agent=None) is not None
    assert c.reserve(task="c", agent=None) is None
    assert c.live_count() == 2


def test_finish_frees_a_slot_and_emits_finished():
    c = _coord(max_live=1)
    h = c.reserve(task="a", agent=None)
    assert c.reserve(task="b", agent=None) is None
    c.finish(h.handle_id, RUN_DONE, result="answer")
    assert c.live_count() == 0
    assert c.reserve(task="b", agent=None) is not None
    kinds = [e.kind for e in c.drain_events()]
    assert kinds == [FLEET_STARTED, FLEET_FINISHED, FLEET_STARTED]
    done = c.get(h.handle_id)
    assert done.status == RUN_DONE and done.result == "answer"
    assert done.finished_at is not None


def test_finish_is_idempotent_first_writer_wins():
    # A child abandoned after a join timeout can finish LATE; the
    # coordinator must not let it overwrite a terminal status.
    c = _coord()
    h = c.reserve(task="a", agent=None)
    c.finish(h.handle_id, "cancelled")
    c.finish(h.handle_id, RUN_DONE, result="late answer")
    assert c.get(h.handle_id).status == "cancelled"
    assert c.get(h.handle_id).result == ""


def test_attach_run_records_run_id():
    c = _coord()
    h = c.reserve(task="a", agent=None)
    c.attach_run(h.handle_id, "run-123")
    assert c.get(h.handle_id).run_id == "run-123"
    c.finish(h.handle_id, RUN_ERROR, error="boom")
    assert c.drain_events()[-1].run_id == "run-123"


def test_snapshot_returns_copies_not_internals():
    c = _coord()
    h = c.reserve(task="a", agent=None)
    snap = c.snapshot()
    snap[0].status = "tampered"
    assert c.get(h.handle_id).status == "running"


def test_all_finished_reflects_live_state():
    c = _coord()
    assert c.all_finished() is True
    h = c.reserve(task="a", agent=None)
    assert c.all_finished() is False
    c.finish(h.handle_id, RUN_DONE)
    assert c.all_finished() is True


def test_concurrent_reserve_never_exceeds_cap():
    c = _coord(max_live=5)
    got = []
    lock = threading.Lock()

    def worker():
        h = c.reserve(task="t", agent=None)
        with lock:
            got.append(h)

    threads = [threading.Thread(target=worker) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(1 for h in got if h is not None) == 5
    assert c.live_count() == 5
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Agents/test_fleet_coordinator.py -v`
Expected: FAIL / ImportError (no `fleet_coordinator` module).

- [ ] **Step 3: Implement**

Write `tldw_chatbook/Agents/fleet_coordinator.py`. Requirements the tests pin:
- Module docstring stating: pure state machine, stdlib only (`threading`, `dataclasses`, `uuid`, `typing`), no DB/Textual/I/O — mirrors `agent_runtime`'s purity rule; the impure thread launching lives in `agent_service`.
- One `threading.RLock` guarding all state. `reserve` checks the cap and inserts atomically under the lock (the concurrency test proves it).
- `finish` ignores a second call on an already-terminal handle (**first-writer-wins**) and only then appends `FLEET_FINISHED`; document that this exists because an abandoned thread can finish after the coordinator already cancelled it.
- `drain_events` returns and clears the queue under the lock.
- `handle_id` from `uuid.uuid4().hex`.
- `snapshot`/`get` return `dataclasses.replace(...)` copies.
- Type hints on every public method; Google-style docstrings (repo style).

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Agents/test_fleet_coordinator.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/fleet_coordinator.py Tests/Agents/test_fleet_coordinator.py
git commit -m "feat: FleetCoordinator handle/state machine" && git push
```

---

### Task 2: Terminal-status guard on `set_status`

**Files:**
- Modify: `tldw_chatbook/DB/AgentRuns_DB.py` (`set_status`, `:647-664`)
- Test: `Tests/DB/test_agent_runs_db.py` (append)

**Interfaces:**
- Produces: `set_status` becomes a no-op when the run is already terminal; adds `-> bool` (True when a row changed). Existing callers ignore the return value — verify `agent_service.py:524` still compiles unchanged.

- [ ] **Step 1: Write the failing tests**

```python
def test_set_status_first_terminal_write_wins(db):
    # A child abandoned after a join timeout can persist LATE; it must not
    # overwrite the terminal status the coordinator already recorded.
    run_id = db.create_run(conversation_id="c", agent_kind="subagent", task="t")
    assert db.set_status(run_id, "cancelled") is True
    assert db.set_status(run_id, "done", result="late answer") is False
    run = db.get_run(run_id)
    assert run["status"] == "cancelled"
    assert run["result"] is None


def test_set_status_still_updates_a_running_run(db):
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    assert db.set_status(run_id, "done", result="ok") is True
    assert db.get_run(run_id)["status"] == "done"
    assert db.get_run(run_id)["result"] == "ok"


def test_set_status_missing_run_returns_false(db):
    assert db.set_status("nope", "done") is False
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/DB/test_agent_runs_db.py -v -k set_status`
Expected: FAIL (the late write currently wins; return is `None`).

- [ ] **Step 3: Implement**

In `set_status`, import `TERMINAL_RUN_STATUSES` from `..Agents.agent_models` (the module already imports from there for `AgentDefinition`) and add the guard to the UPDATE:

```python
        placeholders = ",".join("?" for _ in TERMINAL_RUN_STATUSES)
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE agent_runs SET status = ?, "
                "result = COALESCE(?, result), updated_at = ? "
                f"WHERE id = ? AND status NOT IN ({placeholders})",
                (status, result, _now_iso(), run_id, *sorted(TERMINAL_RUN_STATUSES)),
            )
        return cursor.rowcount > 0
```

Update the docstring: state that a run already in a terminal status is never rewritten (first-writer-wins), because an abandoned child thread can persist after the coordinator recorded `cancelled`, and add a `Returns:` section.

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/DB/test_agent_runs_db.py Tests/Agents/test_agent_runs_db_connection_reuse.py -v`
Expected: ALL PASS (0 failures — the 3 trace-spy tests were repaired on dev; a failure here is yours).

- [ ] **Step 5: Check `reconcile_orphaned_runs` still behaves**

Run: `pytest Tests/DB/test_agent_runs_db.py -v -k reconcile`
Expected: PASS. (It updates `WHERE status = 'running'`, so the new guard cannot affect it — confirm, don't assume.)

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/DB/AgentRuns_DB.py Tests/DB/test_agent_runs_db.py
git commit -m "fix: first-writer-wins guard on terminal run status" && git push
```

---

### Task 3: `run_id` on `on_step`

**Files:**
- Modify: `tldw_chatbook/Agents/agent_service.py` (`__init__:288`, the `LoopDeps` `on_step` at `:1379-1383`)
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (the `on_step` closure passed at `:2063`)
- Test: `Tests/Agents/test_agent_service_on_step.py` (append), `Tests/Chat/test_console_agent_bridge.py` (verify unchanged)

**Interfaces:**
- Produces: `on_step` callback signature becomes `(step: AgentStep, agent_kind: str, run_id: str) -> None`. Every caller must be updated in the same task — an un-updated caller raises `TypeError` at runtime, not import time.

- [ ] **Step 1: Write the failing test** (append to `Tests/Agents/test_agent_service_on_step.py`; read the file's existing helpers first and reuse them)

```python
def test_on_step_receives_run_id_for_primary_and_child(db):
    seen = []
    service, _chat = make_service_with_on_step(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "child work"}),
            "child answer",
            "parent answer",
        ],
        on_step=lambda step, kind, run_id: seen.append((kind, run_id, step.kind)),
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    primary_ids = {r for k, r, _ in seen if k == "primary"}
    child_ids = {r for k, r, _ in seen if k == "subagent"}
    assert primary_ids == {run_id}
    assert len(child_ids) == 1 and child_ids.isdisjoint(primary_ids)
    # Every step carries a non-empty run id.
    assert all(r for _k, r, _s in seen)
```

(`make_service_with_on_step` may not exist — if the file builds the service inline, follow that shape instead and keep the assertions identical.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Agents/test_agent_service_on_step.py -v`
Expected: FAIL — `TypeError: <lambda>() takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Implement**

- `agent_service.py:288`: `on_step: Callable[[AgentStep, str, str], None] | None = None,`
- `agent_service.py:1379-1383`: `(lambda s: self._on_step(s, agent_kind, run_id))` — `run_id` is already in scope in `_run_one` (assigned at `:541`). Verify that by reading, not assuming.
- `console_agent_bridge.py`: find the `on_step` closure passed at `:2063` and widen its signature to accept `run_id`. **Do not** change what it does with it in this task (PR 2b consumes it); just accept and ignore, with a comment saying PR 2b routes fleet rows by it.
- Grep the whole repo for other `on_step=` call sites (tests included) and update every one: `grep -rn "on_step" --include=*.py`.

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Agents/ Tests/Chat/test_console_agent_bridge.py -q`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add -u && git commit -m "feat: attribute steps to their run id" && git push
```

---

### Task 4: Registry cache lock

**Files:**
- Modify: `tldw_chatbook/Agents/tool_catalog.py` (`ToolCatalogRegistry.__init__`, `_ensure_catalog_cache`, `reset_catalog_cache`, `register_provider`, `resolve_name`, `_owner_and_id`, `_source_for`)
- Test: Create `Tests/Agents/test_tool_catalog_concurrency.py`

**Interfaces:**
- Produces: no signature changes — internal `threading.RLock` only. `invoke_by_name`'s two reads become one locked snapshot read.

- [ ] **Step 1: Write the failing test**

```python
"""Registry cache under concurrent lookups (fleet PR 2a).

The registry's own comment (tool_catalog.py:893-907) documents that
`_owner_cache` and `_name_to_id_cache` are rebuilt without a lock, so two
concurrent lookups can observe different generations. With N children on
their own threads sharing the bridge's long-lived registry, that stops
being exotic.
"""

import threading

from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry


def test_concurrent_resolution_never_sees_a_torn_cache():
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    known = [e.name for e in registry.list_catalog()]
    assert known, "catalog must be non-empty for this test to mean anything"

    errors = []
    barrier = threading.Barrier(8)

    def hammer():
        barrier.wait()
        for _ in range(200):
            registry.reset_catalog_cache()
            for name in known:
                tool_id = registry.resolve_name(name)
                if tool_id is None:
                    errors.append(f"resolve_name({name}) -> None")
                    return

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
```

- [ ] **Step 2: Run to verify it fails (or is flaky) before the fix**

Run: `pytest Tests/Agents/test_tool_catalog_concurrency.py -v` — repeat up to 5 times.
Expected: at least one run FAILS (a `resolve_name -> None` while another thread reset the cache). Record what you observed. **If it never fails in 5 runs, say so in your report** — implement the lock anyway (the race is documented in the code and the fleet makes it routine), but do not claim you reproduced it.

- [ ] **Step 3: Implement**

- `__init__`: add `self._cache_lock = threading.RLock()` (import `threading` at module top).
- `_ensure_catalog_cache`: take the lock around the check-and-rebuild. Return the three maps as a tuple so callers get one coherent generation instead of re-reading fields.
- `reset_catalog_cache` and `register_provider`: take the lock around the `= None` invalidations.
- `resolve_name`, `_owner_and_id`, `_source_for`: read via the locked snapshot.
- `invoke_by_name`: get name→id and owner from **one** locked call so a reset between the two reads can no longer produce the "provider is None" fallback. Keep that fallback branch (defense in depth) and update its comment to say the lock now makes it unreachable in-process.
- Update the `__init__` warning comment (`:893-907`) to record that the lock closed this, keeping the historical explanation.

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Agents/test_tool_catalog_concurrency.py -v` — run it 5 times; all must pass.
Then: `pytest Tests/Agents/ -q` — ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/tool_catalog.py Tests/Agents/test_tool_catalog_concurrency.py
git commit -m "fix: lock the tool-catalog cache for concurrent children" && git push
```

---

### Task 5: Per-run scoping for BOTH permission gates

**Files:**
- Modify: `tldw_chatbook/Agents/builtin_tool_gate.py` (`_stamps`, `begin_turn`, `stamp`, `stamped/resolve` readers, `stamp_scope`)
- Modify: `tldw_chatbook/Agents/mcp_tool_provider.py` (`_stamped_decisions`, `apply_batch_decisions`, `stamped_decision`, `stamp_scope`)
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`begin_turn` call at `:518`; the `apply_batch_decisions` calls at `:360,365,535,543,681,744,753`)
- Test: `Tests/Agents/test_gate_run_scoping.py` (create); `Tests/Agents/test_agent_service_review_state_scope.py` (must still pass)

**Interfaces:**
- Produces: both gates key their per-turn verdict state by `(run_id, name)` instead of `name`. Public methods gain a `run_id: str` parameter. `stamp_scope` remains for backward compatibility but becomes a no-op-shaped context manager scoped to one run id.
- The **current** design is snapshot/restore (LIFO), which is only sound for a nested inline child. With N children on their own threads there is no LIFO: a child's `begin_turn` clears stamps its siblings and parent are still consuming.

- [ ] **Step 1: Write the failing tests** (`Tests/Agents/test_gate_run_scoping.py`)

```python
"""Both permission gates must key verdicts per RUN, not per tool name.

Pre-fix, `BuiltinToolGate.begin_turn()` cleared a dict keyed by tool name
on a SHARED gate instance, and `MCPToolProvider.apply_batch_decisions`
REPLACED its dict wholesale — so with concurrent children, one child's
turn wipes or overwrites a verdict a sibling (or the parent) already
decided and has not yet consumed.
"""

import threading

from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider


def test_builtin_gate_stamps_do_not_leak_across_runs():
    gate = BuiltinToolGate(service=None)
    gate.begin_turn("run-parent")
    gate.stamp("run-parent", "calculator", "proceed")
    # A concurrent child starts its own turn on the SAME gate instance.
    gate.begin_turn("run-child")
    gate.stamp("run-child", "calculator", "deny")
    # The parent's verdict must survive the child's turn untouched.
    assert gate.stamped("run-parent", "calculator") == "proceed"
    assert gate.stamped("run-child", "calculator") == "deny"


def test_builtin_gate_begin_turn_clears_only_its_own_run():
    gate = BuiltinToolGate(service=None)
    gate.begin_turn("run-a")
    gate.stamp("run-a", "calculator", "proceed")
    gate.begin_turn("run-b")
    assert gate.stamped("run-a", "calculator") == "proceed"
    gate.begin_turn("run-a")  # a's NEXT turn clears a's stamps
    assert gate.stamped("run-a", "calculator") is None


def test_mcp_decisions_do_not_clobber_across_runs():
    provider = MCPToolProvider.__new__(MCPToolProvider)
    provider._init_decision_state()  # helper added by this task
    provider.apply_batch_decisions("run-parent", {"srv__tool": "proceed"})
    provider.apply_batch_decisions("run-child", {"srv__tool": "deny"})
    assert provider.stamped_decision("run-parent", "srv__tool") == "proceed"
    assert provider.stamped_decision("run-child", "srv__tool") == "deny"


def test_concurrent_runs_keep_their_own_verdicts():
    gate = BuiltinToolGate(service=None)
    errors = []

    def worker(i):
        run = f"run-{i}"
        gate.begin_turn(run)
        gate.stamp(run, "calculator", f"verdict-{i}")
        for _ in range(200):
            if gate.stamped(run, "calculator") != f"verdict-{i}":
                errors.append(run)
                return

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Agents/test_gate_run_scoping.py -v`
Expected: FAIL — `begin_turn()` takes no run_id (TypeError), which is the point.

- [ ] **Step 3: Implement**

Builtin gate:
- `self._stamps: dict[tuple[str, str], str]` keyed `(run_id, tool_name)`; `self._payload` may stay global (it is config, not per-run) — verify by reading `_load_payload` before deciding, and say which you chose in your report.
- `begin_turn(run_id)`: drop only that run's keys.
- `stamp(run_id, tool_name, decision)`, `stamped(run_id, tool_name)`, and any resolver that reads a stamp: thread `run_id` through.
- Guard all mutation with a `threading.RLock` (concurrent children mutate one instance).
- `stamp_scope(run_id)`: keep as a compatibility context manager that snapshots/restores only that run's keys; document that per-run keying is now the real mechanism and this exists for the nested/inline path.

MCP provider: same treatment for `_stamped_decisions` → `(run_id, llm_name)`; `apply_batch_decisions(run_id, decisions)` merges into that run's slice instead of replacing the whole dict; `stamped_decision(run_id, llm_name)`. Add `_init_decision_state()` (used by the test to build a bare instance) that initializes the dict + lock; call it from `__init__`.

Callers: `console_chat_controller.py:518` `begin_turn` and the seven `apply_batch_decisions` call sites must pass the run id. **The controller does not currently know the run id** — thread it from the review hook. Read `build_tool_review_hook`/`build_mcp_review_hook` and `agent_service`'s `review_tool_calls` invocation first: the cleanest seam is for `AgentService` to pass its `run_id` into `review_tool_calls`. If that requires widening `LoopDeps.review_tool_calls`, do it and update every call site; state the chosen seam in your report.

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Agents/test_gate_run_scoping.py Tests/Agents/test_builtin_tool_gate.py Tests/Agents/test_agent_service_review_state_scope.py Tests/Chat/test_console_chat_controller.py Tests/UI/test_console_mcp_approval.py -q`
Expected: ALL PASS. `test_agent_service_review_state_scope.py` is the C1 regression suite — it MUST stay green.

- [ ] **Step 5: Commit**

```bash
git add -u && git commit -m "fix: key permission-gate verdicts per run" && git push
```

---

### Task 6: Threaded spawn + `wait_agents`/`check_agents`

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py` (two tool-name constants + `RUNTIME_TOOL_NAMES`)
- Modify: `tldw_chatbook/Agents/tool_catalog.py` (two `ToolSchema`s)
- Modify: `tldw_chatbook/Agents/agent_runtime.py` (`LoopDeps` fields + dispatch branches)
- Modify: `tldw_chatbook/Agents/agent_service.py` (coordinator wiring, threaded spawn, tool implementations, end-of-turn join)
- Test: `Tests/Agents/test_fleet_runtime.py` (create); existing spawn suites must pass unmodified

**Interfaces:**
- Consumes: Task 1's coordinator, Task 2's guard, Task 3's run_id.
- Produces: `WAIT_AGENTS_TOOL_NAME = "wait_agents"`, `CHECK_AGENTS_TOOL_NAME = "check_agents"`; `spawn(...)` returns immediately with a handle id in its `ToolResult.content`.

- [ ] **Step 1: Write the failing tests** (`Tests/Agents/test_fleet_runtime.py`; reuse `Tests/Agents/test_agent_service.py`'s `ScriptedChat`/`make_service`/`fence`/`CFG` helpers — import or copy their shape)

```python
def test_two_children_run_concurrently_and_wait_collects_both(db):
    # Parent spawns two, then waits; both results come back.
    service, chat = make_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "task one"}),
            fence(SPAWN_TOOL_NAME, {"task": "task two"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "combined answer",
            # child scripts are consumed by the child threads:
            "answer one",
            "answer two",
        ],
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert db.count_subagent_runs("c") == 2
    # Both children's results reached the parent's wait result.
    wait_results = [
        m for m in chat.calls[-1]["messages_payload"]
        if "answer one" in str(m.get("content", ""))
    ]
    assert wait_results


def test_spawn_returns_handle_without_blocking(db):
    # The spawn tool result must name a handle, not the child's answer.
    ...


def test_check_agents_reports_status_without_blocking(db):
    ...


def test_live_cap_refuses_beyond_max_live_subagents(db):
    ...


def test_end_of_turn_waits_for_stragglers(db):
    # Parent finishes its text without calling wait_agents; the service
    # must not return until children are finished or cancelled, and no
    # run row may be left 'running'.
    ...


def test_single_child_path_is_unchanged(db):
    # Byte-identical AC: one spawn + wait behaves exactly like the old
    # inline path — same run rows, same result text.
    ...
```

Fill in every `...` with real code before running — a plan-shaped stub is not a test. Model each on the existing `test_spawn_creates_linked_child_with_clean_context` arrangement.

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Agents/test_fleet_runtime.py -v`
Expected: FAIL (no `wait_agents`).

- [ ] **Step 3: Implement**

1. `agent_models.py`: add both name constants and include them in `RUNTIME_TOOL_NAMES`.
2. `tool_catalog.py`: `WAIT_AGENTS_SCHEMA` (optional `ids: array of string`; omitted = all) and `CHECK_AGENTS_SCHEMA` (no params).
3. `agent_service.__init__`: accept `fleet_coordinator: FleetCoordinator | None = None`; when None, build one per `run_turn` sized from config (`[agents] max_live_subagents`, default 3) — read the config the way the service already reads settings, or take it as a constructor arg from the bridge if the service has no config access (check first; state which you chose).
4. Spawn closure: replace the inline `with scope: self._run_one(...)` block (`:814-830`) with: `reserve` on the coordinator (refuse with a clear `ToolResult` when at cap), start a daemon `threading.Thread` running the child's `_run_one` with the SAME arguments, `attach_run` when the child's run id is known, `finish` in a `finally`. Return `ToolResult(ok=True, content=f"started {handle_id}: <task snippet>")`. **Keep `sub_agent_spawns` and the budget check exactly where they are** — spawns-per-turn is unchanged; the coordinator cap is a separate, additional bound.
   - The child thread must catch every exception and `finish(handle_id, RUN_ERROR, error=...)` — an uncaught exception on a daemon thread would otherwise strand the parent's join.
   - `scope` (`review_state_scope`) is **no longer** the mechanism protecting gate state (Task 5 replaced it); keep acquiring it around the child for the non-fleet/local-provider scopes, but note in a comment that per-run keying is the load-bearing protection now.
5. `wait_agents(ids=None)`: poll the coordinator until the named (or all) handles are terminal, `should_cancel` each poll, bounded by the parent's remaining wall-clock; on cancel/timeout, cooperatively cancel outstanding children. Compose the result: one entry per child, each truncated to `max_subagent_result_chars`, and the WHOLE result additionally budgeted to `max_tool_result_chars` split evenly across children (spec §5 — otherwise 5 children × 4000 chars silently truncates mid-result). Tell the model it can re-fetch one child in full via `wait_agents([id])`.
6. `check_agents()`: format `coordinator.snapshot()` as compact lines (handle, agent, status, elapsed).
7. `_run_one` schema pinning: add both schemas next to `build_spawn_schema(...)` (`:543` area) gated on `agent_kind == AGENT_KIND_PRIMARY` **and** `max_subagents > 0`, mirroring `install_skill`'s primary-only gate.
8. `agent_runtime.py`: add `wait_agents`/`check_agents` to `LoopDeps` and dispatch branches beside `SPAWN_TOOL_NAME`. Both must be dispatched **in-loop** (like spawn), not through `invoke_tool`'s per-call timeout wrapper.
9. End of `run_turn`: after `_run_one` returns, if any child is still live, wait (bounded by remaining wall-clock), then cooperative-cancel, then abandon after a **5s join timeout**, marking abandoned handles `cancelled` (Task 2's DB guard makes a late child write a no-op).

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Agents/test_fleet_runtime.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Prove the no-fleet path is untouched**

Run: `pytest Tests/Agents/ Tests/Chat/test_console_agent_bridge.py Tests/Agents/test_skill_tool_spawn.py -q`
Expected: ALL PASS — the pre-existing spawn/skill suites are the byte-identical guard and must be unmodified. If you had to change ANY existing test, stop and report it as a deviation with the reason.

- [ ] **Step 6: Commit**

```bash
git add -u && git commit -m "feat: concurrent sub-agents with wait_agents/check_agents" && git push
```

---

### Task 7: Cancellation revokes pending approval cards

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (round registry; add a revoke entry point)
- Modify: `tldw_chatbook/Agents/agent_service.py` (call revoke when a child is cancelled/abandoned)
- Test: `Tests/UI/test_console_mcp_approval.py` or `Tests/Chat/test_console_chat_controller.py` (append — put it where the round-accounting tests already live)

**Interfaces:**
- Consumes: Task 6's cancellation path; Task 3's run_id.
- Produces: `ConsoleChatController.revoke_approval_rounds_for_run(run_id) -> int` — resolves every outstanding round belonging to that run as `"deny"`, sets its event, and removes the card.

**Why (spec §6, safety item):** the approval wait happens inside `_call_with_timeout`'s per-call daemon thread. If a child is cancelled or abandoned while a card is on screen, the user can still approve it and the tool executes for real (file written, message sent) while the run reads `cancelled`. The documented invariant `approval_timeout < max_tool_call_seconds` exists for exactly this class of hazard.

- [ ] **Step 1: Write the failing test**

Model it on the existing approval-round tests. Shape:
```python
def test_revoking_a_run_denies_its_outstanding_rounds(controller_fixture):
    # Arm a round for run-A and one for run-B, revoke run-A, assert:
    #  - run-A's decisions are all "deny" and its event is set
    #  - run-A's round is gone from _pending_approval_rounds / _pending_approvals
    #  - run-B's round is untouched and still pending
```
Write it fully against the real controller fixtures in that file — read two neighboring round tests first and match their construction exactly.

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/UI/test_console_mcp_approval.py -v -k revok`
Expected: FAIL (no such method).

- [ ] **Step 3: Implement**

- Record the owning `run_id` alongside each round when it is armed (`:2763-2768` stores a dict already — add the key; the run id arrives via Task 5's threading of run_id into the review hook).
- `revoke_approval_rounds_for_run(run_id)`: under `_approval_state_lock`, find matching rounds, fill every undecided name with `"deny"`, set the event (so the waiting thread returns immediately), discard from `_pending_approvals`, and remove the card from the UI the same way a resolved card is removed (find that path; do not invent a second removal mechanism).
- Wire it: when the fleet cancels or abandons a child, call it for that child's run id. If `AgentService` has no controller reference, add an optional `revoke_approvals: Callable[[str], None] | None = None` constructor seam wired by the bridge — same pattern as the other injected seams.

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/UI/test_console_mcp_approval.py Tests/Chat/test_console_chat_controller.py -q`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add -u && git commit -m "fix: revoke pending approval cards when a child is cancelled" && git push
```

---

### Task 6.5: Turn the fleet ON (default flip + convert ordered spawn scripts)

**Added 2026-08-09.** Task 6 shipped the runtime behind `max_live_subagents=1`, i.e. dark:
at the default no coordinator is built, spawn stays inline, and neither fleet tool is
offered. Without this task the PR merges with its central feature unreachable and Task 8's
live verification ("spawn two agents in one reply") permanently unperformable. Do not fold
this into Task 8 — it is a suite conversion, not a config tweak.

**Files:**
- Modify: `tldw_chatbook/Agents/agent_service.py` (`DEFAULT_MAX_LIVE_SUBAGENTS` 1 → 3)
- Modify: the 11 test files below
- Test: `Tests/Agents/test_fleet_runtime.py` (re-pin the two default-dependent tests)

**Interfaces:** no API change — this flips a default and converts test harnesses.

- [ ] **Step 1: Measure the true baseline before changing anything**

Flip `DEFAULT_MAX_LIVE_SUBAGENTS` to 3, run the full battery, and record the exact failing
test list. The review measured **23 failures / 11 files**: `test_agent_service.py` (8),
`test_skill_tool_spawn.py` (3), `test_console_agent_bridge.py` (2),
`test_search_run_log_runtime_tool.py` (2), `test_fleet_runtime.py` (2, self-inflicted),
and 1 each in `test_install_skill_runtime_tool.py`, `test_run_log_cross_run_search.py`,
`test_run_log_sandbox_isolation.py`, `test_run_log_stats_slice_runtime_tools.py`,
`test_run_log_workspace_isolation.py`, `test_run_skill_script_runtime_tool.py`.
If your list differs, reconcile it before proceeding — a new failure is a real regression.

- [ ] **Step 2: Convert ordered reply queues to addressed ones**

`ScriptedChat` pops off one shared list (`self.replies.pop(0)`) and tests index
`chat.calls[1]` positionally; with concurrent children neither is deterministic. Use the
`FleetChat` pattern already in `Tests/Agents/test_fleet_runtime.py` (address replies by
which agent/turn asked for them, not by arrival order). Convert each failing test.
**Preserve every assertion's meaning.** Where a test asserted the *spawn* tool_result
carries the child's capped text, that assertion moves to the `wait_agents` result — same
guarantee, new carrier; say so in a comment so a future reader doesn't read it as a
weakening.

- [ ] **Step 3: Re-pin the two default-dependent fleet tests**

`test_without_a_coordinator_spawn_stays_inline`,
`test_fleet_tools_are_not_offered_without_a_coordinator`, and
`test_config_of_one_or_junk_keeps_the_inline_path` (its `"nonsense"`/`None` cases resolve
through `DEFAULT_MAX_LIVE_SUBAGENTS`, so they flip meaning when the default does) all rely
on default==1. Re-pin each on an explicit `max_live_subagents=1` (monkeypatched config or
injected `None`), so they keep guarding the inline branch after the default moves. Sweep
for any other test whose expectation is "no coordinator" without saying so explicitly —
grep `_coerce_max_live_subagents` and `DEFAULT_MAX_LIVE_SUBAGENTS` in Tests/.

- [ ] **Step 4: Retire the contradicted bridge test**

`Tests/Chat/test_console_agent_bridge.py::test_run_reply_wires_mcp_provider_stamp_scope_around_a_spawned_child`
pins the behavior Task 6 deliberately removed on the threaded path (holding
`review_state_scope` around a child). Task 5's per-run keying is the replacement
protection, and `LocalToolProvider.stamp_scope` *clears* the parent's slice on entry, so
holding it across siblings would wipe the parent's live verdicts. Replace it with a test
asserting the parent's verdicts SURVIVE a concurrent child (the protection that actually
holds now) rather than deleting the coverage.

- [ ] **Step 5: Gate**

```bash
pytest Tests/Agents/ Tests/Chat/ Tests/MCP/ -q
pytest --collect-only -q | tail -2
```
Expected: 0 failures. READ and report the counts. Any test you touched: state in the report
what its assertion guaranteed before and after.

- [ ] **Step 6: Commit**

```bash
git add -u && git commit -m "feat: enable the agent fleet by default (max_live_subagents=3)" && git push
```

---

### Task 8: Provider thread-safety audit + config + docs

**Files:**
- Modify: `tldw_chatbook/Agents/mcp_tool_provider.py` (execution lock, pending audit outcome)
- Modify: config defaults (`[agents] max_live_subagents`) wherever provider defaults live — find it, do not guess
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Test: whatever the audit requires; `Tests/Agents/test_fleet_runtime.py` (append a cap-from-config test)

- [ ] **Step 1: Audit each provider's `invoke` for shared mutable state**

For each of `MCPToolProvider`, `BuiltinToolProvider`, `LocalToolProvider` (and any other registered provider): read `invoke` and every attribute it touches; list what is shared mutable state across concurrent calls. Write the findings into your report as a table (provider → shared state → safe/unsafe → decision).

- [ ] **Step 2: Lock what is not proven safe**

Per spec §5: an unaudited or unsafe provider gets a per-provider `threading.Lock` around `invoke` (serializes that provider's calls across the fleet — a throttle, not a break). **MCP starts locked until proven otherwise.** Add the lock with a comment naming what it protects and what would let a future task remove it.

- [ ] **Step 3: Config knob**

`[agents] max_live_subagents` already exists (Task 6) and is already 3 (Task 6.5) — this
step is now only: make sure it is documented wherever the repo lists config defaults, and
that an out-of-range/garbage value degrades safely.
**The original step's test ("a config value of 1 makes a second concurrent spawn refuse")
is unwritable and must not be attempted**: 1 means no coordinator is built at all, so the
second spawn runs inline rather than refusing. The equivalent coverage — a cap refusal —
already exists in `Tests/Agents/test_fleet_runtime.py` via an injected `max_live=1`
coordinator.

- [ ] **Step 4: Docs**

Update `Docs/User_Guide/console/agent-runs-and-tools.md`: sub-agents can now run in parallel within a reply; the supervisor collects results with `wait_agents`; approval cards name the agent; cancelling cancels its pending cards; the new config knob. Update the "Verified against" stamp (`*Verified against dev @ <short-sha> — <YYYY-MM-DD>*`) using the sha of your final content commit.

- [ ] **Step 5: Full targeted battery**

```bash
pytest Tests/Agents/ Tests/DB/test_agent_runs_db.py Tests/Chat/test_console_agent_bridge.py \
  Tests/Chat/test_console_chat_controller.py Tests/UI/test_console_mcp_approval.py -q
pytest --collect-only -q | tail -2
```
Expected: 0 failures; collect-only clean. READ and report the counts.

- [ ] **Step 6: Live verification** (per `backlog/docs/lessons-live-verification.md`)

tmux, scratch `TLDW_CONFIG_PATH` (never the live config), real provider key from a repo-root `*-api-key.txt` (note: `openrouter-api-key.txt` returns 401 on chat completions — use another). Verify and capture panes: (1) a prompt that makes the supervisor spawn two agents in one reply, (2) both appear in the rail's sub-agent section, (3) the reply incorporates both results, (4) `sqlite3` shows two child run rows, both terminal, (5) Stop mid-fleet leaves no run row in `running`. If a check cannot be completed after 2-3 genuine attempts, STOP and report exactly what you saw — never fabricate evidence.

- [ ] **Step 7: Commit**

```bash
git add -u && git commit -m "feat: provider locks, fleet config knob, docs" && git push
```

---

## Self-review notes (already applied)

- Spec §5 phase-2 rows all covered: coordinator+threads (T6), wait/check (T6), both gates per-run (T5), registry lock (T4), on_step run_id (T3), set_status guard (T2), card revocation (T7), provider audit/locks (T8).
- **Approval-round keying is already correct on dev** — rounds use `uuid4()`, so the spec's "audit that round keys are globally unique" item is resolved with no code change; T7 only adds run ownership so revocation can target them.
- `clamp_child_budget` deliberately untouched (spec §5: the swap belongs to PR 3a).
- Deferred-to-PR-2a items from task-13154 that are IN scope here: convert the spawn-closure `assert` to a raise (do it in T6 while rewriting that closure); add a load-once-per-turn call-count guard (add it in T6's test file).
- Known live-surface weakness for PR 2b (not this PR): live `SubAgentSummary` rows are appended once at `STEP_SPAWN` and never updated, so they always read "running" until the historical path re-derives them.
