# Trace v2 Event Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Project every observable Console and agent event into one deterministic causal Trace without duplicating all history into a new database.

**Architecture:** Extend the pure trajectory projection with a stable event envelope and adapters for its existing local owners. Persist `AgentStep` rows incrementally through the existing append-only `AgentRunsDB.agent_run_steps` seam; capture non-agent Console facts in their current trajectory/context/retrieval owners. Legacy v1 inputs remain valid.

**Tech Stack:** Python 3.11+, dataclasses, SQLite existing repositories, pytest, Textual-independent projection code.

**Spec:** `Docs/superpowers/specs/2026-08-22-task-19907-trace-v2-exhaustive-collaboration-design.md`
**ADR required:** yes
**ADR path:** `backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md`
**Reason:** changes the durable event and cross-module projection contract while preserving multiple storage owners.

---

### Task 1: TASK-19907 — Define the causal event envelope and adapters

**Files:**
- Modify: `tldw_chatbook/Chat/trajectory.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (snapshot-builder inputs only)
- Test: `Tests/Chat/test_trace_event_projection.py`
- Test: `Tests/UI/test_trajectory_live.py`

- [ ] **Step 1: Write failing contract tests**

Cover stable source-derived IDs, conversation ID, immutable owner `source_seq` distinct
from display position, human labels, actor/run/turn fields, status,
parent/source/replacement links, per-field state, sensitivity, deterministic causal
ordering, concurrent tie-breaking, unknown sidecar kinds, and legacy empty adapters.

```python
def test_causal_parent_precedes_concurrent_child() -> None:
    snapshot = derive_trajectory(
        messages=[], usage_by_id={}, traj_rows=[], variant_sets=[],
        compaction_records=[], agent_runs=[parent_run, child_run],
        agent_steps=[child_step, parent_spawn_step], retrieval_runs=[],
    )
    ids = [record.event_id for turn in snapshot.turns for record in turn.records]
    assert ids.index("agent-step:parent:4") < ids.index("agent-run:child")
```

- [ ] **Step 2: Run the new file and confirm it fails**

Run: `.venv/bin/pytest -q Tests/Chat/test_trace_event_projection.py`
Expected: failures for unsupported adapters/fields.

- [ ] **Step 3: Extend `TrajectoryRecord` minimally**

Add optional fields with legacy-safe defaults: `event_id`, `conversation_id`,
`source_seq`, `label`, `status`,
`actor_kind`, `actor_id`, `run_id`, `parent_event_id`, `source_event_id`,
`replacement_event_id`, `observed_at`, `field_states`, and `sensitivity`.
Keep `kind`, `seq`, and existing fields so screen/import callers remain source-compatible.

- [ ] **Step 4: Add pure adapter functions in `trajectory.py`**

Implement `_records_from_agent_runs`, `_records_from_agent_steps`,
`_records_from_retrieval_runs`, a generic `_record_from_sidecar_event` that preserves
unknown/new event kinds, and deterministic `_causal_order`. Inputs are plain
dicts/dataclasses; do not import DB or Textual modules. Source IDs use fixed prefixes.

- [ ] **Step 5: Thread optional sources through the real snapshot builder**

Load agent run metadata/steps and retrieval provenance in the existing off-thread
builder in `chat_screen.py`; the `TrajectoryScreen` still receives a completed snapshot
and never queries a DB.

- [ ] **Step 6: Verify compatibility and mutation strength**

Run:

```bash
.venv/bin/pytest -q Tests/Chat/test_trajectory_projection.py Tests/Chat/test_trace_event_projection.py Tests/UI/test_trajectory_screen.py Tests/UI/test_trajectory_live.py
```

Temporarily remove the causal-parent ordering branch and confirm its test fails.

- [ ] **Step 7: Commit**

`git commit -m "feat(trace): add exhaustive causal event projection"`

### Task 2: TASK-19907 — Persist agent steps when observed

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py`
- Modify: `tldw_chatbook/Agents/agent_runtime.py`
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Modify: `tldw_chatbook/DB/AgentRuns_DB.py` only if a one-row convenience method is needed
- Test: `Tests/Agents/test_agent_step_incremental_persistence.py`
- Test: `Tests/DB/test_agent_runs_db.py`

- [ ] **Step 1: Write failing crash-survival and no-duplicate tests**

Assert the first `on_step` call is readable from `agent_run_steps` before run completion,
`created_at` is UTC and not derived from the monotonic budget clock, a simulated first
write failure is recovered at terminal persistence, and successful completion does not
append the same step twice.

- [ ] **Step 2: Confirm failure with the focused tests.**

- [ ] **Step 3: Stamp steps at creation**

Add a dedicated injected UTC `wall_clock` to `LoopDeps`; do not reuse `clock`, which is
monotonic and only suitable for durations. In `run_agent_loop.add`, set `created_at`
from `wall_clock`. Keep the callback containment rule: a raising persistence callback
never aborts a run.

- [ ] **Step 4: Persist through the existing service `on_step` seam**

Add an explicit-index `append_step(run_id, step.index, payload)`/batch equivalent using
conflict-safe inserts. Compose it with the current UI callback. At terminal persistence,
insert the complete step list by explicit index so failed incremental writes are filled
and successful ones are no-ops; then write status/result. Do not use the allocating
`append_steps()` API for this path and do not add a second event store.

- [ ] **Step 5: Verify**

Run: `.venv/bin/pytest -q Tests/Agents/test_agent_step_incremental_persistence.py Tests/DB/test_agent_runs_db.py Tests/Agents/test_agent_service.py`

- [ ] **Step 6: Commit**

`git commit -m "feat(agents): persist trace steps incrementally"`

### Task 3: TASK-19908 — Capture Console, model, tool, approval, and context events

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py`
- Modify: `tldw_chatbook/Agents/agent_runtime.py`
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/trajectory.py`
- Modify: `tldw_chatbook/Chat/console_context_repository.py` only at existing outcome seams
- Reuse: `tldw_chatbook/Chat/citation_trace_repository.py`
- Test: `Tests/Chat/test_trace_capture_matrix.py`
- Test: `Tests/Agents/test_trace_approval_capture.py`

- [ ] **Step 1: Write a table-driven event-family matrix test**

The matrix must cover conversation user/system/assistant/feedback/edit/regenerate/
branch-selection; model start/first-token/completion/retry/error/cancel; tool
proposal/approval-request/approve/deny/revoke/execution-start/success/failure/timeout/
cancel; retrieval start/candidate-selection/outcome; context attach/injection; and
compaction start/outcome/failure. Assert generic sidecar projection, causal links, and
safe payload classification.

- [ ] **Step 2: Confirm uncovered families fail.**

- [ ] **Step 3: Add only the missing `STEP_*` constants and emit at owned seams**

Reuse existing tool/model steps where their semantics are sufficient. Add approval,
retrieval/context, retry, cancellation, and compaction kinds only when no existing
record distinguishes the transition. Store safe summaries; never persist hidden
reasoning or credentials.

- [ ] **Step 4: Add best-effort containment and diagnostics**

Every capture call catches/logs failure with conversation/run/event context and leaves
the user operation untouched. Project an incomplete marker when a durable owner is
still writable.

- [ ] **Step 5: Drive one real-seam integration run**

Use the real AgentService, real in-memory/temp databases, real approval callback, and a
fake provider only at the external provider seam. Assert the joined Trace order rather
than a hand-written fake contract.

- [ ] **Step 6: Verify**

Run: `.venv/bin/pytest -q Tests/Chat/test_trace_capture_matrix.py Tests/Agents/test_trace_approval_capture.py Tests/Chat/test_trajectory_capture.py Tests/Agents/test_agent_service.py`

- [ ] **Step 7: Commit**

`git commit -m "feat(trace): capture console model tool and context events"`

### Task 4: TASK-19910 — Capture child-agent lifecycle and lineage

**Files:**
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Modify: `tldw_chatbook/Agents/fleet_coordinator.py`
- Modify: `tldw_chatbook/DB/AgentRuns_DB.py` (schema v13→v14 `spawn_event_id`)
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Chat/trajectory.py`
- Test: `Tests/Agents/test_trace_agent_lineage.py`
- Test: `Tests/Chat/test_trace_event_projection.py`

- [ ] **Step 1: Write failing parallel-lineage tests**

Cover reserve/spawn/start, precise spawn-event correlation, attach durable run ID,
steering/handoff, finish, error,
cancel, supersede, resume, and primary completion with two parallel children.

- [ ] **Step 2: Confirm failure.**

- [ ] **Step 3: Project existing durable lineage first**

Add nullable `spawn_event_id` to AgentRunsDB v14 and `create_run`. Allocate the parent
spawn step/event ID before dispatch and store it on every inline/fleet child run.
Continue using `parent_run_id`, `resumed_from_run_id`, status, and append-only steps.
Emit new fleet events only for transitions not recoverable from those owners; handles
remain process-local implementation identities.

- [ ] **Step 4: Add parent/source event links and safe task summaries**

Child `parent_event_id` points to the spawning step; continuation `source_event_id`
points to the prior terminal run. Parallel children retain per-run sequence and visible
concurrency.

- [ ] **Step 5: Verify**

Run: `.venv/bin/pytest -q Tests/Agents/test_trace_agent_lineage.py Tests/Agents/test_fleet_runtime.py Tests/Agents/test_fleet_continuation.py Tests/Chat/test_trace_event_projection.py`

- [ ] **Step 6: Commit**

`git commit -m "feat(trace): expose durable agent lineage"`
