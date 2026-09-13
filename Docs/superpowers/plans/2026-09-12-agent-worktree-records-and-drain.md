# Agent worktree records and physical completion

> **For agentic workers:** Use subagent-driven-development for this task. Root owns Git and Backlog; implementation and review use fresh workers.

**Goal:** Provide durable worktree ownership and positive writer completion for confirmed recovery.

**Architecture:** Add a structural worktree repository within AgentRunsDB and expose one-shot completion callbacks from the existing ExecutionOwner. This is the storage/lifecycle prerequisite; creation and recovery call-site integration follows in a separate slice.

**Tech Stack:** Python 3.12+, SQLite, existing RuntimeCapacity and targeted pytest.

**Spec:** Docs/superpowers/specs/2026-09-12-agent-worktree-restoration-design.md, Confirmation and durable records.

ADR required: yes, existing decisions
ADR path: backlog/decisions/155-agent-worktree-recovery.md; backlog/decisions/158-agent-runs-migration-order-after-worktree-qualification.md
Reason: durable original-base/ownership and reuse of existing physical lifetime, with recovery migration19→20.

## Global Constraints

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/agent-orchestration-pr`; root owns Git, Backlog and governance. No overlapping source workers. This slice must not modify AgentService, controller, bridge, local provider or Git operations.
- Existing interpreter `.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python`; product imports only under Tests/conftest. Use this plan's stdlib `run_pytest.py` with unique labels/basetemp and captured exact exits. Targeted tests only, no dependencies, network/providers/live config or foreign cleanup.
- The existing ExecutionOwner remains the sole physical ownership ledger. No second registry, PID-based liveness guess, time heuristic, daemon or task replay. DB terminal status and absence from a snapshot alone cannot prove completion.
- Worktree locators and identities stay in a separate local-only table, outside ordinary run projections/export and log bodies. Store no conversation text, model arguments, task bodies or transcript duplicates. Unknown/legacy work is never adopted.
- Transactional claim requires exact conversation ownership, unresolved state, terminal run and positively drained writer. Held/uncertain and in-flight rows remain non-actionable across reopen. Do not reset or steal a live operation when another DB instance opens.
- Callback exceptions must not prevent other callbacks or resource release. No callback, database call, join or logging under the capacity lock. Cleanup uncertainty is sticky and must never be overwritten by a later clean release.

### Task 1: Durable repository and reusable drain callback

**Files:**
- Create `tldw_chatbook/DB/agent_worktrees.py`: narrow structural repository.
- Modify `tldw_chatbook/DB/AgentRuns_DB.py`: schema version20 migration and repository access if needed.
- Create `tldw_chatbook/DB/migrations/agent_runs_v19_to_v20_worktree_recovery.sql`: reference migration matching runtime schema.
- Modify `tldw_chatbook/Agents/execution_capacity.py`: callback and uncertainty semantics only.
- Create `Tests/DB/test_agent_worktree_recovery.py`; extend `Tests/Agents/test_execution_capacity.py`; adjust directly affected schema assertions in existing DB tests without weakening historical migration checks.

**Repository interfaces:** `AgentWorktreeRepository(db)` borrows AgentRunsDB and uses its existing `connection()`/`transaction()` context managers. Do not open an independent database or close the caller's handles.

```python
record_created(*, run_id: str, workspace_id: str, binding_id: str,
               locator_fingerprint: str, repo_root: str, repo_identity: tuple,
               git_common_dir: str, git_common_identity: tuple,
               child_path: str, child_identity: tuple, branch: str,
               base_sha: str, execution_id: str) -> None
get_for_conversation(run_id: str, conversation_id: str) -> dict | None
list_for_conversation(conversation_id: str, *, workspace_id: str,
                      binding_id: str, limit: int = 50,
                      after_run_id: str | None = None) -> list[dict]
mark_writer_finished(run_id: str, execution_id: str, *, cleanup_proven: bool) -> bool
claim(run_id: str, conversation_id: str, *, operation_id: str, action: str) -> bool
finish_operation(run_id: str, operation_id: str, *, state: str) -> bool
```

Record fields above are immutable after insertion. Add created_at, updated_at, writer_state (`held`, `drained`, `uncertain`), mutation state (`unresolved`, `applying`, `merging`, `discarding`, `applied`, `merged`, `discarded_cleanup_pending`, `uncertain`) and nullable operation_id. Foreign key run_id references agent_runs; derive conversation/parent/status in metadata-only joins. Enforce valid state/action/ID/base/branch/identity shapes at the repository boundary; exact SQL parameters only. Definition cap data and every existing table survive migration. Unknown or duplicate records cannot overwrite ownership.

Listing is keyset-paged by run_id (ascending; limit1..100) and filters exact conversation/workspace/binding. Include stored structural status/writer fields and joined run status so the controller can explain non-actionable rows; never hydrate steps. Fresh authority/fingerprint/root checks remain the later recovery service's responsibility. Rows persist in their current state across DB reopen; in-flight rows are unavailable for a new claim. This prevents replay without pretending constructor-time reconciliation can distinguish a crash from another live connection.

`claim` maps apply/merge/discard to its in-flight state in one transaction, requiring exact conversation and terminal run plus unresolved/drained. `finish_operation` requires matching operation ID and expected corresponding in-flight state; accepted results are the matching success state or uncertain. Wrong, stale or duplicate operation completions return False. A clean writer callback may only change held→drained; an unproven callback changes held/drained→uncertain and a later clean callback cannot overwrite it. Wrong execution IDs affect no row.

**Physical-owner interfaces:**

```python
ExecutionOwner.on_drained(callback: Callable[[bool], None]) -> None
ExecutionOwner.mark_cleanup_unproven() -> None
```

Under the existing capacity lock, atomically detect root-finished plus zero operations, remove the owner and detach the callback list exactly once. Invoke detached callbacks after releasing the lock, passing `cleanup_proven = not sticky_uncertainty`. Late registration invokes immediately outside the lock with the same recorded outcome. Repeated finish calls do not re-run a registered callback. A failed callback does not suppress later ones. Preserve current capacity snapshots, reservations, cancellation and child-limit behavior.

- [ ] Write and run RED DB tests using real temporary SQLite: v19→v20 retains a saved definition wall cap; unknown run/duplicate owner rejected; reopening preserves base, identities and held/drained/in-flight states; two conversations cannot read/claim each other's rows; listing is bounded and step hydration forbidden; only one of two competing claims succeeds; stale completion cannot finalize another operation.
- [ ] Write and run RED execution-owner tests: root-first and operation-first release, callback can call `capacity.snapshot()` (therefore outside lock), one failing callback does not suppress its sibling, late registration, repeated release, and cleanup-unproven stays false through actual final completion. Gate an existing real `_call_with_timeout` worker past logical timeout and show the callback waits for its actual thread finally.
  ```python
  observed = []
  owner.on_drained(lambda proven: observed.append((proven, capacity.snapshot())))
  owner.finish_root()
  assert observed == []
  operation.finish()
  assert len(observed) == 1
  assert observed[0][0] is True
  assert observed[0][1].executions == ()
  ```
- [ ] Implement the small repository/migration and outside-lock callback semantics. Keep this slice independent of runtime call sites; its behavior is tested through real SQLite and owned threads. Do not invent Git operations or enable recovery tools.
- [ ] Verify a callback can persist drained in the real repository only after an owned delayed tool finishes; inject unproven cleanup and verify reopening never makes it claimable. Validate foreign operation/execution IDs with explicit unchanged-row assertions.
- [ ] Run the focused new tests and existing execution-capacity module, plus directly affected migration/definition-cap DB neighbors. Use this plan's runner and preserve exact output. Compare changed diagnostic identities and edited-hunk formatting against this task's actual source base; `git diff --check` must pass. Qualify inherited warnings rather than broadening scope.
- [ ] Write report with RED/GREEN evidence, API details and limitations; leave source unstaged for root commit and independent review. This prerequisite does not complete TASK-31210 or TASK-31211 until real creation/recovery call sites use it.
