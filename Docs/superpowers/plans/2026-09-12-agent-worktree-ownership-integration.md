# Agent worktree ownership integration

> **For agentic workers:** Use subagent-driven-development. Root owns Git and Backlog; workers leave source unstaged.

**Goal:** Persist each new worktree before child execution and tie its recovery eligibility to the actual child's physical completion.

**Architecture:** Connect the existing AgentWorktreeRepository and ExecutionOwner directly at fleet admission. The local provider reports unproven worker cleanup to the captured owner before returning an error; no new lifetime registry is introduced.

**Tech Stack:** Python 3.12+, SQLite, ordinary local Git, existing provider/owner interfaces.

**Spec:** Docs/superpowers/specs/2026-09-12-agent-worktree-restoration-design.md

ADR required: no new decision
ADR path: backlog/decisions/155-agent-worktree-recovery.md
Reason: direct integration of the approved durable ownership and ordinary Git decision.

## Global Constraints

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/agent-orchestration-pr`. No Git/Backlog/docs edits by implementers. No subagents from workers.
- Existing interpreter `.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python`; product imports only under Tests/conftest. Copy the prior plan's stdlib runner into this plan's workspace, use unique labels/basetemp, targeted tests only. No provider/network/live configuration, dependency changes or foreign cleanup.
- Exact selected writable named authority controls creation. Preserve every successfully created checkout and branch on later failure. Automatic retirement removes routing only.
- Terminal DB state alone never authorizes recovery. Only the actual root plus all owned operations completing can mark drained. Cleanup uncertainty is sticky, including when persistence fails.
- Callbacks must close only DB connections created on their own callback thread; never close another thread's borrowed handles or a connection still used by an enclosing DB operation. Never silently claim persistence succeeded.

### Task 1: Connect creation records and provider cleanup proof

**Files:**
- Modify `tldw_chatbook/Agents/agent_service.py`: exact child owner passed at admission, durable creation before routing/thread start, callback registration.
- Modify `tldw_chatbook/Agents/local_tool_provider.py`: optional `RunAdmittedWorkspaceRoot.on_cleanup_unproven: Callable[[], None] | None = None`, invoked for a `WorkspaceToolExecutionError` with `cleanup_unproven` before translating the error.
- Modify `tldw_chatbook/Agents/agent_worktree.py` only if a small Git common-directory identity capture helper avoids duplicating fixed structural queries.
- Tests: `Tests/Agents/test_fleet_runtime.py`, `Tests/Agents/test_local_tool_provider.py` or a new narrowly scoped `Tests/Agents/test_agent_worktree_ownership.py` when shared fixtures permit it.

**Interfaces:**

```python
def _admit_agent_worktree(self, handle, child_run_id: str,
                         execution_owner: ExecutionOwner) -> str | None: ...

# On the exact precreated child, before Git creation:
child_owner.bind_run(child_run_id)
# On successful creation, before provider admission or child execution:
repository.record_created(
    run_id=child_run_id, workspace_id=source.workspace_id,
    binding_id=source.binding_id, locator_fingerprint=source.locator_fingerprint,
    repo_root=str(source.root), repo_identity=source.root_identity,
    git_common_dir=str(common_dir), git_common_identity=common_identity,
    child_path=str(created.worktree_path), child_identity=child_identity,
    branch=created.branch, base_sha=created.base_sha,
    execution_id=child_owner.execution_id,
)
child_owner.on_drained(persist_actual_drain)
```

Repository methods and identity tuple shape already exist in `DB/agent_worktrees.py`. Resolve and compare source/child Git common directories using fixed local `rev-parse --git-common-dir` queries, canonicalizing relative output against each checkout. Capture full `(path, dev, ino, mode)` chains with the existing helper. Guard and identity checks bracket queries. If a post-create failure prevents a trustworthy record, retain the in-memory checkout and refuse routing/execution; it remains manual-only rather than inventing durable ownership. If a complete record can be safely captured before a later admission failure, it persists and the actual failed-start owner completion drains it.

The admitted child's optional cleanup callback closes over its real ExecutionOwner (not a ContextVar lookup in the tool thread). It first calls `mark_cleanup_unproven`, then persists `cleanup_proven=False`; failure logs only a bounded error type and leaves held/uncertain. The provider contains observer errors without suppressing the original tool error. The drain callback persists the recorded outcome and contains DB errors; on failed persistence the row remains held/uncertain. Register it only after a successful record insertion. Do not infer drain from fleet.handle terminal events.

- [ ] Write RED real-service/temporary-Git/SQLite regressions before production edits: row exists while child is gated before its first write; exact base, binding and owner IDs survive reopening; after real owner completion it becomes drained. A gated timed-out child tool keeps the record held after terminal status, then actual release drains it.
- [ ] Write RED provider regression for `cleanup_unproven` delivery before the owned operation finishes, then verify its durable record remains uncertain after the real owner drains and DB reopens. Observer exceptions preserve the original refusal and no path/log content leaks.
- [ ] Implement the narrow call-site integration. Failed record insertion must stop child execution/routing and preserve checkout/branch. Failed provider admission and thread start with a complete record preserve ownership and drain only through the owner callback.
- [ ] Run focused new tests and the directly affected creation/failed-start nodes. Inspect exact outputs; compare new Ruff identities and edited-hunk formatting against the recorded base; `git diff --check` passes. No broad repeat of the earlier 260-test selection.
- [ ] Report exact RED/GREEN evidence, persisted structural fields and callback ordering; leave source unstaged. Confirmation and recovery UI remain separate deliverables.
