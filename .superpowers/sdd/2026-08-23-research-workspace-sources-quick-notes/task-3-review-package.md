# Task 3 review package — Research source readiness and selection

Review base:

`7ad8db35bc88aa9c94db5af10a7df78b569b0e86`

Expected commit subject:

`feat: add Research source readiness and selection`

Exact review command after commit:

```bash
git diff --stat 7ad8db35bc88aa9c94db5af10a7df78b569b0e86..HEAD
git diff 7ad8db35bc88aa9c94db5af10a7df78b569b0e86..HEAD -- \
  ':!backlog/tasks/task-21508 - Add-Research-Sources-ingest-association-and-Quick-Notes.md'
```

The excluded TASK-21508 file is controller-owned dirty state and is not part of
this implementation or commit.

## Review order

1. `contracts.py`, `source_readiness.py`, and `Chat/rag_scope.py`: identity,
   readiness, retrieval, and explicit-empty invariants.
2. `local_adapter.py` and `server_adapter.py`: authority isolation, real Media
   scope calls, durable attach, typed capabilities, and association-only remove.
3. `notes_workspace_schemas.py`, `client.py`, and
   `server_notes_workspace_service.py`: strict bounds, quoted paths, actual
   `{ok}` then GET traces, and reconciliation failures.
4. `source_operation_store.py`, `source_association.py`, and `app.py`: bounded
   readiness receipt recovery after association and restart.
5. `controller.py`: qualified per-surface generations and late-result fencing.
6. Focused tests and `task-3-report.md`: inverse guards and WebUI parity gaps.

## High-risk invariants to challenge

- Local desired IDs are canonical Media IDs; Server desired IDs are
  association IDs. Cross-space substitutions must fail.
- Select none must survive Local restart as explicit empty while Console's
  ordinary empty save remains unscoped.
- Server selection/reorder never invent a revision: PUT `{ok}` must be followed
  by GET, and a failed/mismatched refetch must not return stale rows.
- Missing vector readiness is FTS-only and cannot enter Hybrid retrieval.
- Readiness retry must make zero catalog-ingest or association calls.
- Late ABA, preview, readiness, and pre-write source-list results cannot enter
  current caches or visible state.
- Unknown/missing capability projection must be discoverable typed unavailable,
  never inferred from method presence.

## Verification snapshot

- Named Task 3/controller: `100 passed`.
- Touched neighbors: `173 passed`, `138 deselected`.
- Only accepted `RequestsDependencyWarning`.
- Scoped Ruff, new-file format check, changed-production compileall,
  `git diff --check`, and Impeccable detector pass.
- Five inverse mutations turned their intended guards red and were reverted.
- Full pytest intentionally not run.
