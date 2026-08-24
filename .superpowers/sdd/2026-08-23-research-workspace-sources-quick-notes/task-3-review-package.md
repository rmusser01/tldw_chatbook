# Task 3 review package — Research source readiness and selection

Review base:

`ee7185115ba2d37e32d30f4e933bab934d51ae82`

Expected commit subject:

`fix: harden Research source ownership boundaries`

Exact review command after commit:

```bash
git diff --stat ee7185115ba2d37e32d30f4e933bab934d51ae82..HEAD
git diff ee7185115ba2d37e32d30f4e933bab934d51ae82..HEAD -- \
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
- Existing malformed Workspace scope rows must resolve explicit empty while
  ordinary Console zero-save and conversation metadata retain legacy behavior.
- Missing Server media must retain association identity with no invented
  canonical ID; rows/status beyond 100 remain valid within the finite owner cap.
- Server duplicate association cannot become terminal until its desired
  selection is version-reconciled; a failed update must retry after restart.
- A public catalog page crossing a 100-row backing boundary cannot skip or
  duplicate rows, and Local updated-time sorting must use owner vocabulary.

## Verification snapshot

- Fix-round focused gate: `130 passed`.
- Expanded restored-tree owner/consumer gate: `489 passed`, `1 skipped`
  (existing Windows-only boundary).
- Only accepted `RequestsDependencyWarning`.
- Scoped Ruff, readiness-copy format check, changed-production compileall,
  `git diff --check`, and Impeccable detector pass. The whole changed-inventory
  format probe still identifies 12 legacy whole-file reformat candidates; none
  were mechanically reformatted into this fix.
- Nine reviewed defect families turned their intended guards red and were
  restored before the final gates.
- Full pytest intentionally not run.
