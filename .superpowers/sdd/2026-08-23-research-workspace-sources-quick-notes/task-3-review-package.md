# Task 3 review package — Research source readiness and selection

Review base:

`67114c7ed8a20a085afe33ce814ad84a1de2147b`

Expected commit subject:

`fix: close Research source ownership races`

Exact review command after commit:

```bash
git diff --stat 67114c7ed8a20a085afe33ce814ad84a1de2147b..HEAD
git diff 67114c7ed8a20a085afe33ce814ad84a1de2147b..HEAD -- \
  ':!backlog/tasks/task-21508 - Add-Research-Sources-ingest-association-and-Quick-Notes.md'
```

The excluded TASK-21508 file is controller-owned dirty state and is not part of
this implementation or commit.

## Review order

1. `contracts.py`, both adapters, and `controller.py`: source-specific bounded
   pages carrying the independently exact owner selection and the post-write
   generation fence.
2. `server_adapter.py` and readiness coordinator tests: malformed projections
   versus exact identity mismatches and their terminal-versus-pending receipts.
3. `notes_workspace_schemas.py`, preview/readiness adapters, and client/service
   tests: actual missing-media zero transport with no invented canonical ID.
4. `server_adapter.py` reorder tests: owner GET before PUT, exact set checks,
   the 100-ID write boundary, and PUT-then-GET reconciliation.
5. `task-3-report.md`: Round 3 RED/GREEN, inverse, split-gate, and static evidence.

## High-risk invariants to challenge

- Local desired IDs are canonical Media IDs; Server desired IDs are
  association IDs. Cross-space substitutions must fail.
- Select none must survive Local restart as explicit empty while Console's
  ordinary empty save remains unscoped.
- A 101+ source owner mutation must never be repackaged as a 100-row page.
  Exact desired IDs remain ordered and unique up to 10,100; the optional source
  row reconciliation is independently capped at 100.
- Local selection reads back canonical `RagScope` state without listing page 1.
  Server selection derives exact IDs from the validated post-PUT owner refetch.
- Controller selection preserves the current visible page, validates exact
  desired identities, caches only source rows the owner returned, and fences a
  source refresh that began while the owner write was in flight.
- Every source page carries the same exact ordered owner desired IDs regardless
  of page offset; a bounded page never becomes the global selection authority.
- Server readiness rejects both top-level and every row-level workspace mismatch
  before normalization. A typed identity mismatch remains pending/retryable;
  ordinary refresh failures remain terminal.
- Missing-media preview keeps the Server association ID and carries
  `catalog_item_id=None`; no blank or fabricated catalog identity is allowed.
- Transport `media_id=0` is accepted only for actual missing-media status or
  preview shapes, then normalized to nullable canonical identity. Available
  rows and domain ID `"0"` remain invalid.
- Server reorder performs no PUT unless one exact, unique owner order is
  expressible within 100 IDs. Owners above 100 and stale/missing/duplicate
  request sets fail typed `reorder_precondition_unavailable`.
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

- Round 3 focused owner/consumer gate: `219 passed`.
- Independent Research neighbors: `265 passed`.
- Exact Task 3 ingestion/DB neighbors: `275 passed, 1 skipped` (existing
  Windows-only boundary).
- Seven Round 3 inverse mutations across page/race, readiness, preview zero,
  and reorder families each made the intended guard red and were restored.
- Scoped Ruff, changed-production compileall, `git diff --check`, and the
  Impeccable detector pass. The same 10 scoped files are legacy Ruff-format
  candidates at the review base and current tree; none were mechanically
  reformatted.
- Full pytest intentionally not run.

- Fix-round focused gate: `199 passed`.
- Expanded restored-tree owner/consumer gates: `440 passed`, plus Library
  ingestion `145 passed, 1 skipped` (existing Windows-only boundary).
- Only accepted `RequestsDependencyWarning`.
- Scoped Ruff, readiness-copy format check, changed-production compileall,
  `git diff --check`, and Impeccable detector pass. The whole changed-inventory
  format probe still identifies 12 legacy whole-file reformat candidates; none
  were mechanically reformatted into this fix.
- All three Round 2 defect families, plus the identity-receipt branch, turned
  their intended guards red under inverse mutation and were restored. The prior
  nine guard families are included in the representative regression gates.
- A diagnostic all-in-one run crossed the repository FD sentinel at +208. The
  isolated cause is the pre-existing Library ingest test process (+190 alone),
  not changed production or focused tests (+3); split default-threshold gates
  above are pristine apart from the accepted dependency warning.
- Full pytest intentionally not run.
