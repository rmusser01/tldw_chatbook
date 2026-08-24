# Task 3 review package — Research source readiness and selection

Review base:

`60a7bf27c15b285b4a80a1f9f6f1d0737835ac80`

Expected commit subject:

`fix: reconcile Research source owner projections`

Exact review command after commit:

```bash
git diff --stat 60a7bf27c15b285b4a80a1f9f6f1d0737835ac80..HEAD
git diff 60a7bf27c15b285b4a80a1f9f6f1d0737835ac80..HEAD -- \
  ':!backlog/tasks/task-21508 - Add-Research-Sources-ingest-association-and-Quick-Notes.md'
```

The excluded TASK-21508 file is controller-owned dirty state and is not part of
this implementation or commit.

## Review order

1. `contracts.py`, `local_adapter.py`, `server_adapter.py`, and `controller.py`:
   exact owner selection versus the independently bounded visible source page.
2. `notes_workspace_schemas.py` and source-client/service tests: the 10,100-ID
   owner selection cap and preservation of top-level/row workspace identities.
3. `server_adapter.py` and `source_readiness.py`: pre-normalization identity
   validation and pending-versus-terminal readiness receipt behavior.
4. `ResearchSourcePreview`, Server preview projection, client fixture, and
   controller cache tests: honest nullable canonical identity for missing Media.
5. Focused tests and `task-3-report.md`: RED/GREEN and inverse-mutation evidence.

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
  desired identities, and caches only source rows the owner returned.
- Server readiness rejects both top-level and every row-level workspace mismatch
  before normalization. A typed identity mismatch remains pending/retryable;
  ordinary refresh failures remain terminal.
- Missing-media preview keeps the Server association ID and carries
  `catalog_item_id=None`; no blank or fabricated catalog identity is allowed.
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
