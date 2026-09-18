# TASK-32780 — Tool Profile review lifetime

Delayed import inspection and export preparation now belong to the Settings
visit that started them. Category departure (including leave-and-return),
screen removal, unrelated modal suspension and a newer workflow invalidate
old preparation. Only the current workflow's own modal preserves its intent.
The worker rechecks that intent before opening another dialog or admitting a
write. Existing archive, profile and publication authority remains unchanged.

Once a mutation is admitted, its outcome still matters. Independent review
reproduced an initial regression that suppressed a late publication error
after a category roundtrip. An explicit admission flag now preserves those
failure/uncertainty receipts, as well as the existing success receipts; only
obsolete preparation errors are silent.

## Targeted evidence

74 distinct cases pass. No full suite or provider requests were run.

| Scope | Cases | Evidence |
| --- | --- | --- |
| Delayed import/export across departure, return, unrelated modal, removal and cross-type replacement | 10 | [Lifecycle log](lifetime.txt) |
| Same-type import/export replacement | 2 | [Follow-up log](follow-up.txt) |
| Owned import cancel, accept and revise | 3 | [Lifecycle log](lifetime.txt) |
| Delayed valid/invalid destination capture after departure | 2 | [Lifecycle log](lifetime.txt) |
| Obsolete preparation errors versus admitted failure/uncertainty receipts | 6 | [Lifecycle log](lifetime.txt), [follow-up](follow-up.txt) |
| Existing Tool Profiles workflows and action/loading ownership | 42 | [Regression log](existing.txt) |
| Existing export recovery and terminal publication outcomes | 9 | [Export regression log](export.txt) |

Eight of the first ten lifecycle cases failed before repair with a stale
review over the new view; the two removal controls already passed.
[Pre-fix summary](red-summary.txt). Mounted tests use production CSS at 80×24,
real category controls and real review/options dialogs. File choices and
held service operations are controlled inputs; these cases qualify UI
lifetime and do not claim real import activation. The export regression
matrix separately retains real capture/publication for its six recovery
cases.

The first 21-case lifetime run passed 20 cases. Its import-uncertainty fixture
used `outcome_uncertain`, the removal category, which `ToolPackError` correctly
normalized to `payload_invalid`. The fixture now uses `activation_uncertain`.
The follow-up passed both uncertainty cases and the two newly added same-type
replacement cases. Those overlapping passes qualify 23 distinct lifetime
cases. The export log's initial ten lifetime passes are not counted again.

Scoped Ruff introduces no diagnostics (Settings baseline remains 114). The
new test file and changed workflow methods pass formatting; backlog and diff
guards pass. Independent review found no remaining introduced blocker after
the admitted-outcome correction. [Verification receipt](verification.json).

No visual values changed and no new native captures were taken. Focus through
refresh and complete native import/removal journeys remain in the
[Tool Profiles ledger](../../reports/2026-09-18-tool-profiles-review.md).
The earlier export gallery remains historical evidence of its captured
source; this change adds mounted lifetime qualification. ADR-107 applies.
