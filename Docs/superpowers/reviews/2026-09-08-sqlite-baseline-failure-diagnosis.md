# SQLite baseline-failure diagnosis — 2026-09-08

TASK-32160 continuation at `e1a2e859127eb83a100b22387e47a8ffbc95d424`.
The user approved tackling the 26 baseline failures after the reviewed Task9
inventory correction. The diagnosis below records the pre-repair evidence.
The approved repair is now committed as `ad5c02e5a8`; implementation and current
qualification status are recorded in `Docs/Canvas/V2_VERIFICATION.md`.

## Twenty-two core-owner failures: stale tests

The existing immutable BASE control reproduced all26assertions. Fresh scoped
representatives trace the22owner failures to three test assumptions:

- Two exact-kwargs expectations omit the existing
  `factory=_QuiescentSQLiteConnection`. That factory is required for actual
  same-file quiescence; the exact expectation predates its introduction.
- Thirteen unconditional `Path.resolve` traps intercept resolution of the fixed
  helper executable, not a database/backup path. Seven owner and six backup
  parameters use the overbroad trap.
- Seven relative-path cases spy on the old local `_prepare_artifact` route,
  while live preparation now calls `prepare_in_helper(PrepareRequest(...))`.
  That process boundary is required by ADR-125; restoring local descriptor work
  to satisfy the spy would reintroduce the lock hazard.

Fresh reproduction used collected repository isolation: the exact-kwargs case
failed as expected; a three-node selection reproduced both path traps and the
empty preparation observation (3failed,1warning,1.07s). The existing central
selected-target-only path guard passed (1passed,1warning,1.73s).

Proposed repair is test-only in
`Tests/DB/test_core_sqlite_owner_privacy.py`: retain exact kwargs equality,
target the path guard at selected database/source/destination paths, record and
delegate the actual helper request, and forward actual owner arguments without
self-correcting them in the recorder. Keep real SQLite operations, private modes,
copied contents, unsafe-alias refusals and cleanup. Negative controls must prove
the revised guards still detect wrong factories/owners/paths/helper policies and
helper/backup bypass, not merely accept current code.

## Four compaction failures: missing schema function

One existing real compaction test reproduced `admitted=True` and
`reason_code='vacuum_failed'` at the expected completion assertion. The isolated
synthetic probe confirmed the underlying cause is
`sqlite3.OperationalError: no such function: canvas_revision_payload_valid`.
The ordinary ChaChaNotes connection registers that deterministic three-argument
validator; the separate maintenance connection does not. Canvas table CHECK
constraints reference it.

The probe registered only the existing real validator on each maintenance open.
The same logical-GC result then completed compaction and
`PRAGMA quick_check(1)` returned `ok` (1passed,1warning,1.62s). No schema change,
new backup authority, trigger bypass, altered lease, checkpoint or exclusion
policy was needed. This establishes the shared causal omission; it is not yet
the four original tests passing against a production repair.

Proposed repair is scoped maintenance connection setup, reusing the existing
validator implementation and owning native close if setup fails. Do not install
semantic mutation or deletion grants merely to enable VACUUM. Required regression
coverage includes real populated Canvas revision preservation after shrink/reopen,
continued rejection of invalid payloads and unauthorized deletes, and setup-failure
closure, alongside the existing admission/cancellation/lease/integrity cases.

## Boundaries and next step

Existing ADR-029/125 govern private file/helper ownership, trace ADR-097 governs
same-file maintenance and exclusion, and ADR-121 requires fail-closed Canvas
payload and mutation constraints. The proposed repairs implement those contracts;
no new architecture or security policy is proposed. The user approved the precise
repair after diagnosis. Canonical Task10 records its test-first implementation
and independent review requirements. Committed focused regressions and startup
guards now pass; independent task review approves with no Critical or Important
finding. The minor test-teardown improvement and inherited verification noise
are disclosed in the verification document.

The two diagnosis reports and retained synthetic probe are in the existing
git-ignored plan workspace:
`.superpowers/sdd/2026-09-07-sqlite-lock-safe-private-validation-implementation/`
(`task-10-owner-diagnosis.md`, `task-10-compaction-diagnosis.md`, and
`task-10-compaction-probe.py`). They record exact commands and scope. No full
suite, Semaphore11rerun, host/account/dependency changes, external PR action or
Canvas V2 admission was performed. TASK-32160 stays In Progress; scoped AC10/11
are now checked with AC8/9, while the original final criteria remain open.
