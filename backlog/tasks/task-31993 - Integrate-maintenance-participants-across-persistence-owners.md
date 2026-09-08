---
id: TASK-31993
title: Integrate maintenance participants across persistence owners
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-07 23:53'
updated_date: '2026-09-08 10:44'
labels:
  - backup-recovery
dependencies:
  - task-31978
  - task-31988
  - task-31989
  - task-31990
  - task-31991
  - task-31992
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Every participating persistence owner drains safely and is covered by the shared admission protocol.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every participating persistence owner drains safely and is covered by the shared admission protocol.
- [ ] #2 Unsaved drafts and unfinished cross-store work are neither discarded nor falsely reported captured.
- [ ] #3 Real multi-process evidence proves coherent ownership boundaries and safe resumption without deadlocks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Establish the uncovered persistent owner behavioral RED regression and exact installed producer census.
2. Bind installed app and headless participants to the reviewed shared admission protocol, including raw writers and shared files; propose coordinator API changes before implementing them.
3. Close new mutation admission, drain actual transactions, handles and cross-store publication, stop watchers at safe boundaries, and preserve dirty editor save/discard decisions.
4. Preserve unresolved/pending work, release normal holds only after proven retirement, and resume in reverse dependency order after exclusive capture release.
5. Verify real two-process coherent DB/assets and config/registry boundaries, timeout, app-close, failures and deadlock ordering.
6. Run required focused participant, SQLite, census and service-composition guards plus scoped static checks and self-review.
7. Record exact evidence, source handoffs and ADR-126 in notes/report; retain In Progress and unchecked criteria pending independent review.
ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation of approved persistence, admission and recovery lifecycle contract.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-02-inventory-admission.md#task-10)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

## Implementation Notes

Foundation phase only; Task10 remains In Progress and all criteria remain unchecked.
Under ADR-126 and controller rulings50–51, added the native pause-hint observation
and moved process lease acquisition/retirement waits outside the coordinator RLock.
Retiring native holds remain explicit; startup is not retired and no producer is
promoted from participant_pending. Added the Participant protocol and conservative
missing-coverage guard. Owner/app/headless integration remains the next phase.

Behavioral RED: first missing-owner assertion failed with DID NOT RAISE; then native
gate and blocked-acquisition lock-order tests failed (2 failed/1 passed). GREEN:
16 focused cases and final79 combined participant/admission/bootstrap cases passed.
Required guards:379 passed/1 skipped (existing Windows-only functional posture),
using the approved Python3.12 interpreter and private pytest caches. Existing
requests and AST SyntaxWarnings remain; no full suite ran. Exact commands, evidence
limits, source handoffs and remaining work are in the task10 execution report.

Files: Backup_Recovery/admission.py, storage_admission.py, new participants.py,
Tests/Backup_Recovery/test_participants.py, owner inventory documentation and this
task. No new ADR: direct implementation of ADR-126, retaining ADR-004 restart-required
config mapping and ADR-036 ownership boundaries. No automatic startup-token release,
GC drain, owner allowlist, output publication or diagnostic inventory refresh.

### Phase2 implementation plan (before code)

1. Establish behavioral RED for Event/Sync file connection native retirement and private local pause gate/pending acquisition refusal.
2. Add reservation before path/authority lookup, opaque process-local pause and live installed repository operation provenance (PID, actual Thread/task identity, exact path/native scope); track actual SQLite resource retirement and same-holder probes.
3. Integrate Event/Sync transaction scopes with finally-native-close on their creating thread; retain in-memory connection semantics. Never cancel writes as a drain mechanism.
4. Verify real native SQLite/operation lifetimes, late/cancelled acquisition, descendant confinement, task/thread and stale capability refusal, close failures, and incomplete runtime-coverage refusal with startup retained.
5. Run affected focused guards/static checks; self-review and commit this bounded phase. Preserve foundation evidence, In Progress status and unchecked ACs for whole Task10 review.
ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation of approved lifecycle authority under controller rulings50–52. No new ADR; no startup-retirement success or production responder until later runtime cohorts.

### Phase2 implementation notes

Under ADR-126 and controller rulings52–53, implemented the private process-local
pause gate, pre-root/bootstrap/authority acquisition accounting, live repository
operation provenance and actual SQLite resource metadata/retirement. Event/Sync
transactions now commit/rollback and finally-close file connections on the creating
thread, retaining memory behavior. Ordinary custom constructor allocation and
close/reinitialize failures retain exclusion conservatively. No startup lease is
retired, no responder is installed, and runtime coverage remains incomplete.

Evidence:32 focused cases pass, including real worker SQLite lifetimes and separate
native maintenance observer processes.145 earlier affected admission/bootstrap/
repository guards passed. Final required SQLite/census/service guards passed in a
combined run (469 passed/1 existing Windows skip overall), whose additional core
capture checks had1 failure/2 setup errors. Core alone passes62; the three failing
cases alone pass3 and reduced service-plus-three ordering passes7. The exact combined
command on immutable clean phase BASE0f4ff4ba reproduces the failure family (468
passed/1 skip/1 failure/4 errors), establishing pre-phase2 app callback/bootstrap
concurrency debt for the remaining Task10 lifecycle cohort. BASE's unexpected RAG
metadata network attempts were blocked by Tests/network_guard; no download or
network permission change occurred. Do not claim the combined-order run is green.

Scoped ruff fatal checks, new-module formatting and diff checks pass. Self-review
covered actual operation/handle accounting, cancellation/retirement, native scope,
constructor and thread-affinity behavior, memory/rollback semantics and coverage
limits. Source syntax counts are unchanged; inventory guards pass without source
row changes. Updated owner inventory and the evidence lesson from the actual
wrong-thread closed-handle assertion incident. Detailed APIs, exact commands,
source handoffs and remaining cohorts are in task-10-phase2-report.md. Task31993
stays In Progress with all acceptance criteria unchecked for whole-task review.

Final self-review added three behavioral RED cases for forged/instance-shadowed
validation callbacks, then enforced exact token type and direct class validation
for operation/pause identity. The final32 focused cases and scoped static checks
pass; no caller-supplied validation callback can confer authority.

### Phase3 implementation plan (before code)

1. Inspect the five concrete core SQLite owners and propose exact operation, borrower, native retirement and safe reuse APIs to the controller before shared edits.
2. Establish behavioral RED for failed native-close reference retention, late mutation refusal, transaction ownership and native cached-handle drain; retain unresolved borrowers and actual thread-affinity blockers truthfully.
3. Integrate the controller-approved bounded lifetime strategy for ChaChaNotes, Media, Prompts, Library Collections and Library Ingest Jobs. Preserve memory, managed/nested/borrowed transaction semantics and all-thread evidence; no generic close, rollback, garbage collection or caller-selected owner token qualifies drainage.
4. Verify actual domain constructors/transactions and independent native maintenance exclusion, then targeted affected SQLite/domain/admission/census checks and scoped static/self-review checks.
5. Record exact supported boundaries, remaining limitations and evidence, commit only this phase, and leave Task31993 In Progress with unchecked ACs.
ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation of approved runtime lifetime contract, preserving ADR-004 mapping and ADR-036 ownership. No startup retirement, full runtime-coverage promotion or production responder in this phase.

Phase3 plan refinement, controller ruling54 (before shared edits): retain native APIs and implement conservative explicit source-thread borrower retirement. Managed transaction exit does not invalidate escaped native connections/cursors. Installed all-thread handle records and actual ordinary leases remain until successful explicit native close; raw/escaped/foreign-idle consumers truthfully block drainage. Later Task10 caller/app/headless job ownership boundaries must positively retire those caches at proven source-thread safe points before idling. This is a phase division, not reduced Task10 acceptance or a permanent unsupported declaration. No generic facade or executor redesign.

### Phase3 implementation notes

Under ADR-126 and controller rulings54–55, five exact installed core SQLite owners
now gate cached getters and managed scopes, strongly retain actual native handle/
lease associations and reserve explicit source-thread retirement. Existing native
connection/cursor, nested/borrowed transaction and memory APIs remain intact.
Managed exit does not revoke escaped borrowers. Failed native close or live liveness
probe keeps references and exclusion. Distinct thread caches retire independently;
Ingest's shared cache protects every accessing thread. Independent nested installed
operations preserve ordinary multi-instance calls only through fresh gate-checked
admission; no descendant authority transfers. Native registry association cannot
be stolen by a second same-path instance. Uninstalled subclasses remain ordinary,
unqualified callers; exact raw/unmatched returns remain actual blockers.

Behavioral RED/GREEN covers five owners' close failures, native observer exclusion,
late getters, admitted work/escaped cursors, active-work close, safe cache reuse,
source-thread races, liveness failure and same-path association. Combined focused
core/phase2 lifetimes:91 passed; final association fix's8-case affected subset passed
(no claimed92-case whole rerun). Final private SQLite guards:316 passed/1 existing
Windows skip after16 test spies were corrected to honor their supplied real native
factory. Final affected Prompts/Chatbooks guards:43 passed. Earlier capture checks:
62 passed. Full exact commands, overlapping-run limits and failed intermediate
runs are retained in task-10-phase3-report.md; no full suite was run.

The earlier domain check had257 passed/3 failed: two new real Prompts trace callback
conflict regressions were fixed under ruling55; the remaining Media historical-schema
fixture independently fails on immutable phase BASEb778f3883e7174bba0a1887363d53cf02af49240
with duplicate chunk_engine_version. Controller carries that malformed fixture to
Task13 historical-schema qualification; no migration change or replacement fixture
was made here. The known phase2 combined-order app callback/bootstrap failure remains
an explicit Task10 app/lifecycle obligation. Removed incorrect new-directory test
oracle/speculative constructor work is disclosed as abandoned self-review, not
behavioral regression proof. Scoped fatal lint, touched focused-module formatting
and diff checks pass; the actual callback incident is recorded in testing lessons.

Phase3 is complete, **Task10 is incomplete**. Startup lifetime remains installed,
runtime coverage refuses, and no responder is enabled. Actual Library pool-service
jobs, app-owned ingest persistence/parser callbacks, Chatbooks/Persona compound
producers and remaining raw/config/model/TTS/operational/app/headless cohorts need
full mutation and positive source-thread handle retirement before idle. Idle foreign
caches, escaped raw objects and dead affine threads remain truthful blockers; these
are required next integration work, not reduced final acceptance. Updated owner
inventory names the exact caller handoffs without promoting syntax rows. Existing
ADR-126/design references remain authoritative; all31993 ACs remain unchecked and
status In Progress pending whole-task implementation and independent review.

### Phase4 implementation plan (before code)

1. Establish behavioral RED for full Feedback/Grammar mutation refusal before cached state changes, template mkdir/save admission, and actual emoji worker lifetime.
2. Add separate exact source-bound raw participants and pre-selector reservations. Admit fixed target/publication sidecar/missing-directory scope before memory or storage mutation; pin and recheck existing ancestor identity, retaining actual native leases and resource-close uncertainty. No repository token reuse or caller callback authority.
3. Integrate constructors/direct helpers and full service mutations; preserve ordinary custom/built-in template destinations without expanding capture ownership, subclass ordinary behavior without installed pause descendants, and emoji app-owned unsynchronized last-write-wins behavior.
4. Verify real temporary files/workers, native maintenance observers, pause/acquisition races, same-source serialization, failure/close uncertainty, cancellation and safe resume. Run only new focused tests, affected producer/domain/admission/participant and exact census guards.
5. Self-review, record exact evidence/remaining cohorts, update inventory, and scoped commit. Retain In Progress and all unchecked ACs; no runtime coverage promotion, startup retirement, Complete capability or responder.
ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct approved raw storage lifetime integration, retaining ADR-004 mapping and ADR-036 ownership; controller approved separate raw registry and directional multi-target scope before shared edits.

### Phase4 implementation notes

Under ADR-126 and the controller-approved separate raw-source strategy, Feedback and
Grammar full mutations/constructor reads/direct helpers, default Chunking template
mkdir/read/save and actual emoji workers now retain source-bound ordinary admission.
Fixed file/sidecar/mkdir paths are admitted before state or IO changes; native pins
and descriptor-relative IO preserve parent identity, while real descriptors/wrappers
and leases remain until positive native retirement. Cached service mutations roll
back on refusal/error. Existing sidecars are preserved, successful publication
consumes exact temporary inode ownership, and failed close/publication/cleanup
preserves unresolved evidence and refuses later conflicting writes. No GC/callback/
Future-cancellation authority or core-token transfer was introduced.

Custom/built-in template paths and subclasses retain ordinary selected-path behavior
without installed default-owner or pause-descendant coverage. Missing raw pinning
primitives select ordinary path IO before access under ruling57; actual admission
leases (including stronger native leases) and resource accounting remain, but no
installed participant, pause-descendant grant, capture or startup qualification is
created. Simulations do not qualify Windows. Existing trusted parent aliases retain
physical identity checks; caller-owned aliases on the pinned route still refuse.
Emoji stays app-owned,
best-effort and unsynchronized last-write-wins; a canceled worker can remain an actual
native drain blocker. Exact selector/callback/Thread/Task/native-scope identity is
checked. All participant_pending declarations, startup holds and incomplete runtime
coverage remain; no Complete/replacement capability, responder or app-wide coverage.

Evidence: final raw+Feedback/Grammar domain run64 passed (58 raw cases +6 service
cases); final exact producer census11 passed; earlier affected producer run30 passed/2 pre-existing server-only module
skips; affected admission/participant/core/bootstrap/census guards182 passed.
Behavioral RED/GREEN covers paused mutation/constructor/mkdir, native observers,
parent/selector races, exact-file sidecar refusal, cancellation, path/native-close
uncertainty, spoofed callback identity and actual two-process sidecar ownership.
The initial native-observer test selected the wrong ambient hold and was corrected;
three native-failure red checks required missing explicit descriptor metadata while
already retaining native exclusion. These are disclosed separately from behavioral
false-drain regressions. The portable disjoint fixture initially omitted its claimed
config file; the alias fixture initially assumed an untrusted caller-owned alias
was permitted. Corrected fixtures preserve bootstrap/link policy; the positive
trusted-alias case explicitly simulates trust classification only. Requests/Pydantic/AST and joblib semaphore ENOSPC warnings
remain; skips are not substituted for release evidence and no full suite ran.

Changed six production modules including new raw_participants.py, the focused raw
lifetime tests, source inventory and the actual sidecar-race testing lesson. Exact
commands, overlapping run counts, scoped static/self-review checks, platform limits
and remaining full Task10 integration are recorded in task-10-phase4-report.md.
Task31993 remains In Progress, all ACs unchecked, pending remaining cohorts and
independent whole-task review; ADR-126/design references are preserved.
