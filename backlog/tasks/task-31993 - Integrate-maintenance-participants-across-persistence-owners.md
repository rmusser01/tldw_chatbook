---
id: TASK-31993
title: Integrate maintenance participants across persistence owners
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-07 23:53'
updated_date: '2026-09-08 21:57'
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

### Phase10 implementation plan (before production)

1. Reproduce real Persona/dictionary/citation core-file and cached-state pause gaps using actual SQLite/files and independent native observation.
2. Obtain controller ruling on exact installed source binding and simultaneous existing core/raw scopes before implementing the new compound interface.
3. Bind actual canonical app-created source relationships, preselect every core/file/temp member before effects, preserve ordinary custom/no-file/memory behavior and source-thread native borrower retirement.
4. Cover source reads/mutations/cache rollback/publication and concrete dictionary threaded scope dispatch through actual completion, including repeated cancellation.
5. Run targeted behavioral/domain/shared/census and scoped static checks, self-review, update exact source handoffs and evidence, and commit only this phase. Preserve all unchecked ACs and unavailable runtime/startup/Complete coverage.
ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct source-lifetime implementation; any approved compound interface decision will amend ADR126. ADR024 citation provenance and ADR037 persona separation remain unchanged.

### Phase12 implementation plan — actual Persona Visual lifetimes

1. Establish behavioral RED at actual file publication/core activation and workspace/native/UI boundaries; obtain controller source-backed ruling before helper expansion.
2. Bind exact configured Persona Visual sources and finite preselected publication, workspace, import and runtime members; preserve custom ordinary routes and source-native uncertainty.
3. Integrate actual screen creator, executor and result lifetimes, source-thread DB retirement, dirty editor preservation and retained cleanup evidence.
4. Verify targeted source/domain/native/census guards and static checks, self-review and commit this bounded phase; preserve In Progress and unchecked ACs.
ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation of approved persistence lifecycle; preserve ADR074/067/032 and record the bounded controller ruling.

### Phase13 implementation plan — Shared Visual Identity source/native lifetimes

1. Reproduce actual post-rename/core pause RED, then obtain the controller ruling for this concrete source/native bridge before production.
2. Bind exact configured profile/core/source identities; reserve before candidate, seed, runtime and publication selectors; preserve existing candidate/seed locks and public ordinary behavior.
3. Preselect finite source/output/cleanup members, retain actual native resources and committed/uncertain outcomes, and integrate concrete reaction-pack save/cleanup caller retirement.
4. Verify actual pause, identity, dirty draft, fork/stale graph, native uncertainty and ordinary portability with targeted source/domain/UI/native cases only.
5. Update ADR126 rationale, source inventories and concrete remaining generation/app/headless routes; self-review scoped diff, run focused guards/static checks, commit and dual-report. Keep Task31993 In Progress and all ACs unchecked.
ADR required: yes (reuse existing).
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: direct implementation of approved maintenance lifetimes under ruling74; ADR067/074 separation and ADR032 package immutability remain.


### Phase14 implementation plan — TTS repository lifecycle foundation

1. Preserve actual first-open and paused-CRUD behavioral REDs, then implement controller ruling79 using the existing owner loop, state/lifecycle locks and serialized executor.
2. Reserve before first-open selectors/native creation; close new admission separately from terminal/generation, retain actual worker futures and explicit result-publication completion, and clean only on the source worker.
3. Prove bounded reversible drain, timeout/repeated cancellation, exact native retirement, stale source refusal, competing repository exclusion and definitive close compatibility with synthetic real SQLite/private native observers.
4. Keep finite migration/backup/reference native scopes and actual app/runtime/dirty editor aggregation as exact immediate Task10 continuation; this foundation grants no across-pause callback capability or repository/Complete qualification.
5. Run focused affected domain/shared/census/static checks, self-review, update ADR126/source inventory/evidence and scoped commit plus identical dual report. Keep In Progress and every AC unchecked.
ADR required: yes (reuse existing).
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: direct approved maintenance lifecycle implementation under ruling79, preserving ADR028/040/051/023 and existing schema/native authority policy.

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

### Phase5 implementation plan (before code)

1. Establish behavioral RED on actual Workspace, AgentRuns and ClientNotifications cached getters, native close/reference failures and active-work retirement refusal; extend to actual ScheduledTasks, Research and Writing scopes.
2. Extend only the exact installed repository declarations and native policy associations. Preserve independent nested admission and actual wrapper/lease/path/Thread/Task association; leave BaseDB and ordinary subclasses unqualified.
3. Gate/register fresh and cached native getter routes. Count the three cached sources' existing read/transaction scopes; reserve explicit source-thread close and retain live/uncertain caches. Count ScheduledTasks lexical read/transaction lifetimes through native close, AgentRuns' explicit fresh-read route, and Research/Writing path-backed lexical native contexts. Setup failures positively close or retain native evidence; memory/external injection behavior stays ordinary.
4. Run focused behavioral cases with independent native observers for gates, pending/native setup, exact source association, nested scopes, threads, escaped native objects, close failures and reopening. Run only affected six-source/domain and shared participant/admission/private-SQLite/census guards; no full suite or collection sweep.
5. Update exact owner/census and concrete caller handoffs, scoped fatal lint/new-test format/diff checks, self-review, report and scoped commit. Keep Task31993 In Progress with all three ACs unchecked; startup/native-unqualified holds and runtime coverage refusal remain.
ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation of the approved native storage lifetime contract with existing ADR-004/036 boundaries; no schema/migration or new subsystem decision.

### Phase5 implementation notes

Under existing ADR-126, the six actual operational/domain SQLite sources now bind
exact installed instances, selected paths, native policies and real handles to
existing participant lifetimes. Workspace/AgentRuns/ClientNotifications managed
read/transaction scopes retain per-thread caches through exit and preserve them on
failed close/live probe; source-thread explicit close remains the retirement route.
ScheduledTasks retains fresh closing read/transaction boundaries and counts its
actual migration sequence. AgentRuns get_run_fresh retains its separate finally
close. Research/Writing add counted lexical native contexts, preserve file close-on-
exit and retained memory/external injection behavior, and bind relative files once.
Native setup failure positively closes or leaves registered uncertainty; raw handles
remain native APIs and independently block drain until explicit retirement.

Controller-directed extension reproduced raw allocation and setup participant-gate
races in the five earlier core owners plus Event/Sync. Their exact concrete source
routes now recheck new allocations before returning; Event/Sync and Collections
raw getters are gated and Event/Sync native file policy remains db.base. Ingest
cleanup keeps the exact allocated handle local so mutable cache replacement cannot
redirect refusal close to a pre-existing borrower. No BaseDB-wide authority or
cross-store grant was added. Shared pre-admission missing-parent lookup preserves
the existing typed PrivatePathError/UNSAFE_PARENT/missing_parent refusal; no parent
is created and alias/trust checks are unchanged.

Behavioral REDs: initial cached9, scope9, setup-close6, raw-allocation gate6,
previous-owner gate7, all-source PRAGMA/setup gate13, and Ingest native-identity1.
Final focused96 cases are included in the final222-case affected lifetime/interop/
Ingest run (all passed). Earlier same-source domain167 passed; owning-thread fixture
cleanup97 passed; previous-owner domain195 passed; Media55 passed with only its
independently baselined malformed historical-schema case explicitly deselected.
Shared guards452 passed/1 existing Windows skip/2 missing-parent failures; both
failures were fixed and their complete interop file passed in the final222 run.
No inflated sum of overlapping runs or full-suite claim. Exact commands/raw logs,
initial wrong Scheduled first-row-vs-MAX test oracle and corrections are retained
in task-10-phase5-report.md. The initial domain run's +371-descriptor warning exposed
GC-dependent fixtures and worker tests; real owning-thread finalizers/close calls
were added in five affected test files. The97-case cleanup run emitted no such
threshold warning; this is not a zero-growth or production lifecycle claim.

Self-review checked exact source/native associations, pending and nested gate races,
Thread/Task identity, ambiguous closure retention, native borrowers, memory/external
compatibility and concrete source callers. Scoped fatal lint, new-test/helper format
and diff checks pass. Owner inventory records six-source lifetimes plus the earlier
race corrections, with no producer-count/schema/capture promotion. Testing lesson
records actual fixture-native retention evidence. Startup/native-unqualified holds,
participant_pending rows and incomplete runtime coverage remain; no responder or
Complete/replacement exposure. Task31993 stays In Progress, all three ACs unchecked.
Required next work includes Console/settings/retention AgentRuns callers, Workspace
service and Home notification pool jobs, notification/scheduler/research execution
and cross-store followups, multi-scope Writing jobs, remaining storage cohorts,
app/headless lifecycle and one independent whole-task review. The clean-BASE app
callback/bootstrap issue remains Task10; Media historical-fixture debt stays Task13.

### Phase6 implementation plan (before code)

1. Establish behavioral RED for actual Evals/Subscriptions/FileNotes cached reads and safe close, constructor directory pause, Receipts and installed Kanban source scopes.
2. Extend exact source declarations and native association only; bind Subscriptions read-only mode on every use and preserve different native write/read policies. Count existing lexical CRUD/schema/read/transaction bodies without changing borrowed native commit or cursor semantics.
3. Integrate FileNotes constructor as two independent safe stages: exact constructor-only fixed directory admission and positive raw resource retirement, then fresh SQLite repository admission. Pause between stages may leave admitted directories but refuses SQLite creation. Preserve unsupported-platform ordinary behavior; no compound authority transfer.
4. Count SiteConfigManager setup only through its actual lazily available exact source, retaining standalone helper ordinary status and separate CharactersRAGDB hybrid connections. Register exact allocation before setup, recheck gates, positively close failures or retain uncertain resources.
5. Verify real SQLite/native process observers, active/foreign-thread refusal, pause/allocation/setup/selector races, native close failures/reopen, memory/relative/read-only/hybrid compatibility, and fresh per-operation close semantics. Run affected targeted domain/lifetime/private-SQLite/census guards, static checks and self-review.
6. Update source inventory and exact caller handoffs, append evidence, scoped commit. Task31993 stays In Progress with three unchecked ACs; no startup retirement, responder, runtime coverage or Complete capability.
ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation of approved persistence lifetime contract; controller approved exact source strategy and two-stage directory boundary, preserving ADR-004/036.

### Phase6 implementation notes

Under ADR-126 and controller-approved exact source strategy, Evals/Subscriptions/
FileNotes/receipt/Kanban native scopes now use installed path/policy/Thread/Task
associations and positive source-owned retirement. SiteConfigManager initial schema
setup has a distinct exact source; hybrid core handles retain separate ownership.
Lazy production-module lookup avoids eagerly loading these added sources. New
cached/raw access is gated; fixed admitted scopes can finish; failed native close
or live errors preserve references. FileNotes shared cache refuses foreign/active
close, maps semantic notes.file_notes to native notes.file_notes_replica and keeps
original memory-close behavior. Receipt/Kanban commit/rollback/finally.close and
all pre-existing literal/schema payloads remain intact.

FileNotes directory construction uses fixed ordinary raw admission and native pins,
then positively retires before fresh SQLite admission. Pause may leave successfully
admitted directories but refuses SQLite creation; path retarget and uncertain
native directory retirement remain blockers. No composite raw-to-SQLite authority,
new schema/migration, coverage promotion or startup retirement was introduced.
Subscriptions resolves only an omitted config default before its DB operation so
cold config values survive; explicit overrides win and intervening pause refuses
mutation. Its entire actual schema tail remains in the original admitted scope.

Evidence: final source run167 passed including83 focused cases; earlier combined
source/domain302 passed. Shared earlier lifetimes/privateSQLite/census637 passed,
with1 existing Windows posture skip. Exact Evals capture10 passed; the4 Subs
catalog/capture/audio checks initially failed from changed SQL indentation and now
pass after original plain/joined literal restoration. The incident is recorded in
lessons-testing-evidence.md. Final FileNotes/new-lifetime94 passed before the last
config/semantic tests. Counts overlap, never additive. Fatal lint, new/participant
format checks and diff checks pass. No full suite ran.

Retained test corrections include the unsafe initial native trace-callback close
(exit139, not rerun or claimed OS evidence), intermediate Evals SyntaxError, wrong
mkdir/policy/memory/nested-writer assumptions and one malformed census row. True
retarget/schema-tail/memory/cold-config/semantic regressions have behavioral
RED/GREEN. Explicit fixture/source-thread cleanup removed the initial+246 descriptor
threshold warning in scoped reruns; this is not production worker qualification.
Self-review used full/whitespace-insensitive diffs plus exact AST literal/control-flow
comparison and real native/capture evidence. No schema allowlist was widened.

Files: participants/raw_participants/storage_admission, actual five sources plus
Kanban native helper and SiteConfigManager, focused tests, exact source fixture
cleanup, owner inventory and evidence lesson. Full commands/results/source handoffs
are in task-10-phase6-report.md. File Notes workspace shutdown currently may invoke
close on a different executor thread then clear its reference; actual caller jobs
must positively retire those handles. Evals/Subscriptions/Site jobs, Notes import/
file/Git/sync/dirty editors and Kanban multi-scope methods remain later Task10 work,
along with prior cohorts and app/headless responder/startup aggregation. Task31993
stays In Progress with all3 ACs unchecked; no claim of Task10 completion.

Final phase6 boundary review also reproduced a real Kanban DELETE-to-WAL setup
mutation after exact source admission closed. The native helper now registers and
checks only the supplied actual service before PRAGMAs; standalone calls retain
ordinary semantics. The focused source rerun passed12 cases and the final Kanban/
private-interop/mutation checks passed36; final source census passed11. Full exact
late RED/GREEN commands and the corrected report boundary are recorded in the phase
report. No additional runtime coverage or Task10 completion is claimed.


### Phase7 implementation plan (before code)

1. Inspect actual PromptHistory load/append and Console warm-load/recall/accepted-send callers, plus ChatScreen sidebar debounce, direct IO and unmount flush. Settle the exact two-source pending/queued/native cancellation strategy with the controller before production edits.
2. Add behavioral RED tests for predispatch pause, queued cancellation and running cancellation without cache/dirty loss. Use source-bound pending reservations with independent worker admission; never transfer raw authority across tasks or threads.
3. Integrate immutable history snapshots and serialized complete operations, fixed selected paths and pinned native IO, plus sidebar revision/failure tracking and a usable safe-point flush. Preserve ordinary custom/subclass behavior without installed authority and preserve best-effort accepted-send behavior.
4. Exercise actual asyncio/Textual callers, late writes/cap rewrites, refusals, source/path/task provenance and positive/uncertain native retirement with independent native observers. Run focused domain/shared and source-census checks only.
5. Update exact owner census and caller obligations, record RED/GREEN and fixture corrections, self-review, run scoped static checks, and commit this bounded phase. Keep Task31993 In Progress, all three ACs unchecked and earlier notes/Design references intact.

ADR required: no (existing ADR applies).
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: direct implementation of the approved full-operation admission/retirement contract for two existing async file sources. No startup retirement, runtime coverage promotion, responder, Complete or replacement capability.

Phase7 controller refinement: retain the no-transfer/no-ack job strategy. The creator's post-IO work is only cache/dirty/error bookkeeping under its original pending reservation. Actual process startup retains the native hold until future whole-process local drain sees all pending work retired; this phase does not retire it. All file writes/publication/cleanup/native-close effects stay inside the worker raw scope. An initial test demanding a standalone per-job native hold during cache delivery omitted actual startup; replace that unsupported assumption with actual startup plus local pending drain and independent native observer evidence. Do not add a worker waiting on event-loop acknowledgement or its shutdown deadlock risk.


### Phase7 implementation notes

Implemented only PromptHistory and ChatScreen sidebar full async file jobs under
ADR-126 (`backlog/decisions/126-complete-local-backup-and-recovery.md`). Source-bound
queued records register pending work before selectors/snapshots, and actual worker
threads acquire their own fixed raw scopes; no token/context/callback transfers
permission. Queued cancellation atomically prevents source entry. Running and
repeated cancellation retain source serialization through actual callback completion
and result bookkeeping. A separate actual-worker result signal fixes independent
executor-wrapper cancellation without waiting for an event-loop acknowledgement.
Failure/uncertain close retains source evidence; startup retirement stays disabled.

History load/append now share a lock, immutable writes publish cache on success,
capped rewrites use an owned sidecar, and newly stashed drafts survive delayed writes.
`persistence_safe_point()` exposes pending/error state without saving/discarding text.
Sidebar keeps revision-safe dirty state across exclusive Textual worker cancellation,
flushes through its actual source lock, preserves unrelated TOML via atomic sidecar
publication, and refuses queued config-profile redirection. Direct sidebar read/save
routes are admitted too. Initial state tracking is initialized before its first read.
Accepted-send recording remains best effort; composer warm-load/recall uses the same
source lifetime. Exact source bindings cannot demote to bypass a gate after config
changes; raw IO accepts only explicit r/w/a modes with operation permission checks.

Behavioral REDs covered paused writes, queued cache pollution, premature running
cancellation, newer live-draft loss, an independently cancelled executor wrapper
spinning in an isolated child, bound-source demotion and read-scope r+ file creation.
A proposed per-job native-tail assertion omitted actual startup and was corrected
per controller ruling: supported process startup plus pending/local-drain already
protect creator bookkeeping. Two fixture corrections delayed the native observer
until after admission and limited a json.loads gate to actual string history lines;
no production admission was weakened. The old sidebar race fixture now gates actual
TOML serialization instead of a spoofable instance writer callback.

Final covering verification: 54 passed (20 new async/native, 25 existing history,
6 prior Feedback/Grammar and 3 prior raw mode/scope cases); latest real Textual
cancellation/profile-retarget cases 3 passed. Earlier complete focused UI/sidebar/
composer domain 31 passed, accepted-send history subset 5 passed, shared raw/
participant guards 74 passed and source census 11 passed. Counts overlap and are
not additive. All subprocesses were reaped; actual native-close uncertainty and
independent maintenance were exercised in private children. Final scoped static,
format and diff checks are recorded in task-10-phase7-report.md. No full suite,
network/download, real user data, shared environment mutation or per-phase reviewer.

Changed paths: async_file_participants.py, raw_participants.py, prompt_history.py,
chat_screen.py, composer warm-load comment, the new focused async test module,
sidebar domain tests, accepted-send caller test, exact owner/source census and the
cancellation lesson. Source owners and actual caller obligations are detailed in the
inventory and phase report. Controller owns independent whole-Task10 review.

Phase7 is complete; Task10 remains In Progress with all three ACs unchecked.
App/headless composition must observe source safe-point failures/late dirty state,
retain startup until all actual producers drain, reacquire before reopening, and
qualify remaining source/jobs and persistent/queued diagnostic sinks. Neither
logging nor the whole app is claimed IO-free/drained. Earlier phase2 combined-order
callback/bootstrap and Task13 Media schema-fixture debts remain; focused phase7
runs did not reproduce them. No runtime coverage, responder, Complete or replacement
capability is activated.

### Phase8 implementation plan — settings, definitions and templates

1. Bind only the actual RuntimeSourceStateStore, default EvalConfigLoader, notes template module/CLI/import job, SettingsThemeEditor and default ConfigFileStorage source routes. Keep custom exports/storage ordinary and unqualified; revalidate previously installed selectors.
2. Extend existing pending/raw native accounting for complete serialization, read/modify/write, selected directories/sidecars, verified descriptors, publication and positive cleanup identity. Preserve private runtime path security/posture behavior and raw portability.
3. Collect CLI input before admitted template reread/merge; keep the actual template worker reserved through queued/running cancellation and outcome bookkeeping. Preserve unrelated templates and failure results.
4. Preserve evaluation mutable configuration with truthful dirty/error safe point; retain theme is_modified/ThemeModifiedStatus and existing confirmation flows. Account for Tamagotchi backup pruning and repair wrappers without capture-time constructors.
5. Establish behavioral RED/GREEN for real source routes, native/process and cancellation boundaries; run focused domain/shared/census guards and scoped static/diff checks. Self-review, update owner census and exact remaining app/job handoffs, scoped commit and report. No Task10 completion or runtime/startup/Complete activation.

ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation of existing ADR-126 installed owner lifetime contract and phase4/7 raw/pending APIs; no new storage or recovery authority.

### Phase8 implementation notes — partial Task10 continuation

Implemented source lifetimes for runtime private state, installed Eval YAML, shared user note templates (actual reader/CLI/import worker), canonical theme editor and default pet JSON/recovery/backup routes. Serialization/RMW and selected sidecars remain admitted until positive native retirement; dynamic member preflight retains ordinary parent admission and pins before bounded exact-member selection. Runtime private helpers retain original posture checks and public signatures with narrow source discovery, descriptor/stream tracking and explicit positive close. Eval mutable dirty/error state and theme `is_modified` block participant drain; failed reload/write/native retirement preserve draft state. CLI input precedes fresh locked template merge and async import success follows final publication. Pet backups preserve exact bytes and existing same-second overwrite policy; stale sidecars and replaced prune entries remain intact.

Existing ADR-126 applies. Runtime/startup/responder/Complete composition stays unavailable; Task10 is not complete and all three ACs remain unchecked. Controller owns independent review after the remaining cohorts. Changed source census and detailed API/evidence report are in `backlog/docs/backup-recovery-owner-inventory.md` and `.superpowers/sdd/2026-09-07-complete-local-backup-restore/task-10-phase8-report.md` (external backup `/private/tmp/chatbook-backup-execution-hgp11i7t/task-10-phase8-report.md`).

Verification: initial actual-route RED4; later runtime/temp/async cancellation, dirty-drain, same-second backup and native-close regressions verified RED/GREEN. Affected domain rerun51 passed after selector-before-constructor and exact matching-entry bound corrections; earlier75 other domain cases passed. Covering source/shared/native/domain/census run286 passed2 failed81.75s: both failures are profile path census debt reproduced unchanged on immutable phase BASE (2 failed21.78s), not passing guard coverage. Exact failures are `test_production_profile_owned_path_inventory_is_exact` and `test_cli_prints_the_real_source_census_and_enforces_it` in Tests/Architecture/test_profile_owned_path_inventory.py, from untouched TTS/recovery.py111/115 literals `~/.config/tldw_cli/chatterbox_voices` and `~/.config/tldw_cli/higgs_voices`. Carry these two entries into the pending Task10 TTS cohort; no phase8 TTS edits. Logs retained externally. Narrow late runtime/import/preflight and source census/shared checks recorded in full report; no unchanged broad repeat or full suite.

Existing ordinary cross-process template concurrent-writer limitation remains: concrete post-read ready-handshake schedule fails on current source and immutable838b949ac7aa9c92988d79fe725025e8ffc21a79 with old CLI entry adapted. Per ruling62 no new write mutex or Admission contract; no xfail or passing snapshot-safety claim. Preserve scratch harness/logs for later source-specific concurrency policy work. Maintenance exclusion remains covered independently. Remaining work includes config helpers/portalocker source lifetime, non-template import DB/Notes/Git compound jobs, TTS and other remaining cohorts, app/headless pause/dirty/error handoffs and full independent Task10 review.

### Phase9 implementation plan (before code)

1. Bind the actual configuration source, current/lock/backup/snapshot members and fixed random temps before effects.
2. Preserve the process RLock and cross-process portalocker, with pause-aware waits and explicit native stream/FD retirement.
3. Cover bootstrap, read/merge/write, cache/generation, revisions, encryption and shutdown; retain truthful state on failure.
4. Keep concrete independently admitted canonical derived-directory effects and existing startup order; no post-pause scope widening.
5. Run behavioral RED/GREEN and targeted config/private/native/census checks; self-review, scoped commit and full report.
ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation; ADR004 restart boundary, ADR012 secret precedence and ADR033 commit models remain unchanged. Task10 stays In Progress with unchecked ACs.

### Phase9 implementation notes

Configuration persistence now binds the exact loaded config module and fixed current, lock, backup, snapshot and random temporary members. Existing process RLock and cross-process portalocker serialization remain; lock waits observe pause, and native streams/FDs require positive retirement. Direct private helper routes and cached reads are covered. Failed writes, reloads and native retirement preserve prior usable cache/settings/generation/password state and persistent error evidence. A pause after bytes commit reports `file_replaced=True`, `caches_reloaded=False` and refuses fresh derived directory effects. Separate concrete data/chat/model directory operations retain existing mkdir behavior and fix/recheck generation/path under the process lock. No DB/model service relocation or runtime/startup/responder qualification.

Behavioral config/native/interprocess and affected shared checks cover normal, paused and uncertain outcomes. The covering run had 228 passed and one stale config-module fixture failure; the corrected actual theme/app fixture plus private-path/runtime-store subset passed83. The isolated source fixture imports a fresh actual config module before selecting app consumers; it never clears registries. The original serialization association RED is supplemental introspection; actual pause-during-serialization and two-process write-lock tests provide behavioral evidence. Final affected config/native/persistence rerun passed37; retained actual startup+pending composition passed1 after the same narrow fixture correction. Scoped Ruff/new-module format/diff checks passed. Exact commands are in the phase9 report.

ADR126 applies with ADR004 restart, ADR012 credentials and ADR033 commit models unchanged. Task10 remains In Progress and all three ACs unchecked. Remaining cohorts and actual Settings/app/headless gates, dirty/error drain, startup release/reacquire, plus independent whole-task review remain required. Inherited phase2 combined-order app/RAG and phase8 TTS profile-census debt are unchanged.

### Phase10 implementation notes — concrete chat source continuation

Implemented exact configured Persona/dictionary/citation core-sidecar pairs and
actual Chat_Dictionary_Lib parser/import/export/listing lifetimes. Existing scopes
are preadmitted and rechecked before simultaneous same-PID/Thread/Task discovery;
no generic callback/owner token or post-pause widening. Seven Persona cache groups,
dictionary history and compatibility citation cache preserve prior usable state on
failure; possible mixed effects and known native uncertainty retain sticky drain
blockers without claiming durable rollback or restart repair. Citation companion
binding preserves actual canonical generation fencing and ordinary concurrency.
Library IO fixes external input/output/temp identities without widening inventory,
materializes default exports before writes, preserves copy metadata using retained
FDs (including Darwin libc.fchflags), and keeps unsupported operations unqualified.
The real dictionary scope executor waits for actual completion under cancellation
and closes only a newly opened worker-native connection on its source thread.

ADR126 is amended for approved concrete compound semantics; ADR024/037 and schema
formats remain unchanged. Behavioral RED/GREEN includes real committed-row versus
history pause, cached-state refusal, optimistic no-change conflicts, actual citation
read, running/queued/repeated/executor cancellation, native close before/after,
external metadata/preflight/identity, missing-input/listing behavior and late worker
outcome/sidecar replacement defects found during self-review. Targeted covering
cohort231 and shared raw/native/startup/census84 passed; later changed-boundary
checks and exact logs are in the phase10 report (counts overlap, not additive).
Fixture corrections use only fresh actual config modules, actual descriptor fault
selection and isolated child observer import paths; no registry clearing or user IO.

Task10 remains In Progress with all three ACs unchecked. Actual app/headless safe
points, remaining UI Textual jobs/cohorts, startup release/reacquire, responder and
Complete qualification remain required. Sticky mixed-source recovery needs explicit
validated reconciliation; none is invented. Existing phase2 app callbacks, phase8
TTS literals, Task13 Media historical fixture and Task26 diagnostics remain pending.
Controller performs independent whole-Task10 review. Exact APIs, source-only native
versus startup evidence, commands and limitations are in task-10-phase10-report.md.

### Phase11 implementation plan — actual MCP stores (Task10 continuation)

1. Read binding brief, actual five source families and caller jobs; establish behavioral RED before production.
2. Obtain controller approval for exact source/private-helper expansion, then bind full constructor/read/RMW lifetimes and preselect every concrete sidecar/parent.
3. Verify permission corruption backup, history migration/rotation, native retirement, exact identity and caller-state failure behavior with targeted real files/processes.
4. Record pending service/job/credential boundaries, update source census/docs, run affected targeted domain/shared guards, self-review and scoped commit.

ADR required: no new ADR.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md; existing ADR-053 standalone runtime, ADR-032 permission boundary and ADR-080 advisory summaries apply.
Reason: direct implementation of approved installed-source lifetimes; no new authority or data model. Task remains In Progress, all three ACs unchecked, runtime/startup/responder/Complete unavailable. CLI plan command succeeded; prior plan/notes/design references restored before appending this phase.

### Phase11 implementation notes — partial Task10 continuation

Implemented actual five MCP local-source constructor/read/RMW lifetimes and exact
history private-helper native resource ownership under controller ruling69 and
existing ADR-126 (phase11 addendum), ADR-053/032/080. Canonical config/profile/path
binding refuses retarget/demotion; ordinary custom/subclass/private-platform behavior
remains. Permission corruption backup, caller timestamp native completion, history
migration/rotation and owned inode/temp expectations now share complete admitted
source lifetimes. Partial generation/native uncertainty remains sticky; no imported
approval or runtime activation, cross-process mutex or automatic repair is added.

Actual source matrix plus seven affected domain modules: **164 passed**; additional
local/target/context pause and exact-file/None-native boundaries: **5 passed**
(across two targeted runs after correcting a legacy-target fixture). Shared raw,
private helper/runtime, startup pending bookkeeping and source census: **153 passed**.
Late identity RED demonstrated adoption of a foreign published inode; carrying the
actual owned publication identity made the nine affected edge tests green. Native
before/after cases and independent rotation observer ran only in private test
children, with quiescent child startup retired solely for source-only native proof.

Source census and precise remaining MCP service/job/credential routes are documented
in backlog/docs/backup-recovery-owner-inventory.md. Diagnostic guard ran and remains
failed on inherited clean-BASE drift; own diagnostics/sinks are AST-equivalent and
already match checked manifest after old wrapper indentation removal. No unrelated
manifest refresh. Exact commands/results/fixture corrections and immutable external
exports are recorded in task-10-phase11-report.md. Full runtime/startup/responder and
Complete coverage remain unavailable; Task31993 stays In Progress, ACs unchecked,
and whole Task10 independent review remains controller-owned.

Final self-review added actual REDs for constructor reentry retarget/error clearing
and history private-posture demotion; bound reinitialization now refuses before
mutation and private posture is rechecked. Final affected selector/constructor/
migration/local-target-context matrix: **22 passed**. New/changed Python fatal
Ruff, new-module format checks and diff whitespace checks passed. A testing lesson
records the real native-callable fault-injection provenance trap. No additional
broad sweep or runtime qualification was performed.

### Task10 phase12 implementation notes — Persona Visual source/native and UI lifetime

Implemented the actual canonical Persona Visual repository, immutable publication,
workspace/import candidate and runtime-asset lifetimes using existing storage/core
admission. Rulings70–73 retain exact source/issued-object identity, fixed native
paths, positive FD retirement and dirty/error/cleanup ownership. The real UI
publish worker composes only its bound local Persona source for the late guard.
An actual public-guard/UI deadlock RED required omitting the new visual mutex only
from authenticated publication; existing optimistic activation and native scopes
remain. The shared thread-drain helper now observes actual callback completion;
concrete Persona jobs close only newly created source-thread DB borrowers.

Actual custom/optional-clear/replace/import UI saves activate the next version;
required-state clear preserves the dirty draft and refuses invalid publication.
Native partial/before/after uncertainty excludes an independent maintainer; normal
worker retirement lets it observe matching real core locators and asset bytes.
Concurrent publications retain one optimistic winner and owned loser cleanup;
concurrent source cleanup cannot activate stale bytes. Exact positive cleanup
reconciles only its own source/UI blocker.

Targeted covering evidence: publication/lifetime69passed; cleanup/workspace/import
55passed; raw/core/private186passed; Persona/shared widgets42passed; affected Shared
Visual cancellation7passed; clear contract3passed; final cleanup-reference1passed.
Counts overlap. Remaining final static/caller evidence and exact commands/log paths
are in `.superpowers/sdd/2026-09-07-complete-local-backup-restore/task-10-phase12-report.md`.
The source census and retired-owner guard passed; two inherited unchanged TTS path
literals and inherited diagnostic owner/config-sink drift remain explicit. Only
the changed Persona UI diagnostic digest was reconciled after AST-equivalence
review. Reused ADR-126; no new schema/visual feature/archive capability or broad
manifest refresh. No full suite, external generation, network, real user data,
startup release, responder or Complete qualification. Task31993 remains In Progress
with all three ACs unchecked; Shared Visual Identity and whole Task10 review remain.

### Task10 phase13 implementation notes — Shared Visual Identity source/native and UI lifetimes

Implemented actual Shared Visual Identity candidate issuance, selected manual/builtin readers, immutable publication/core activation, exact cleanup, public repository operations, config Samira seed and concrete Personas reaction jobs through the separate visual_identity_participants bridge. Strong native descriptors/streams and existing pending/core lifetimes survive pause and cancellation; close uncertainty and original cleanup/result associations remain retained. Existing candidate/seed locks, ordinary injectable APIs, shared/builtin forks, binding guards and partial card-then-pack seed outcomes remain intact. Canonical UI row restoration is issued only to the original candidate and remains dirty before replacements; maintenance never discards it.

ADR required: yes. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md. Rulings74–78 record exact source/native authority, path-free canonical restoration, suppression of admission discovery during public atomic replacement/repository guard evaluation and only three authenticated internal repository result-read edges. No generic callback authority, global source mutex, schema change or installed package mutation.

Actual REDs covered file-rename/core pause, runtime read/UI selector admission, canonical restoration, actual duplicate publication, copied candidate/same-path other DB cleanup, public callback/guard borrowing and source-error plus borrower-close error association. Final source/repository iteration: 65 passed and two internal result-read failures, both corrected; exact amended source/guard/native/UI covering selection14passed and callback selection7passed. Earlier ordinary domain141passed with a Windows-only skip and one subsequently corrected error-category failure; cleanup/fork10passed, seed/canonical8passed, shared helper6passed and selected UI27passed. Counts overlap. Exact commands, intermediate fixture corrections and full logs are in .superpowers/sdd/2026-09-07-complete-local-backup-restore/task-10-phase13-report.md and its identical external copy.

Changed-source census is exact. The combined census guard has24passed and3inherited failures: two unchanged TTS/recovery.py profile literals and existing13diagnostic-owner/config-sink drift. Only the changed visual source diagnostic digest was refreshed after identical diagnostic-call AST proof; unrelated debt remains visible. Fatal Ruff, new-file formatting and scoped diff checks pass. Self-review retained ordinary callback behavior and sole cleanup/native references.

INCOMPLETE — phase complete, Task10 continuation required. Task31993 stays In Progress with all three ACs unchecked. Mandatory remaining routes include external image-generation runtime, the analogous earlier PersonaVisual public callback seam, TTS/runtime/credential cohorts, actual app/headless aggregation/startup/responder retirement and whole Task10 independent review. No full suite, live generation, network, user data, startup release or Complete promotion.

### Task10 phase14 implementation notes — TTS lifecycle foundation only

Implemented ruling79 through the existing TTSProfileRepository owner loop,
state/lifecycle locks and serialized executor. Pure construction is preserved;
ordinary first-open/CRUD/restore entry is reserved, admitted generation/results
settle before source-worker native cleanup, and successful maintenance can reopen
only the same previously open source. Public close stays definitive. Strong local
source blockers preserve before/after-close and resume uncertainty; timeout and
cancelled waiters cannot clear ownership. Resume checks original native identity on
the worker, and cancelled successful resume settles admission before redelivery.
No new callback/IO authority, installed TTS config binding, schema or migration
policy was introduced. Existing ADR126 (phase14 ruling79 note), ADR028/040/051/023
apply; no new ADR. Changed production is only TTS/profile_repository.py, with a
focused20-case test module, ADR126 and source/remaining-owner inventory updates.

Actual REDs cover paused first-open executor allocation, paused real CRUD success,
post-native-close false local drain, queued first-open typed refusal, late worker
source substitution, ongoing/cancelled resume and failed-resume native uncertainty.
Fixture corrections compare preexisting temporary entries and TTSProfilePage.profiles;
intentional sticky failures run only in private children. Final focused lifecycle/
shared run57passed, ordinary affected domain116passed with3 host SemLock allocation
failures before repository code, and reference/backup integration96passed. A small
stdlib-only spawn Event probe independently reproduces OSError28; no host cleanup.
Final source census1passed. Earlier architecture23passed/3failed: intermediate
resume open-call census drift now removed/rechecked, plus the two unchanged TTS
voice-path literals carried to the immediate voice/backend Task10 cohort. Do not
report the profile guard Green. Static/self-review evidence and exact overlapping
commands/results are in the dual task-10-phase14-report.md.

INCOMPLETE — phase14 foundation complete, Task10 continuation required. Exact
finite migration/backup/reference/BLOB/materialization/bundle native scopes, ordinary
backup failed-close retention, app-owned source/config registration, dirty editors,
service/runtime jobs and full startup/responder aggregation remain immediate
Task10 work (phase14b), not passive exclusions or Task26 deferral. Independent
native lock observations distinguish known diagnostic-child startup from actual
production startup. No Complete/source qualification, full suite, live generation,
network, user data, shared environment mutation, merge or publication. Status stays
In Progress and all three ACs remain unchecked for controller whole-task review.
