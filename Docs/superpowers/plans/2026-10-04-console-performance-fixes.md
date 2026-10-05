# Console performance fixes implementation plan

> For agentic workers: use systematic debugging and test-driven development for each independent fix; integrate and request a whole-branch review.

**Goal:** Remove measured Console pauses and native refusals, verify complete captured conversations on all three platforms, and create one PR against dev.
**Architecture:** Preserve existing storage admission and owned-operation boundaries. Deduplicate layout, batch finite reads, reduce refresh fan-out with fenced snapshots, and qualify native Windows reuse with differential evidence.
**Tech stack:** Python 3.12+, Textual 8, SQLite, Windows NTFS facade, targeted pytest/native CI.
**Spec:** Docs/superpowers/specs/2026-10-04-console-performance-fixes-design.md
**ADR required:** yes for Windows reuse/owner changes and confirmed trace GC contract changes.
**ADR path:** backlog/decisions/126-complete-local-backup-and-recovery.md and backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md.
**Reason:** Evidence reuse and trace revision lifetimes are governed runtime/security contracts. UI batching/layout preserves existing contracts (no new ADR).

## Global constraints
- Keep private path ownership, native qualification, maintenance, epoch/provenance, scope revision, and cancellation lifetime checks.
- Targeted tests only; do not change unrelated dirty main checkout files, raise performance ceilings, or suppress normal timers to pass.
- Use token-backed existing UI geometry; verify actual rendered recovery-row geometry.
- Every task begins with a reproduced failure and ends with passing meaningful checks and concise notes.

## Review focus
- ACL/owner/path changes between reuse observation and counting must run full derivation and retain original refusal.
- Character/profile/session changes while worker reads finish must reject stale publication.
- Cancellation must not close a worker connection still in use.
- Startup GC must not delete the exact admitted revision before provider reservation.
- Repeated recovery state must avoid CSS work while external geometry mutations are corrected.

## TASK-34403: Control layout and complete native performance verification
Files: Widgets/Console/console_control_bar.py; Tests/UI/test_console_recovery_height_idempotence.py; Tests/Performance/test_console_native_pause_probe.py; .github/workflows/console-pause-native-evidence.yml; qa/console-pause-investigation-2026-10-04/.
- [x] Count stylesheet work for unchanged real control state; run to confirm failure.
- [x] Add class/inline-constraint equality guard; test recovery visible/hidden geometry and externally changed constraints.
- [ ] Integrate independent fixes, investigate remaining sampled stalls, and add ratchets for complete native sends without provider latency.
- [ ] Run targeted regressions and three-platform native receipts including ordinary timers/GC; record exact source, counts, timing limits, and ownership negatives.
- [ ] Verify live DeepSeek three-turn conversation on the completed source.
- [ ] Review all work, update task evidence, and create/attach the combined PR against dev.

## TASK-34404: Windows native admission and SQLite sidecar ownership
Files: Utils/windows_files.py; Backup_Recovery/storage_admission.py; DB/private_sqlite.py as necessary; focused native and differential tests; ADR-126.
- [ ] Reproduce elevated-runner owner/default-token behavior and NTFS change-time invalidations using native receipts.
- [ ] Write RED tests for safe Windows reuse and owner fix; retain rejection of foreign/shared objects and pause/provenance mutations.
- [ ] Implement minimal native fix and measured safe reuse; preserve exact full-derivation refusal/fallback semantics.
- [ ] Run targeted native oracle tests and report handle count reduction and remaining risks.

## TASK-34405: Character and availability finite read batching
Files: UI/Console_Modules/character_context.py; UI/Console_Modules/workspace.py; relevant focused tests.
- [ ] Reproduce owned connection/helper fan-out with real finite reads; write RED batching checks plus interleaving authority/revision negatives.
- [ ] Batch under one finite owned callback and remove redundant runtime-binding reads with unchanged retirement and freshness boundaries.
- [ ] Verify all existing character scope and workspace publication/cancellation tests, lint, and helper count reduction.

## TASK-34406: UI refresh fan-out and startup trace GC race
Files: UI/Screens/chat_screen.py and Console modules excluding character_context/workspace; Chat/console_trace_service.py and DB trace repositories as necessary; focused tests; ADR-097 for lifetime amendments.
- [x] Reproduce refresh fan-out and deterministically confirm/rule out TASK-33621.47 startup GC race.
- [x] Add failing checks for reduced reads and scope-fenced publication; fix confirmed GC revision race with appropriate lifetime protection.
- [ ] Implement minimal shared refresh deduplication and bounded worker reads, preserving fresh post-await snapshots and live status.
- [ ] Verify targeted capture/retry/refresh/GC/maintenance regressions and identify any remaining broader app performance root.

## Execution rulings
The user's explicit instruction is to fix the identified and broader defects and open the PR. Continue authorized implementation without a new design/execution permission loop. Independent subsystem work is delegated under dispatching-parallel-agents; integration, layout, native evidence and final PR remain with the primary agent. Targeted native CI is part of verification. Preserve main checkout and all other task edits.


## Registered integration budgets
Before the final source measurement, register each complete captured send at <=15s and each send-phase UI heartbeat stall at <=1s. Each send permits <=200 main-thread storage acquisitions, <=40,000 real native Windows file opens or <=16 POSIX private-SQLite helper starts. These are bounded regression checks for the real immediate-adapter flow with ordinary timers enabled; no existing ceiling is raised. The first combined Windows receipt (before persistent presentation sharing and native tree observations) is RED against these limits: 53.23/51.05/40.66s sends, maxima 8.58/9.06/8.10s. Earlier Linux/macOS native sends were about3-4s. Remaining failures must drive another root-cause fix, not larger limits.


## TASK-34367.1 through .5: Existing provider audit reconciliation
- [x] Reproduce each documented complete/SSE failure through its actual adapter with only the HTTP boundary mocked.
- [x] Reconcile provider-specific profiles and normalization; retain nested Groq streamed usage and errors, reject conflicts and unknown/malformed fields.
- [x] Run affected provider, generic preset and strict hosted-parser regressions; update each task and the audit with precise qualification.
- [ ] Include these fixes in the final combined review/PR and native whole-source verification.
ADR required: no new ADR for reconciliation within existing provider-specific contracts. ADR path: ADR-179 and ADR-062/063; amend an existing ADR before implementation if a new normalization contract is required.


## Additional measured background root: scheduler stop-path resolution
The second complete native receipt samples Scheduling.scheduler.loop._emergency_stopped resolving default_emergency_stop_path on the event loop before offloading only the sentinel read. Reproduce actual resolver thread identity and fresh real sentinel transitions; move resolution and read into one existing scheduler-owned finite offload, retaining fail-safe errors and canceled-worker drain ownership. Verify scheduler stop/maintenance/native-pause targeted tests. ADR required: no. ADR path: N/A (existing scheduler-owned offload and emergency-stop contract). Reason: move an existing blocking lookup into its already-owned worker without new authority, cache or dispatch policy.

## Latest dev integration
Provider fixes were committed and dev49206beea9 merged before final source measurement. An auto-merge duplicated Together choice_allowances; the duplicate narrower keyword was removed, retaining the documented superset and target stream_include_usage=True. The merged provider/discovery/live-fixture regression selection passed644 tests. Final UI/native integration remains pending.

## Additional measured background root: subscription backfill starter
The third native receipt samples app.start_subscriptions_fts_backfill resolving get_subscriptions_db_path and execution_allowed on the UI loop before starting its thread worker. The actual _backfill_subscription_items_fts worker already resolves the path and enters a fresh execution_scope. Reproduce starter thread identity and preserve the worker refusal control; remove only the duplicated UI preflight. Verify existing stagger policy and actual FTS driver tests. ADR required: no. ADR path: N/A. Reason: preserve the existing worker-owned admission policy and move redundant blocking setup off the event loop.

## Additional measured root: MCP catalog store reads
The fifth valid native receipt identifies synchronous permission-store reads in the main-loop MCP composition and its controller preflight. Keep the async catalog service and provider publication on the caller's main loop, and move only documented thread-safe synchronous get_kill_switch/get_inventory/effective_tool_states reads into finite workers. Capture profile kwargs before the await; do not memoize permission payloads. Verify native source retirement on cancellation, fresh deny/kill transitions and actual thread identity plus existing provider/controller regressions.
ADR required: no
ADR path: ADR-126 and the existing MCPToolProvider threading contract
Reason: execute existing worker-safe file methods on their already documented permitted thread; no new authority, transport or persistence boundary.

## TASK-34404 bootstrap creation retry amendment

ADR required: yes. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: preserve exact created-entry durability across failed initialization and
process death while avoiding unrelated Windows ancestor barriers.

1. Retain the real Windows rights-denial retry RED and original HEAD _pending-only
   control; label that scoped comparison separately from the earlier whole-HEAD run.
2. Add a bounded strict caller-child/actual-parent creation intent in control_records
   only. Use private native file creation, actual exclusive file locks, named binding
   checks and full file/parent barriers before creation and intent retirement.
3. Recheck exact intent names throughout the requested lexical chain on retry;
   refuse live, incomplete/corrupt, foreign, moved/substituted records and changed
   parents. Preserve external FileExists acceptance only after its exact entry barrier.
4. Verify actual crash and required-barrier failure retry, no lower/public mutation
   before required completion, and finite cleanup after both success and failure.
5. Re-run bounded native tests, lint and the three original mounted restore controls
   in the parent's coordinated window; then freeze all changed source for final probes.


## Additional measured root: finite paired-witness control observation
An actual installed native Windows MCP permission source made 3,003 file opens for one get_kill_switch: 18 selected-path queries and 19 witness reads. Each witness repeated three validated control-record reads and two registry reads; direct raw directory loops contributed only 56 opens. TASK-34403 AC7 requires one fresh validated record/registry observation per witness call, shared only by that call's source-scope, startup and paired-generation decisions.
- [x] Add a real native read-count RED and differential refusal fixtures using actual records and native leases.
- [x] Add a fault barrier that writes pending evidence during the actual observation and prove refusal.
- [x] Extract existing decision helpers, preserve public fresh readers, and retire the finite observation before returning. Recheck native source identities/change stamps before publication; no cross-call cache or empty-witness shortcut.
- [x] Verify source/generation/native regressions and freeze precise source hashes for independent review.
ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: fresh metadata sharing changes a governed recovery-reader boundary; the amendment retains all native/source and paired-generation rules and adds an observation-completion fence.

## Additional measured root: native source validation under coordinator lock
Actual installed MCP source validation performs its selected-generation native reads inside raw._check's storage coordinator lock, despite the established prohibition on disk I/O under that lock. Observe the original guarded functions by code-object profiling, block one actual witness-read boundary, and verify another thread can acquire the coordinator. Move only the potentially blocking bound-source proof outside the lock, with unchanged pure issued-operation/lease checks before and after it and exact state/participant/source correspondence. Preserve pause, wrong-thread/task, source retarget and native retirement refusals with differential race controls.
ADR required: no new ADR
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: restore the existing no-disk-I/O-under-coordinator invariant and retain current source and native ownership gates, without adding authority or observation reuse.

## Continuous installed MCP custody follow-up (TASK-34403 AC8)

The corrected real permission RMW pause fixture fails on literal HEAD and the coordinator repair: repeated selected-generation proof requests a new acquisition after pause. Seed actual bytes and qualify the original failure, then retain only the exact installed raw operation's canonical observation lease before acceptance. Check issued source/participant/process/thread/task/native custody before and after each fresh observation; retain existing exact execution-context path checks and all child, foreign, retarget, native-demotion and pause refusals. Move remaining native participant proofs outside coordinator sections with exact metadata/closed/pause rechecks. Verify accepted RMW completion and negative ownership/cancellation cases alongside the four coordinator race controls; freeze source for native integration.
ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: a finite retained canonical observation interface replaces an invalid new acquisition during already admitted work without adding authority or caching witness/permission data.


### Finite witness observation verification receipt (frozen production)
The focused native RED reported two failures and eleven passes: three record/two registry reads instead of one pair, and pending evidence inserted after the actual registry read still returned a witness. After independent review repairs, the final Windows bundle passed 61 cases with two intentional POSIX-policy skips in 38.76 seconds, covering actual paired/shared generations, required-owner damage, surviving record loss, registry publication, real replacement, mutation during observation and fresh empty queries. The skipped POSIX shared-parent and containing-ancestor rename cases remain native Linux/macOS CI obligations.

An isolated actual paired native lease counted legacy 119 native opens (three records/two registry) versus bundled 89 (one/one). A separate installed-config/MCP comparison kept storage/raw/config/MCP/native source hashes unchanged: get_kill_switch 3,003 to 2,414 opens, load 2,561 to 2,065, selected_path 142 to 111. Permission checks still perform eighteen selections/nineteen witness queries; this change does not cache across them. Single-run unprofiled timing is diagnostic, not a whole-Console budget receipt.

Frozen SHA256: generation_witnesses.py `10d4686ede16826b880c39e2cee49ec3b4a2451a06f3b470363c570abb5d189d`; bootstrap.py `624249b195edb3524b5bd44a78c2ec1f1bd3df00889f6f788d517991116a49ef`; activation.py `242db8cbd6e6e9c23f5c652fbb0dfe02a16ae4ac8416e8c5f43149d4a5e94fe0`. Formatter passes. The source Ruff diagnostic multiset matches HEAD exactly (fourteen existing strict type-comparison/lambda diagnostics); new generation code/tests have none.

The broader exploratory targeted run was 147 passed/53 failed, a classification receipt rather than a clean regression claim. Windows-only fixture failures include unavailable symlink privilege, select on a pipe, fork and POSIX chmod expectations. A verified pristine c78b9dd81c whole-source baseline reproduces the representative projection first-search failure and incomplete-journal startup refusal. The edited-selector baseline is blocked earlier by its known Windows directory-flush rights refusal; actual current fixture bytes instead prove its intended LF-only post-restore edit was a CRLF no-op (no edit marker, current hash still matches profile). Normalizing that fixture boundary retains all assertions and makes its current-generation case pass in 8.70 seconds. The four projection TOML path serialization edges were independently proven invalid on Windows and corrected identically in baseline/fixed fixtures; the remaining first-search behavior is unchanged and remains outside this witness fix.

Local receipts: witness-red.xml, witness-final.xml, witness-review-repair.xml, witness-ancestor-red.xml, witness-edited-selector.xml, witness-regression.xml, witness-baseline.xml, witness-native-dd40867eb5/result.json and raw-source-profile-6ad584e3cc/result.json under the authorized private console-pause-34402 evidence directory. Native matrix, independent review, mounted restore and whole captured-send budget verification remain required before the combined task is done.


### Independent witness completion review repairs
Independent read-only review found that POSIX completion used held directory metadata without rechecking its actual lexical name, and Windows ignored ancestor policy receipts that its native tree had already freshly observed. The real Windows final-snapshot mutation added a World full-access DACL to an isolated containing ancestor, verified its shared mode, and reproduced a one-failure RED (1.48 seconds) without replacing OS or guard callables. The repaired Windows branch requests the same native snapshot's observed paths and ancestors, applies the existing private-path owner and writable/sticky rules to fresh directory receipts, then compares the original fixed record/root/marker/registry stamps. It adds no native tree nodes: final paired witness opens remain 119 to 89, three/two reads to one/one.

POSIX completion now reopens admission then bootstrap root through the existing verified private native walk while the original pins remain live, comparing actual named device/inode against held device/inode before publishing stamps. Its native control renames a common containing ancestor after both completion pins are acquired, leaving the descendant inodes/stamps unchanged; this exact scenario is intentionally skipped on Windows and must run on real Linux/macOS. No os.name spoof or fabricated filesystem receipt is used. The final local repaired bundle is 61 passed/two platform skips (38.76 seconds); the Windows DACL mutation is GREEN and all final source hashes are recorded above. Both POSIX controls and overall cross-platform custody/performance remain pending native CI.


### Continuous installed MCP custody verification receipt

The corrected real installed permission RMW fixture seeds True then False to ensure actual bytes and a reached pause barrier. Literal HEAD raw checker reproduces the accepted RMW new-acquisition refusal (one failure in 4.80 seconds); no source guard is replaced. The remaining coordinator entry proofs reproduce two failures in 5.41 seconds against all three literal HEAD production files. Removing only the final exact source/canonical fields from retained custody reproduces two failures in 4.87 seconds at a real StorageLease.execution_context return barrier. Each temporary baseline restores exact current bytes in finally. Malformed fixture and missing-helper/import attempts are excluded as evidence.

The final formatted targeted bundle passes 21 cases in 43.45 seconds, normal exit zero: four original coordinator lock/revocation controls; actual participant/scope native-reader entry checks; maintenance closure/drain/resume during accepted pause; exact wrong-path native lease refusal; final source/canonical mutation fences; seeded accepted RMW and same-path foreign refusal; installed/custom and profile/native demotion; foreign thread/task/sibling; and actual canceled native worker custody. Actual accepted continuation makes no new acquisition after pause. No selected-generation lease is broadened to another canonical path; each generation observation remains fresh, uncached and owned by the same issued operation. An actual distinct restored-generation integration and native cross-platform matrix remain final qualification obligations.

Scoped production and the new coordinator regression module pass Ruff; new module formatting and diff whitespace checks pass. Source-lifetimes Ruff diagnostic multiset is identical to HEAD (13 existing diagnostics after normalizing only embedded shifted definition line numbers). Production/test source is frozen for coordinated mounted and full native measurement. SHA256: raw_participants.py 451bde0e5a47aa7371612f8b349c45aed65c7e8067ef1263f9c9cc80e8d883e5; mcp_source_participants.py 809212856647086100231fb962d57983c05ae5d2b36c6ac5fdf690af95a9f770; MCP/recovery_activation.py f120eb3747210791ca513ab2c59fd75bf0f30cd7ed3c36d9ff3a41aca82d1b23; test_mcp_source_lifetimes.py 78a0f3e735d858a0fa80968e5a048eb3ef9a96f3d3acb4828c479930150ce8ee; test_raw_source_coordinator_io.py 119040923116b0aae7433651c781b7d680e794a8c6cefbf8f99c5f4f26f51ce2. ADR-126 and TASK-34403 AC8/plan/notes record the custody boundary; task stays In Progress pending native matrix, whole-Console budgets, review and combined PR.


## Additional measured root: duplicate checked display policy reads (TASK-34403 AC9)

The exact-source seventh receipt counts fourteen context-control presentation config
lifetimes on send one and three (4.526 and 3.326 cumulative entry seconds), alongside
thirteen/twelve checked readiness config reads. These inclusive worker timings may
include lock contention and overlap native child costs; they are not summed wall time.

1. Reproduce the redundant lifetime with a real controller/database and original
   config operation observed by code object, retaining a fresh live-action control.
2. Capture immutable sparse CLI Console policy alongside settings inside the existing
   readiness worker, between both checked source tags; do not substitute merged
   application defaults for the CLI lookup semantics.
3. Let only the screen-owned presentation reader share the published exact source,
   generation, profile/session/workspace/settings owner. Cold modal warming waits for
   the checked worker; expiry keeps the existing one-second refresh and one pending
   worker. Source/owner changes reject pending results and invalidate display data.
4. Preserve every live controller/action/dispatch policy read. Verify source-edge/ABA,
   payload/owner/expiry/cancellation and closed modal fences, then freeze leaf source.

ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: sharing checked data across two display readers changes a cross-module
interface; it must remain disposable presentation data and never admission authority.
ADR-097 ratchets and registered whole-send budgets remain unchanged.


### TASK-34403 AC8 finite MCP source metadata repetition

The exact 0ecd8327 native seventh capture remains RED. An actual installed seeded permission read performs eighteen bound selections and nineteen fresh witnesses, seven full raw checks, twelve acquisitions and 1,930 native opens on an isolated shorter private path. Remove only five redundant selections: immutable actual-kind owner classification in members, preflight and the history-directory branch; and each same-source nested scope's duplicate path selection immediately after its full fresh raw check. Retain every full raw source/native check, exact selected_read refusal and final acceptance binding proof; no witness/permission cache, altered lease selection, skipped native checks, helper authority or timer changes. First reproduce the real native count RED, then qualify the prior 21 custody/retarget/refusal controls plus actual restored-generation integration. Parent owns storage evidence and DB changes; config combined acquisition is deferred.
ADR required: yes (existing amendment), ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md. Reason: document pure owner metadata classification and use of the exact just-validated scope selection without introducing an authority/cache boundary.


### TASK-34403 AC10 consecutive same-path repository proof

The actual installed EventStateRepository nested operation performs two full native checks of the same previous.path under one coordinator interval. Qualified RED: one failure/two control passes in 1.64 seconds, with real Windows native open/security call-through counts recorded. Keep the first fresh previous.path proof always; omit only the immediately consecutive second proof when the participant and exact target path match. A differing target path retains both previous and target checks, independent owners receive independent admission, and restoration/retirement/cancellation fences remain unchanged. Verify the actual path-count regression plus cross-owner pause, retarget/provenance/thread/task and established independent nested failure controls. Scope is only storage_admission._repository_operation; parent independently owns lower-file creation-intent evidence.
ADR required: no new ADR. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md. Reason: remove one identical consecutive check without moving, caching or weakening the existing native proof or authority boundary.

## Native matrix follow-up: creation inputs and fixture ownership

ADR required: no new ADR
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: complete existing admission evidence inputs and correct test fixture construction; production security and platform boundaries stay intact.

1. Add exact ancestor creation-intent paths to admission evidence after a genuine warmed reuse/full-derivation divergence RED; verify unchanged refusal and preserved marker bytes.
2. Run the shared Windows tests with a test-only native TokenOwner=TokenUser launcher, restore the original owner in finally, refuse required elevated-custody mode, and verify real stdlib and child-created fixture owner. Keep the separate three Python custody jobs on the unmodified elevated token.
3. Add call-site diagnostics for macOS tracer relative names without suppressing any dependency; qualify the exact inputs on native macOS.
4. Run all three actual native probes even if an earlier control is red, retain red job status and evidence, and rerun the complete bounded matrix at a reviewed exact source commit.


### Checked sparse display policy verification receipt

The real-controller/code-object control reproduced the duplicate display lifetime
(1 failed in 4.62 seconds); no config/native/guard method identity was replaced.
Shared display now makes one checked config operation and zero live global-policy
calls; a live context action independently adds its own checked operation. A forged
pure display policy leaves both the live action and actual compaction dispatch
preflight unchanged, each making a fresh policy read.

Final focused Windows bundle: 51 passed in 36.63 seconds, normal exit zero, across
shared-policy/readiness/context/settings-owner modules. It covers both wrong source
tags, an actual checked native B callback while the cheap UI selector changes A/B/A,
source changes during an actual DB read, actual session/workspace/settings/closed
owners, one-second expiry with one shared pending worker, cancellation ownership,
and actual cold/closed settings-modal paths. Two initial unmounted modal fixture
failures incorrectly assigned the runtime-backed store property and instantiated an
unrelated incomplete runtime; only that test fixture property seam was corrected.
The modal expiry control additionally reproduced a 1-failure RED in 4.47 seconds:
a failed refresh must not satisfy a cold modal wait with expired same-owner data.
Its new completion age fence is GREEN in the final bundle.

Spend projection and new test Ruff/format pass. Controller Ruff matches all 61 HEAD
diagnostics exactly; unrelated baseline formatter differences were restored, and
the edited presentation method independently matches formatter output. Diff whitespace
check passes. Frozen SHA256: spend projection
`685e0378c4da7340845dd394a27940c1d5ffed17a2fa49aeef40b7adf0968ccd`;
new test `02eed79b51154fa506709f89ac7d1646ef630777aec0d9966f8024f2f902313c`;
presentation method text `b4ce65b79c64c943a47049c514ddef9da018407cd1458f1dbf26f861096653a2`.
The controller file is concurrently owned by a separate agent only at its nonoverlapping
finite DB callback method; final combined full-file hashes belong to integration.

Receipts in private console-pause-34402: shared-context-red.xml,
shared-context-expiry-red.xml, shared-context-final.xml. Native Linux/macOS, independent
review and registered whole-send budgets remain integration obligations. No timing
ceiling, timer cadence or live authority cache changed in this scoped fix.


### Frozen finite-repetition receipts

MCP native RED1/control1pass7.04s to GREEN2/6.22s: source selections18 to13, fresh witnesses19 to14, full raw checks unchanged7, native opens2,304 to1,903 and security1,206 to924. The final original custody/retarget/cancellation plus real restored distinct canonical observer bundle passes24 in57.73s; the final lint-clean fixture import module passes2 in6.08s. The fixture aliases remain the same original actual-source fixture objects.

Repository native RED1/control2pass1.64s to GREEN3/1.45s: same-path checks2 to1, opens16 to8, security2 to1. Differing targets still show both actual checks; admitted same-owner continuation across pause and foreign-owner refusal stay green. Latest combined core provenance/thread/task/copy/stale/path, nested rollback/cancellation and native MCP controls passes18 in11.32s, normal exit zero. Scoped Ruff, both new-module formatters and diff whitespace checks pass.

Source/tests are frozen: raw428e8977067b1c2f289659dd79c579ae7b3bc80163ac8df916d5e32aa29ec7bb; mcp0d20c7432edfdea70d569d3d6170ae15583bbd294f6deb827fbc5af2fcd6f540; MCPcounttest7c13b10aba986dc0e787f99492fe2ec59eb2adfee163a8ed6fc433c871ed4337; nestedtestcacae210cf374c5b8d89dd8fe8a7879728f8fd79493c36660a9c2f8402fd8dd9. Only the decorated storage_admission._repository_operation span belongs to this fix, hashc645ba34a6f889694202d02ed739465c472231244c762e6410d014db3a9971b3; parent independently owns creation-intent evidence lower in that shared file. XML receipts are mcp-metadata-repetition-{red,green,final,final-import}.xml and nested-repository-check-{red,green,final}.xml under private console-pause-34402. No test/profiler/source writes follow freeze during whole native measurements.

Qualification: isolated installed warm config read_current uses201opens/3raw checks and global policy189/1. Its one pinned parent raw check uses6opens; a fresh stat-many union uses12, so parent union batching was rejected for this case. Actual finite AgentRuns query/newworkerclose uses221opens/2acquisitions/12corechecks; first diagnostic lacking required conversation argument is excluded. Config combined acquisition is deferred because exact-file bound companion ownership must remain valid. The broader whole-Console40k-open/15s-send/1s-UI and cross-platform budgets remain RED/pending; these receipts are bounded syscall reductions, not a claim that global performance is fixed. AC10 and its plan preceded repository implementation; MCP plan/ADR126 preceded implementation, while AC9 was subsequently formalized within already authorized AC8/AC2 scope.


## Bounded ordinary Character and count presentation cadence (TASK-34403)

ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: the named presentation facade introduces a finite observation contract across screen/controller owners; actions keep the existing fresh native authority boundary.

The seventh actual Windows receipt records 19/1/20 count callbacks and 24/0/23 paired Character metadata callbacks over three sends. Active count refresh currently expires after 0.2 seconds, and general UI sync always captures Character metadata before comparing its fingerprint. These are display callbacks, not dispatch permission decisions.

1. Before production, add actual native regression controls comparing fresh legacy observations with repeated same-owner display calls. Count exact metadata pairs, SQL count calls, native opens/helper starts and elapsed diagnostics using original code-object observers, with no guarded/native callable replacement.
2. Use the existing two-second subagent display TTL during active runs as well as idle. Retain immediate row/run/child/bridge/library-owner invalidation and add the exact captured AgentRunsDB to the key, query receiver and final publication fence. Preserve custom/in-memory and live action behavior.
3. Introduce a separately named Character presentation refresh used only by general screen sync. Serialize/coalesce it with an async lock; recompute the ambient owner after lock entry, cache only successful stable generation/owner paired observations for at most two seconds, and retain both real metadata reads and their UI midpoint on each fresh callback.
4. Include exact config generation/selected source, app/config mapping, DB receiver, controller generation, active store/session/workspace/conversation binding/current-character inputs. Publication from a display-triggered refresh additionally checks its captured owner; action refresh, capture/commit/midpoint checks and dispatch remain freshly validated. Do not cache failures or cancellation, and do not accept a retired or replaced owner.
5. Verify real changed-owner/generation barriers, queued waiters, expiry, fresh actions and native retirement/cancellation. Run only relevant suites; native whole-app budgets and three-host acceptance remain parent-owned and pending. No cadence budget, UI token or existing performance ratchet increase.


## Repository native proof outside the coordinator (TASK-34405)

ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: distinguish synchronized issued ownership from each fresh native path observation; preserve exact authority and retirement.

1. Reproduce actual blocked parent stat/resolve under repository nesting/restoration, core access/getter and acquisition initialization using original code-object observers and installed real SQLite owners. Independently attempt coordinator entry; do not substitute guard methods or OS flags.
2. Add pure issued-state/installed-owner/lease/native-hold fences before and after the native proof. Recheck the exact captured operation, repository, path, resolved path, parent identity and live lease before acceptance. Native observations remain fresh per real boundary and run outside the shared coordinator.
3. Move only confirmed callers outside the lock, with final pure state validation before reuse, leader publication or thread-local restoration. Keep accepted same-owner pause continuation and refuse fresh independent owners, final revocation/retarget and copied/thread/task tokens.
4. Preserve the one real same-path nested proof and independent proof for a differing target. Verify blocked-native revocation, existing finite callback/retirement, rollback/cancellation and provenance cases. No cross-call path/permission cache or private RLock release manipulation.


## Exact process-owned live fleet precedence (TASK-34403)
ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: make the display fallback honor its existing live-over-history service contract at the actual bridge ownership boundary.

Summary presentation already prefers non-idle live snapshots; fleet presentation instead queries durable history whenever live handles are empty. The real bridge setup contract explicitly suppresses the previous turn before a current run exists, and its live primary binding is process-local. This both displays prior children under the current run and repeats finite historical reads on live setup/run token changes.

1. Before production, prove actual seeded historical rows remain visible on resumed/terminal/unknown and custom bridges, while actual process-owned setup and bound running primary empty snapshots currently schedule native history and leak previous children.
2. After nonempty live handles, allow only exact ConsoleAgentBridge/AgentRunsDB and either its real setup marker with setup status or a bound actual live primary with running status to render live summaries, including empty. Terminal, restored, unknown and custom fallback remain unchanged.
3. Prove nonempty real fleet handles stay first, inline live children render, current primary rebind/finish restores existing fallback, and native callback counts drop to zero only where live ownership is established. No permission, repository authority or dispatch change.


### Frozen repository coordinator receipts (TASK-34405)

Original installed EventStateRepository and actual issued tokens/live leases, observed at original WindowsOS.stat code boundaries, reproduced native lock monopoly in all five entry/restoration/access/getter/initializing cases and completed-proof counter/lease/participant revocation: 8 failed, 2 existing path refusal controls passed in 3.24s. Initial repair passes 13 in 4.50s. Five supplemental full-proof-return barriers prove final pure publication fences; an initial initializing observer incorrectly stopped path=None pure construction and was narrowed to actual path proof.

The final combined finite callback/provenance/pause/cancel/borrowed ownership bundle passes 56 in 37.63s, normal exit zero. After preserving the original out-of-scope resolve short-circuit, exact-source scope passes 32 in 10.30s, normal exit zero. Native same-path nesting still reports one full proof, eight opens and one security read; differing targets keep two checks. No native callable/OS substitution or cross-call path/permission cache was used.

Frozen production: storage_admission.py 003a9525ebe138c0cdd419669077c28567f890199274d1b821c15533827d025b; participants.py 6cfe600c3b2ba4ada71ce54e53f3033b0f9709e8e97787fe5dea8983c9fea7ac; new test_repository_coordinator_io.py a706f1c1378291e787ded35551b8343c760beca1a8a8dff0a6fd5d6b705dd049. The owned storage classes/helpers span lines56-308 has semantic text hash11ba452095aea6b8801c2e75768e831815daa0b7b948a5469302cc62d30326b7; lower evidence changes remain parent-owned. Scoped lint matches all eight existing storage HEAD diagnostics exactly, with zero in participants/new test; changed spans/new test format and scoped whitespace checks pass. Private XML receipts repository-coordinator-{red,green,final,frozen,final-scope}.xml. Task34405 AC6 and plan/ADR126 amendment preceded production; integration/native budgets/review remain pending.

macOS relative tracer diagnosis is source-based pending native stack evidence: the test real-profile write audit guard sends a leaf-only open event to realpath(raw), and CPython3.12.10 _joinrealpath issues lstat(rawleaf) before final abspath. That guard-origin CWD probe is separate from the admission open already mapped through F_GETPATH. Unknown inputs remain in the tracer and relative_callers diagnostics await native macOS; no local POSIX or fake-platform qualification is claimed.


### Display cadence/live-owner frozen receipts

TASK-34403 AC13/AC15 remain under the combined native matrix/whole-send acceptance. Actual native legacy cadence RED3/7.63s precedes production; initial GREEN3/6.15s. Six warm unchanged Character ticks:6 real paired callbacks/1878opens/0.594s→1/313/0.094s. Active successful count ages0.3s:6callbacks/4164opens/1.109s→1/1124/0.265s, with no 2s TTL change. These are isolated diagnostic timings, not whole-send budget proof. Final new cadence29pass57.89s includes actual owner changes, queued waiter recapture, real SQLite metadata failure/recovery, both pairs+midpoint, actual opened native handle repeated cancellation and fresh action positive controls. Exact standard count callbacks bind the captured AgentRunsDB; custom/file-backed bridge callbacks keep their declared semantics and are captured independently, with the exact receiver/publication fence.

Live-empty actual setup and bound running controls reproduce native historical reads (each1145opens) and setup's previous-child leak; actual inline child projection is also RED. The initial real-handle fixture was corrected to use the actual fleet service seam, handle ID and secondary task field; its unchanged positive control separately passes1/4.05s, and its failed draft is excluded from functional RED. Exact standard disk bridge chooses process-owned setup or bound running snapshot after nonempty handles, allowing meaningful empty; terminal/restored/unknown/custom fallback remains native. Final live suite is separately recorded below.

Independent review exposed same-ID field-equal actual session replacement accepting the old checked display owner, custom bridge override bypass, and repeated readiness cancellation setting settled/pending early while an actual raw state and two leases remained live. Actual local RED2/8.03s plus corrected same-shape context owner RED1/4.72s; repeated native cancellation RED1/5.98s. Owner keys now retain id+strong session/config mapping references, preserving readiness's fixed session index5; default convergence excludes only settings revision index7 and compares every other owner field. Checked readiness cancellation loops until its actual worker retires before settling, with cancellation precedence and no publication. Final59pass65.65s includes the original eligible pristine convergence, before/after actual source tags, same-ID pending replacement, modal/source/ABA/expiry, fresh action isolation, custom semantics and exact standard count receiver barriers. Earlier83case run's one real owned-convergence failure (index shift) was repaired and the original control rerun GREEN; not claimed as a clean earlier receipt.

The related mounted Character/Agent modules produced79pass/5fail100.81s: three source-lifetime fixture failures before changed code, one removed-widget identity assertion, one global mounted count-spy ordering assertion. These remain unqualified, with exact IDs/traces delivered for isolated original0ecd baseline review; no assertions weakened and no baseline pass inferred. Existing refresh/paired metadata control bundle is green (41pass70.16s with19cadence-owner deselections), not a full mounted pass.

Final live-empty native receipt:8pass10.01s normalexit0, including actual setup/running empty, current inline child, nonempty real fleet handles and all four terminal/restored/unknown/custom positive controls. Literal live setup/running display observer now0historicalcallbacks/0nativeopens/0helpers. Source and tests frozen for parent checkpoint; no local full app or tests remain active.


Native macOS observer follow-up (TASK-34404 AC7): run 37244879946 captures exact real-profile audit hook -> _protected -> CPython realpath relative lstat probes, separate from actual descriptor-relative admission reads. Classify only exact installed code identity and the actual raw relative write-open input; retain unknown relative reads (including identical control-like names) in dependency completeness. Add actual macOS descriptor-write and same-name raw-read controls. ADR required: no. ADR path: N/A. Reason: test-only attribution correction, no admission or profile guard behavior changes. Original three native dependency cases are qualified RED; native positive/negative classification and original oracles must pass in the next full three-OS checkpoint.
