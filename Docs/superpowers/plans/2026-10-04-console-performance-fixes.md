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

TASK-34403 AC13/AC15 remain under the combined native matrix/whole-send acceptance. Actual native legacy cadence RED3/7.63s precedes production; initial GREEN3/6.15s. Six warm unchanged Character ticks:6 real paired callbacks/1878opens/0.594sÃ¢â€ â€™1/313/0.094s. Active successful count ages0.3s:6callbacks/4164opens/1.109sÃ¢â€ â€™1/1124/0.265s, with no 2s TTL change. These are isolated diagnostic timings, not whole-send budget proof. Final new cadence29pass57.89s includes actual owner changes, queued waiter recapture, real SQLite metadata failure/recovery, both pairs+midpoint, actual opened native handle repeated cancellation and fresh action positive controls. Exact standard count callbacks bind the captured AgentRunsDB; custom/file-backed bridge callbacks keep their declared semantics and are captured independently, with the exact receiver/publication fence.

Live-empty actual setup and bound running controls reproduce native historical reads (each1145opens) and setup's previous-child leak; actual inline child projection is also RED. The initial real-handle fixture was corrected to use the actual fleet service seam, handle ID and secondary task field; its unchanged positive control separately passes1/4.05s, and its failed draft is excluded from functional RED. Exact standard disk bridge chooses process-owned setup or bound running snapshot after nonempty handles, allowing meaningful empty; terminal/restored/unknown/custom fallback remains native. Final live suite is separately recorded below.

Independent review exposed same-ID field-equal actual session replacement accepting the old checked display owner, custom bridge override bypass, and repeated readiness cancellation setting settled/pending early while an actual raw state and two leases remained live. Actual local RED2/8.03s plus corrected same-shape context owner RED1/4.72s; repeated native cancellation RED1/5.98s. Owner keys now retain id+strong session/config mapping references, preserving readiness's fixed session index5; default convergence excludes only settings revision index7 and compares every other owner field. Checked readiness cancellation loops until its actual worker retires before settling, with cancellation precedence and no publication. Final59pass65.65s includes the original eligible pristine convergence, before/after actual source tags, same-ID pending replacement, modal/source/ABA/expiry, fresh action isolation, custom semantics and exact standard count receiver barriers. Earlier83case run's one real owned-convergence failure (index shift) was repaired and the original control rerun GREEN; not claimed as a clean earlier receipt.

The related mounted Character/Agent modules produced79pass/5fail100.81s: three source-lifetime fixture failures before changed code, one removed-widget identity assertion, one global mounted count-spy ordering assertion. These remain unqualified, with exact IDs/traces delivered for isolated original0ecd baseline review; no assertions weakened and no baseline pass inferred. Existing refresh/paired metadata control bundle is green (41pass70.16s with19cadence-owner deselections), not a full mounted pass.

Final live-empty native receipt:8pass10.01s normalexit0, including actual setup/running empty, current inline child, nonempty real fleet handles and all four terminal/restored/unknown/custom positive controls. Literal live setup/running display observer now0historicalcallbacks/0nativeopens/0helpers. Source and tests frozen for parent checkpoint; no local full app or tests remain active.


Native macOS observer follow-up (TASK-34404 AC7): run 37244879946 captures exact real-profile audit hook -> _protected -> CPython realpath relative lstat probes, separate from actual descriptor-relative admission reads. Classify only exact installed code identity and the actual raw relative write-open input; retain unknown relative reads (including identical control-like names) in dependency completeness. Add actual macOS descriptor-write and same-name raw-read controls. ADR required: no. ADR path: N/A. Reason: test-only attribution correction, no admission or profile guard behavior changes. Original three native dependency cases are qualified RED; native positive/negative classification and original oracles must pass in the next full three-OS checkpoint.


## Character browser message-pump lifetime follow-up (TASK-34403 AC17)

ADR required: no
ADR path: backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md
Reason: restore the framework-owned message-pump task without changing the existing controller, presentation, action, or surface ownership contracts. This is a mechanical lifetime bug fix under ADR-120. The design language and component-pattern documents were read; no visual values, CSS, geometry, interaction copy, or keybindings change.

1. Qualify the unchanged removed-browser assertion on actual current and isolated original source, and observe the real Textual task ownership.
2. Add a controlled suspended controller refresh: awaited removal must finish while that unrelated read remains held, and late state must preserve the exact prior state identity.
3. Rename only the widget-owned controller-task slot so Textual retains its own message-pump task and removal awaits the correct task.
4. Verify the original removed-browser assertion plus mounted Character/controller ownership tests. Qualify three real-app config-selection fixture failures and give only those nodes the canonical bootstrap_profile marker; keep their behavioral assertions unchanged.
5. Record actual regression, fixture qualification, and scoped verification separately; the combined native matrix and whole-send budgets remain open.


## Actual finite Settings writer lifetime (TASK-34405 AC8)

ADR required: no new ADR
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: direct implementation of the existing finite Notes callback boundary at the actual generation and context-policy persistence writers.

An original-code native lifetime census reproduces one ordinary worker SQLite lease surviving the new shared-policy settings mutation fixture and causing later pause drains to remain false. The existing Settings generation writer and context-policy writer use raw asyncio.to_thread callbacks; neither closes its newly opened thread-local Notes handle. Keep the actual persistence fixture and all pause assertions.

1. Add typed real Settings submission regression tests and original-code observers for both standard native writers; reproduce live worker handles after successful CAS and native pause/cancellation ownership before production.
2. Capture the exact standard ChatPersistenceService bound writer and file-backed CharactersRAGDB before scheduling; context writes additionally require the exact same-DB ConsoleContextRepository. Qualify only the original installed writer methods.
3. Count one complete _core_operation inside operation_owned_connection for each qualified callback. Recheck exact captured persistence/database/repository bindings before and after the body and before accepting its result. Native close follows interval exit and retires only newly opened worker handles.
4. Shield each qualified native worker through repeated cancellation until physical retirement; the existing settings drain task and lifecycle ownership remain registered. Preserve CAS reconciliation, committed metadata/policy, source/session publication fences and custom/subclass/memory execution contracts.
5. Verify actual committed rereads, borrowed transactions, same-owner pause continuation and fresh-owner refusal, source replacement, cancellation/error retirement, then rerun the original ordered prefix without global lease clearing. Native Linux/macOS matrix and whole-send budgets remain combined obligations.


TASK-34404 raw-parent hypothesis rejected: the actual permission scope has a single named parent. A fresh stat-many ancestry snapshot increased its opens from1903 to1973 and security reads924 to987 while retaining7checks/14witnesses/13selections. The prospective1100 bound failed; the new helper and bound are withdrawn. This repeats the earlier isolated6-to12-open result already recorded above. No production parent-check change remains; the original checks and whole-Console budgets remain unchanged. Accessor setup ImportError receipts are excluded. Native XMLs raw-parent-native-{red,green}.xml preserve the failed experiment.

AC8 body-time source clarification: actual native duplicated-library RED proves that a standard bound writer can dereference persistence.db after a transient A-to-B-to-A retarget and report success under the restored A binding. The eligible async helper supplies private expected captured database/repository receivers to the actual original standard methods. Those methods reject changed identity at entry and hold the strong captured receiver through every CAS read/update or policy transaction; the exact captured policy writer is retained. Default synchronous and custom/subclass/memory calls retain their existing arguments and route. Verify body-entry and repository-entry ABA with actual duplicated native libraries, original installed code observers, no mutation or worker lease on B, and committed rereads from A.


### AC17 native removal verification and mounted failure qualification

The installed Textual message pump owns `node._task`; `App._prune` awaits that
exact task for removal. The Character widget assigned an unrelated controller
refresh to the same slot. Qualified native current-source controls reproduce the
unchanged late-presentation state-identity failure and a real removal blocked by
a suspended controller refresh (2 failures, 3 canonical fixture controls pass,
52.31s). Identical removal controls against the immutable original
0ecd8327f79c1b14ea18c15f704850cc41b0ec9f archive reproduce both failures in 7.32s;
all loaded project modules are confined to that archive. The production repair
only names the widget-owned slot `_controller_task`, preserving the framework
message pump, existing action/service contracts and original state assertion.

Three real-app Character nodes require the existing node-specific
`bootstrap_profile` fixture contract: their unmarked per-case configuration
redirect conflicts with the actually installed raw source and correctly refuses
before their behavioral assertions. Those three controls pass with the canonical
markers before the widget production repair. No native source or identity guard
is replaced. The native config-sync worker fixture now supplies the exact new
presentation facade and accepts its declared optional arguments; production
wiring already constructs the exact service. Its missing-facade RED is one
failure in 7.06s; the five actual worker/pause/teardown controls pass in 26.76s.

The original broad 79-pass/5-failure receipt is superseded as a classification:
the exact original archive qualification is 4 failures/1 pass, with all 1,510
project/test file hashes matching. The badge node passes that original baseline,
so its current failure remains actionable until a real identical overlap barrier
qualifies the fixture. The final three targeted mounted modules now give 84
passes and only that unchanged badge global-call assertion fails (120.97s).
Its uninstrumented current standalone node passes in 19.70s; that isolated pass
does not erase the mounted failure. Initial setup errors and faulty diagnostic
observer attempts are excluded from behavioral proof.

Production widget SHA256 is
3499dd1f22831f2afd853db308e1709a687c950390748fa449b5a53d52e9e52b.
Scoped Ruff, edited-method formatting and diff checks pass. Textual teardown
warnings remain in the mounted receipt; this is not a warning-free whole-suite,
native-matrix or captured-send budget claim. TASK-34403 remains In Progress.


## Exact checked warm configuration display (TASK-34403 AC18)

ADR required: yes.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: clarify the existing detached, checked display contract and its pure
installed-source identity fence; this supplies no native read or actor authority.

The ninth native receipt identifies 38 general/poll enclosing config entries in
the first send, 2.793s including waiting and native admission. Including coalesced
and image/control callers gives 46 entries/3.168s. Under those enclosing display
stacks, the recorded public warm settings calls cost 0.012s. The selected change
skips only the redundant outer native lifetime when the exact standard readiness
reader has already published its own checked mapping within the original one
second. A stale owner continues the original native route; the projection age,
refresh worker, cold deferral and live/default helper remain unchanged.

1. Add a genuine installed-source passive code-object RED for repeated actual
   warm screen entry, retaining the real config/native callable identities.
2. Only the original standard factory reader may issue a detached display proof
   after its existing checked source-before/after observation. Retain exact
   module, installed raw participant/state, factory reader/loader, strong mapping
   and full owner/source key plus its original UI loop/thread. No operation,
   lease, path authorization or config lock is retained.
3. The separately qualified helper route checks issuance and exact active
   projection/mapping, loop/thread, installed module/raw participant identity,
   closure/local pause, current source/owner and age before and after the
   synchronous body. Custom, injected, replaced, cold or expired state never
   selects this route. Body errors remain visible and post-fence errors retain
   cleanup precedence/chaining. Maintenance replay remains unchanged.
4. Keep every genuine nested guarded reader/writer on its own fresh native
   lifetime. Live actions and dispatch never consume the display proof.
5. Verify zero redundant scopes/opens for real warm entry, genuine default and
   nested disk-reader positive controls, and expiry/owner/source/participant/
   loop/pause/replacement/injected/body-error negatives. Run the unchanged
   original config lifetime and checked readiness/context control suites, then
   freeze for independent review and the coordinated full native probe.

Removing the wrapper unconditionally, raising the one-second age, trusting a
caller flag, borrowing a retired native operation or caching permission/native
decisions are rejected. The proof selects only rendering of its own detached
mapping and does not authorize source I/O.

### AC19: native normal-Send message delivery

ADR required: no new ADR
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md; backlog/decisions/148-console-run-hooks.md
Reason: preserve the existing finite native custody and hook consent contracts while restoring delivery of ordinary keyboard and button messages.

The heartbeat probe measures loop stalls, but Textual separately awaits a coroutine handler on the app or screen message pump. Verify actual Enter and Send-button delivery with the original installed native hook configuration body held in its worker; the source profiler reads original code/receiver identity without replacing a guarded callable. Driver-style input and bounded original pump callbacks must run before the native worker is released. Keep both failed and passing receipts.

If that original path is RED, the standard screen wiring may provide an optional actual-pump task predicate. Only exact app/screen pump callers hand the existing captured snapshot/review/normal Send continuation to a Textual worker before any native await. Direct callers and the spoken worker continue to await their actual outcome. Keep the synchronous busy/dedup gate, original exact chat and draft checks, fresh independent native permission reads, repeated cancellation drain and shutdown ownership. Reuse existing interaction states and worker ownership; no visual, keybinding or permission authority changes. Test default/custom paths and native source lifetime as well as mounted input delivery. Whole-app registered budgets and final native matrix remain pending.

### Badge fixture consumer qualification (TASK-34403, 2026-10-04)

The original badge node passed uninstrumented at exact original `0ecd8327`
and also passed a current standalone run. The earlier mounted bundle failure
therefore remained actionable until its ownership was qualified. Identical
native current and immutable-archive barriers preserved every original
assertion and made the actual mounted browser's native-ID row query complete
between the manual AB/A row reads. Both hosts of the same Windows interpreter
then failed the original last-call assertion, with actual returned mappings
and both ordering barriers reached (29.40 s current, 26.42 s archive). This
proves competing legitimate row-set consumers, not a production count loss.
The archive's 1,510 project/test file hashes had matched exact original source;
the controlled launcher also confined all 1,505 loaded project/test modules
to that archive. The initial overbroad trace and observer-None fault attempts
are excluded from this qualification.

The fixture now supplies only the unrelated mounted browser's declared
`_subagent_counts_for_rows_fn` dependency with an empty mapping. The manual
real AB/A AgentRunsDB, bridge, cache, worker, and every calls/value/count
assertion remain unchanged. Its exact native node is GREEN in the four-node
`checked-display-scope-red.xml` receipt (12.22 s test call; 21.36 s bundle).
No production filtering, global spy suppression, or assertion relaxation was
introduced. The full three-host original module verification remains a final
matrix requirement; the prior mounted bundle was 84 GREEN plus this race.


### Warm display native RED before production (TASK-34403 AC18)

Actual installed-source control `checked-display-scope-red.xml` settled two
expected failures and two positive controls in 21.36 s on native Windows.
Six actual warm ChatScreen display entries opened six original configuration
operations and 1,662 main-thread Windows native handles despite returning
the same checked readiness mapping. The genuine nested disk reader also
observed the redundant enclosing raw operation. The direct default helper
control entered its original fresh native scope and retired normally; the
approved badge fixture control passed. Observation used original code-object
profiles without changing installed guard or native callable identities.
Spend and screen production were unchanged for this RED receipt.

## Exact startup initializing publication fence (TASK-34404 AC8)

ADR required: no.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md (existing startup readmission contract).
Reason: restore the existing exact pause-owned startup route after a new blanket ordinary-pause refusal; no new native admission authority.

The original three mounted Windows restore nodes all fail in 85.993s at the new initializing publication fence. An original-code observer records storage_locally_paused from initializing line279 in each real readmission thread. The exact _StartupReacquisition, coordinator pause and actual worker match; cancellation is false, TokenOwner equals TokenUser, real-profile refusals are zero, and final native counters are zero. The redacted mounted-startup-readmission-red.json receipt retains actual module/source/script hashes and original exception traceback metadata. Original assertions and timeouts remain unchanged.

Preserve actual cancellation, issued pending membership, PID, Thread, Task and operation identity under the existing coordinator lock. Only type(self) is _StartupReacquisition may recheck the original exact class metadata check with no file path; this route requires path=None and operation=None. Ordinary paused callers and subclass/lookalike attempts remain refused. Keep original native I/O outside the coordinator and original scope/publication checks intact. Add source-aware native controls for post-check actor/pause/cancellation/issuance changes and an exact-type unissued impostor; preserve the original mounted three nodes and coordinator race suite. Source changes wait for root's pump RED release, then freeze before serialized native GREEN and the final native matrix.


### AC16 final source and actual UI preparation controls

ADR required: no new ADR. ADR path:
`backlog/decisions/126-complete-local-backup-and-recovery.md`.
Reason: completes its existing finite source versus loop-owned projection and
exact actor contract. No permissions or native authority are cached.

Qualified original-source controls reproduce six borrowed actual method
receivers and an in-place pending attachment ID mutation (7 failures, 34.05s).
The standard worker route requires actual bound MethodType plus original function
and exact receiver. UI preparation captures the attachment IDs before its await
and compares them before the existing atomic runtime acceptance.

Three further actual catalog-body barriers reproduce inventory callback,
field-equal governance and copied exact source-registry binding replacement
being absorbed between separate composition awaits (3 failures, 29.54s).
One ephemeral captured-source keeper now spans the standard composer, retaining
its original catalog callback and all source bindings. Private helper inputs
share that keeper without duplicating reads; only its own lazy permission/log
publication may update its retained binding slot. Custom routes, actual loop
inventory/governance, fresh invocation gates and cancellation drain remain.

Receipts: `async-mcp-borrowed-receiver-qualified-red.xml/log` and
`async-mcp-composition-owner-qualified-red.xml/log` under the authorized
`deepseek-uat` evidence root. Final native module, compatibility checks and
independent review remain pending; these RED receipts do not establish final
performance budgets or cross-platform completion.


### Native observer import-order qualification (TASK-34403 AC18)

ADR required: no new ADR.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: preserves its existing exact checked display/source contract and the
call-through observer; no runtime interface or authority decision changes.

The public loader observer replaces only config.load_settings. If the actual
Console screen was imported first, its unchanged public alias no longer
matches that module attribute, so the display factory cannot issue its proof
and warm rendering returns to enclosing native scopes. Qualify the real early
and late imports in separate fresh private-profile children before changing
the observer. Synchronize only an already-loaded screen alias that is exactly
the captured original public loader, using monkeypatch restoration. Keep all
private enrolled guarded loaders, original budgets, heartbeat and whole-probe
body unchanged. Both controls must warm the original factory, render without
redundant main-thread native entries, and retain a genuinely fresh nested disk
reader and complete native retirement. No wrapper introspection, permission
cache, injected proof flag, source swapping or relaxed identity check qualifies.


Native observer RED qualification: the two fresh children settled in 27.52 s.
Early import retained the old alias and opened 6 warm scopes / 2,058 actual main
native opens; its nested reader also borrowed the redundant display scope.
Late import already produced a real proof and 0 warm scopes/opens. Its separate
failure was a new test expectation error: the original serialized reader enters
three guarded functions, not one. Preserve all three gates and additionally
observe their yielded exact raw operation identity: one new owner must be
shared only inside the real nested read and retired before completion. This
expectation correction is excluded from production regression evidence.
Receipt: checked-display-observer-red.xml/log in the private evidence TEMP.


### Checked display completion evidence (TASK-34403 AC18)

Native Windows targeted receipt checked-display-final.xml/log settled normally
with 104 PASS, 0 failures/errors/skips in 440.12 s. The bundle contains 35 new display
controls, 19 original readiness controls, 18 shared policy controls, 28 original
configuration lifetime controls and 4 observer controls. Every production/test
source hash below matched at start and finish. The ordinary native fixture
launcher restored its actual original TokenOwner on exit; no test process or
finite native worker remained. Native source guards were not replaced.

Six actual warm display calls now record 0 main configuration entries and 0 native
opens, versus 6 entries / 1,662 opens in the original installed-source display RED.
Both genuinely fresh private-profile screen import orders also record 0/0 under
the public call-through observer. The early-import observer RED had 6 entries /
2,058 opens; alias synchronization is restricted to its exact captured original
public loader and monkeypatch restoration. The real nested serialized reader
keeps all 3 original guarded entries, shares only its own newly issued raw owner,
performs actual native opens and retires that owner before returning. The
default helper still owns a fresh native configuration scope.

Qualified identity/view RED receipt checked-display-identity-red.xml/log had
8 genuine failures and 1 positive control before correction: permissive equality,
borrowed key reader, replaced app/DB/store receivers, an arbitrary injected
proof and borrowed/retargeted screen could retain or mis-handle the old view.
Final controls require exact type and issued membership, original real bound
MethodType and receiver, element-wise alias identity, strong exact owner/map
references and the original screen. Spoofed bound attributes keep native
fallback. Original 1 s age, cold/expiry/foreign-loop fallback, source/owner/pause/
closure pre-refusal and post-body coalesced retry, actual concurrent saved
generation invalidation, body-error precedence and repeated worker-cancellation
drain all pass. Live action/dispatch does not consume the display proof.

The first observer RED's late-import failure was solely a new test expectation
error, excluded from production-regression evidence: serialized config reading
has 3 nested original guards, not 1. Its warm display already recorded 0/0. The
corrected control additionally verifies one fresh exact raw owner and actual
retirement. Four final report warnings concern pytest record_property/xunit2;
the existing unset asyncio fixture-loop scope deprecation is also printed.
No application RuntimeWarning was reported in this bundle.

Frozen SHA256 receipts:
- console_spend_projection.py: AEBDC7701C3147E7FC696A9AFEFC85A035DEAD782B699F2FE422F992F26E4221
- chat_screen.py: 283BEAFC8121CEDEEBA121088EF22E0CDDD38FF5E4F4DED3132972A4B91AA2C2
- test_console_checked_display_scope.py: C2F5F23FB5F07041F419A984B2C3F94A2761E3AEA7C7056671384F303F8EB0DF
- test_console_native_pause_probe.py: 0AA03FFB0FE7537FE1CAA6639BAEE5A097C88D866B2C55C26B06F98A9C027F45

Whole captured-send and heartbeat budgets, native macOS/Linux/elevated custody
matrix, final rebased source review and live UAT remain separate pending gates.
This targeted native receipt establishes this display/lifetime fix, not final
performance or cross-platform completion. ADR required: existing ADR-126
amendment applies; no additional ADR is needed for the observer alias repair.


### Hook indicator physical-read ownership (TASK-34403 AC20)

ADR required: no new ADR.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md and
backlog/decisions/148-console-run-hooks.md.
Reason: preserves existing finite native producer and independent fresh Send
authorization contracts. Presentation coalescing is limited to one in-flight
read, with no permission or settled-snapshot cache.

The unchanged eager Allow-all Send-button control failed once in the native
46-case bundle (45 pass), with its original five-second post-answer deadline.
A passive original-code diagnostic also failed: actual visit_snapshot reads
overlap the required Send and approval reads. Its generic contextmanager
wrapper was unintentionally observed, generating 52,583 additional events;
its wall timings therefore remain diagnostic, not performance qualification.
The source independently shows that cancelling refresh abandons its actual
asyncio.to_thread producer. Qualify the original real HookPermissions and
native raw-config body before implementation: concurrent refresh, repeated
cancellation, owner/reader replacement, refresh during Send, and manual review.
Coalesce accepted physical reads, retain cancellation until retirement, fence
strong source identity before publication, and avoid redundant visits while
Send owns its required read. Preserve custom readers, reader error/cancellation
precedence and manual review state refresh. Re-run the unchanged eager control
after the fix; do not change any original deadline or guard.

The three-host CI job ceiling becomes 75 minutes because its expanded targeted
bundle has measured component runtimes exceeding 52 minutes on this native
Windows host. Individual test deadlines and all Console performance caps stay
unchanged; this job ceiling only allows the entire targeted bundle to finish.


### Hook snapshot and error completion (TASK-34403 AC21)

ADR required: no new ADR; existing ADR-126 and ADR-148 apply.
Original-native direct Send/manual review cancellation controls both fail on
premature task completion while real issued configuration leases remain live.
Five custom callback controls also fail: an incompatible source inherits the
old source failure; a surviving joined caller hides publisher failure; and a
deferred presentation scheduler replaces sent, original-error and cancellation
outcomes. Receipt hook-refresh-error-red.xml/log: 7 qualified failures in
20.75 seconds; Native and custom worker controls are distinguished.
Use the same physical task drain in all three remaining snapshot callsites,
retain one shared publication error for joined callers, discard only the old
source's ordinary reader failure after complete incompatible-flight retirement,
and preserve the current waiter's cancellation. Presentation scheduling stays
best effort after busy release; its error cannot change Send. Physical witness
cleanup observes original synchronous snapshot AND complete visit call/return
frames, since both continue beyond the captured configuration lease's exit.
Native GREEN, unchanged eager approval timing and the whole performance probe
remain pending; no original deadline or private source guard is weakened.


### AC16 exact-native custom catalog compatibility refinement

ADR required: no new ADR; existing ADR-126 applies.
Actual exact Unified service with an overridden async local_external_catalog
swaps its local service during the await. The previous provider contract resolves
that inventory receiver after the custom catalog completes. The new joined
source keeper incorrectly treats this as standard-source drift: native RED
async-mcp-native-custom-catalog-qualified-red.xml/log is one genuine failure
in 8.79 seconds, with original permission admission retained. Retain the original
Unified catalog callable at defining-module completion. Use a separate provider
composition qualifier requiring its real original bound receiver in addition
to standard source qualification; custom catalog callbacks retain the preceding
late receiver and worker inventory behavior. Maximum/local snapshot qualifiers
do not depend on this unrelated async callback. Re-run custom and actual standard
provider controls with unchanged native guards and finite custody.

The source-frozen 82-case native receipt had 81 pass and one new descriptor-test
expectation error, 545.82 seconds, with all 15 source hashes unchanged. That test
supplied global_default=off, which is not an accepted store state. Correct only
the new control to a valid deny policy and retain its empty MCP maximum,
alternate actual permission-source/caller-loop and custom-fallback checks.
The 33.57-second three-case follow-up had one standard positive and two new
assertion errors: a store-only probe saw no local-store leases on the custom
catalog route, and a revised assertion confused agent:builtin's independent
resolver with builtin:tldw_chatbook's MCP resolver. Correct the observer to
capture actual permission-body leases and keep valid-deny empty semantics;
these failures provide no additional production-regression evidence.

### AC19-21 focused native hook verification

The native Windows qualification passes all24 selected cases in157.53 seconds
with normal exit0 and eight unchanged source hashes:18 new Hook lifetime/error
controls, five actual mounted Send-pump controls and the original unchanged
allow-all eager Send-button approval node. Cancellation observes complete
original snapshot and visit retirement, including repeated cancellation and
surviving joiners. Same-source readers share publication failures; other owners
wait for physical retirement and recapture; fresh manual/Send review and original
settled outcomes are preserved. Whole-app budgets and all-platform evidence
remain separate outstanding gates. Existing ADR-126 and ADR-148 apply.


AC16 final catalog ledger refinement: native receipt
async-mcp-catalog-static-qualified-red.xml/log has three genuine failures in
23.07 seconds. A class-level async catalog property returning the previously
captured original bound method executes its getter after an admitted native
bundle wait and passes callback equality. Actual maximum and local preparation
also unnecessarily dereference a preexisting unrelated custom catalog property.
Provider composition alone opts into a catalog keeper; this keeper qualifies
original class/callback provenance before any dynamic catalog lookup. Maximum
and local keepers do not resolve that unrelated async callback. Qualify the
three controls plus original standard provider/lazy custody and custom late
inventory behavior, without changing guarded reads, callbacks or deadlines.


### Proven original local-review unit harness repair

ADR required: no; test-only fixture correction preserves all production contracts.
The unchanged original missing-master-key and without-bridge composer nodes fail
identically on the current source and immutable full-package 0ecd8327f79c archive:
raw_source_selection_changed and missing _character_read_guards respectively.
Receipt legacy-composer-current-baseline-pair.json records exact base/archive
identity, both process results and six stable source/test hashes per process.
Neither failure demonstrates an async preparation regression.
The unit harness fakes Console settings and MCP but leaves LocalToolProvider's
documented module-level settings consumer reading a collection-selected real
config after the test changes profile. Declare supplied tool-setting defaults
at that existing consumer seam only; retain the actual config module functions,
guards, profile admission and all native-source controls. Initialize the real
constructor's existing empty _character_read_guards field in the __new__ bare
controller. Preserve every original assertion and runtime-source policy test.
Qualify the complete affected original module after the startup native slot;
keep it in final CI. No production or binding/authority fallback is changed.


AC16 final native qualification receipts: the source-frozen 82-case run completed
81 pass / 1 newly introduced invalid-off descriptor expectation failure in
545.82 seconds, all 15 source hashes unchanged. The corrected valid-deny
descriptor and exact-native custom late catalog receiver controls then passed
2/2 in 19.55 seconds, with original permission-body issued leases observed and
retired. Three catalog ledger negatives qualified independently (3 failures in
23.07 seconds), then the final five-case native bundle passed all 5 in 37.38
seconds: static descriptor refusal, unrelated custom maximum/local callback
contracts, exact-native late inventory and standard lazy keeper publication.
New-file Ruff format/lint pass, formatting preserves AST identity, all 15 owned
source/test files compile and the scope diff check passes. Frozen hashes are in
async-mcp-final-frozen-source-static.json.
No full local 53-case repetition is claimed: final exact-head three-host CI
retains the full async, first-import, source-contract and original modules.
Earlier mixed legacy failures remain honestly recorded and have a separate
immutable/current representative-pair receipt and fixture-only repair plan.
Whole app timings, source rebase, final native CI and live UAT remain parent
qualification gates; targeted receipts do not establish those broader outcomes.

### TASK-34404 AC9 finite Samira metadata qualification

ADR required: no new ADR. Existing ADR-126 and ADR-067 apply.
AC9 is registered before production: cold standard Samira seeding must read
all 35 resources, persist the complete card and 31-asset pack, and use at most
12,000 actual Windows native opens. The original source-passive native receipt
records 18.406 seconds, 16,653 opens, 3,923 security reads and 1,482 identity
observations. All 35 issued source/repository/resource witnesses matched; guards
and network attempts were zero and callback-owned resources retired.

First qualify the original seed budget RED and actual selected-file,
carrying-parent, source, pause, custom-facade, large-DACL and protected-handle
close controls. The close control uses real HANDLE_FLAG_PROTECT_FROM_CLOSE;
guarded functions remain installed. Snapshot try/finally currently ignores
CloseHandle BOOL results, so physical retirement cannot be assumed.

After qualified evidence, use one fresh existing snapshot per original fixed
identity observation with exact defining-module/class/MethodType receiver and
function provenance before and after it. Retain every original comparison,
source/issuance/pause fence and per-file lease. The 4,096-byte descriptor cap
stays unchanged; unsupported optional metadata requires a complete fresh scalar
fallback only after confirmed retirement. If a genuine close failure is proved,
retain exact failed metadata handles and uncertainty on the already-issued
visual state and its existing raw/capture exclusion; do not retry an uncertain
close or accept fallback. This proposed repair remains unimplemented pending RED.

Preserve native-open/fstat/named-leaf checks, image/hash/budget validation,
immutable resources, card-before-pack commits and customized/tombstoned/fork
states. Verify the count ratchet and boundary controls, then original mounted,
coordinator and all ten issued startup publication controls under unchanged
20/45-second limits. Whole-app budgets and three-host gates remain outstanding.

AC9 original native qualification is complete before implementation:
samira-observation-red.xml has three genuine failures (17,053 opens against
12,000, actual source replacement accepted and actual Windows pause accepted),
five original positive/path/custom/large-DACL passes, and one explicitly
unqualified missing integrated snapshot boundary. The independent original
snapshot control, samira-observation-close-red.xml, proves that it returned
metadata while the same protected physical HANDLE and file identity remained
live. Fixture-only flag removal and positive close were verified; guarded
functions, eight source hashes and original deadlines stayed unchanged.

The approved checked-close repair tracks each actual opened handle incarnation
with its captured native owner and fresh identity. Each close is attempted once.
All remaining handles are retired even on body/ENOTSUP errors; any uncertain
close raises the specific defining exception carrying failed incarnations.
The actual issued visual state retains these handles and uncertainty in its
existing raw/source exclusion before fallback or scope retirement. It never
retries a failed close. Ordinary unavailable metadata can fall back only after
positive retirement. This refines existing ADR-126 finite custody; no admission
cap, other WindowsOS operation, seed semantics or deadline changes apply.


### AC16 legacy unit harness qualification after the original 59-case run

The complete unchanged-assertion module exposed 12 failures (47 passed / 12 failed,
22.59 s); this is retained as RED, not a successful receipt. Eight original
representatives were then run against current source and the extracted immutable
0ecd8327f79c1b14ea18c15f704850cc41b0ec9f package with identical original test
bodies and only the already approved local-provider default consumer / existing
character-guard fixture corrections. Both sides failed identically (8 / 8,
10.797 s current / 10.563 s baseline). Ten hashes per side remained stable and
the archive fixture was restored. Evidence: legacy-composer-unmasked-current-
baseline-pair.json plus its current/baseline XML and logs in the UAT evidence root.

ADR required: no new ADR for these fixture-only corrections.
ADR path: backlog/decisions/032-local-agent-tool-permission-boundary.md and
backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: declare the existing unit harness defaults and constructor fields;
production producer/permission/native identity contracts remain unchanged.

Before changing the remaining harness, apply these bounded corrections:
1. Declare supplied defaults at each settings consumer this unit harness calls
   (controller, LocalToolProvider, BuiltinToolProvider, SubscriptionsDB), preserving
   the original config readers and all native source guards.
2. Give the bare controller the existing constructor's empty ConsoleChatStore.
3. Read the existing ToolReviewDecision verdict through normalize_tool_review;
   preserve the original proceed, deny-only hook, and approval-round assertions.
4. Skip the original symlink substitution node only when Windows reports the
   actual missing symlink capability (WinError 1314); retain every other error
   and all original assertions on supported hosts.
5. Keep the three actual fs_list root-pin refusals unresolved by fixture edits:
   a real original retained Windows pin demonstrated a 64-bit stat volume serial
   compared against a 32-bit HANDLE volume field. Qualify and repair that existing
   production defect under AC22 separately, preserving full identity/refusal/close.
6. Re-run the entire original module on the final source after actual worker
   origin qualification; do not exclude old cases or raise their deadlines.

AC9 review refinements, native proof before implementation: the actual held
standard snapshot accepted a retargeted stat binding, made 59 replacement
callback calls and returned original bytes (binding-red, one genuine failure).
After initial stock qualification, recheck its exact bindings before every
scalar fallback, including successful, ENOTSUP/OSError and ValueError exits;
mid-read drift refuses without invoking replacements. Preinstalled custom
facades retain the original scalar contract.

The admitted metadata control uses a real enrolled native profile and installed
_observe_stamps inside the issued visual source. Its protected physical HANDLE
survived while bytes were returned and raw/source state retired (one genuine
failure, 4.91 seconds). Ordinary metadata errors keep prior evidence fallback;
only the exact defining close-uncertainty exception passes through selector,
path, reused evidence, candidate observation and public acquisition wrappers.
The actual visual state retains its failed incarnations before scope retirement.

The original opener's real junction/reparse refusal also leaves its protected
pre-return HANDLE alive without reporting close uncertainty. Check only that
validation-failure close; retain the exact native owner/handle (identity may be
unknown if info failed), preserve the original validation error on successful
close, and transfer failed incarnations through the snapshot's existing ledger.
Fixture-owned physical cleanup was positively verified; all guarded functions,
source hashes and deadlines stayed unchanged. Prior authority-path and absent-
handle observer errors were setup gaps, never authority RED. Existing ADR-126
applies; no other native close site, admission cap or deadline changes apply.



### AC9 scalar close uncertainty refinement (2026-10-04)

ADR required: no new ADR. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md. Reason: preserve the existing finite native resource retirement boundary for the original custom scalar route.

The actual custom-facade scalar reader delegated to the original Windows rejected-junction opener. Its genuine protected HANDLE remained alive after the defining metadata-close exception, but the issued visual state/raw record was removed with uncertainty false (one native RED, 3.77 seconds; guards/network zero, captured source hashes unchanged, physical fixture cleanup verified). Before implementation, scope the repair to catching that exact defining close exception around the existing scalar observation and passing its actual opened-handle records to the existing uncertainty retainer. Ordinary custom/POSIX errors and successful custom/native/large-DACL fallback contracts retain their current behavior. Verify the new native negative with the existing custom positive, standard rejected-open, large-DACL, and unsupported/body-error controls; preserve the original startup 20/45 second bounds.


## AC22 Windows retained-root identity repair

Proposed outcome: An unchanged admitted Windows workspace root remains usable when Python reports a 64-bit volume identifier or 128-bit file identifier. Complete expected/current identity comparisons still refuse replaced roots, changed high bits and reparse points, and all retained/verification handles retire with the prior directory restored on success or error. Original cross-platform root-pin and isolated-helper controls remain in final native CI.

ADR required: no new ADR.
ADR paths: backlog/decisions/101-one-shot-pinned-workspace-tool-execution.md and backlog/decisions/032-local-agent-tool-permission-boundary.md.
Reason: This routine bug fix preserves the existing complete identity tuple, retained-handle lifetime, current-directory verification and local tool authority contract; it corrects the native metadata width used to implement that contract.

The original real Windows diagnostic on Python 3.12.10 measured st_dev=10718190542197972492 and legacy HANDLE volume=3198770700 for the same directory, with identical inode=65583669577539523. The original pin refused that unchanged root. The diagnostic retained original reader/callback bodies and closed the actual native handles; root-pin bytes matched managed, primary and immutable baseline. Receipt: legacy-original-native-root-pin-diagnostic.json.

1. Register the outcome in TASK34403 before production. Preserve the original source hashes and actual mismatch receipt.
2. Verify the private no-pip/no-network same-3.12 test interpreter imports the managed worker/root-pin/filesystem/profile-core under ordinary -I, with dependency/ABI versions matching the primary interpreter. This isolates tests only; no product loader or primary environment changes.
3. Run six original-native regression/refusal/retirement controls against the unchanged implementation and retain RED before source edits.
4. On the same retained HANDLE, keep the existing BY_HANDLE_FILE_INFORMATION attributes/reparse read and additionally attempt GetFileInformationByHandleEx(FileIdInfo). Read its complete unsigned 64-bit volume and little-endian 128-bit file identifier. If the extended query is unavailable, preserve the original fresh legacy projection; the existing complete comparison remains fail closed. Never truncate or normalize an expected identity.
5. Run the six native controls, all original root-pin controls, stdlib-only worker import closure and the complete original local-review module with approved baseline-qualified fixture corrections. Keep existing deadlines, permissions and source guards intact.
6. Record static/source receipts and final test results, then freeze for independent parent review and exact-head three-host CI. Do not claim the broader performance task complete from this leaf qualification.

Primary sources: [CPython 3.12 Windows stat changes](https://docs.python.org/3.12/whatsnew/3.12.html), [CPython v3.12.10 fileutils.c](https://raw.githubusercontent.com/python/cpython/v3.12.10/Python/fileutils.c), [FILE_ID_INFO](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_id_info), and [GetFileInformationByHandleEx](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-getfileinformationbyhandleex). CPython reads complete FileIdInfo volume/ID and retains a legacy fallback when the extended query fails.


### AC16 local-review fixture declaration correction

The first exact-managed full-module run stopped at setup with 59 errors because the new fixture named a nonexistent tool_catalog.get_cli_setting alias. This is a harness mistake, not production RED (legacy-composer-complete-source-current-green-source.json; stable 15 source hashes). Source review confirms BuiltinToolProvider imports its getter inside the optional _GATEABLE_BUILTINS iteration; calculator/datetime are separate always-on entries. Before rerunning, remove that nonexistent alias, retain the three actual consumer aliases, and explicitly declare an empty optional catalog for this local/hook unit harness. Optional gates default False and this module does not assert optional built-in availability. Retain the actual constructor, gates, original config readers/native guards and every original outcome assertion, plus only the already qualified Windows1314 capability skip. No production change; existing ADR32/126 contracts apply.


The corrected complete unit module reached 57 PASS /1 actual Windows1314 capability SKIP /1 old raw-config failure (legacy-composer-complete-source-current-green-second-source.json; stable15 sources). The sole scripted run_reply hook case supplies an empty session prompt and no overrides; it now reaches the original Internal_Prompts resolver config-override lookup. Resolver bytes match immutable baseline (841fb4d9ae3228fe027a4784682727afbcB803cdbf9bfdd27dd369b2e40abaf1). Before the final complete-module rerun, only that case will declare shipped prompt defaults at the existing console_agent_bridge.get_internal_prompt consumer alias via original CATALOG[id].default. Keep the actual resolver, config guards and compose_agent_system_prompt intact; every hook-before-permission/dispatch assertion remains. This is a unit-input declaration, not a production repair.


The third full local-review run retained 57 PASS /1 capability SKIP /1 caught provider error. One bounded passive original-node exception diagnostic (13.94s, five stable source hashes, all original callables retained) observed52 refusals:51 existing caught budget/log/fallback/webhook defaults and one fatal _stall_timeout_seconds configuration read. The helper documents ENV-first TLDW_STREAM_STALL_TIMEOUT_SECONDS before config. Only the scripted hook case will now explicitly set that variable to the original DEFAULT_STALL_TIMEOUT_SECONDS constant; retain the actual watchdog and its unchanged positive ceiling, every other caught path and all guards. This is one documented unit input, not a timeout relaxation or production change. Full original module qualification queues after startup release; no further consumer changes without new failing evidence.


### AC22 and final legacy qualification receipt

The original Windows implementation produced three genuine failures and three refusal positives in the six-control RED (pytest 1.08 s; 16 stable source hashes). The minimal FileIdInfo 64/128 projection then passed all 16 controls: six actual Windows identity/refusal/retirement cases, all nine original cross-platform root-pin cases and the stdlib-only isolated-worker import gate (pytest 1.97 s; 18 stable source hashes; normal exit 0). The actual full expected/current comparisons, fresh legacy reparse metadata and original close/cwd semantics remain unchanged. Root-pin source SHA 9808d5404ac84c5d22aaffc8b347cef0ded160f09b92c28e7e36cd33666874ed.

The final complete original local-review module passed 58 cases and skipped only the actual Windows 1314 symlink capability case; every supported original assertion passed. Its15 source hashes stayed stable. A brief overlap with startup31828 is explicitly recorded, so its elapsed time is functional/source evidence only and is not performance qualification. The new actual junction refusal and unchanged root-pin controls passed in the earlier exclusive run. Earlier47/12,59 setup errors and 57/1/1 receipts remain retained and honestly classified; no source guard or timeout budget was changed to satisfy them.

Ordinary private 3.12 -I imports proved five exact managed origins and identical Python/ABI/SQLite plus six dependency versions. The private no-pip/no-network environment changes no primary checkout/environment or product loader. Final static receipt confirms three syntax checks, formatted new leaf/tests, git diff check and no new Ruff diagnostics; existing strict-type E721 and unused-import F401 match HEAD exactly. Receipts: source-current-interpreter-origins.json, workspace-root-native-identity-qualified-red/green.xml and -source.json, legacy-composer-complete-source-current-final.xml/-source.json, workspace-root-identity-final-static.json. Production/fixture source is frozen for parent review and final exact-head native CI; broader performance/UAT completion is not claimed.


### AC8 startup observer qualification refinement (2026-10-04)

ADR required: no new ADR. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md. Reason: test-only observation of the existing exact native startup publication contract.

The original three mounted restore checks and all fifteen coordinator controls passed after the seed fix. The ten new controls missed their intended check: seven failed the original initial-finalization 20-second wait and three reached the 45-second child bound. The passive stage receipt showed app construction completed in 17.922 seconds and actual managed restore fingerprint/native admission work still running before the target. The 35-second faulthandler diagnostic exited natively before the ownership receipt; a subsequent metadata-only snapshot had brief contention with the separately authorized legacy module, so neither diagnostic is performance/authority RED evidence. No ownership cycle is claimed.

Before the managed fixture edit, narrow only the test observer. The original `_LocalPause.reacquire_startup` assigns its exact Thread before calling `start` (storage_admission.py lines422â€“427, with no intervening await). Install a first-call profile selector after the original app constructor/setup and before monitor/restoring tasks. Only the exact current thread owned by the current pause selects the existing check-return observer; all other newly created threads immediately restore the previous profile. Preserve every original guard, finalization control, publication revocation, 20-second wait and 45-second child bound. First qualify one cancellation control against the actual issued native attempt; only a genuine GREEN permits the full ten-control run. Record actual source paths, selected target thread, refusal and final native retirement counters. No production change or timeout increase is part of this refinement.


### AC8 native observer event handoff qualification (2026-10-04)

The late exact-owner selector's single cancellation control still missed initial finalization (one setup failure, 36.14 seconds), so the ten-control run was not launched. The actual unchanged original pending-failure node on the verified same3.12 source-current isolated runtime passed in 20.33 seconds; all captured source hashes stayed stable and final native counters were zero. No production regression or authority failure is inferred from the new setup miss.

Before this test-only edit, preserve the selector and every original guard/control/deadline, but replace the new fixture's one-millisecond preboundary polling with one captured-loop asyncio.Event handoff. The same actual native check-return observer schedules only the event notification with loop.call_soon_threadsafe, then retains its existing ten-second publication hold; the original revoker still changes the exact captured metadata under the coordinator and has the same thirty-second bound. The actual initial wait remains twenty seconds and the child forty-five seconds. Qualify one cancellation control; full ten requires a genuine target/source/refusal/retirement GREEN. ADR required: no new ADR; existing backlog/decisions/126-complete-local-backup-and-recovery.md native publication contract applies.

Final integration: freeze all owned source and tests; commit, rebase onto latest origin/dev preserving command dispatch and Buddy resume ownership; verify actual isolated helper origins. Run the original committed native performance probe alone, all three host CI jobs and all three genuine elevated Windows custody jobs, then a fresh private-profile DeepSeek/deepseek-chat three-message UAT. Publish one combined PR against dev after reviewing final results. ADR required: no additional ADR; implements the existing ADR126/097/101/32 boundaries.

### AC2 Home Resume fixture and UTF-8 child failure reporting qualification

Preimplementation scope: the exact integrated children for cold and reused Home both stop at test_home_console_resume.py:98 with WorkerCancelled before pressing Resume. The unchanged Buddy baseline contains the same fixture/helper/Home bytes; that proves source provenance, not a baseline runtime outcome. Textual pop_screen updates the active stack before queued ScreenResume dispatch, and Home's resume hook starts the same exclusive snapshot group. After qualified cancellation-owner evidence, use the actual Pilot queue barrier before the explicit worker, bounded to the fixture's existing eight-second condition limit. Retain the original post-worker pause for HomeCanvas recomposition, all saved-history/identity/draft/reuse assertions and every original test/child deadline. No cancellation suppression, repeated workers or product changes.

The isolated Windows parent's locale decoder masks this native child failure with CP1252 UnicodeDecodeError. First add a focused reporting RED control that runs the real wrapper failure branch with a small UTF-8 child and locale-bound parent log read, without app/native imports. Then explicitly pin child PYTHONIOENCODING=utf-8 and log open/read encoding=utf-8, preserving exact child selection, XML/return-code refusal, profile guards, coverage and log-tail length. Keep the helper and Home source unchanged during root's next isolated diagnostic; implement only after its source release. Existing TASK-34403 AC2 covers this fixture/evidence qualification; no new task or acceptance scope is introduced. ADR required: no. ADR path: N/A. Reason: routine test synchronization and diagnostic encoding preserve application/security boundaries. Root serializes the pure reporting RED/GREEN and exact Home cold/reused verification. Earlier integrated RED remains retained.
### AC16 visible draft mirror before finite native preparation

The tenth exact-source diagnostic qualifies a real pre-custody refusal at wiring._prepare_console_turn_to_runtime: the retained session object and settings revision stayed unchanged, while its stored draft changed from empty to the same 28-character live composer draft during the native await. Regular Console sync mirrors the visible composer into the store. Enter already mirrors its captured text synchronously; button/direct capture can enter preparation before that mirror. This is a draft synchronization defect, not changed native authority, and the three original whole-probe failures remain recorded.

ADR required: no additional ADR. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md, fresh async Console definition maximum preparation / AC16 actual UI boundary. Reason: synchronize the already owned in-memory composer mirror before capturing existing publication fences; no new source contract, admission, permission cache or native lifetime.

1. Add a real ConsoleComposerBar to the existing actual screen-wired native source fixture without replacing its lookup, session sync, guarded source readers or acceptance callbacks. Hold the original admitted MCP read with the existing passive code-object observer. Before production, show that an explicit original session draft sync changes only the stored mirror while exact composer/stash/session/settings/prefill/attachment/evidence owners and revisions remain unchanged, then observe the current preparation refuse.
2. Add unchanged ownership positives and four independent negatives: a real edit, edit away and back with the serial advanced, same-text load_draft with generation advanced, and a store-only different draft. Each negative must refuse before custody and retain the draft; all started native leases must physically retire. Keep the existing broader source/actor/attachment/cancellation controls unchanged.
3. After the qualified positive RED and native-slot release, mirror only the exact currently visible owning composer's current text synchronously before stored_draft is captured and before any preparation await. Retain all post-await store/composer/stash and native source checks. Do not accept arbitrary post-await store changes, call broad session creation/switch synchronization, relax edit/generation checks, or change queued/custom/synchronous contracts.
4. Run the focused positive/negative native bundle and required existing actual-wired drift/queued controls exclusively with the source-current isolated interpreter. Record source hashes, normal exit, real original worker/lease evidence, Ruff/format and diff checks; freeze the source for parent review and the next unchanged-budget whole probe. No full sweep or performance claim is part of this leaf fix.

AC16 visible-draft mirror verification (frozen source): the original seven-node native RED settled normally with two genuine unchanged-mirror refusals and five passes in 162.42 seconds. Empty and previous stored text changed solely through the original session sync while exact captured composer/stash/session/settings/runtime/prefill/attachment/evidence comparisons remained equal; the already-synced positive and all four real-drift negatives passed. Six source hashes were stable. The minimal production fix establishes the exact owning visible composer's in-memory mirror before stored_draft capture and the first await; every existing post-await check remains unchanged. Nonvisible/no-composer, custom synchronous callbacks, actual native authority and runtime acceptance remain on their original routes.

The final exclusive native run passed all ten focused controls in 184.90 seconds, normal exit0, with six stable start/end source hashes: three unchanged-mirror positives, four genuine real edit/edit-away-and-back/same-text-scope/store-only negatives, both original normal/queued actual-wiring controls and the original screen-owned joined-snapshot positive. Every new case asserted actual held MCP leases live before release and physically retired after settlement; accepted test turns were canceled/drained and runtime disposal awaited in finally. This proves the finite native preparation leaf, not mounted pump responsiveness, global ordinary-counter census, the full older53-case module, whole-app budgets or three-host CI. Existing deprecation warnings remain unchanged. Native/source slot was returned to root after physical process settlement.

Ruff check and scoped git diff check pass. The test module is formatted. Whole wiring formatting still requests exactly the unchanged HEAD edits outside this four-line behavior change; literal HEAD fed to Ruff reproduces those requests, so no unrelated formatter rewrite was applied. Receipts in deepseek-uat: draft-mirror-native-red.log/.xml/-source.json and draft-mirror-native-green.log/.xml/-source.json. ADR126 AC16 already governs this finite source versus UI snapshot boundary; no additional ADR or permission caching was introduced.

Frozen six-source SHA256:
- tldw_chatbook/Chat/console_chat_controller.py: C1427C4D89D3ED1EB4988183C6EBFBE465C63E6E1B1F582850EF0DDF923C3C7B
- tldw_chatbook/UI/Console_Modules/wiring.py: 58A88C4AFE1397096503F5898ACC84301A540BEAE219C543CB8FBA8F15E6EF49
- tldw_chatbook/MCP/console_snapshot.py: EF3AD7B16264B8DFE5C6C6B3FDB489B8629BC2BA954596FE2A49FFDC2A487587
- Tests/Chat/test_console_async_mcp_snapshot.py: D2A653EBA22DD980F452138C5866F296ED8B33DAA4C0D486F4F7C441A37EC5B4
- tldw_chatbook/UI/Console_Modules/session.py: ECC4DB85B4EA65B48D2050112B489C58023D3DBCF18B0023BD7835C8FA78F783
- tldw_chatbook/Chat/console_chat_store.py: 8A99FAB424DC60356CE36A1A5FBF5D9956553CAFC4022427435B2A9EC5828DB8


## Whole cold/warm startup completion requirement and remaining critical preparation

TASK-34404 AC10/11 extend the existing user-authorized broader performance pass. The 12000-open Samira bound is an interim resource-specific regression guard. Completion requires measured total native file/directory opens, content reads and unique paths, real input acceptance and UI heartbeat across separate cold and warm processes. Declare instrument coverage explicitly; Native facade calls and POSIX helper starts are not every operating-system syscall. Record an evidence-based whole-startup operation budget before implementation and demonstrate materially less redundant I/O and better launch responsiveness on actual Windows/macOS/Linux. Preserve every existing permission, file-change, cancellation, recovery and native retirement check and every deadline. Existing warm seed preflights may already avoid unchanged resources; qualify actual reads/reseeding before claiming that defect.

Prior committed tenth Windows startup: 70488 main +46401 worker native facade opens; max observed startup heartbeat stall3.632s. No complete content-read/unique-path/first-usable-input or warm-process baseline exists yet. Broader metrics/coverage and untouched baseline are a prerequisite to accepting further startup optimization. Do not present AC9's 9490 seed opens as whole-app startup improvement.

Passive selected-worker diagnostic with the actual mirror fix reaches three COMPLETED native turns, three user/assistant pairs, three complete traces/response links and no dispatch checkpoint. Extra all-worker profile adds substantial CPU cost: phases159.682/69.901/116.572s are diagnostic only, and the original15s assertion fails. Process settles naturally exit1; no stop action was performed. All production/probe/helper/guard/plugin source hashes remain unchanged, no production native guard is replaced, every selected span finishes and original anchors are complete. Receipts: console-critical-bridge-diagnostic.json, perf-critical-bridge-diagnostic-source.json/log/xml. Small mechanism qualifier separately passed actual DB/native work and existing-worker hook restoration in2.432s.

Critical original worker spans carry27623/26752/13680 native opens. The actual explicit log-root selection has zero native opens; subsequent argument-free global selections cost2000/1129/1129. First sensitive raw-input construction has13760 native opens on sends1/2; the second same-bind context costs688. Both contexts cost688 on send3. Thus sharing only the two contexts saves the688 repeat, not the expensive first derivation. Investigate the exact memo-key/source and equivalent getter validation before changing it. Preserve strong cache/source identities and current native source proof; never retain resolved sensitive context between binds or convert metadata into permission. The existing critical worker root covers the bridge/service guarded bodies but starts after the outer worker_guard admission; unowned phase counts are explicitly outside its selected coverage.

Before additional implementation, record scoped source-fence design and native RED for exact injected writer/root/callback/module/actor drift, actual current-root review/refusal, revoked scratch, custom/subclass/borrowed receiver and original migration/no-overwrite behavior. Follow the Evidence-only scoped-run-log-fix-plan-draft.md after actual profile qualification. Supported stock base/run containment may share one fresh context within one bind; original legacy migration has a third independent check unless separately qualified. Custom two-argument checkers retain their call shape, stock resolver failure retains logging-disable behavior, and publication refusal remains outside the publisher's observer-exception handler.

ADR required: yes for changed runtime source selection or finite configuration interfaces.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md (narrow amendment before implementation).
Reason: retain existing actual-source recovery, actor and native custody boundaries while removing equivalent work. Routine mirroring/reporting fixes require no new ADR.

## Remaining refresh attribution and hidden Console lifecycle

TASK-34404 AC12 covers the source-proven hidden credential timer candidate before implementation. A suspended Console stops transcript, survivor, cost and draft timers, while its credential timer remains. The presentation decorator schedules readiness work before the credential body's hidden-screen guard. Require an original mounted hidden-versus-visible native RED before a minimal lifecycle fix; preserve original timers, visible freshness, source/owner/recovery checks, resume and retirement. Separately attribute actual run_owned_db_call callbacks and native work; overlapping diagnostic profiler durations do not prove wall time or that all database calls come from readiness.

ADR required: no for a routine timer lifecycle repair preserving existing authority and UI structure.
ADR path: N/A; existing finite custody remains governed by backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: scheduling only when the original presentation target is active repairs the existing hidden-body contract. Any broader runtime-boundary change requires an ADR amendment before code.


## AC10 original reader body binding qualification

The cold-bundle native candidate cannot yet be accepted: its first targeted
verification exits during collection because the existing cached effective-path
resolver is a concrete lru wrapper, without a function's `__globals__`. Preserve
that original wrapper invocation/cache contract and qualify its definition-time
wrapped-body metadata before rerunning the fifteen cases. This is an import
regression, not a fifteen-case native result.

Separately, exact FunctionType identity and globals alone do not distinguish a
replaced `__code__` body on the same unguarded accessor. Add two bounded private
native controls before accepting an ordinary-function anchor completion: a
replacement before the sensitive helper's first import retains the original
custom no-added-scope contract; a replacement at the original single-file
builder return refuses before invoking the changed body or publishing a memo.
Retain original seeded SQLite physical close, actual native census, source
hashes, guards, hook restoration and the existing 45-second child bound.
The positive cloned original must preserve keyword-only defaults as well as
positional defaults; its literal 13-database path assertion qualifies the test.

ADR required: no new ADR.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: complete the already approved exact defining-reader/custom-source
qualification; no new operation, permission, source lifetime or cache.


### Original run-log regression fixture qualification

ADR required: no. ADR path: N/A. Routine test fixture repairs preserve the existing configuration admission boundary. The original two-node code-local witness confirms `raw_source_selection_changed` before provider entry, with source identities unchanged and global monitoring events zero. Keep the real admitted collection profile for the survivor and service-wiring modules using the existing `bootstrap_profile` marker; keep all provider, logging, survivor and privacy assertions unchanged. The activation subprocess writes a TOML literal with an unescaped Windows path; render that same path with `as_posix()`. Rerun the original six affected modules with original timeouts and no guard replacement.


### Hidden credential poll regression and native cleanup

ADR required: no. ADR path: N/A (routine lifecycle correction under existing screen reuse). The actual native RED observed two original credential readers while Home was current, plus positive visible and same-instance resumed native reads. Original reader/guard/source identities and monitoring retirement passed. A final zero-lease fixture assertion also found seven ordinary App-held caches after screen shutdown; this is not evidence that the credential reader owns those caches. Preserve the final zero census by retiring the actual installed current-thread private SQL caches using the original coordinator cleanup after original App shutdown. Capture each real repository/participant/connection/lease pair and prove native SQLite closure; leave unknown leases visible. Then stop credential polling on suspend and restart the same cadence on reconciled resume, retaining visible expiry/source/recovery behavior. No cache TTL, native guard, deadline or performance budget changes.


## Registered whole-launch startup outcomes

Root accepted the measured proposal below before further startup production changes. These are acceptance targets, not achieved results. Existing Send/UI/custody budgets and deadlines remain. The original unobserved liveness route decides timing; code-local observer time is diagnostic.


Evidence-only proposal for root review. No production behavior, test deadline, or acceptance criterion is changed by this file.

ADR required: no new ADR.
ADR path: existing `backlog/decisions/126-complete-local-backup-and-recovery.md` (profile ownership and native custody).
Reason: preserve the existing permission, source, actor, cancellation, recovery, and physical-retirement contracts while reducing redundant startup work. The proposed limits below describe outcomes, not permission-result caching.

## Retained baseline

Both launch pairs use the original App and Pilot, a pristine private profile for cold startup, the same durable profile in a second fresh interpreter for warm startup, and original shutdown. No App/global cache survives between launches. OS page cache is uncontrolled.

The unobserved liveness pair is `startup-liveness-baseline-2`. The code-local diagnostic pair is `startup-code-local-baseline-1`. Both cold/warm processes exited normally; the code-local pair retained original source hashes, all twelve original body anchors, no source mismatch, and original monitoring cleanup.

| Observation | Cold | Warm |
|---|---:|---:|
| Liveness Pilot-return upper bound from parent spawn | 21.578 s | 16.883 s |
| Code-local first original key/composer insertion entry from child entry | 24.428 s | 17.390 s |
| Code-local Pilot-return upper bound from child entry | 24.992 s | 19.327 s |
| Selected original Native entered calls through that upper-bound phase | 103,586 | 78,138 |
| Selected original Native successful HANDLE returns, main | 51,770 | 44,292 |
| Selected original Native successful HANDLE returns, worker | 42,224 | 26,042 |
| Selected original Native successful HANDLE returns, total | 93,994 | 70,334 |
| Observed lexical unique paths through that upper-bound phase | 2,641 | 2,442 |
| Samira original buffer returns / unique lexical resource paths | 35 / 35 | 0 / 0 |
| Samira original returned resource bytes | 5,390,397 | 0 |
| Pixel original buffer returns / unique lexical resource paths | 3 / 3 | 0 / 0 |
| Pixel original returned resource bytes | 12,883 | 0 |
| Original pack activation entries | 2 | 0 |
| Original seed-both entries / normal returns | 1 / 1 | 1 / 1 |
| Path.read_text normal returned buffers / lexical unique paths | 1,042 / 150 | 840 / 92 |
| Unfinished selected frames at observer stop | 27 | 30 |
| Exception or phase-crossing completion gap rows | 16 | 15 |

Warm seed entry is a preflight invocation, not evidence of rereading or reseeding bundled content. The actual observed warm resource bytes and pack activations are zero.

The counter boundary is conservative: `mount_until_first_input` ends when the unchanged Pilot call returns. The original insertion body entered 0.564 s / 1.938 s earlier. The current receipt does not contain an exact I/O counter snapshot at that original input event. A future separately qualified Evidence-only observer can copy its existing in-memory counters at the expected composer's first original insertion normal return. This does not require changing App/Pilot behavior.

## Proposed outcome limits for review before implementation

1. Unobserved fresh-process startup reaches the unchanged normal Pilot key path and preserves the one-character draft delta within 15 s cold and 10 s warm from parent spawn. These require approximately 30% / 41% improvement against the retained liveness baseline; they are proposed targets, not results.
2. For the Windows selected-original-Native coverage window above, entered calls are at most 60,000 cold / 45,000 warm, and successful HANDLE returns are at most 55,000 cold / 40,000 warm. Both categories remain independently visible. These require roughly 40% or greater reductions, not a cap restricted to the seed helper alone.
3. Once the real mounted input surface is available, continuous original-loop heartbeat gaps are at most 0.200 s, and a delivered original key reaches original composer insertion within 0.500 s. Import/constructor time before a usable loop is separately reported and cannot qualify this heartbeat gate.
4. An unchanged ordinary warm profile performs zero original Samira/pixel bundled buffer reads, zero returned bundled bytes, and zero pack activations. Changed files, permission revocation, recovery, cancellation, and failed-close cases remain fresh and fail closed under their existing limits.
5. Cold/warm ordinary launch measurements are obtained on Windows, Linux, and macOS. The existing three elevated Windows Python custody jobs retain their original role; startup need not be duplicated there absent evidence. All existing six jobs must pass.

Limits must be accepted and registered in the appropriate task/plan before new startup production code. Timing acceptance uses the unobserved liveness route; observer timing is diagnostic only.

## Coverage and remaining questions

Successful Native facade returns are actual observed HANDLE returns. They are not total kernel I/O, SQLite/WAL/SHM/mmap I/O, helper-process I/O, or all direct native APIs. Relative HANDLE-parent paths are explicitly unresolved, not reconstructed from a reusable numeric-HANDLE ledger. Unique paths are observed lexical spellings only.

Code-local PY_START/PY_RETURN has no exceptional unwind completion on Python 3.12. Unfinished frames and phase-crossing/exception gaps remain explicit; no successful interval or completed refusal is fabricated. Distinct read APIs can observe overlapping work, so buffer/byte categories must not be summed as a unique total.

POSIX facade/audit helper counts must not be described as actual syscall totals. Platform-specific native-open limits require an actually qualified open observer on that platform. Until then, report coverage gaps and the observed categories alongside liveness.

The whole-launch pressure is proved by the measured counts. Which repeated reads/validation chains are unnecessary still requires selected original-code ownership and call ancestry evidence. The warm asset-reread hypothesis is not supported by these receipts.


### Session-tab publication after parent disposal

ADR required: no. ADR path: N/A. This repairs the existing Textual parent lifecycle without changing UI structure or authority. Actual Windows CI reports three stock tab-strip MountErrors after parent removal. The original sync awaits its lock, child removal and each mount, then publishes against a retained strip without checking whether the same parent is still attached. Add four bounded real-Textual controls at these actual await boundaries, requiring disposed sync to retire without mounting and a fresh attached surface to retain normal tabs. Verify RED before a minimal attachment/source fence at each continuation. Preserve tab order, mounted controls, overflow hints, counters, session state and original navigation deadlines; rerun the original tab-strip tests and failed command/Home nodes. These controls refine existing AC12 lifecycle and AC10 performance acceptance.


## AC10 scoped run-log original body binding qualification

The revised two-case native RED is genuine: scoped-body-native-red-2 settled
with two failures in 11.55 seconds (17.453-second driver), source unchanged.
Both same-function code replacements ran after scoped qualification; the
original provider completed, seeded SQLite physically closed, guarded callbacks
and modified unguarded bodies were restored, and every final ordinary/core/raw/
pending/retiring count was zero. The initial source-drift-native-red-1 scoped
failures were fixture precondition errors before mutation and are explicitly
excluded from product RED. The separate tuple-source failures remain genuine.

ADR required: no new ADR.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: complete the existing original producer/body and custom callback
qualification, without a new source operation, permission or native lifetime.

Before production, complete definition-time stock service wrapper/body, writer
method and selector metadata with exact original code/globals. Retain supported
finite function/MethodType/partial callback bodies and strong receivers. The
actual contextmanager factory is recognized only by the original stdlib helper
code/globals and its one original wrapped-function closure; retain and compare
that wrapped body too, with no arbitrary unwrap or recursive closure authority.
Preinstalled changed stock bodies preserve the original fallback; drift after
issuance refuses before invocation/publication. A dispatch checkpoint must follow
the caller's original check-return boundary, because that boundary itself can
change the mutable function code. Keep publication refusal outside the optional
publisher exception handler and original nonfatal binder behavior.

Actual-root/root-object admission, thread/task/source/live-bit retirement,
partial-keyword checks, unsupported custom/default routes and independent
foreign-worker/survivor authority remain. No guarded activation body, callback,
source cap, deadline or cache is replaced. Root must run the unchanged original
nine source controls and both revised body controls on frozen source; native
GREEN and integrated budgets remain pending.


## Literal Windows hook argv serialization repair

Root registered this bounded existing persistence/consent repair before production code. Product RED must precede the new helper.

# Preserve literal Windows argv in config serialization

> Evidence-only implementation plan for parent review. No managed source or tests have been changed and no regression draft has been run.

**Goal:** Save and reload valid command arguments containing literal `\\x` without changing their characters, order, keys, permission fingerprints, or config ownership.

**Architecture:** Keep the existing TOML encoder for all ordinary types, keys, and string values. Select a per-call encoder only when a nested string value contains literal backslash followed by lowercase x. Its string formatter uses TOML basic-string escaping only for those affected values; every other formatter remains the original formatter. Both the canonical writer and the literal Hook Save preflight use the same serializer. All native admission, actor/source checks, write locks, parse-back validation, atomic replacement, encryption, and publication remain in their existing owners.

**Tech stack:** Python 3.12+, existing toml 0.10.2 and stdlib tomllib. No new dependency or runtime.

**Spec / evidence:** Parent's retained `qualify-windows-toml-command.json`, CI run 37273707121 Windows hook failures, and this task's existing authorized UAT/failure repair scope. Proposed task outcome must be registered by root before managed implementation.

ADR required: no new ADR.
ADR paths: `backlog/decisions/033-settings-commit-models-three-honestly-labeled.md`, `backlog/decisions/126-complete-local-backup-and-recovery.md`, and `backlog/decisions/197-console-hook-configuration-review.md`.
Reason: routine codec repair within the existing config persistence and exact-argv consent contracts. No new owner, storage format, permission rule, or public persistence route is introduced. Original section-stamp hashing and raw-text restore paths remain unchanged.

## Evidence and distinction

The actual CI executable `C:\\hostedtoolcache\\windows\\Python\\3.12.10\\x64\\python.exe` fails original tomllib parse-back after original toml.dumps. The original encoder `_dump_str` splits repr text at `\\x` and collapses doubled backslashes in the preceding prefix, introducing unescaped Windows separators. The retained pure witness proves the installed encoder/reader pair, without App import.

Product `_write_raw_cli_config_unlocked` at config.py:7426/7472 performs parse-back before atomic commit: the product refuses the valid Save rather than committing malformed config. Literal/Hook Save additionally parses stock serialization at config.py:8770 before calling that writer, so repairing only line7472 is insufficient. `HookPermissions.save_configuration` delegates to `replace_hooks_config_snapshot`; the command's ordered argv is the original fingerprint input.

The shared `Tests/Agents/test_hook_permissions.py` fixture writes stock toml.dumps directly, so it commits malformed test data. Converting the executable to forward slashes in that fixture would conceal the product problem and change the exact argv under review. After qualified product repair, use the same corrected serializer in its three direct fixture serialization sites, retaining actual sys.executable and every original assertion/deadline/source guard.

## Proposed acceptance outcomes

- A valid literal backslash-x string survives canonical config replacement and Hook Save unchanged, including ordered argv, Unicode, quotes, repeated backslashes, and nested list/table values.
- An execution-definition change still invalidates consent; a purely cosmetic re-save of the same command preserves its exact original fingerprint and grant.
- Ordinary values/types and quoted keys retain the original serializer's output. Keys are never normalized or renamed. Existing section-stamp hashes are unchanged.
- Existing malformed-output parse refusal, stale snapshot/profile refusal, encryption rejection, publication failure, cancellation, and native ownership controls remain intact.
- The shared hook fixture stays valid on the actual Windows CI executable path while retaining all original tests, original argv, and original deadlines.

## Review focus

- Uppercase backslash-X and ordinary Windows paths continue through the original codec; the qualified lowercase branch is the repair target.
- Control characters and Unicode scalar handling follow TOML basic-string escaping only within the affected value; no general fidelity rewrite is claimed.
- A literal backslash-x in a mapping key is not rewritten by the value encoder. Its existing failure remains guarded; fixing arbitrary key serialization is outside this measured command defect.
- No global monkeypatch of toml.encoder or toml.dumps is introduced. A separate encoder instance prevents unrelated serializers from changing behavior.
- No command process is launched by the dedicated regression; exact definitions, original fingerprints, and actual consent persistence establish the behavior.

## Files and sequence

1. [ ] Root registers the outcome and plan before implementation. Publish the Evidence draft `test_config_windows_argv_roundtrip_draft.py` to a dedicated Tests module, unchanged product body/encoder/guards.
2. [ ] Run only its three actual product cases first. Expected qualified RED: canonical replacement rejects parse-back; replace_hooks_config_snapshot and actual HookPermissions.save_configuration do not replace the file. Preserve pre/post source/config receipts and unchanged bytes on refusal. A setup failure is not product RED.
3. [ ] Add one pure shared leaf (proposed `tldw_chatbook/Utils/toml_serialization.py`, `dumps_cli_config`) after RED. Inspect nested built-in dict/list/tuple values with a cycle-safe predicate. Ordinary input calls original toml.dumps with the same single argument, retaining existing one-argument fault-injection controls. For affected input, construct a per-call original TomlEncoder and replace only its string-value formatter with the conditional formatter.
4. [ ] Use that helper at config.py:7472 canonical serialization and config.py:8770 literal mutation expected_raw parse-back. Use it at8257 only for the same missing-file snapshot serialization fallback; exact existing raw file snapshot copying remains unchanged. Leave7671 hook section-stamp hash, encrypted rollback bytes, raw text replacement, and unrelated serializers unchanged.
5. [ ] After product GREEN, replace only the shared hook fixture's stock dumps calls at test_hook_permissions.py:41/71/1191 with the same helper. Do not normalize sys.executable, change command input, patch guards, or relax assertions.
6. [ ] Run pure conditional-encoder cases, all three actual product cases, and targeted original config parse-back/participant/encryption controls plus the original Hook permission/admission/review/lifetime modules. Keep existing native serialization and exact-source receipts. Final three-host CI retains the original failed modules; no full-suite sweep is authorized by this plan.
7. [ ] Review resulting diff and write exact qualification notes. Preserve the old CI and pure witness as RED evidence. Record fixture bypass versus product guarded rejection separately.

## Draft locations

- `test_config_windows_argv_roundtrip_draft.py`: actual canonical writer, Hook expected_raw preflight, and consent regression. Imports no proposed helper, so these are suitable for original product RED.
- `test_conditional_toml_serializer_draft.py`: independent pure round-trip and ordinary-codec compatibility tests, to publish after the helper's intended API is accepted.

No implementations or test results are asserted by this document.


### Context-capacity display metadata inside the issued snapshot

# Keep context-capacity rendering inside the checked display snapshot

Evidence-only proposal; no production or managed test changes and no native launch.

The callback98158 receipt identifies an actual second configuration route inside an accepted display body: `ChatScreen._console_settings_context_estimate_for_session` calls the runtime's gateway `cached_context_window`; both its cold metadata branch and `_context_window_target` call the gateway's original `functools.partial(_provider_config_for_app, app)` source. That source calls original `config.load_settings`. This bypasses `provider_readiness_app_config`, which already returns the issued display mapping in the same synchronous body. One selected main load took 1.123 seconds inside the settings-summary render. The duration is diagnostic, but the original caller ancestry is concrete.

The finite callback diagnostic does not attribute readiness Native opens: the observer failed to bind the lazily published spend module. No readiness open savings or permission-caching opportunity is claimed from that gap.

## Proposed outcome and boundary

An exact standard gateway's context-capacity rendering uses only the same freshly issued display mapping while its original owner/source/actor proof is valid. This pure serving-metadata projection opens no second config scope or native path merely to derive the same model/endpoint/credential identity. Direct actions, dispatch, provider readiness, metadata network refresh, and arbitrary injected/custom gateway callbacks keep their existing fresh reads and signatures. Source replacement, app mapping/session/settings change, pause, retirement, expiry and post-body drift preserve the original refusal/fallback behavior.

ADR required: no new ADR if implemented only as an explicit synchronous metadata projection under the already issued display proof.
ADR path: existing `backlog/decisions/126-complete-local-backup-and-recovery.md` and the registered Console presentation plan.
Reason: this extends the existing detached display value consumption to its identified context-capacity consumer. It must not introduce a thread-local ambient authority or change the gateway's live configuration provider contract. Root must register the precise outcome before production changes.

## Qualified RED first

The companion test draft constructs the actual original `ConsoleRuntime`, gets its actual standard gateway via `ensure_provider_gateway`, and runs the original screen context estimate inside `ChatScreen._run_console_config_sync`. It reuses the original checked-display fixture's real private Notes/store/controller and original native observation helper. It tests both no metadata cache and an existing original metadata cache. No config/native/permission reader is replaced. Expected RED is actual redundant main Native opens inside an otherwise warm accepted render, not an attribute/setup failure.

Original native profiling in this bounded control is an observer only. App startup is not used. Runtime/gateway, selected worker tasks and the actual DB are explicitly retired. Existing checked-display owner/expiry/copy/custom/foreign-loop controls remain required; do not raise their deadlines.

## Minimal design to review after RED

1. Add a named presentation-only consumer helper in `console_spend_projection`. It verifies the existing issued `_CheckedDisplayProof` in the active projection, including the same app, mapping, screen, loop/thread, source/participant, session/settings/recovery owner and before/after freshness. Only an exact `ConsoleProviderGateway` with its original bound metadata methods and the actual runtime-produced original `functools.partial(_provider_config_for_app, same_app)` qualifies. Class/instance/custom/subclass/borrowed callback paths retain the original one-argument `cached_context_window` invocation.
2. Add a narrowly named private gateway metadata projection accepting that supplied mapping. Thread this mapping through cold capacity fallback, target memo-key derivation and target projection, so it is consumed once without calling `_config_provider`. The original live `cached_context_window`, `_context_window_target` default route, `resolve_context_window` and `resolve_for_send` keep their existing provider getter. Endpoint/custom credential and serving-cache keys remain identical.
3. Change only the two original screen context-estimate call sites to the helper. The helper may fall back to the original gateway callback when no current proof qualifies. Never supply the mapping as a final provider or permission decision, invoke a whole coroutine in a worker, or retain it beyond this display interval.
4. Qualify the actual native RED nodes, both cold/warm metadata positives, injected callback/subclass compatibility and the original checked-display/source drift/expiry/foreign-loop refusal suite. Add a direct/live read control and original gateway send/config tests. A post-body owner change must not publish an accepted estimate.

The exact API naming is provisional. Root review must resolve callable provenance before implementing it; no existing reader or guard is to be weakened.


### Tab retirement before physical detach

ADR required: no. ADR path: N/A. The disposed-parent leaf passed four controls, but real Textual pruning keeps attachment while stock mount becomes a no-op. Add exact original App._prune controls and same-ID strip replacement before a minimal retirement-state fence; preserve real pump retirement, tab order and mount-churn evidence. Initial is_mounted-only fixture failures remain excluded.


### Original Ubuntu trace-settlement observation

ADR required: no. ADR path: N/A; existing ADR126 and the original trace settlement contract apply. Passive test-only diagnosis preserves every production owner, guard, test assertion, three messages, .25-second settle, and original deadlines. Before tracked installation, root reviewed the exact source-fenced local observer and finite launcher plus pure controls. Use a separate opted-in Ubuntu CI job; the original six native jobs and three-platform probe remain the acceptance route. Hash all managed application, test and bundled profile-core sources and actual installed helper/plugin origins before and after, including failures. Actual child loaded sources must match that manifest. Retain real pending/failed/owned settlement state and handled exception categories at the original assertion and teardown, with nonblocking snapshots and explicit coverage gaps. Diagnostic timings cannot establish performance acceptance. Native qualification and cause classification are pending.

The old original Windows activation failures stopped before their intended effect boundary. The corrected fixture waits for actual effect entry under the already existing ten-second effect/release bound, retaining original .2/.5-second waiter timeouts, abandonment/cancellation result, .04-second maintenance refusal, native close and fresh reopening. Native corrected original three cases pass 14.049 seconds; the other 119 original run-log regression cases already passed on the preceding frozen source. The original tab-strip module needs the existing bootstrap-profile marker because its imported guarded config otherwise points at a different per-test profile. Original 18 plus new six lifecycle cases pass 10.299 seconds; the three real close journey failures remain distinct and unresolved.

The hidden Console reader check passes visible native expiry, hidden zero reader/schedule starts and same-instance native resume, but ordinary SQL coverage still fails. A supported-local observer installed after actual App construction missed getter acquisition because it selected the original decorator wrapper's self-less frame; its clean callbacks and physical-close records do not qualify acquisition attribution. Preserve that receipt and observe the exact original getter body before identifying or changing an owner. No unknown connection is closed or excluded. Original config compatibility run is also retained: 15 pass/34 fail, with explicit retarget and Windows pipe-monitor fixture failures under investigation. These are not claimed as codec regressions or passing compatibility.


### Finite unread-row worker connection ownership

ADR required: no. ADR path: existing ADR126. Qualified native observation hidden-sql-original-getter-acquisition-1 observes the exact getter wrapper, body and registration with no missing spans/global hooks: ConversationLocalMarksService.unread_ids_for opens a worker connection that an unrelated later browser callback closes on the reused executor actor. Earlier hidden lifecycle runs leave one such worker lease live. The later passing observer run does not establish deterministic retirement. Before production, exercise the original _load_manual_unread_rows with the actual service/file DB in a dedicated single-worker executor, observe original registration and physical close, and prove new-versus-borrowed lifetimes. Preserve its result/profile/revision publication and use the existing run_owned_db_call boundary only for qualified stock unread-row producers. Add the same finite ownership to the browser-history unread producer sharing this original service route, preserving original custom/in-memory calls. Run targeted RED/GREEN, existing manual-mark service controls and the actual hidden/resume lifecycle; no unknown foreign connection is closed and no timer/deadline/budget changes.


Context-display refinement after qualified RED: project an ephemeral serving target from the issued mapping without reading or writing the live target memo. Preserve the original cold family/model fallback and original one-argument direct/custom gateway routes. Pin defining gateway/runtime/cache method bodies, original partial receiver, static owner method slots and plain instance fields before any dynamic method binding. Attribute cold73 native opens to first-use tokenizer initialization, not the provider getter. Original cold/warm controls fail on one actual redundant getter/load each; direct route passes its original fresh getter/load. Exact post-body source/cache/partial changes refuse display publication through the existing coalesced retry. Source review complete; focused native GREEN remains pending.


Current-source native verification update: context display55PASS51.94s; finite unread-row2PASS16.97s with physical inspection before cleanup plus original manual service42PASS39.26s. Original tab navigation with a passive diagnostic1PASS47.33s and actual replay recovery,13 selected local codes/global0/overflow0/live0. Original nearest40 CI prefix40PASS66.66s/source stable, obstruction not yet reproduced. Keep these distinct from the final unobserved original three journeys, six-job matrix, whole performance/startup budgets and live UAT. Supplemental CI matrix and real Linux3.10 bundle controls qualify new leaves; they do not replace original acceptance. Raw and normalized installation/source hashes are labelled distinctly.


Unread reader provenance follow-up (before implementation): TASK-34404 retains the original custom/memory worker lifetime. Qualify pre-UI class replacement and same-function body replacement with actual worker/SQLite lifetime checks, then retain stock class/function/code/globals/defaults in the defining marks-service module before UI import. Stock file-backed producer closure and borrowed-owner preservation remain required. ADR required: no; ADR path: existing ADR-126; reason: routine refusal-metadata repair within the existing ownership boundary.


Original tab deadline observation refinement: retain the exact original-sync Task and root coroutine at entry; observe supported local yield/resume for that code only and capture its actual CPython await chain at the unchanged original settle deadline. The actual 3.12 Task/Future/coroutine mechanism controls pass and all six synthetic tasks settle; no global tasks/frames/payload enumeration. Production holding await remains pending the native original case.

Startup and shared-CI cohort diagnostics (registered before installation): track the reviewed original-prefix observer/launcher and the separately qualified complete cold/fresh-process warm liveness helpers. Original six acceptance jobs, original 240/900 second App/process bounds and all performance budgets remain unchanged. Source snapshots independently enumerate app, Tests and profile_core at each edge. Add separate Ubuntu original-prefix diagnosis and ordinary three-OS startup liveness; diagnostics do not replace acceptance. Normal selected Windows redirector ancestry/exits are qualified; forced-timeout descendant custody remains explicitly unqualified pending a separate no-App mechanism check. ADR required: no; ADR path: existing ADR-126; reason: test-only source/cohort observation retaining original authority and process boundaries.


2026-10-05 source checkpoint: genuine wrapped-getter body2RED then2GREEN (9.040s/9.621s XML; source stable); integrated sensitive-config21PASS155.59s with original native5000 ceiling and physical/census retirement. Defining getter metadata now pins actual body/code/globals/defaults plus body/wrapper closure cells at decoration time; custom prior replacements retain the ungrouped original route and qualified mid-build changes refuse before execution/memo publication. Unread source provenance2FAIL/2PASS21.80s then4PASS20.172s XML/source stable proves pre-UI class replacement and same-function body mutation no longer inherit stock worker closure. The stock reader is retained in its defining module and rechecked at actual worker entry; memory and instance overrides retain their original lifetime. Both repairs reviewed without remaining concrete source blockers. First unread4setup errors are excluded and its parent-only bootstrap marker correction does not alter the independent real child profile or native guards.

Original whole tab journeys remain1PASS2FAIL236.63s; no GREEN is inferred from the prior observed navigation pass. At the original failure/retry ten-second deadline, actual original sync Task is alive, uncancelled, global0/source stable and awaiting character_context.refresh_presentation_if_scope_changed at the shielded owned-task await. Original task/await-chain attribution is qualified; the owned child stage remains pending. The actual source-bound observer retains original assertions/deadlines, strong task/actor/code refs and explicit unmatched exception gaps. Separate exact-prefix Ubuntu/shared-startup and ordinary cold/warm three-OS diagnostics are now tracked; original six acceptance jobs remain unchanged. Installed tiny startup helper/process qualifier passes8.08s/noApp/source unchanged. Forced timeout descendant custody remains unqualified, so no complete startup-custody acceptance is claimed. Differential AST/Ruff against dev179Python files:639baseline/639current diagnostics, zero new. All final integrated budgets, remaining original failures, native matrix and real private DeepSeek UAT remain required; PR3023 remains draft.
