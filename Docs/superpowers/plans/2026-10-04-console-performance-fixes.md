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
