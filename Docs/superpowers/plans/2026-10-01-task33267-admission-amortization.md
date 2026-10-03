# Ordinary admission amortization implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development or executing-plans to implement this plan task-by-task. Each task receives its own immutable review package.

**Goal:** Complete TASK-33560's remaining approved admission work and original performance targets without changing backup/recovery authority or freshness.

**Architecture:** Retain verified predecessor chains and immutable parsed observations on existing native holds. Ordinary borrowers revalidate current paths, security posture and bytes outside the coordinator mutex; capture and recovery keep their cold paths. Use existing MCP fences and monitor ownership.

**Tech Stack:** Python, existing ctypes Windows filesystem interface, native flock/LockFileEx, SQLite, pytest and existing private-profile/network-guard helpers. No new dependency.

**Spec:** `Docs/superpowers/specs/2026-10-01-task33267-admission-amortization-design.md`, approved by the requester on 2026-10-01; ADR-126 ordinary admission amendment.

ADR required: yes.
ADR path: `backlog/decisions/126-complete-local-backup-and-recovery.md`.
Reason: hold-owned evidence changes ordinary admission ownership and concurrency; the approved amendment preserves its security and retirement boundaries.

## Global Constraints

- Cache entries never grant authority. Fresh pathname-to-object bindings cover every verified ancestor, owner/mode/ACL, exact source selection, native gates, registry publication and pending intent before dependent I/O.
- Reuse current bounded bytes only after full byte comparison; inode/mtime, process epoch, TTL or a watcher alone cannot authorize reuse. Return detached mutable caller values.
- Evidence belongs to the actual live native `_Hold`, PID, exact bootstrap/config/source selection and namespace group. Preserve trusted-link rules/hop limits and Windows native identity/ACL/reparse checks.
- No filesystem validation, native waiting or unrelated DB transaction runs under the global coordinator mutex. Cold initialization remains single-flight; warm borrowers reserve and recheck liveness.
- Closing fences new borrowers; accepted operations and native resources retire positively before descriptors close. Failed retirement stays visible. Fork, source/group change, pause, maintenance and failed validation invalidate reuse.
- Capture, preview, restore, publication and arbitrary materializer paths retain their fully checked paths. No execution-log or credential/backend-discovery cache.
- Reuse the upstream one-second monitor cadence; complete an initial and relevant immediate local lifecycle probe with race-free bounded scheduling. Keep cross-process probes at most one second apart when the prior probe has settled, cancellation joining and all existing child/maintenance deadlines.
- Preserve missing/corrupt/unreadable MCP behavior, strict permission parser/generation distinctions, existing mutation fences and detached returns. Oversize input may be ineligible for caching and use its existing parse behavior; it must not acquire a new corruption/reset policy.
- Existing <0.5 ms complete ChaChaNotes transaction boundary (entry and retirement, excluding the measured SQL body), at least 80% boot-open reduction, and at least 50% further real idle-open reduction criteria remain unchanged. Unmet goals do not justify removing checks or claiming completion.
- Use the existing isolated worktree and associated TASK-33560 before edits; preserve upstream TASK-33267 Done/part 1. Official Backlog tools only. Normal commits/hooks, exact-head protected integration; no full suite, CI cancellation, blind retries or real-profile/credential access.

### Task 1: Reconstruct and retain the paired probe (Stage 1)

**Goal:** Obtain reproducible current and historical measurements before admission code changes.
**Status:** Complete
**Files:** Retain the existing `Helper_Scripts/Benchmarks/backup_admission_benchmark.py`; retain `Tests/Performance/test_backup_admission_benchmark.py`; record subsequent evidence on TASK-33560 through official tools.
**Interfaces:** Consume `CharactersRAGDB.transaction`, `_RepositoryParticipant.operation`, `TldwCli._ui_ready`, existing `Tests.network_guard`, Null keyring and private-profile environment semantics. Produce a standalone CLI with `--source`, `--phase transaction|boot`, `--iterations` and a metadata-only JSON receipt; subsequent tasks run the identical committed probe against their immutable source snapshots.

- [x] Read the real transaction, participant and full-app startup callers before instrumentation. Use call-through timing and the existing audit-hook pattern from `test_console_keystroke_work_census.py`; never monkeypatch `os.open` or bypass admission. Include entry and retirement in ordinary admission overhead, separate DB-body time, and report native Windows handle/ACL work rather than counting it as zero.
- [x] Write meaningful failing tests for private environment establishment before app import, fail-closed network isolation, exact source selection, preserved errors/exit code, bounded metadata and counters including retirement. Run those tests red using the existing private runner.
- [x] Implement the minimum child probe using stdlib and existing helpers. Seed identical synthetic data and a disclosed fixed path depth, disable splash with existing settings, observe actual `_ui_ready` and positively close app/owners. Never export application values or raw child logs. Retain source/probe hashes, iteration counts and actual limits in private receipts.
- [x] Run the same probe at historical `840ed2ca58` and current `495e0abbbf3e81f5042c985d9e06e797ab77b883` from separate retained source snapshots. State that September scripts were lost; this is a reconstruction, not a reuse of missing artifacts. Preserve unsuccessful runs as unsuccessful. Do not infer a host-load cause or waive numeric targets.
- [x] Run targeted helper tests, scoped Bandit/lint/compile/diff; commit the probe and official Task notes normally. Obtain independent spec/quality review of probe implementation and its receipts before changing admission behavior.

Task1 and I1 fix reviews passed; fixed probe SHA34278facac896ecc0e4ed8a3319243d3501272e87692a858779c6b449a475428 is retained at `/private/tmp/task33267-probe-i1/retained/Helper_Scripts/Benchmarks/backup_admission_benchmark.py`. Historical495/probe identities remain historical. Fresh identical fixed-probe controls at historical840 and currentf337 are retained under `/private/tmp/task33560-paired-vtq2fabm`; independent `/private/tmp/task33560-fixed-baseline-independent-review.md` passed all source/probe/receipt joins. Complete transaction medians are3.446000/1.159021ms; boot-open medians126252/19640 (84.44381079% reduction). The <0.5ms complete boundary remains unmet; native Windows/Linux and idle outcomes remain unqualified. Later dev changes boot imports and scheduling, so these measurements keep their exact historical identities and disclosed mode/receipt-retention/stat limits.

### Task 2: Reuse checked hold evidence and remove warm serialization (Stage 2)

**Goal:** Ordinary borrowers avoid repeated root walks and record parsing while retaining every freshness barrier.
**Status:** Complete
**Files:** Modify `tldw_chatbook/Backup_Recovery/storage_admission.py`, `bootstrap.py`, `admission.py`, the actual shared validator callers in `participants.py` and, only at required pinning seams, `native_files.py` / `tldw_chatbook/Utils/private_paths.py`; extend the existing `Tests/Backup_Recovery/test_admission_evidence_reuse.py`; extend existing participant/related-path/admission tests only where behavior needs coverage.
**Interfaces:** Preserve public `acquire_storage(path=None, *, related_paths=())`, `StorageLease.execution_context(path)`, participant operation and maintenance APIs. Use existing `_Hold` ownership and pending/live/retiring sets. Internal evidence fields must be documented in the task report for Task 3; they must not be transferable admission capabilities.

- [x] Trace every caller of changed pinning/control functions. Write red tests with real retained native holds for directory and ancestor rename/replacement, private-to-unsafe mode/ACL changes, same-inode record changes, new pending intent, registry-lock/gate replacement, related-path escape, selector/group change and fork. Assert refusal before actual dependent read/write, preserving foreign bytes.
- [x] Add deterministic warm-borrower/close/pause races and unrelated-owner concurrency tests using events. A validator blocked on one owner must allow an unrelated owner to validate/transact; failed or unknown closes remain observable to drain. Cold followers still wait for the original initializer and incomplete foreign evidence still refuses.
- [x] Retain root-to-leaf predecessors only on actual live holds. Revalidate current edge identities and current security posture through native handles before use. Paths whose complete existing trusted-link semantics cannot safely reuse pins keep their existing fully checked route. Do not replace Windows checks with POSIX stat or turn a cache failure into admission fallback.
- [x] Read current bounded control bytes under existing registry lock/intent barriers, reuse only immutable equal-byte parsed records, and share one checked observation within the operation where existing before/after barriers permit it. Preserve all foreign registry, alias/absence, fingerprint, candidate-schema and qualification checks. Never accept a changed post-lock observation using the pre-lock result.
- [x] Reserve warm holds under brief bookkeeping, run filesystem/native validation outside `_lock`, then recheck actual generation, cancellation, selection and scope. Cold-only initialization is single-flight. Keep tokens counted through native retirement and close descriptors only after the last accepted borrower positively retires. Capture/preview/recovery readers remain cold.
- [x] Run new red-to-green tests and affected admission/participant/startup/related-path modules through the private runner, with unchanged timeouts. Run paired Bandit and lint/compile/diff; compare the retained probe. Commit scoped changes and official notes; obtain independent concurrency/security/spec review before proceeding.

### Task 3: Bound monitor scheduling and reuse MCP parsing (Stage 3)

**Goal:** Complete immediate lifecycle/native monitor scheduling around the existing one-second cadence and remove redundant parsing of the four guarded MCP JSON stores.
**Status:** In Progress
**Files:** Modify `storage_admission.py`, `runtime_maintenance.py`, `mcp_source_participants.py`, `MCP/local_store.py`, `MCP/server_target_store.py`, `MCP/unified_context_store.py`, `MCP/permission_store.py`; extend `Tests/Backup_Recovery/test_runtime_native_poll.py`, `test_mcp_source_lifetimes.py`, and permission read-error tests.
**Interfaces:** Keep `_poll_local_pause_requested()` and its cancellation/native joining unchanged. Use Task 2's actual hold ownership plus raw `state.holds`, `reader(source)`, `complete(state)` and existing source locks. Keep strict snapshot hashes based on exact observed bytes and parser-policy identity.

- [ ] Read `/private/tmp/task33267-json-monitor-map-R1qbzk/report.md` and actual caller bodies. Write red tests for parse reuse with a fresh byte read every time, same-inode/metadata-preserving edits, deep caller mutation, permissive cache followed by unreadable/missing/corrupt current data, strict/inventory generation differences, source/hold/fork/pause changes and failed retirement.
- [ ] Use a finite hold-owned immutable parsed payload cache; publish reusable observations only after positive raw retirement and current hold/selection recheck. Preserve UTF-8 decoding and permission duplicate/non-finite hooks. Each caller retains its existing fallback and schema policy; do not cache fallback defaults as successful records. Unqualified/custom sources keep existing behavior. Oversize reads are ineligible for cache reuse without changing their existing read/error policy.
- [ ] Add red monitor tests for one-second idle spacing, external intent detection, immediate relevant local lifecycle wake, snapshot/wait race, no overlapping native probes, subscription cleanup and cancellation/error retries. Pulse only relevant hold/startup/pause/retirement events, not every transaction bookkeeping notification.
- [ ] Implement initial native probe then bounded monotonic timeout/event scheduling with race-free subscriber retirement. Local events request a fresh native probe and grant no authority. Preserve same-task pause/resume, existing 30-second settlement and all short native maintainer deadlines.
- [ ] Run affected source/permission/monitor/runtime handoff/refusal tests, paired scoped Bandit and lint/compile/diff. Commit normally and obtain independent spec/quality review; rerun the identical probe without relaxing goals.

### Task 4: Qualify and integrate the finite change (Stage 4)

**Goal:** Demonstrate actual numeric improvements and all preserved platform contracts, then integrate reviewed work.
**Status:** In Progress
**Files:** Update this plan's stage statuses, approved spec verification record if needed, official TASK-33560 evidence the minimum separate real-idle benchmark and its focused contract tests, and existing finite/native workflow selectors required for changed code.
**Interfaces:** Consume retained historical/current/probe receipts and immutable stage commits; preserve their original source identities. Consume existing macOS/Linux/Windows native runner and finite product selection, with unchanged preflight, isolation and deadlines.

- [ ] Run identical retained boot/transaction probes on final source and compare to the disclosed historical reconstruction. Require <0.5 ms complete admission overhead and at least 80% fewer opens to actual `_ui_ready`; report full measurements and failures without inventing causes. Verify unrelated DB progress and include platform-native costs.
- [ ] Measure three paired real full-app 60-second idle windows per OS against the disclosed contemporary one-second/schema75 pre-optimization source, with normal live timers, identical readiness settlement, all-thread/native counters and positive cleanup. Require at least50% reduction; no manual polls, parked timers, cadence-only substitution or denominator change. Reuse the existing runner/wheel/isolation seams and preserve the fixed transaction/boot probe unchanged.
- [ ] Run targeted security/lifetime tests and appropriate actual native macOS, Linux and Windows execution for changed routes. Audit receipts/source/installed joins independently. No blanket original matrix rerun, duplicate artifact download, incomplete qualification or full suite.
- [ ] Run final scoped static checks and strongest available independent whole-branch review using an immutable diff; resolve verified issues with finite tests. Keep prior failed controls visible and distinct from passing current evidence.
- [ ] Publish a separate PR against latest dev, inspect review comments and current-head required CI, assess/rebase relevant upstream changes and qualify any changed executed routes. Merge only with exact head match and protections satisfied, under the requester's existing integration authorization.
- [ ] Finalize TASK-33560 through official tools after verified integration; preserve TASK-33267 Done/part 1; preserve normal commits. Close only workstream items actually complete. Remove only this plan's disposable review workspace after final review; retain source/measurement evidence and other agents' artifacts.

## Preflight rulings

PR #2955's config/test/benchmark fixes remain separate. This branch is rebased onto published reviewed followupbca3e27b9d477daa5b7c4a4957451e9c6bdd4f9b/devfccf70d3b0cd21b0d44a906ec3a68b0633887684. Its7d19 affected432, d616 expanded42 and bca quit-flow6 phases retain their identities and are not admission-performance qualification. Dev already supplies PERF-06, PERF-08 part1, the one-second monitor constant and schema75; Task2 strengthens the existing Hold/Evidence and differential oracle rather than recreating them. Read-only reconciliation reports `/private/tmp/task33560-reconciliation-report.md` and `/private/tmp/task33560-expanded-dev-reconciliation.md` define remaining shared-seam gaps. Later quit guards do not change the Task2 admission sources; preserve their current cleanup behavior. Task3 retains schema4 MCP migration/error policy and positive HTTP/subprocess retirement. Historical measurements cannot be relabelled, and no claim reproduces the lost September absolute timings.


## Task2 safety handoff and current source

Task2 implementation/concurrency gates passed scoped SPEC and whole-stage QUALITY, including F1 custody, F2 counted pre-observation, F3 selection-error cleanup and F4 outside-mutex relative path normalization. Reports: /private/tmp/task33560-task2-fix3-independent-spec-review.md and /private/tmp/task33560-task2-fix3-independent-quality-review.md. Current normal-rebase source cdbb0f3cc51557b5efc922a191bf7dbe891bba93 on reviewed followup ba7388/current dev ecc0 preserves15 own patch matches/blobs-modes and fixedprobe34278. New affected phase /private/tmp/task33560-perf07-finite-t3j52cq4/combined/summary.json passes28 (12 upstream POSIX memo plus16 Task2), zero failures/errors/skips/parentnetwork,7752 stable Python/SQL pins and420 actual imports. Independent current-result/rebase attestation PASS /private/tmp/task33560-perf07-result-independent-review.md verifies exact selectors, current Git/source/import/JUnit/rebase joins, final executed storage bytes and unchanged original guards; no new issue. The final lint-normalized storage source is now directly executed in this phase, distinct from earlier9-case semantic/static carry.

Stage2 completion is implementation/safety completion, not numerical/native acceptance. Original65b complete3.7872705ms and historical boot arithmetic79.9013084941% failed their unchanged goals; later corrections are unmeasured. Broader failed controls/intentional failed-close owners stay recorded without resets. Task4 must qualify the <.5ms complete boundary, >=80% boot and >=50% further real idle goals with actual macOS/Linux/Windows costs. Windows warm storage reuse stays disabled where complete native pins cannot be safely reused; cold/native costs are measured rather than waived.

Task3 consumes actualHold/BorrowFrame positive-retirement interfaces and the accepted PERF07 applicability /private/tmp/backup2955-perf07-upstream-review.md. Input/result memos are never Hold authority. Use lifecycle-only native wakes with prior-start+1.0s due times and no overlap, and four store parse observations only after fresh complete bytes plus positive raw retirement/current servingHold validation. Read the stage handoff brief; no new cache authority, fallback/reset policy, dependency or deadline. Task4 minimal native/idle preparation /private/tmp/task33560-task4-minimal-qualification-plan.md stays NOT GO until accepted Task3 source.


## Task3 safety and Task4 implementation handoff

Task3 SPEC and QUALITY pass at faeaf4a5, with exact final38/175/12 covering phases225PASS. Earlier unsuccessful85/BASE/isolated controls remain visible; numerical/native/real-idle targets are not accepted. Normal performance-only rebase onto published reviewed PR2955 f6c71d91 / deve6ab66 yields fbf0e93e,17equal patches and23own blob-types-modes retained, fixed34278 probe unchanged. The new upstream trace work hint/write-generation loop is part of the contemporary idle baseline.

Stage3 stays In Progress only for its final identical probe/numerical gate; its code safety/quality is complete. Stage4 begins minimum harness/finite-selection implementation plus local contract checks while PR2955 requiredCI settles. Actual native and numerical qualification requires accepted harness reviews and appropriate protected integration. No source identity, target, platform requirement or deadline changes.
