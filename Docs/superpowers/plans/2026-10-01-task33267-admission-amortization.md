# Ordinary admission amortization implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development or executing-plans to implement this plan task-by-task. Each task receives its own immutable review package.

**Goal:** Meet TASK-33267's existing admission-performance targets without changing backup/recovery authority or freshness.

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
- Replace 10 Hz monitor scheduling with a cross-process native probe at most one second apart when its prior probe has settled, plus relevant immediate local lifecycle probes. Preserve cancellation joining and all existing child/maintenance deadlines.
- Preserve missing/corrupt/unreadable MCP behavior, strict permission parser/generation distinctions, existing mutation fences and detached returns. Oversize input may be ineligible for caching and use its existing parse behavior; it must not acquire a new corruption/reset policy.
- Existing <0.5 ms ChaChaNotes admission-overhead and at least 80% boot-open reduction criteria remain unchanged. Unmet goals do not justify removing checks or claiming completion.
- Use the existing isolated worktree and associated TASK-33267 before edits. Official Backlog tools only. Normal commits/hooks, exact-head protected integration; no full suite, CI cancellation, blind retries or real-profile/credential access.

### Task 1: Reconstruct and retain the paired probe (Stage 1)

**Goal:** Obtain reproducible current and historical measurements before admission code changes.
**Status:** Not Started
**Files:** Create `Helper_Scripts/Benchmarks/backup_admission_benchmark.py`; create `Tests/Performance/test_backup_admission_benchmark.py`; update TASK-33267 through official tools.
**Interfaces:** Consume `CharactersRAGDB.transaction`, `_RepositoryParticipant.operation`, `TldwCli._ui_ready`, existing `Tests.network_guard`, Null keyring and private-profile environment semantics. Produce a standalone CLI with `--source`, `--phase transaction|boot`, `--iterations` and a metadata-only JSON receipt; subsequent tasks run the identical committed probe against their immutable source snapshots.

- [ ] Read the real transaction, participant and full-app startup callers before instrumentation. Use call-through timing and the existing audit-hook pattern from `test_console_keystroke_work_census.py`; never monkeypatch `os.open` or bypass admission. Include entry and retirement in ordinary admission overhead, separate DB-body time, and report native Windows handle/ACL work rather than counting it as zero.
- [ ] Write meaningful failing tests for private environment establishment before app import, fail-closed network isolation, exact source selection, preserved errors/exit code, bounded metadata and counters including retirement. Run those tests red using the existing private runner.
- [ ] Implement the minimum child probe using stdlib and existing helpers. Seed identical synthetic data and a disclosed fixed path depth, disable splash with existing settings, observe actual `_ui_ready` and positively close app/owners. Never export application values or raw child logs. Retain source/probe hashes, iteration counts and actual limits in private receipts.
- [ ] Run the same probe at historical `840ed2ca58` and current `495e0abbbf3e81f5042c985d9e06e797ab77b883` from separate retained source snapshots. State that September scripts were lost; this is a reconstruction, not a reuse of missing artifacts. Preserve unsuccessful runs as unsuccessful. Do not infer a host-load cause or waive numeric targets.
- [ ] Run targeted helper tests, scoped Bandit/lint/compile/diff; commit the probe and official Task notes normally. Obtain independent spec/quality review of probe implementation and its receipts before changing admission behavior.

### Task 2: Reuse checked hold evidence and remove warm serialization (Stage 2)

**Goal:** Ordinary borrowers avoid repeated root walks and record parsing while retaining every freshness barrier.
**Status:** Not Started
**Files:** Modify `tldw_chatbook/Backup_Recovery/storage_admission.py`, `bootstrap.py`, `admission.py` and, only at required pinning seams, `native_files.py` / `tldw_chatbook/Utils/private_paths.py`; create `Tests/Backup_Recovery/test_admission_evidence_reuse.py`; extend existing participant/related-path/admission tests only where behavior needs coverage.
**Interfaces:** Preserve public `acquire_storage(path=None, *, related_paths=())`, `StorageLease.execution_context(path)`, participant operation and maintenance APIs. Use existing `_Hold` ownership and pending/live/retiring sets. Internal evidence fields must be documented in the task report for Task 3; they must not be transferable admission capabilities.

- [ ] Trace every caller of changed pinning/control functions. Write red tests with real retained native holds for directory and ancestor rename/replacement, private-to-unsafe mode/ACL changes, same-inode record changes, new pending intent, registry-lock/gate replacement, related-path escape, selector/group change and fork. Assert refusal before actual dependent read/write, preserving foreign bytes.
- [ ] Add deterministic warm-borrower/close/pause races and unrelated-owner concurrency tests using events. A validator blocked on one owner must allow an unrelated owner to validate/transact; failed or unknown closes remain observable to drain. Cold followers still wait for the original initializer and incomplete foreign evidence still refuses.
- [ ] Retain root-to-leaf predecessors only on actual live holds. Revalidate current edge identities and current security posture through native handles before use. Paths whose complete existing trusted-link semantics cannot safely reuse pins keep their existing fully checked route. Do not replace Windows checks with POSIX stat or turn a cache failure into admission fallback.
- [ ] Read current bounded control bytes under existing registry lock/intent barriers, reuse only immutable equal-byte parsed records, and share one checked observation within the operation where existing before/after barriers permit it. Preserve all foreign registry, alias/absence, fingerprint, candidate-schema and qualification checks. Never accept a changed post-lock observation using the pre-lock result.
- [ ] Reserve warm holds under brief bookkeeping, run filesystem/native validation outside `_lock`, then recheck actual generation, cancellation, selection and scope. Cold-only initialization is single-flight. Keep tokens counted through native retirement and close descriptors only after the last accepted borrower positively retires. Capture/preview/recovery readers remain cold.
- [ ] Run new red-to-green tests and affected admission/participant/startup/related-path modules through the private runner, with unchanged timeouts. Run paired Bandit and lint/compile/diff; compare the retained probe. Commit scoped changes and official notes; obtain independent concurrency/security/spec review before proceeding.

### Task 3: Bound monitor scheduling and reuse MCP parsing (Stage 3)

**Goal:** Eliminate permanent 10 Hz monitoring and redundant parsing of the four guarded MCP JSON stores.
**Status:** Not Started
**Files:** Modify `storage_admission.py`, `runtime_maintenance.py`, `mcp_source_participants.py`, `MCP/local_store.py`, `MCP/server_target_store.py`, `MCP/unified_context_store.py`, `MCP/permission_store.py`; extend `Tests/Backup_Recovery/test_runtime_native_poll.py`, `test_mcp_source_lifetimes.py`, and permission read-error tests.
**Interfaces:** Keep `_poll_local_pause_requested()` and its cancellation/native joining unchanged. Use Task 2's actual hold ownership plus raw `state.holds`, `reader(source)`, `complete(state)` and existing source locks. Keep strict snapshot hashes based on exact observed bytes and parser-policy identity.

- [ ] Read `/private/tmp/task33267-json-monitor-map-R1qbzk/report.md` and actual caller bodies. Write red tests for parse reuse with a fresh byte read every time, same-inode/metadata-preserving edits, deep caller mutation, permissive cache followed by unreadable/missing/corrupt current data, strict/inventory generation differences, source/hold/fork/pause changes and failed retirement.
- [ ] Use a finite hold-owned immutable parsed payload cache; publish reusable observations only after positive raw retirement and current hold/selection recheck. Preserve UTF-8 decoding and permission duplicate/non-finite hooks. Each caller retains its existing fallback and schema policy; do not cache fallback defaults as successful records. Unqualified/custom sources keep existing behavior. Oversize reads are ineligible for cache reuse without changing their existing read/error policy.
- [ ] Add red monitor tests for one-second idle spacing, external intent detection, immediate relevant local lifecycle wake, snapshot/wait race, no overlapping native probes, subscription cleanup and cancellation/error retries. Pulse only relevant hold/startup/pause/retirement events, not every transaction bookkeeping notification.
- [ ] Implement initial native probe then bounded monotonic timeout/event scheduling with race-free subscriber retirement. Local events request a fresh native probe and grant no authority. Preserve same-task pause/resume, existing 30-second settlement and all short native maintainer deadlines.
- [ ] Run affected source/permission/monitor/runtime handoff/refusal tests, paired scoped Bandit and lint/compile/diff. Commit normally and obtain independent spec/quality review; rerun the identical probe without relaxing goals.

### Task 4: Qualify and integrate the finite change (Stage 4)

**Goal:** Demonstrate actual numeric improvements and all preserved platform contracts, then integrate reviewed work.
**Status:** Not Started
**Files:** Update this plan's stage statuses, approved spec verification record if needed, official TASK-33267 evidence and only existing finite/native workflow selectors required for changed code.
**Interfaces:** Consume retained historical/current/probe receipts and immutable stage commits; preserve their original source identities. Consume existing macOS/Linux/Windows native runner and finite product selection, with unchanged preflight, isolation and deadlines.

- [ ] Run identical retained boot/transaction probes on final source and compare to the disclosed historical reconstruction. Require <0.5 ms complete admission overhead and at least 80% fewer opens to actual `_ui_ready`; report full measurements and failures without inventing causes. Verify unrelated DB progress and include platform-native costs.
- [ ] Run targeted security/lifetime tests and appropriate actual native macOS, Linux and Windows execution for changed routes. Audit receipts/source/installed joins independently. No blanket original matrix rerun, duplicate artifact download, incomplete qualification or full suite.
- [ ] Run final scoped static checks and strongest available independent whole-branch review using an immutable diff; resolve verified issues with finite tests. Keep prior failed controls visible and distinct from passing current evidence.
- [ ] Publish a separate PR against latest dev, inspect review comments and current-head required CI, assess/rebase relevant upstream changes and qualify any changed executed routes. Merge only with exact head match and protections satisfied, under the requester's existing integration authorization.
- [ ] Finalize TASK-33267 through official tools after verified integration; preserve normal commits. Close only workstream items actually complete. Remove only this plan's disposable review workspace after final review; retain source/measurement evidence and other agents' artifacts.

## Preflight rulings

PR #2955's config/test/benchmark fixes remain separate. This branch starts from its latest-dev rebase; its 707-case qualification is not relabelled as admission-performance qualification. Current dev includes PERF-06 warm config changes, so measurements must use that baseline. No assertion that historical absolute timing reproduces the lost September protocol is permitted.
