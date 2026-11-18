---
id: TASK-33267
title: 'PERF-08: Amortize Backup_Recovery storage admission (ADR-126 amendment)'
status: Done
created_date: 2026-09-28 18:02
dependencies:
- TASK-33260
labels:
- performance
- backup-recovery
- database
- adr
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
updated_date: 2026-10-02 15:02
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every outermost guarded call re-derives admission evidence from disk. It walks directory chains from / with one open() per component, reads registry.json about 7 times and takes a flock, under a process-wide initializing section with a 10 ms poll-wait. That is about 245 opens and 4-15 ms per DB transaction on about 12 DB owners, 207k open() calls to reach _ui_ready, about 3,400 open()/s at idle, and 240 admissions per MCP visit. The backup-maintenance monitor probes at 10 Hz forever. Needs owner decision D1: an ADR-126 amendment allowing generation-scoped admission evidence re-checked through held descriptors. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-08; every issue with file:line is listed under PERF-08 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ADR-126 carries an approved amendment describing generation-scoped admission evidence and the preserved invariant
- [x] #2 Reused evidence never admits what the full derivation would refuse: a moved, replaced or re-permissioned admitted directory or verified ancestor sends the call back through the derivation before dependent I/O (security tests cover path-level swaps, not only fstat of held fds)
- [x] #3 Per-transaction admission overhead on ChaChaNotes is under 0.5 ms (benchmark), and transactions on unrelated DBs no longer serialize
- [x] #4 open() calls to reach _ui_ready fall by at least 80% versus the 840ed2ca58 baseline on the same probe
- [x] #5 The 10 Hz maintenance monitor and warm MCP store reads are split to TASK-33560 (PERF-08 part 2); this task delivers acquire_storage reuse
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
**Part 1: reusable admission evidence in `acquire_storage`,** under the ADR-126 amendment approved 2026-09-29 (PR #2911).

**What reuse requires**
- A warm ordinary acquisition on a live native hold may reuse the allowed result of the unmodified derivation while two sets of stamps stay identical on the same call:
  - posture `(dev, ino, type, mode, uid)` of every component of every chain the derivation walks;
  - content `(dev, ino, size, mtime_ns, ctime_ns)` of the bootstrap and admission dirs, registry.json/.lock, the enrollment marker, every control record, the selector (bound) and native_qualification.json.
- Evidence lives on `_Hold`. It is confirmed only after two consecutive full derivations bracket identical stamps, with content change times at least 1 s old.
- Evidence is never kept with:
  - pending records;
  - absence-proved roots;
  - symlinked chains;
  - unqualified storage;
  - startup reacquisition;
  - on Windows (no raw `st_ctime` there).

**What the reuse path does**
- Re-runs every in-memory gate.
- Re-checks the read-only mount flag.
- Counts the lease before a final content check.
- Never enters `initializing`, so unrelated DBs stop serializing.

**Epoch.** In-process writers advance `bootstrap._admission_epoch`: `Admission._write`, `Admission._write_new_record`, the two activation publishers, and the profile replace in `preserve_owned_binding`. This is hardening; the stamps are the detection mechanism.

**Measured** (scratch profile; dev 82c905000d as base)
- Warm acquire+close:
  - unbound 2.36-3.66 -> 0.075 ms median;
  - bound 3.60-4.95 -> 0.10-0.14 ms (AC #3: < 0.5 ms).
- Four threads on unrelated paths:
  - unbound 291 -> 3,284 acquisitions/s;
  - bound 154 -> 2,166/s (AC #3: no longer serialized).
- open() calls to `_ui_ready`:
  - 194,696 -> 40,838 (-79%) alone;
  - 35,402 (-82%; -83% against the audit's 207k at 840ed2ca58) with PERF-06 (#2903), which removes the warm config handshakes (AC #4).
- Settled idle opens/s (the probe waits for the staggered boot fleet to drain):
  - dev 3,024;
  - this branch 729 (-76%);
  - with PERF-06, 311 (-90%).
- An earlier idle figure (19,805 -> 2,016) counted first-boot one-shots as idle and was wrong. The largest of those one-shots is the bundled Buddy install, which is TASK-33561 and not an idle loop.
- The remaining idle sources are:
  - the 10 Hz monitor and the MCP store/raw-participant paths (TASK-33560);
  - `get_user_data_dir` (PERF-07, TASK-33266).

**Correction.** The amendment's first draft said a moved or replaced admitted directory "is still refused". The oracle shows the derivation admits a data directory renamed away and recreated in place, while refusing a re-permissioned ancestor. The invariant and AC #2 now state what is preserved: equivalence with the derivation.

**Bug found by benchmark.** Concurrent derivations kept replacing each other's confirmed evidence with unconfirmed copies, so reuse never engaged under concurrency. Fixed in 359ad4a346, and pinned by a concurrency test that fails 3/3 against the old logic.

**Tests** (`Tests/Backup_Recovery/test_admission_evidence_reuse.py`)
- A 12-mutation oracle, bound and unbound, including subprocess writers:
  - pending records, related and unrelated;
  - registry replace and registry intent;
  - in-place profile edit;
  - marker replace;
  - selector edit;
  - group-writable ancestor;
  - data dir renamed and recreated;
  - data dir swapped for a symlink;
  - bootstrap root removed.

  Reuse must match the derivation exactly, and five mutations are pinned as refusals.
- A macOS dependency-completeness trace: every path the derivation reads is stamped. It fails when registry.json is dropped.
- Engagement, settle-margin, epoch and concurrency tests.
2026-10-02 integration reconciliation: dev113e435 merged PR2911 (ADR amendment) and PR2919 (ordinary acquire_storage evidence reuse) while the separate approved backup followup/performance work was being prepared. Preserve this Task's upstream Done/part1 acceptance record and all prior measurements; they are not the reconstructed complete-transaction probe. The requester approved the retained-source/full-byte/native-hold amortization design on2026-10-01. Its reconstructed Task1 probe and independently reviewed signal-failure fix retain separate original identities. Remaining approved monitor/MCP/raw/current-byte/native-platform work will build on the merged implementation under existing TASK-33560; do not duplicate it or relabel old receipts as dev113 qualification. The proposed design doc added by PR2955 is historical input and must be reconciled with the already accepted ADR before further production edits.
Historical branch proposal notes retained during latest-dev rebase; their awaiting-approval state predates the requester's later yes and the merged part1 implementation:
Concrete proposed ADR-126 amendment: Docs/superpowers/specs/2026-10-01-task33267-admission-amortization-design.md. Independent read-only review /private/tmp/task33267-admission-design-review-wMGTxE/report.md. Hold-bound descriptor/parsed-record reuse retains fresh pathname, ancestor, ACL, intent and native gate barriers; no global mutex across disk validation/transactions, one-second bounded native monitor probe, byte-fresh MCP parse cache. Existing numeric goals remain unproven and will not be met by weakening checks. Awaiting owner approval of this architectural contract before performance implementation.
2026-10-01 proposal remains awaiting the owner's explicit amendment decision; no admission performance implementation made. Independent design review is linked in the committed proposal. Original September probe scripts were lost, as the source audit itself records; any reconstruction must be a disclosed identical protocol on historical/current source rather than falsely claiming reuse of preserved scripts. Existing <0.5ms /80% /concurrency targets remain unmeasured and unwaived. TASK-33370/33373 finite fixes and unchanged benchmark budgets now pass separately; earlier current/prior timing failures remain retained without an invented host-load or regression cause.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
