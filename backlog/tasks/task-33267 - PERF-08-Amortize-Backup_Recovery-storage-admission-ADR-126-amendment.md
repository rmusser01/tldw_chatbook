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
updated_date: 2026-10-02 16:29
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
Historical Task1 record from prior performance head6c90d1c1ed39ef89b3412686058d2a0e78832091, preserved after normal rebase. The following notes keep their original dates/source/probe/test identities and any then-pending status; they do not reopen upstream33267 or qualify current33560. Remaining approved work belongs to33560.

Concrete proposed ADR-126 amendment: Docs/superpowers/specs/2026-10-01-task33267-admission-amortization-design.md. Independent read-only review /private/tmp/task33267-admission-design-review-wMGTxE/report.md. Hold-bound descriptor/parsed-record reuse retains fresh pathname, ancestor, ACL, intent and native gate barriers; no global mutex across disk validation/transactions, one-second bounded native monitor probe, byte-fresh MCP parse cache. Existing numeric goals remain unproven and will not be met by weakening checks. Awaiting owner approval of this architectural contract before performance implementation.
2026-10-01 proposal remains awaiting the owner's explicit amendment decision; no admission performance implementation made. Independent design review is linked in the committed proposal. Original September probe scripts were lost, as the source audit itself records; any reconstruction must be a disclosed identical protocol on historical/current source rather than falsely claiming reuse of preserved scripts. Existing <0.5ms /80% /concurrency targets remain unmeasured and unwaived. TASK-33370/33373 finite fixes and unchanged benchmark budgets now pass separately; earlier current/prior timing failures remain retained without an invented host-load or regression cause.
2026-10-01: The requester explicitly approved the admission-amortization design with 'yes'. Proceed under Docs/superpowers/specs/2026-10-01-task33267-admission-amortization-design.md; architectural approval is satisfied. Preserve fresh pathname/ancestor/owner/ACL/current-byte/native checks and all deadlines. Numeric <0.5ms /80% criteria remain unmeasured. Implementation will be separate from PR #2955; baseline must include latest dev ab4df99959545e37d8d2048c1c1ce15fd914721d and disclose reconstruction of lost September probe scripts.
2026-10-01 Stage 1 private probe implementation: Helper_Scripts/Benchmarks/backup_admission_benchmark.py plus Tests/Performance/test_backup_admission_benchmark.py. Reconstructed protocol explicitly discloses lost September scripts, hashes exact git-archive snapshots, isolates HOME/USERPROFILE/XDG/config and Null keyring before app imports, installs Tests.network_guard, times participant entry AND retirement, and captures actual _ui_ready synchronously. Existing stdlib-only Notes/git_process_containment.py supplies POSIX process-group / suspended Windows Job Object admission and positive empty-tree proof; outer probe supervision is a new 300s bound, not a changed product/native deadline. TDD missing-probe red7FAIL (/private/tmp/backup-followup-check-7_l5rrd5), explicit-exit red1FAIL (/private/tmp/backup-followup-check-b_zgyl0z), group-ownership red1FAIL (/private/tmp/backup-followup-check-8digekdm), final targeted green9PASS (/private/tmp/backup-followup-check-cyxpche9), zero undrained egress. Ruff check/format, compile2 and scoped Bandit0 findings/errors pass (B101 only exempt for pytest assertions; both paths absent in baseline). Frozen baseline495 and historical840 snapshots retained under /private/tmp/task33267-paired-gcigzter; numeric measurements and independent review still pending. No admission behavior or budgets changed.
Stage 1 retained first committed probe f6d233df939cd02f207815e8d5c3bdd8b8310e1c (SHA256 07bc8efe69d385ac97bd2e5a8f9ce0a9c61385d90a5a7e2ed0a2c10364c9afc1) and all controls under /private/tmp/task33267-paired-gcigzter/receipts. Both100-transaction controls completed/retired: historical participant-boundary median2.1449375ms, full transaction2.9312085ms, SQL body0.0113955ms; baseline3.1827085ms/4.331792ms/0.014979ms;197opens/transaction each. Both boot warmups reached actual readiness but failed ownership-retirement census (historical148262opens and9.739780333s, baseline120175opens and7.651455292s), hence no warm3-run/80% qualification. All failed receipts remain failed, no cause attribution. Harness amendment uses existing _begin_local_pause/_retire_current_thread_caches after public app/loop teardown, retains all outstanding set counts and live-child count, and reports conservative complete transaction-boundary median (total minus SQL body, including SQLite BEGIN/COMMIT and every guarded manager check). New cleanup test failed before implementation (/private/tmp/backup-followup-check-fgd304mp); final10PASS (/private/tmp/backup-followup-check-40690t2q), scopedBandit0/compile2/ruff/diff pass. Identical amended-source measurements still pending; admission behavior unchanged.
Stage 1 amendment a30e1ec8ec7535fb5ec7305258f39b5153aeb16c retained (probeSHA fd69777c6d8dfe07d858ee99019f3fe42fbd01839f8fa9a4038dda631da3fefa). V2 transaction100 pairs positively retired with complete-boundary median3.2431245ms historical/3.600042ms baseline. Both v2 boot warmups remain unsuccessful: historical9.102021s/148893opens, baseline10.113087584s/120447opens; each census retains1hold/2leases, no pending/active/raw/retiring operations and no live helper children; OS supervisor proves each owned process tree empty. Bounded code-owner diagnostic /private/tmp/task33267-paired-gcigzter/owner-diagnostic/code-owner-receipt.json proves residual db.chachanotes.primary registered handle on a dead foreign worker, not a logging sink. Both frozen source versions already expose CharactersRAGDB.quiesce_connections(timeout_seconds=0), which fences acquisitions, requires tracked uses and native transactions drained, and positively closes registered same-file handles. Final harness-only amendment uses that exact public method under existing local pause; no forced raw/foreign close, admission change or deadline change. Meaningful public-quiescence cleanup red1FAIL (/private/tmp/backup-followup-check-vycmjg79); targeted green11PASS (/private/tmp/backup-followup-check-dhtiupaa), scopedBandit0/compile2/ruff/diff pass. Third protocol attempt is final; any remaining ownership issue must stop measurement retries for reassessment. Prior unsuccessful receipts/probes stay preserved, numerical targets unwaived.
Stage 1 v3 probe ee2d20a1817f4534f46d574688a557cdd6a86a1b (SHA c1dc5b4fd0f04c3cdca7d5459f3c10316844a1b69209286ab698e7c378f2517f) completed both100-transaction and both warm3-boot controls with every resource census0 and positive OS tree retirement. These immutable v3 receipts remain preserved. A separate receipt-integrity fail-first control then proved an abrupt zero-exit child could reuse an older owned success JSON. Minimal removal of that owned receipt before launch now rejects MissingReceipt; red1FAIL /private/tmp/backup-followup-check-mbsdlr0z, green12PASS /private/tmp/backup-followup-check-mnwl3x76 (zero egress). Hardened probe SHA ba84865344410ad3e4432945b2763aa0a9a0ed9dce1ea8dcbacce956656333e5; Ruff/format, compile2, scoped Bandit0 findings/errors (pytest B101 only excluded), diff checks pass. Retirement algorithm and all original inputs/deadlines unchanged. One final paired run under this exact hardened probe remains pending so subsequent tasks use its identical reviewed identity; this distinct receipt-correctness fix is not another speculative cleanup protocol.
Stage 1 hardened final paired probe 0730a5a52bce69479ad749b0dbf72b7878a9aa1f (SHA ba84865344410ad3e4432945b2763aa0a9a0ed9dce1ea8dcbacce956656333e5) completed unchanged inputs:100 measured transactions/source;8 identical synthetic notes; private-profile depth7/DB depth13; warmup plus3 measured full-app warm boots/source to synchronous actual _ui_ready. Exact source digests revalidated unchanged. Historical participant median5.790ms /complete boundary7.6593335ms /full transaction7.691ms /SQL body0.0298955ms; baseline6.0054375ms /8.039750ms /8.066771ms /0.0290415ms.100 participant entry+retirement successes each,19700 os.open calls each. Warmboot opens historical[126856,126856,126646], baseline[99253,98078,97893], medians126856/98078 =>22.685564734817433% reduction; times historical[9.095820250,8.853355667,8.714501084]s, baseline[7.545524958,6.922429667,6.745042916]s. Every seed/warmup/run exits0, all six ownership sets0, no live helpers before reaping, positive OS process-tree empty proof, blocked network/foreign source modules0. Boot entry/retirement counters are readiness-cutoff snapshots; later teardown proves complete retirement separately. Both unchanged <0.5ms and80% targets fail at this preoptimization baseline and remain unwaived; no host-load or regression-cause claim. macOSarm64 CPython3.12.11 only; native Windows handle/ACL instrumentation and existing Job Object supervisor are present but Windows/Linux qualification not run. Retained final summary /private/tmp/task33267-paired-gcigzter/paired-final-summary.json SHA e663ff60faf70b7ca1e2718fb3ed5095bbdb96a2425b9dd54fba0693eba52e8d; bounded receipt catalog probe-receipt-catalog.json SHA19631a9845420a5e2793074db6a1d18844364a706143fbdaeac8abb35a4fe77c includes all16 trial receipts and previous failures under their original identities. Final12 helper tests pass; lint/format/compile2/scopedBandit0/diff pass. No production admission changes, full suite/native credential reruns, dependency/CI/benchmark-budget changes or push/merge. Independent Task1 spec/quality review pending; whole TASK-33267 remains In Progress.
Task1 independent review I1 fix round1 at BASE a3e3d257bcbaeedac7c8f51a86a15bb50bb94776: verified that max([0,-15]) hides a measured POSIX signal failure. Replaced only top-level aggregation with int(any(child exit_code !=0)); prerequisite/run raw return codes stay unchanged, nonzero CLI status is1 for negative/positive failures. Isolated main test mocks child returns (no app/native measurement) for successful prerequisite followed by0,-15,23; fail-first /private/tmp/backup-followup-check-tdd7wve1 has3cases/2FAIL (negative and positive normalization), then full targeted helper /private/tmp/backup-followup-check-boh36a4v has15PASS/0FAIL/ERROR/SKIP and0undrained egress using existing private runner/Null/network/original deadlines. Ruff format/check pass, compile2/diff pass, paired baseline/current touched-scope Bandit0findings/0errors with B101 only excluded for pytest assertions; static receipts /private/tmp/task33267-probe-i1. Amended probe SHA34278facac896ecc0e4ed8a3319243d3501272e87692a858779c6b449a475428. No production/admission/native/timing changes, app/native/paired controls reruns, source snapshot edits, dependency/CI/budget relaxation, push or branch change. Original all-zero final controls retain their validity and ba848653 measurement identity; amended probe is a separate unmeasured fix identity, not a relabel of retained receipts. Earlier report/commit identities remain historical after controller rebase onto158b516634468b23de01a05f5001dc526dd4f4aa; review fix and fresh independent review remain controller handoff, numerical/platform limitations unchanged.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
