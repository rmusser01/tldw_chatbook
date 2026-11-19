---
id: TASK-33560
title: >-
  PERF-08 part 2: event-driven maintenance probe, warm MCP store reads,
  raw-participant stamp caches
status: Done
assignee:
  - '@codex'
created_date: '2026-09-29 20:29'
updated_date: '2026-10-07 03:28'
labels:
  - performance
  - backup-recovery
  - perf-audit-2026-09
dependencies:
  - TASK-33267
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Split out of PERF-08 (TASK-33267). Part 1 reuses confirmed admission evidence in acquire_storage (ADR-126 amendment, 2026-09-29), which with PERF-06 cut open() calls to _ui_ready by 82%. Part 1 leaves AC #5 undone, plus the other guarded paths that still re-derive on every call. (1) The backup-maintenance monitor probes native pause state at 10 Hz for the whole session (runtime_maintenance.py ~700; about 240 opens/s at idle, measured 2026-09-29). (2) Guarded MCP store reads (mcp_source_participants) run the full handshake on every call. (3) raw_participants._scope re-walks its pin chain, re-parses the registry in pause_requested, and re-runs companion_guard's direct _scope on every config operation. Apply the same stamp-validated reuse under the same ADR-126 amendment. A slower monitor changes how quickly a backup sees the app pause, so that part needs the owner's call before landing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The maintenance monitor no longer probes at 10 Hz, and the change to backup pause latency is measured and approved by the owner
- [x] #2 Warm guarded MCP store reads skip the admission handshake, with the same oracle and completeness tests as acquire_storage
- [x] #3 raw_participants' pause probe, pin walk and companion_guard scope reuse stamp-validated evidence, and the reuse-vs-derivation oracle still matches under every mutation
- [x] #4 Idle open() calls per second fall by at least a further 50% on the boot/idle probe
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (assessment/amendment of existing reusable-evidence boundary)
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: extend the existing process-local, stamp-validated allowed-evidence reuse to the remaining guarded paths without changing native permission or drain ownership.
1. Trace the actual 1 Hz maintenance probe, MCP/raw/config companion callers and prior owner approval. Measure the current idle/admission cost and pause latency before changing source.
2. Specify the smallest shared reuse path and complete mutation dependencies under ADR-126; document any required amendment before implementation.
3. Add failing warm-path and reuse-versus-full-derivation controls, then implement cache reuse with per-call gates, complete stamps, settle margin, epoch invalidation, exact physical pins and unchanged fallback reasons.
4. Measure a further >=50% reduction in actual idle opens and owner-approved pause latency without changing work, caps or census ceilings. Run targeted security/completeness/cleanup checks and independent review; close only after all criteria are evidenced.

Reviewed concrete reuse design (2026-10-06): independent preflight Ready after explicitly including every consulted historical path-token chain. Use the current hold-owned _Evidence at Backup_Recovery/generation_witnesses._witnesses for its entire positive source-scope/paired-generation derivation; at Admission.pause_requested for parsed registry/groups only; at raw parent pin proof with a fresh independently owned descriptor; and at config companion metadata under its actual retained registry lock. Complete dependencies cover registry/control records/selector/all relevant current and historical roots/activation generation/required.json, with defensive result copies and fresh lease context. Preserve native contention results, selector/member inode/foreign overlap checks, existing counts, epoch/settle and exact full-derivation fallback. History reads retain migration write authority and validated fresh temporary children. Unknown historical/alias inputs remain fresh rather than guessing completeness. Establish paired actual boot/settled-idle baseline before source edits; microbenchmarks alone do not satisfy AC4. Add omitted-history dependency, mutation, publication race, native contention and physical-FD/uncertainty controls before implementation; measured >=50% actual idle reduction and original owner-approved 1Hz pause response remain completion requirements.

Prospective PR3037 integration: preserving merge of exact dev518277133cc1281e446387b06d59dc54fea9b727 after user authorization. Incoming Console failed-attempt settlement/retry and established-store read changes must compose with the unchanged approved hook/storage source. No directly overlapping production files. Qualify only the affected incoming Console/provider tests and exact disposal/warm-reader boundary controls, obtain immutable independent integration review, retain historical metric and NON-GREEN limitations, and close this reopened verification criterion before publication. ADR required: no new decision; existing ADR126 and ADR163/197 boundaries remain exact. Use protected head-matched merge only after fresh PR Fast Lane, all four UI shards, latency, Derived artifacts and current review conversations pass. No moving-dev update while CI runs, full sweep, original budget replay, retry/dispatch or protection weakening.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Extended existing bounded hold-owned _Evidence reuse to complete positive MCP generation witnesses, parsed native pause groups, config companion metadata and raw parent ancestry. Two independently bracketed positive derivations, complete current/foreign/historical/control/activation stamps, one-second settle, epoch/PID/names and defensive copies are required. Every per-call lease context, actual registry lock/gate/flock, member identity and separately owned parent descriptor stays fresh. Fresh children borrow only a separately proven admitted directory and still validate their complete chain/leaf after counting; file-only roots cannot prove parents. Mismatch, uncertainty, aliases, pending/absent controls and excluded platform paths take the original derivation/reason. Filesystem work stays outside the coordinator lock. ADR required: yes; prospective/measured amendment in backlog/decisions/126-complete-local-backup-and-recovery.md. No new cache framework or dependency.

AC1: preserved the existing Sep29 owner-approved1.0s monitor (runtime_maintenance.py's original owner-decision comment). Actual native intent at near-worst scheduling phase: notice0.976281s, real localpause1.069559s, exclusive maintenance1.108681s, no app refusal, normal resume. This stimulated checkpoint is separate from idle and precedes final memory-lock-placement/concurrent-publication corrections; interval/native protocol unchanged.
AC2/3: final targeted safety52PASS/F/E/S0/XML12.642s, exact warm-reader/full-derivation/omitted-dependency mutation oracles, native contention, activation/historical retarget, foreign alias/file-only/fresh-child collision, physical FD retirement/uncertain close, publication/concurrency/settle and no-FS-under-lock controls. Separate persisted permission parse-pause controls2PASS/XML1.390s conserve exact original storage_locally_paused refusal and stored bytes; selections NEVER SUM.
AC4: exact frozen probe da5ba790b5d59549bab2345f2418a98dc47a24fa6c77ab58e251039c4d94539f, real TldwCli/ChatScreen and drained boot fleet, eight actual profile leases at both edges; same10nativeprobes/40credentialpolls/timers/work/privateprofile. Baseline884nativeopens/10.001132s (88.390/s) -> final286/10.000684s (28.598/s),67.64561%lower rate. Boot41034->16571opens; single UI3.51457->3.64624s does NOT certify startup speed. No original completed budgets/suites replayed.

Portable source/evidence/scripts/static/failed-history: Docs/superpowers/qa/2026-10-06-task33560-raw-evidence/{report.md,manifest.json,raw-evidence.zip,boot_idle_probe.py,paired_probe_source.txt}. Manifest06a1a9ab2533bc8d4e3b81d5ad5f1427292b57893e70c941a5ef5f3b836d8bfd and archivea3769948918a501c4a1f008dad379f9ef570c78bfb5be0d8bd50ae849b9b695a; parent authenticated all11currentfiles and87archivedentries. Independent immutable source review recorded below before closure.

Limits/history: first35.06%after failedAC4; unbound/alias/collection/setup attempts, abandoned extra root optimization RED, earlier FD fixture failures, all raw/XML/cache/foreign temporary cleanup warnings retained. Related selection stays NON-GREEN6PASS1defaultFalse-seedFAIL; attempted real persisted RMW finish exposes unchanged post-pause nested binding refusal. No nested permission-RMW finish/cause/bypass/resource certificate or unrelated source fix. Static exactly22inherited signatures, noNEW; new test/probe and changed production ranges lint/format clean, whitespacePASS. No broad clean-suite/native non-macOS/full capture/restore/provider/transport qualification. Native root, SQLite parent and activation authority stay fresh. Task scope stops at the measured target.

Independent source Ready/no actionable findings: QA independent-review.json (SHA c444b42070663677855ce7167d8736b35568e21b38b3b192901a725102a07a87). Separate combined composition/doc review Ready/no extra targeted check: composition-review.json (SHA e75c2c2114d5b3bb8e2cab410b7c9871b30ad677928ffd4da364c3cc3e89b6ba). All source/evidence activity settled before CLI closure; no source edit after approval.

PR3037 authorized preserving integration: prospectiveplan dee6077a165275459136d7a2ba559f20f0ff6391 before exactdev518277133cc1281e446387b06d59dc54fea9b727 merge ebb84e496fe3a3d6cd1bdfdcd819e321efe1f99c. Approved source/tests/QA remain byte-exact; full31128 tree entries authenticate both sides. Fresh incoming58PASS/FES0/XML181.650s and separate affected-boundaries17PASS/FES0/XML110.583s; NEVER SUM. Independent hook/runtime and storage immutable integration reviews Ready/no actionable findings. Portable raw/XML/reviews/tree receipt: Docs/superpowers/qa/2026-10-06-pr3037-integration. Both selections preserve27lateforeignrm_rf warnings; no cleanup. Existing ADR126/163/197/097 preserved; no new decision. Historical67.64561% paired measurement is not a merged-app measurement because incoming established-store demand changes. Earlier NON-GREEN/resource/cause/platform/provider/style limits remain; no original budgets or full sweep replay. ReopenedAC3 closed only after reviews and targeted evidence settled. Hosted current-head gates and actual protected merge remain pending at this local closure.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
