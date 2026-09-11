---
id: TASK-32005
title: Replace selected local data with verified encrypted rollback
status: In Progress
assignee: []
created_date: 2026-09-08 00:00
labels:
- backup-recovery
dependencies:
- task-31978
- task-31993
- task-31997
- task-31999
- task-32000
- task-32001
- task-32002
- task-32003
- task-32004
updated_date: 2026-09-11 03:29
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Replacement cannot mutate live data before exact affected stored data and supported credentials have a verified encrypted rollback copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Replacement cannot mutate live data before exact affected stored data and supported credentials have a verified encrypted rollback copy.
- [ ] #2 Maintenance spans the entire safety-copy/publication/validation interval and final active inventory matches the approved generation.
- [ ] #3 Interrupted or failed replacement retains recovery evidence and never boots ambiguous or automatically active state.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute original component04 Task22 incrementally. First qualify source-preserving native-held SQLite snapshots (read-only SQLite can mutate live SHM): exact main/WAL private materialization, installed owner snapshot and limits/resource checks. Then bind reviewed affected originals including classified sidecars and damaged config, keep one native session through supported credential capture/encryption/authenticated verification, publication/installed validation/activation/commit. Add focused actual SQLite/WAL and interruption/cancellation/failure tests; do not claim replacement complete from prerequisite slices. Activation integration remains separately approval-blocked.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-04-restore-recovery.md#task-22)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Read-only original Task22 prerequisite analysis completed: /private/tmp/chatbook-sqlite-rollback-source-plan.md. Existing held capture cannot directly handle damaged config binding or coherent raw config credential omissions; public path-only rollback verifier correctly refuses unproven SQLite snapshots. Root native SQLite3.49.1 private fixture (abrupt child leaves committed WAL, read-only connection backup to disposable file) copied both main and WAL-only rows but changed original -shm bytes/timestamps. Exact observation /private/tmp/chatbook-rollback-wal-observation.json, disposable fixture /private/tmp/chatbook-rollback-wal-observe-jwsd4qz7. This is diagnostic evidence, not owner/native-session qualification. Therefore replacement safety capture must materialize stable source main+WAL privately before native SQLite reads, preserving reviewed physical originals; do not silently refresh target fingerprints. No Task22 production edits or completion claim.
Original Tasks14/22 SQLite source-preservation prerequisite authorized: add capture-only private main/WAL materialization using real active MaintenanceSession source identity and existing stage, exact installed SQLite owner policy, bounded pinned copying, hot-journal refusal, cancellation, before/after source checks and positive native retirement. No live SQLite read before materialization; preserve main/WAL/SHM and original fingerprints. Propagate existing reviewed ArchiveLimits/byte_budget from held capture, enforce cumulative/member/space bounds, leave ordinary noncapture DB behavior unchanged. Actual installed owner/WAL TDD and focused limits/path/cancel evidence required. No activation, damaged-config binding, replacement executor or semantic rollback verification authority in this slice; retain inherited storage traversal edits.
SQLite source-preservation prerequisite: materialize only native-session-qualified main/WAL into bounded private staging so read-only SQLite never alters original SHM. Bind private/source identities through reuse and native open/backup; quarantine ambiguous descriptor/native connection retirement. Final focused cohort: 57 passed (44 materialization, 8 existing core/domain, 2 capture factory refusals, 3 ordinary factory/copy-close controls); earlier public capture variants 3 passed. Production Ruff 53→53 and Bandit 7→7 existing findings, no new production findings. Exact inventory delta is only three _CaptureScope.sqlite_target rows (mkdir2/open2/write1). Round-two independent review pending. Task remains In Progress: this does not implement replacement execution, rollback receipts, activation association, or fence clearing. Evidence: /private/tmp/chatbook-capture-sqlite-report.md and /private/tmp/chatbook-capture-sqlite-independent-review.md.
Final bounded independent re-review APPROVED the SQLite prerequisite; all three review findings addressed. Original main/WAL/SHM preservation, cached private identity binding, native/descriptor quarantine and short-read behavior qualified. Root is staging only this slice, preserving inherited TTS/traversal/activation work; exact staged ownership inventory verification follows.
Root final gate: unchanged ownership inventory guard passed 11 tests in the existing isolated snapshot with this slice's exact index source/doc updates; staged diff check clean. Only the three new census rows and isolated source-preservation code/test/task notes are staged. Inherited storage traversal (100 lines) and TTS census rows remain unstaged. Independent review approved; 57 focused tests and unchanged production security/static baseline recorded above. This prerequisite is ready to commit; task remains In Progress.
Next original Task22 held rollback unit follows /private/tmp/chatbook-held-rollback-current-contract.md (current-source verified, source-preserving SQLite main/WAL copy already2196768bc, no redo). Implement internal replacement.capture_verify_rollback(candidate,plan,journal,destination,*,session,password,work_root,cancel,acknowledged_credential_issues): exact prepared local source/SQLite-sidecar groups and damaged-config target binding under same real session including unbound guard; existing owner snapshots/rawconfig preservation, supported credential extraction with exact omissions distinct from storage coherence, real age encryption/decrypt verification, typed exact snapshot/group rollback receipt before publication. Public path-only SQLite verifier stays refusing. Minimal newreplacement.py, journal receipt, publication sidecar/receipt checks, one narrowlychecked replacement capture_scope entry adjacent ordinaryscope preservingguards; no capture.py rewrite/newregistry/trustflags. Coordinate disjoint storage scope hunk with concurrentRAG rawlimit only. Focused realWAL/damagedconfig/encryptedroundtrip, nativewriters blocked throughcrypto, wrongscope/group/source andfail/cancelnegatives; scopedcensus/static/Bandit. Fullouterreplacement/rollback executor remains separate originalwork.
Held rollback independent review changes requested before commit: new source grouping/classifier must use installed per-item role, since recovered.media raw payloads currently inherit catalog SQLite policy and fail later verification; .payload-wal lookalikes must stay unclassified. Report /private/tmp/chatbook-held-rollback-independent-review.md. Root also reproduced declared same-path aliases (db.chachanotes.primary + notes.sync_bindings, shared_group shared, one exact main-dependency WAL) refusing rollback_sidecar_unclassified while the single-owner inventory succeeds. Preserve explicit shared alias support using exact installed role/physical same-path source+declared sharedgroup; do not infer cross-path WAL relationships or broaden generic sidecar authority. Add focused actual captured alias/mixedpayload regressions and rerun affected group after minimal fix. Prior final20+118 tests passed, no new static findings; unit remains uncommitted.
Actual default ChaChaNotes alias fixture exposed existing publication-preparation selector mix-up: _prepare requires every plan.selectors relocation locator in context.selectors, so real DB paths are incorrectly fingerprinted as configuration selectors (default schema exceeds bootstrap1MiB record bound). Resolve narrowly in publication preparation using actual verified manifest/config-owned restored selectors, consistent with existing finalize_candidate behavior; keep all relocation paths in plan digest, target fingerprints, registered held roots/pending target coverage. Fixture must pass only true config selectors and keep normal default SQLite page size. Do not raise bootstrap limits, weaken pending/native scope or shrink fixture solely to bypass this defect. This is necessary for actual installedDB heldrollback positive, same existing publication-owned unit.
Held SQLite rollback prerequisite and bounded corrections approved in /private/tmp/chatbook-held-rollback-correction-independent-review.md. Captures checked original main/WAL privately under retained native maintenance and verifies real encrypted rollback before typed journal receipt. Per-item mixed catalog/raw classification, exact same-path declared alias sidecar coverage, actual-config-only preparation with default1.3MiB ChaChaNotes DB; all plan/fingerprint/namespace checks retained.28 unique focused cases have passing evidence (27 initial plus repaired-test2rerun),83 affected preparation/publication/finalization passed; scoped fixRuff0/Bandit0, prior storage14 baseline diagnostics unchanged; +4 private verification producer rows. Reports /private/tmp/chatbook-held-rollback-report.md and ...-fix-report.md contain commands. Remains internal prerequisite: outer replacement/credential apply/projection/later rollback executor not complete.
Exact staged production and owner-inventory snapshot passed11 Architecture/test_backup_owner_inventory.py checks in10.83s (3 existing SyntaxWarnings). Seven-file index excludes concurrent/inherited work; cached diff check clean. Internal held rollback unit ready for scoped commit.
Original Task19 Unit5/Task22 held rollback composition authorized before edits: qualified entire Chroma original roots must use same private-copy group validator already used for capture/staging/installed validation, within actual retained native session and existing encrypted rollback receipt/readback. Current replacement._checked_originals blanket rollback_projection_group_validation_required can be removed only with positive native group evidence; no original engine opens or path-only receipt. Scope coordinated with Task32002 Unit5: restore_plan/replacement/publication/rag_inventory or small projection-specific composition module+focused tests; no RAG recovery.py/factory/ingestion edits (parallel Enhanced unit). Omitted/shared derived indexes need previewed retirement/quarantine and shared scope expansion/refusal, not prefix guesses or automatic rebuild. Outer full replacement/later rollback executor remains pending.
Unit5 before-edit journal seam authorization: held whole-Chroma rollback requires prepared exact group roots and matching group validation evidence, because existing rollback_requires_owner/sqlite_groups only qualifies SQLite and raw hash coverage must not bypass engine group validation. Publication author may minimally extend existing journal typed evidence and exact equality checks; public record/verify paths must refuse caller-supplied projection proof. Reuse session-held private group validation and encrypted readback; no new generic proof framework. Coordinate finalization seam with future isolated executor; no other author currently owns journal.py.
Projection publication Unit5 independently approved by runtime_final_capture: /private/tmp/chatbook-rag-publication-independent-review.md; exact6files/patchhash verified, no actionable findings. Owned15passed21.11s; native controlled80dada+6overlay1passed12s; Task15 excluded-lock correction independently committed3578727a3 with actual fresh/native2pass. Unit5 real whole-root retirement/publication and held raw encrypted Chroma group rollback, no outer executor/laterrollback/UI claim. Root staging exact6files and related tasks for census/commit.
Root exact staged owner census11passed7.33s, frozen6files unchanged and no new census rows needed. Committing bounded projection publication/held Chroma rollback unit after independent approval; original full tasks remain In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->