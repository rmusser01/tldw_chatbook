---
id: TASK-31997
title: Implement staged credential exclusion and encrypted credential recovery
status: Done
assignee: []
created_date: 2026-09-07 23:56
labels:
- backup-recovery
dependencies:
- task-31978
- task-31989
- task-31990
- task-31991
- task-31992
- task-31996
updated_date: 2026-09-12 02:26
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local recovery behavior for this independently reviewable slice: Default archives remove managed secrets from all supported staged locations, including SQLite remnants, without modifying sources.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Default archives remove managed secrets from all supported staged locations, including SQLite remnants, without modifying sources.
- [x] #2 Credential inclusion and exact rollback require encryption and retain supported values with honest omission reporting.
- [x] #3 Restoration isolates credential scopes and never overwrites shared entries or claims remote authentication was recovered.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute original component03 Task14 only: typed installed credential policy inventory, staged exclusion by default, encrypted include/rollback, known semantic fields and aliases, keyring scope export/rebind without reading unrelated credentials. Read approved Task14/spec credential boundaries and existing adapters/config histories/server_credentials. TDD specific leaks and mode/copy-source immutability; retain unknown format refusal. No runtime publication or scope expansion. Focused security/lint/review evidence; no full suite.
<!-- SECTION:PLAN:END -->

## Design references

- [Approved specification](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-07-backup-recovery-03-capture-archives.md#task-14)
- [ADR-126](../decisions/126-complete-local-backup-and-recovery.md)

ADR required: yes

ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md

Reason: direct implementation of the approved recovery ownership, archive, and lifecycle contract; reuse ADR-126.

Before implementation, move this task to In Progress and copy its linked task steps into an Implementation Plan section. Keep implementation notes and completion evidence for after the work is finished. Do not mark criteria complete from this planning document.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Original Task14 excludes credentials by default. Root verified need for narrow setup-required guard in actual common Image_Generation/config.py and Video_Generation/config.py _resolve_secret functions: excluded restored config must not silently reuse fixed shared keyring slots. Implement staged marker recognized before implicit fallback, keeping ordinary unmarked resolution unchanged; focused no-keyring-read tests. This is existing credential exclusion scope under ADR126, no provider UI or new feature.
Implemented staged credential exclusion/include/rollback and scope planning/apply APIs under ADR126/component03 Task14. Exclusion reconstructs exact validated SQLite into fresh bytes, preserving FTS/hidden rowids and removing freed-page secrets; config histories/semantic locations handled without source mutation. Exact server/image/video setup-required guards prevent implicit keyring reuse. Supported explicit scopes plan/recheck isolated writes; unsupported implicit material is retained encrypted with honest manual-recovery issues for Task20. Unknown MCP args refuse exclusion rather than claim sanitization. Root source review plus independent review found/fixed destination keyring backend exception leakage. Final40 credential tests passed4.84s; related281 passed25.82s and exactSQLiteinventory29 passed46.50s. Ruff90→90/Bandit10LOW→10LOW no new existing-file findings, newmodulesclean/Bandit0. Reports /private/tmp/chatbook-credentials-report.md and /private/tmp/chatbook-credentials-review.md document exact interfaces, scoped coverage and future orchestration boundaries.
Reopened original Task14 on source-backed review findings while completing public capture: citation HMAC key/live DB reference, keyring-only generation provider config, and skill trust key-cache coverage are not yet represented. Report /private/tmp/chatbook-runtime-credentials-discovery-review.md. Existing staged exclusion/inclusion remains tested; do not remove runtime.credentials placeholder until exact affected-source coverage is resolved. No keychain enumeration or activation of restored grants authorized.
Incremental source-backed credential gaps implemented and reviewed: exact citation key references across all five whole-ChaChaNotes payload aliases, fixed generation keyring slots even without provider subsections, and recognized RAG profile secret fields. Excluded citation marker is refused by actual load/provision methods before ambient key access or regeneration; encrypted citation bytes retained for explicit manual recovery. Root/reviewer corrected earlier overly conservative trust interpretation: disk trust files contain historical metadata/MACs/encrypted skill content, not the derived keys in optional keyring cache. Default exclusion preserves those bytes with no cache read/error; include/rollback requires acknowledgement of unexported key-cache/manual-unlock coverage. Exact three-mode trust test3 passed. Native staging integration fixed by normalizing only the new copy to DELETE on its already-owned destination handle before validators; source WAL and ordinary-copy behavior unchanged. Reports /private/tmp/chatbook-task14-credential-gap-report.md and /private/tmp/chatbook-task14-incremental-review.md. Final RAG physical-sharing tag correction pending; no broader blanket credential scope claimed.
Final combined credential/capture/publication/RAG cohort146 passed28.22s. Citation policy covers every physical staged alias and preserves include material for manual recovery; default exclude reconstructs copies with setup-required identity. Native capture destination normalizes to DELETE only within exact capture staging so validators do not create unprocessed WAL/SHM. Skill-trust default exclude retains historical metadata/MAC/encrypted bytes without reading keyring cache; include/rollback reports manual-unlock coverage. RAG policy uses actual owner and relative JSON shape, with no semantic misuse of physical shared_group. Production credential Bandit0; task remains In Progress pending remaining integration.
Original Task14/19 bounded retained-definition credential plan: credentials.py plus new test_rag_definition_credentials.py only. Support actual legacy profiles list blobs and installed pipeline TOML with existing config semantic sanitation/encrypted-value material; recognize actual experiment config/results shapes as historical noncredential data and preserve arbitrary query/metric/history bytes. Unknown shapes remain explicit unsupported. Read source serializers first, TDD actual private fixtures, source immutability and no keyring authority, then focused tests/Ruff/Bandit. No definition lifetime or activation claim.
Retained RAG credential-format sub-slice implemented in credentials.py + new test_rag_definition_credentials.py: actual legacy blobs/.migrated and shipped pipeline TOML use existing semantic sanitation/encrypted material; validated experiment config/results remain byte-exact historical content. Unknown paths/shapes refuse coverage; unreadable rollback retains bytes with issue. Real ProfileConfig optional URL null now preserved. Behavioral15red+5controls -> new45pass7.62s; affected new36+existing60=96pass18.41s. Ruff/format/diffclean; productionBandit0->0, tests30LOWassertions only. Report /private/tmp/chatbook-rag-definition-credentials-report.md. Existing start_experiment Path JSON serialization issue documented, not fixed; no RAG definition lifetime/runtime/activation qualification claimed. Source frozen for review, not committed.
Root review correction completed: whole-profile/pipeline metadata could previously reach enc: unlock/sanitation. New9behavioralreds -> final114affectedtests passed18.06s. Config-only sections now preserve profile/pipeline descriptions/name/tags/function identifiers and real encrypted material locationprefix; experimenthistory unchanged. Source credentials.py/newtest only, fullRuff/format/diffclean and productionBandit0. Appended /private/tmp/chatbook-rag-definition-credentials-report.md; frozen for re-review, not committed.
Root final review and independent verification of retained RAG format credential slice complete. Initial review caught whole-profile metadata being treated as encrypted config;9behavioralreds then corrected exact config sections and original material locationprefixes. Root independently reran54new+60existing credentials:114passed16.07s (/private/tmp/chatbook-root-rag-formats-green.log). FullRuff/format clean and productionBandit0 (/private/tmp/chatbook-root-rag-formats-bandit.json; existing nosec-comment warnings unchanged). Legacy/migratedprofiles, shipped pipeline TOML and actual experiment formats now supported; historical names/descriptions/tags/function IDs and experimentquery/metricbytes remain inert and preserved. Exact installed configfields only sanitized/unlocked; malformed syntax/shape explicitrefusal or rollbackraw-preservationdisclosure. No realkeyring or producerlifetime/capture-completeclaim. Report /private/tmp/chatbook-rag-definition-credentials-report.md; ADR126. Scoped credentialformat commit only, task remainsInProgress for remaining originalcoverage.
Original Tasks14/22 SQLite source-preservation prerequisite authorized: add capture-only private main/WAL materialization using real active MaintenanceSession source identity and existing stage, exact installed SQLite owner policy, bounded pinned copying, hot-journal refusal, cancellation, before/after source checks and positive native retirement. No live SQLite read before materialization; preserve main/WAL/SHM and original fingerprints. Propagate existing reviewed ArchiveLimits/byte_budget from held capture, enforce cumulative/member/space bounds, leave ordinary noncapture DB behavior unchanged. Actual installed owner/WAL TDD and focused limits/path/cancel evidence required. No activation, damaged-config binding, replacement executor or semantic rollback verification authority in this slice; retain inherited storage traversal edits.
Independent review of native SQLite source-preservation slice requested one fix: cached private main/WAL paths were not bound to the actual created objects across reuse, allowing a replacement private database to be accepted while original source state remained unchanged. Report /private/tmp/chatbook-capture-sqlite-independent-review.md. Fix round1 is confined to private cache identity checks before/after native reads and backup, preserving native close quarantine and original source evidence. Eight actual Research/core replacement cases reproduced the issue (main, WAL, numbered directory, native-open swap). Source remains uncommitted pending fix/re-review. No activation or other approval-blocked code is included.
SQLite source-preservation prerequisite: materialize only native-session-qualified main/WAL into bounded private staging so read-only SQLite never alters original SHM. Bind private/source identities through reuse and native open/backup; quarantine ambiguous descriptor/native connection retirement. Final focused cohort: 57 passed (44 materialization, 8 existing core/domain, 2 capture factory refusals, 3 ordinary factory/copy-close controls); earlier public capture variants 3 passed. Production Ruff 53→53 and Bandit 7→7 existing findings, no new production findings. Exact inventory delta is only three _CaptureScope.sqlite_target rows (mkdir2/open2/write1). Round-two independent review pending. Task remains In Progress: this does not implement replacement execution, rollback receipts, activation association, or fence clearing. Evidence: /private/tmp/chatbook-capture-sqlite-report.md and /private/tmp/chatbook-capture-sqlite-independent-review.md.
Final bounded independent re-review APPROVED the SQLite prerequisite; all three review findings addressed. Original main/WAL/SHM preservation, cached private identity binding, native/descriptor quarantine and short-read behavior qualified. Root is staging only this slice, preserving inherited TTS/traversal/activation work; exact staged ownership inventory verification follows.
Root final gate: unchanged ownership inventory guard passed 11 tests in the existing isolated snapshot with this slice's exact index source/doc updates; staged diff check clean. Only the three new census rows and isolated source-preservation code/test/task notes are staged. Inherited storage traversal (100 lines) and TTS census rows remain unstaged. Independent review approved; 57 focused tests and unchanged production security/static baseline recorded above. This prerequisite is ready to commit; task remains In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed staged credential exclusion/inclusion, encrypted safety recovery and independently scoped omission review without overwriting global credentials. Already checked criteria now have downstream public integration evidence: 2196768bc, 26292ed27, ed411564f, fb64239ad and e6150b7d3. Unexportable credentials and remote token validity remain disclosed limitations. Independent AC reconciliation: /private/tmp/task26-dependency-completion-reconciliation.md; committed evidence index: backlog/docs/backup-recovery-release-evidence.md. This closes stale dependency bookkeeping against recorded revisions; it is not a current-build test claim or completion of TASK-32009.
<!-- SECTION:FINAL_SUMMARY:END -->