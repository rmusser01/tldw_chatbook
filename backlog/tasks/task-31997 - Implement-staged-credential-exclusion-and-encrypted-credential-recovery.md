---
id: TASK-31997
title: Implement staged credential exclusion and encrypted credential recovery
status: In Progress
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
updated_date: 2026-09-10 18:26
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Staged managed credentials now exclude safely by default, preserve encrypted recovery material when selected, and avoid overwriting shared scopes. Focused credential, shared resolver and SQLite inventory checks pass; full backup/restore orchestration remains separate original tasks.
<!-- SECTION:FINAL_SUMMARY:END -->
