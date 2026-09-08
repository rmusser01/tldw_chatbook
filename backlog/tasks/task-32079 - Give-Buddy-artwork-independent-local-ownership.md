---
id: TASK-32079
title: Give Buddy artwork independent local ownership
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 19:29'
updated_date: '2026-09-08 20:53'
labels:
  - buddy
  - console
dependencies: []
references:
  - Docs/superpowers/specs/2026-09-08-console-buddy-management-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users select an installed visual companion without creating or retaining a Persona; preserve existing appearances and source attribution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh and migrated profiles can own and select Buddy artwork without a Persona; source Persona edits/deletion do not affect the copy.
- [x] #2 Existing private visual validation/publication/rendering are reused with explicit Buddy authority and versioned schema migration; no dummy Persona is created.
- [x] #3 Legacy selection migration is idempotent, preserves geometry, enabled state and artwork notices, and leaves the old selection usable on failure.
- [x] #4 Built-in and native imported artwork can be published to Buddy ownership with preserved supplied attribution and unknown licensing kept unspecified.
- [x] #5 Targeted SQLite, publication and controller tests verify restart, migration, failure and ownership isolation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md
Reason: independent ownership and explicit default-selection semantics.
Follow Task 2 in Docs/superpowers/plans/2026-09-08-console-buddy-foundations.md.
Read the approved spec and existing relevant ADRs; add failing targeted regression coverage, implement shared authority/creation seams, validate targeted integration, then document evidence and review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented independent local Buddy artwork ownership under ADR-139. Schema 70 adds real Buddy owners/bindings and a read-only explicit-owner projection over shared immutable visual storage. Existing Persona APIs remain compatible; Buddy snapshots have persona_id=None and real buddy_id/revision authority. Shared publication supports first and subsequent versions under CAS.

BuddyLibrary exposes list/get/graph/preview, bounded read-only native archive review, publication, guarded Persona copying, independent built-in seeding, and idempotent legacy selection migration. Copies own separate packs/files. Source notices and unspecified licenses survive native imports and copies. Startup no longer creates a built-in Persona; legacy and independent tombstones are retained. Independent controller selection needs no Persona service. Migration preserves geometry/visibility and prior selection on copy or config-write failure. The management batch writer and revision-guarded rollback support atomic persona_buddy plus buddy_interaction config writes.

Verification: broad targeted ownership/repository/publication/runtime/import/authoring/preferences/schema/startup/packaging run reported 485 passing; six startup/source-inspection failures were rerun with files stable and all six passed. Final focused library/config/startup/lazy-controller check: 21 passed. Existing Persona compatibility subset previously passed 195. One architecture AST guard fails identically on pristine HEAD because the untouched legacy builtin_pixel_migu module imports Character_Chat; reproduced in an isolated HEAD source tree, not introduced here. New library/artwork/snapshot/test modules pass the configured Ruff rules; modified existing runtime modules introduce no new Ruff findings compared with HEAD. Thirteen scoped files pass formatter check, and scoped git diff --check passes. No full-suite run.

Core changed files: Persona_Buddy/library.py, controller.py, preferences.py and facade; Persona_Visual artwork/snapshot/importer/repository/publication/runtime/authoring; ChaChaNotes schema migration and app composition. ADR-139 records the concrete ownership projection. Added a testing-evidence lesson for editing source during inspect.getsource-based test execution. Parent integration review remains pending; task status stays In Progress until that review and overall integration finish.

Parent review reproduced and resolved two startup regressions: inactive or archived legacy builtins no longer seed a replacement independent Buddy, and unclaimed migration now holds the first-controller construction lock across preference admission, disk write and app_config publication. If controller construction wins during the visual copy, migration hands off without overwriting its current selection. Deterministic concurrency/handoff tests and real inactive/deleted/archived legacy seed tests pass; included in the final 183-test speech/startup regression run. No full-suite run or shared-file formatting.

Final integration gate: 813 passed in Tests/Persona_Buddy, Tests/Persona_Visual, WorkspaceDB/default/provisioning and Persona assignment suites (162.62s); 7 lazy packaging/architecture checks passed, with the previously reproduced untouched legacy builtin boundary failure excluded. Independent review startup retirement and constructor/migration ownership races are fixed and covered. Final user guide: Docs/User_Guide/buddies.md. ADR-139 governs genuine Buddy ownership. No new lint findings in task-owned files; only targeted tests were run.
<!-- SECTION:NOTES:END -->
