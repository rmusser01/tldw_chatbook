---
id: TASK-31815
title: >-
  Cross-wave Library modal-inventory repair: the skills-era discovery blocker
  plus three stale delegator rows
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06 04:42'
updated_date: '2026-09-15 04:55'
labels:
  - library
  - decomposition
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_library_modal_dismissal.py maintains a hand-declared (file, class, presenter, modal-type) inventory and rediscovers it by AST-parsing only the files named in _SUPPORTED_OWNER_SCOPES, then asserts the two sets match in BOTH directions. The file has been 1-red at every Library decomposition wave tip since the skills series: an unresolved modal constructor for LibraryScreen._present_library_skills_import_choice_if_needed aborts discovery before the comparison ever runs, so the guard currently proves nothing about ANY subsystem's rows -- it is a blocked guard, not a failing assertion. Three further row clusters are equally stale for the same underlying reason (the named owner is now a one-line delegator, or was pruned outright, while the real presenter moved into a Library_Modules controller): the ingest row keyed on handle_library_ingest_browse, and the two skill-trust passphrase presenter rows. Wave-6 task 3 repointed the four prompts rows and proved them by construction against the new _OwnerScope, but could not prove the FILE green end-to-end because of the blocker; that same blocker is why the recipe's documented-reds list carries a standing 1-red entry for this file. This is cross-wave repair work that belongs to no single subsystem series, which is why five consecutive waves each deferred it.

2026-09-14 Library audit at 2939afda63 reproduces the unresolved SkillImportChoiceModal(snapshot.candidates) constructor at LibraryScreen._present_library_skills_import_choice_if_needed. The selected gesture tests pass, but the inventory still aborts before proving bidirectional coverage. This confirms the existing task; no duplicate was filed. Evidence: Docs/superpowers/reports/2026-09-14-library-workflow-audit.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tests/UI/test_library_modal_dismissal.py runs to completion with zero failures at HEAD -- no declared-but-undiscovered row and no discovered-but-undeclared edge, in both directions of its own bidirectional assertion
- [x] #2 The skills-era discovery blocker is resolved: _discover_library_modal_edges completes without raising an unresolved-modal-constructor error for _present_library_skills_import_choice_if_needed, and the row names whichever owner actually constructs the modal today
- [x] #3 The stale ingest row keyed on handle_library_ingest_browse names its real present-day owner and is rediscovered by the file's own AST walk (adding the owning module to _SUPPORTED_OWNER_SCOPES first, if it is not already listed -- without that, a repointed edge is never discovered and the assertion fails the other way)
- [x] #4 The two skill-trust passphrase presenter rows (_request_library_skill_trust_passphrase and _request_library_skill_trust_bootstrap_passphrase) name their real present-day owners and are likewise rediscovered
- [x] #5 Each repointed row is verified against the modal's CONCRETE type, not just its presenter name, so a row that resolves to the wrong modal class is caught
- [x] #6 backlog/docs/library-decomposition-recipe.md section 7's documented-pre-existing-failures list has the test_library_modal_dismissal.py entry removed, with the commit that fixed it named
- [x] #7 The skill import chooser and review-set picker participate in the same concrete modal dismissal, positive-result and focus/lifecycle contracts as other inventoried dialogs.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the complete modal test-file baseline and trace all currently constructed dialogs and moved presenters in the supported scopes.
2. Register the skill chooser and review-set picker with exact concrete behavior contracts; add Skills, Ingest and Export controller scopes. Repoint the three originally reported edges plus the newly exposed Export destination and File Notes root-picker edges. Preserve strict unresolved-constructor and bidirectional checks.
3. Add focused negative controls proving all five repaired presenter/type mappings reject a wrong modal class, and exercise both newly registered dialogs through existing gesture/positive/focus/lifecycle cases.
4. Run the complete targeted modal file and the two dialog-specific files; compare lint against baseline and format changed code. Record evidence, self-review and remove the obsolete standing-failure documentation with a reference to the repair commit.

ADR required: no
ADR path: backlog/decisions/161-component-pattern-library.md (existing)
Reason: test inventory and coverage maintenance for existing dialog classes and controller boundaries; no production UI or architectural behavior changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repair commit: b3109ba5cf. Added exact contracts and production launch edges for SkillImportChoiceModal and LibraryReviewSetPickerDialog. Added Skills, Ingest and Export controller owner scopes; repointed both skill-trust prompts, ingest browse, export destination and the renamed File Notes root picker. The strict discovery and bidirectional comparison now cover 35 edges, 21 concrete dialog types and ten supported owners.

The original blocker was a missing concrete contract, not an unsupported constructor shape. Once removed, the complete comparison exposed two further stale Export/root-picker rows; the plan was updated and both were repaired. Five source-mutation controls prove each repaired edge detects a wrong modal type while preserving its owner/presenter name. The new dialog contracts inherit gesture, positive-result, exact opener-focus and single lifecycle checks.

Verification: baseline 1 failed / 169 passed. Final complete modal file plus both dialog-specific files: 198 passed. Whole-file Ruff formatting and diff whitespace checks pass; the same two pre-existing Ruff diagnostics remain, with none added. Two pytest cleanup warnings concern unrelated old Kokoro test directories. Self-review completed. No product behavior, dependencies, performance, permissions, security or license changes; no full test sweep or new native/server claim.

Updated Docs/superpowers/reports/2026-09-14-library-workflow-audit.md with bounded evidence and backlog/docs/library-decomposition-recipe.md section 7 to remove both obsolete standing-failure entries and name repair commit b3109ba5cf. Existing blocked-guard lessons already cover this incident; no duplicate lesson needed. ADR required: no; existing backlog/decisions/161-component-pattern-library.md governs the unchanged component contracts.
<!-- SECTION:NOTES:END -->
