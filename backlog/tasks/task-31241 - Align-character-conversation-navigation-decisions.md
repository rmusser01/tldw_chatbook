---
id: TASK-31241
title: Align character conversation navigation decisions
status: Done
assignee: []
created_date: '2026-09-04 02:03'
updated_date: '2026-09-05 19:57'
labels:
  - architecture
  - console
  - roleplay
  - search
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-09-03-character-conversation-navigation-design.md
  - >-
    Docs/superpowers/plans/2026-09-03-character-conversation-navigation-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish the architectural source of truth for the approved character-conversation navigation programme before any implementation begins, so all eight pull requests share one local-only identity, activation, recovery, and indexing contract.
<!-- SECTION:DESCRIPTION:END -->

## Renumbering provenance

Renumbered from TASK-31233 on 2026-09-04. The final pre-commit worktree sweep
found the older `Review selected opens the review it creates` task created at
01:50; it keeps TASK-31233 under the older-arrival rule. This unshipped task was
created at 02:03 and moves with all plan and dependency references.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A new collision-safe ADR records local-only character conversation navigation, typed identity, activation, repair, and semantic-index ownership.
- [x] #2 ADR-004 (`004-personas-destination-native-workbench.md`), ADR-030, ADR-037, ADR-046, ADR-083, and ADR-085 are amended consistently with the approved design.
- [x] #3 ADR-031 and ADR-033 remain preserved contracts and are linked without contradictory changes.
- [x] #4 Terminology distinguishes Data Profile authority from RAG configuration profiles in every changed decision.
- [x] #5 Documentation-only validation and review pass with no implementation behavior change.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reconfirm task and ADR allocations.
2. Create ADR-120 from the approved design.
3. Amend `004-personas-destination-native-workbench.md`, ADR-030, ADR-037,
   ADR-046, ADR-083, and ADR-085.
4. Link preserved ADR-031 and ADR-033.
5. Run documentation and reference checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- ADR required: yes. ADR path: backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md. Reason: ADR-120 remains the programme authority for identity, pagination, semantic-index ownership, and lifecycle; no new ADR was required for this review correction.
- Created ADR-120 and the six scoped amendments, preserving ADR-031 and ADR-033, with documentation-only governance and no runtime or dependency changes.
- Incorporated PR #2429 review corrections after rebasing the six governance commits onto current origin/dev without conflicts. The implementation plan now carries created_at in CharacterConversationCursor and CharacterConversationRow, specifies descending last_modified/created_at/conversation_id keyset ordering, and requires equal-timestamp and unchanged-page no-skip/no-repeat regressions.
- Completed the semantic manifest example with local content authority, installed model artifact digest, chunk-configuration digest, and source content watermark. Compatibility, publication, and query checks now reject changed artifact bytes under the same model ID and changed source watermarks while retaining atomic ready-generation fencing and selected-content exclusions.
- Added durable data_authority_id to semantic index, saved, and draft configuration examples. Build/rebuild and lifecycle checks fail closed across active Data Profile changes and prevent reuse of another authority’s jobs or generations.
- Rechecked ADR-120 and TASK-31241 acceptance wording; neither contradicted the approved spec, so no additional ADR or acceptance-criterion amendment was needed.
- Scoped contract assertions, relative-link checks, git diff --check, and the repository backlog-ID guard passed. The correction diff remains documentation-only. Application tests were intentionally not rerun because runtime was unchanged.
- Runtime carry-forward: TASK-31242 must implement and test the corrected three-field ordering; this documentation correction does not claim the current two-field runtime cursor is fixed.
- Final self-review found no additional actionable review issue or generalisable repository lesson.
<!-- SECTION:NOTES:END -->
