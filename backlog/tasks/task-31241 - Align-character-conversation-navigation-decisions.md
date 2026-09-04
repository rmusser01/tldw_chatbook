---
id: TASK-31241
title: Align character conversation navigation decisions
status: Done
assignee: []
created_date: '2026-09-04 02:03'
updated_date: '2026-09-04 03:19'
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
2. Create ADR-116 from the approved design.
3. Amend `004-personas-destination-native-workbench.md`, ADR-030, ADR-037,
   ADR-046, ADR-083, and ADR-085.
4. Link preserved ADR-031 and ADR-033.
5. Run documentation and reference checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- ADR required: yes. ADR path:
  `backlog/decisions/116-character-conversation-navigation-and-local-semantic-search.md`.
  Reason: the programme needs one authority for durable derived indexes, typed
  local identity, cross-surface activation/repair, semantic consent, and
  long-lived surface ownership before runtime implementation.
- Fetched `origin/dev` and swept every remote ref, registered worktree, and open
  pull request. No older claimant exists for TASK-31241 or ADR-116; the merge
  candidate contains exactly the intended task and ADR paths.
- Created ADR-116 with the approved local-only identity union, selected-branch
  Keyword/Meaning corpus, Console-owned activation, Library-only CAS repair,
  opt-in embeddings-only Meaning generations, and surface boundaries.
- Added narrow dated amendments to ADR-004
  (`004-personas-destination-native-workbench.md`), ADR-030, ADR-037, ADR-046,
  ADR-083, and ADR-085. ADR-031 remains the key/hint authority and ADR-033
  remains the Settings commit-model authority.
- Modified only governance documentation named by the corrected plan; no runtime
  behavior or dependencies changed.
- Initial documentation/reference checks, `git diff --check`, the exact contract
  check, local-link existence checks, and the repository backlog-ID guard passed.
- Initial self-review found and corrected the Roleplay amendment's aggregate
  draft-owner enumeration. Independent review then found the ambiguous duplicate
  ADR-004 selection, incomplete semantic maintenance/Delete lifecycle, and
  canonical ADR filename mismatch.
- Fix Round 1 removed the unrelated storage-restart amendment, amended
  `004-personas-destination-native-workbench.md` for Roleplay per-character
  browse versus Library global/archive ownership, renamed ADR-116 to the
  approved `local-semantic-search` path, and updated every committed reference
  plus the source implementation plan.
- ADR-116 now makes future maintenance effective only after a complete ready
  initial generation and makes Delete atomically remove ready/staging
  generations, disable saved future maintenance, clear semantic query caches,
  synchronize original/draft Settings state to Off, and preserve source chats.
- Fresh review-correction, preserved-contract, lifecycle, filename, link,
  `git diff --check`, and backlog-ID gates passed. The full application suite
  was intentionally not run for this documentation-only task, as ruled in the
  task brief.
- Final self-review found no unresolved finding or generalisable new repository
  lesson.
<!-- SECTION:NOTES:END -->
