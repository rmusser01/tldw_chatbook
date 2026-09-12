---
id: TASK-31692
title: Align Library modal inventory with concrete launch ownership
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:33'
updated_date: '2026-09-05 18:36'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the modal inventory mismatch against current runtime launch owners while preserving exact bidirectional edge coverage and modal dismissal contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every declared Library modal launch edge maps to its actual production owner and presenter
- [x] #2 The inventory continues rejecting undeclared, missing, nested and injected modal edges without count-only changes
- [x] #3 Modal inventory and relevant dismissal tests pass with static checks
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the exact inventory mismatch and inspect missing/extra owner-presenter edges rather than changing only the count.
2. Update the inventory declaration or discovery recognition at the actual launch boundary while retaining exact constructor and owner checks.
3. Add negative detection coverage for any newly recognized presenter form, run inventory and relevant dismissal tests, static checks and review.
ADR required: no
ADR path: N/A
Reason: Test-only inventory ownership/discovery repair preserving existing modal production contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Registered the concrete LibrarySkillsController owner and retargeted its two passphrase modal declarations from obsolete LibraryScreen forwarding methods. Bidirectional discovery remains exact and the total stays 34. No changes to discovery recognition or runtime modal behavior. Added a synthetic injected-edge negative check using the actual Skills local push_screen_wait pattern; moved exact edge diagnostics ahead of the unchanged count assertion.
Baseline inventory 1 failed, 4 passed; exact missing edges were SkillTrustPassphraseModal and SkillTrustBootstrapModal. Inventory plus SkillTrust dismissal coverage: 18 passed, 165 deselected in 7.73s. Ruff check and both touched-range format checks pass, unrelated existing full-file format debt preserved; git diff --check and parent review pass.
ADR required: no, test-only actual-owner inventory correction. The failure illustrates why decomposition inventory must follow concrete presentation owners, not screen forwarding shims.
<!-- SECTION:NOTES:END -->
