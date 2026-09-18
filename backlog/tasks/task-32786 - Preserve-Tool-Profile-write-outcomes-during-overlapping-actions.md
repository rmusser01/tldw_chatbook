---
id: TASK-32786
title: Preserve Tool Profile write outcomes during overlapping actions
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 11:41'
updated_date: '2026-09-18 11:57'
labels:
  - settings
  - tool-profiles
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep completed Tool Profile mutations and their visible outcomes consistent when another management action starts before the first write returns.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Starting another management review during an admitted write cannot discard the first write outcome or its listing refresh.
- [x] #2 Newer review preparation and explicit cancellation retain their existing authority boundaries; no duplicate write is admitted without fresh confirmation.
- [x] #3 Completed, refused and uncertain write outcomes remain truthful while newer navigation retains focus.
- [x] #4 Targeted overlap regressions and representative real-service/native evidence qualify the repair without changing shutdown cancellation behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add mounted regressions that hold import/export/remove after write admission, attempt another same-operation action, and verify actual outcome, listing refresh and newer focus. 2. Track the existing write phase per operation and guard event dispatch before exclusive worker replacement; show local progress, clear admission in finally, and preserve export cancellation/retry boundaries. 3. Verify success/refusal/uncertainty, existing review lifetime and real-service cancellation boundaries; qualify representative real native overlap journeys with private lifecycle receipts. 4. Review independently, update ledgers and save to draft PR2707. ADR required: no. ADR path: backlog/decisions/107-portable-tool-use-packs.md; backlog/decisions/150-design-token-system-and-design-language.md. Reason: Restore existing truthful outcome and review contracts within Settings; no storage, provider, app-worker ownership or shutdown boundary changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Guard Import/Export/Remove event dispatch only during admitted writes, keeping the original exclusive worker and visible outcome while preserving supersedable review preparation and explicit cancellation. Import uncertainty now refreshes current facts and receives truthful recovery text. No shutdown or service ownership boundary changes. Updated Settings, fifteen mounted overlap cases and the existing lifetime assertions. 173 distinct targeted cases pass (95 UI, 52 publication/removal services, 26 governance); no full sweep. Scoped Ruff introduces no diagnostics, changed methods/new files are formatted, and Backlog/diagnostic guards pass. Independent review found no introduced blocker. Four real native theme/size cells used the real lifecycle lock to delay removal, then verified one revision increment, tombstone, refreshed listing and visible continuation; all twelve captures inspected, final hashes and clean private lifecycle verified. Evidence: Docs/superpowers/qa/2026-09-18-tool-profile-write-overlap/README.md. Ledgers and the cancellation lesson updated. ADR required: no; existing backlog/decisions/107-portable-tool-use-packs.md and backlog/decisions/150-design-token-system-and-design-language.md apply. Screen-destruction/shutdown outcome ownership and broader destination review remain open; PR2707 remains draft pending visual approval.
<!-- SECTION:NOTES:END -->
