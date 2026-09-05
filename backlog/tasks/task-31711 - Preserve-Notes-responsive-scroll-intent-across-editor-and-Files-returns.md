---
id: TASK-31711
title: Preserve Notes responsive scroll intent across editor and Files returns
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:53'
updated_date: '2026-09-05 19:40'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the three remaining Notes scroll restoration failures while preserving exact semantic focus receipts and deliberate user overrides across responsive layouts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Files and editor returns restore the existing exact wide browse offsets
- [x] #2 Deliberate compact preview scrolling replaces older responsive memory without stale restoration
- [x] #3 Original scroll assertions and full Notes workspace verification pass or disclose separately owned failures; no budget or timeout increases
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce all three reported responsive-scroll failures together and record exact captured receipts, geometry clamps and latest user intent across each transition.
2. Distinguish premature harness observations from stale runtime restoration; characterize any runtime change through existing real resize/callback seams before implementation.
3. Preserve exact scroll/focus assertions and latest-user guards, using the existing owner boundaries; request parent review for a runtime policy change.
4. Run responsive focus/scroll coverage plus full Notes workspace file, disclose separately owned theme/geometry failures, static-check and review.
ADR required: no new ADR anticipated
ADR path: N/A
Reason: Repair is intended to restore existing semantic scroll and deliberate-user-override contracts; reassess if ownership policy changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified without a runtime change after checkpoint5c915a585e. The three original scroll regressions pass together (3passed10.915s), and expanded real responsive/focus coverage passes (11passed30.19s). Full Notes workspace verification completed:157passed,3failed389.54s. The prior Files6versus7 failure did not reproduce in either focused or fullfile evidence; no offset, focus, editor identity, timeout or ratchet assertion was relaxed.
Remaining separate failures: high-stakes light-theme Save failed contrast1.498versus4.5 at both sizes, and compact Folder Files Export exact copy region painting Saved (cause not yet verified). These remain explicit repair work, not a claim that all Notes UI is green.
Evidence: /private/tmp/library-scroll-resumed-baseline-31711.xml; /private/tmp/library-responsive-resumed-31711.xml; /private/tmp/library-notes-resumed-full-31711.xml. ADR required:no; no implementation or boundary change. No code/lint delta for this task.
<!-- SECTION:NOTES:END -->
