---
id: TASK-67
title: Add Console provider recovery Settings action
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Console provider setup failures directly recoverable from the Console surface so users can open Settings without remembering the command palette path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console shows a visible Open Settings action when provider setup is blocked.
- [x] #2 Clicking the action routes to the Settings destination through the existing app navigation seam.
- [x] #3 The action is hidden when provider setup is ready.
- [x] #4 Focused mounted regressions cover the blocked and ready states.
- [x] #5 Actual Textual-web screenshot evidence is captured before approval.
- [x] #6 Provider recovery renders as one compact warning/action strip instead of repeated floating guidance.
- [x] #7 Empty transcript state tells users where chat output will appear without duplicating provider recovery.
- [x] #8 Console workspace panel presents unwired workspace switching as read-only instead of showing disabled future controls.
- [x] #9 Provider recovery appears before first-run guidance when sending is blocked.
- [x] #10 Empty transcript copy has an explicit heading and blocked-send recovery wording.
- [x] #11 Workspace read-only recovery appears directly under Runtime before conversation rows.
- [x] #12 Left rail gives Convos & Workspaces more vertical room than Staged Context.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add mounted Console regression for visible Open Settings action in provider-blocked state and route invocation to Settings.
2. Add mounted regression proving the action is hidden when provider setup is ready.
3. Implement a Console helper to keep the provider recovery button in sync with existing provider blocker copy.
4. Wire the button through the existing app `switch_tab(Settings)` navigation seam without changing route IDs.
5. Run focused Console tests and diff hygiene, then capture actual Textual-web screenshot evidence.
6. Consolidate provider setup guidance into a single inline recovery strip and remove duplicated setup copy from the first-run transcript guidance.
7. Add an empty transcript placeholder that clarifies where messages appear.
8. Make the Console workspace tray read-only until workspace switching and creation are wired.
9. Move provider recovery above first-run guidance, tighten transcript empty-state copy, and rebalance left-rail section heights.
10. Move workspace recovery copy directly under Runtime, before the Conversations section.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Console provider recovery and polish improvements: a compact blocked-provider Settings action above first-run guidance, explicit empty transcript recovery copy, read-only workspace context recovery under Runtime, and left-rail workspace height rebalancing. Added mounted regressions for blocked and ready provider states, provider-strip ordering, workspace recovery ordering, and left-rail proportions. Captured approved Textual-web screenshot evidence at Docs/superpowers/qa/console-ui/2026-05-21-console-provider-settings-action.png. Verification: focused Console/workspace suite passed with 69 tests and git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
