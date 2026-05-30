---
id: TASK-74
title: 'CDP QA: Console composer paste workflow from current dev'
status: To Do
labels:
- qa
- console
- cdp
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the current-dev Console composer paste workflow through actual textual-web/CDP use so the visible input, paste-collapse token, unfurl flow, and normal typing behavior are proven in the rendered app after the paste-threshold merge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 CDP/browser-driven QA is run against a clean current-dev app instance, not a mockup, SVG, or static layout render.
- [ ] #2 Actual screenshots are captured for Console baseline, visible typed input, collapsed large paste token, first-click Unfurl prompt, second-click expanded paste text, and normal typing over the threshold remaining literal.
- [ ] #3 QA notes record exact commands, browser/CDP URL, app commit SHA, environment, and any failed or uncertain observations.
- [ ] #4 The walkthrough verifies that only paste/insert chunks over the configured threshold collapse, while normal typing remains visible and literal.
- [ ] #5 Any observed Console composer regression is either fixed in the same follow-up PR with focused regression coverage or logged as a separate Backlog task with severity and reproduction steps.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] Current-dev SHA and clean worktree state are recorded before QA begins.
- [ ] CDP/browser screenshots are attached or linked from durable QA notes.
- [ ] The rendered app is verified for visual usability and functional behavior, not only selector presence or clickability.
- [ ] Focused automated regression coverage is added for any bug fixed during the ticket.
- [ ] Task notes identify residual risks and the next follow-up if the workflow is not fully usable.
<!-- DOD:END -->
