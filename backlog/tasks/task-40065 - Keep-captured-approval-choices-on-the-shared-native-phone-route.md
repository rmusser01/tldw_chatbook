---
id: TASK-40065
title: Keep captured approval choices on the shared native phone route
status: In Progress
created_date: 2026-11-09 10:29
labels:
- console
- mobile
- approval
- ci
priority: high
references:
- backlog/decisions/235-console-approval-interaction-and-feedback.md
- backlog/decisions/236-served-desktop-approval-relay.md
- tldw_chatbook/Chat/console_remote_rounds.py
- tldw_chatbook/Web_Server/static/mobile/app.js
updated_date: 2026-11-09 11:15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The latest dev approval work landed while PR #3345 was in CI. The shared phone Console needs to offer and accept the same captured decisions as the owning native round, including narrowed or shared verdicts, without bringing back the retired mobile engine or endpoints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The active phone snapshot and exact native decision command honor the captured approval row choices. One shared verdict has one phone control, and withheld grants cannot be submitted.
- [x] #2 The rebased branch preserves native receipt ordering, approval feedback and terminal-only review, and passes the original software UAT and relevant local checks without higher runtime or source limits.
- [ ] #3 Independent exact-commit review and the published head’s own hosted gates verify the integration.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/235-console-approval-interaction-and-feedback.md and backlog/decisions/236-served-desktop-approval-relay.md
Reason: Compose the existing captured presentation and feedback with the already-authorized shared native command owner. The authority, persistence, permission scopes, and legacy-route retirement remain governed by the existing decisions.

1. Preserve the exact dev and published refs, replay every commit, and identify current workflow, approval, and decision-reference overlaps.
2. Reproduce the mismatch on the real retained approval worker and native phone command. Bind the projected and accepted options to the exact captured native view, and render one control per verdict key with all reviewed call summaries.
3. Keep exact-target checks and receipt ordering, attach the existing advisory feedback after the worker event release, and retain terminal-only paths. Preserve all original browser and mounted assertions.
4. Reconcile the duplicate decision ID and check the merged test rosters, artifacts, and pins. Run focused native and hosted-lane equivalents and the original UAT at existing limits. Require exact review and own hosted gates before closeout.
5. Keep the real execution-fence test on the admitted bootstrap profile via its existing per-node marker after the connected run showed the fixture redirected config before the fence assertion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The native phone snapshot and decision command now use the owning round’s captured choices. Shared verdicts display one control with each reviewed summary, and malformed or withheld choices are refused. Receipt recording still precedes worker release and advisory feedback follows it. The terminal-only review path and legacy-route retirement remain governed by ADR-235 and ADR-236.

The original keyboard/native phone cohort passed 41 cases. The connected approval cohort had 174 passes and one fixture refusal, followed by a pass for that exact execution-fence case after preserving its admitted bootstrap profile. All 118 served-native and authentication cases passed. Original software UAT passed all 15 browser cases and the mounted parent and three children. Exact committed-head review and its own hosted qualification are required for AC3.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
