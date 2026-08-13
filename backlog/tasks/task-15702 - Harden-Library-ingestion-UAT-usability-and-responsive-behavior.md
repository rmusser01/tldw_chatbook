---
id: TASK-15702
title: Harden Library ingestion UAT usability and responsive behavior
status: Done
assignee:
  - '@codex'
created_date: '2026-08-12 20:25'
labels:
  - library
  - ingest
  - ux
  - accessibility
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the latest-dev Ingestion UAT findings so source identity, keyboard operation, responsive layout, field context, action visibility, and queue status remain reliable across supported terminal sizes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preflight and entry focus keep the destination authority, path identity, and complete source value visible without clipping focused controls.
- [x] #2 Clearing a staged source and typing immediately preserves every keystroke in the current path input across repeated structural updates.
- [x] #3 At 80x24, the Ingestion footer exposes truthful Start, Back, and contextual Retry guidance while the Library rail yields enough space for the task.
- [x] #4 Focusing compact Ingestion actions never changes their geometry or scroll position when already visible.
- [x] #5 The final import forecast and Start action remain discoverable alongside preflight details at standard and 80x24 sizes.
- [x] #6 Title, Author, and Keywords retain persistent field identity after values are entered.
- [x] #7 Queue group and row vocabulary distinguishes queued from running work without duplicated status language.
- [x] #8 Collapsed option summaries and active-queue context remain concise, complete, and relevant.
- [x] #9 Focused automated interaction, rendered-frame, responsive-geometry, and state-copy checks cover the remediated behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a bounded usability and regression-remediation pass within the existing Library ingestion architecture and ADR-031 keyboard/footer conventions; it changes no storage, service, ownership, or long-lived application boundary.

1. Add rendered-frame, interaction-race, geometry, responsive-footer, and state-copy regressions for the UAT findings using the real Library shell harness.
2. Stabilize entry/preflight scroll ownership and Clear input preservation so focused controls remain fully visible and immediate typing survives structural updates.
3. Prioritize narrow-terminal task space and local keyboard guidance, including responsive rail collapse and footer hint ordering.
4. Recompose the long form around a persistent source contract, compact metadata labels, concise option summaries, and an always-discoverable commit region.
5. Normalize queue group vocabulary and separate blank-form guidance from active activity context.
6. Rebuild the modular stylesheet, run focused tests and static checks, capture 120x40 and 80x24 rendered evidence, and self-review the complete diff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the complete ingestion UAT remediation without changing storage or service boundaries (ADR required: no; ADR-031 remains the keyboard/footer authority).

- Entry focus now preserves the top-of-form source contract; narrow Ingest automatically collapses the destination rail to its reachable handle.
- Clear removes stale preflight panels in place and suppresses its programmatic empty-value echo, eliminating the remount and late-message windows that swallowed immediate typing.
- A docked review bar keeps forecast, consent/readiness copy, and Start together; it hides after submission so activity receipts own the viewport. Metadata fields now have persistent labels in a compact row.
- Footer fitting preserves the ordered workflow prefix at narrow widths; Retry is advertised only when its real gate is open. Default option-panel titles are concise and queue batch headers use `active` plus exact per-state tallies.
- Independent pre-PR review found and resolved two integration gaps: Clear now preserves visibility and keyboard reachability of the always-valid generic options panel, and the legacy Retry-shortcut test now covers the contextual no-snapshot/active/settled contract.
- Qodo review identified redundant footer reflow on every ingest registry tick; registration is now conditional on an actual shortcut-tuple transition, with a mounted no-op/Retry-transition regression.
- Rebuilt the modular stylesheet and updated the Library import guide.

Evidence: 218 ingest-state tests passed (one Windows symlink-privilege case deliberately excluded); ten standalone mounted interaction/geometry checks passed, including repeated Clear/type races, generic-panel continuity, and conditional footer registration; focused Ruff passed with only the repository's pre-existing F402/E721 classes ignored; py_compile and `git diff --check` passed. The real-shell UAT capture completed at 120x40 and 80x24 for idle, mixed-preflight, and active-queue states; Browse focus geometry remained identical on the entry/preflight frames.
<!-- SECTION:NOTES:END -->
