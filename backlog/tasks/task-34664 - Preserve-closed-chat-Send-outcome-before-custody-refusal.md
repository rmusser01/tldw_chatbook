---
id: TASK-34664
title: Preserve closed-chat Send outcome before custody refusal
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-08 21:08'
updated_date: '2026-10-08 22:27'
labels: []
dependencies: []
priority: high
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the established explicit closed-chat Send response without weakening draft custody, source admission or unrelated chat isolation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An explicit chat closed before dispatch or during manual withdrawal returns the existing informative session-closed result and never sends to the active chat.
- [x] #2 A live review-required handoff still refuses Send with unconfirmed-custody copy and retains its draft, transcript and sibling run state.
- [x] #3 Original controller and per-session send cases pass with admitted fixture profiles; size, static and affected artifact guards keep their original limits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; existing backlog/decisions/219-console-chat-destinations-and-bounded-starts.md applies.
Reason: Routine restoration of the established closed-session response; live custody and authority contracts remain unchanged.
1. Retain original failing-node evidence and trace all shared submission/custody callers.
2. Add real-store RED controls for close after withdrawal and live review-required custody; preserve existing provider and sibling assertions.
3. Reuse the existing closed-session result only after failed handoff drain when the explicit target is absent; keep all live gates.
4. Verify originals and complete affected modules, static/size and Backlog guards; record audited outcomes and local raw hashes.
5. Commit the bounded fix, then run the requested broader native-owner, closed-cursor, mounted Interrupt, startup and Character-switcher qualification matrices without changing limits. External Windows/human evidence stays explicitly open.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored the existing closed-chat result only after failed custody drain and absence of an explicit target. All live custody, hook, origin and provider gates remain; real-store controls preserve sibling transcript/run state and zero provider calls. Valid RED: three closed-target failures/one live custody pass; GREEN 57 checks including all 52 size rows. Complete controller/per-session/new-control aggregate: 385 passed in 195.61s with strict coroutine warnings, no pytest warnings in parent or the actual private child. All eleven artifact guards pass, new test lint/format passes and changed inherited Ruff diagnostic multisets add nothing. Controller ceiling lowered 29270 to 29266. Self-review traced all submission callers; no config/export/signature change. Earlier live-labelled parameter was automatically skipped by repository collection and is retained, superseded by the renamed, actually executed review-required control. Two unused generated helpers removed only after the full green gate, with recoverable local copies/hashes. Raw logs stay local under /private/tmp/closed-send-qualification-UH3Eup; broad qualification and integration remain pending, no Done or push claim.

Fresh whole-branch read-only review at c37690530e includes bounded manual closed-Send behavior and all callers: no Critical or confirmed Important production finding. Closed-Send production is byte-identical to ecf4b2 verified385-test checkpoint. Later ownership checkpoint dc03d61 is locally committed and relevant resources/native tests pass; seven50ms UI breaches and external platform/human gates remain. No push/integration authority inferred; keep In Progress for integration closeout.
<!-- SECTION:NOTES:END -->
