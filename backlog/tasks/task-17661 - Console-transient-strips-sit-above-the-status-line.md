---
id: TASK-17661
title: 'Console: transient strips (staged evidence, prompt queue) sit above the status line'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-18'
labels:
  - console
  - ux
dependencies:
  - task-17659
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Owner request 2026-08-18: the attachments bar — the staged-evidence strip that appears when context is attached from Library — and, per the follow-up clarification, ALL transient strips (staged evidence AND the prompt-queue shelf) should sit ABOVE the status line rather than directly above the composer, so the area around the composer stays visually quiet. New deck order in the default placement: workspace grid → staged evidence → prompt queue → status row → gap → composer → gap → footer.

This overrules the TASK-17659 shelf-adjacency contract (queue glued to the composer's margin): the shelf's nearest lower neighbor is now the status row. In below-placement mode the strips are already first in the deck; the rule "strips at the top of the control deck" holds in both modes. The status-row position mover simplifies: it anchors on the composer alone instead of the strip cluster.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 With the prompt-queue shelf visible in the default placement, the deck renders grid → shelf → status row → gap → composer, pinned by mounted geometry assertions
- [x] #2 The staged-evidence strip occupies the same top-of-deck slot (verified on the running screen with a visible state)
- [x] #3 The status-row placement setting still works in both modes (mover re-anchored on the composer; order and popup contract tests green in both placements)
- [x] #4 The command popup still never paints over any visible strip or the status row
- [x] #5 User Guide Console page updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: flip the shelf-adjacency pin — queue's lower neighbor becomes the status row (DOM-order geometry, harness-proof); watched fail.
2. Compose reorder: staged strip + queue yield BEFORE the chips in above mode (strips are already first in below mode).
3. Simplify `apply_status_chips_position`: anchor on the composer alone with exact-adjacency guards (the strip anchor is obsolete).
4. Sweep, painted probe with a visible staged state, docs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Compose-order move plus a simplification: the transient strips (staged evidence, prompt queue) now always open the control deck, so the status-row mover anchors on the composer with exact-adjacency guards instead of the staged-strip anchor — the loose ordering guards became normalization-correct equality checks in the same change. The TASK-17659 shelf-glued-to-composer contract is deliberately overruled (owner call): the shelf's lower neighbor is the status row, and the composer keeps its quiet gaps. Popup clearance needed no change (its min() loop is position-agnostic across all three strips).

Evidence: RED-first on the flipped shelf pin; 648 passed on the 16-file deck sweep; painted probe with staged sources at 150x44 renders grid border / "Staged for the next send · 2 sources" + rows / status row / blank / composer / blank / footer.

Files: `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/UI/Console_Modules/status_row.py`, `Tests/UI/test_console_prompt_queue.py`, `Docs/User_Guide/console.md`.
<!-- SECTION:NOTES:END -->
