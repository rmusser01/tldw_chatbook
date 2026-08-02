---
id: TASK-1846
title: 'Approval card is unstyled and the tool trace is the faintest text on screen'
status: To Do
assignee: []
created_date: '2026-08-01 19:30'
labels:
  - console
  - ux
  - design-system
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The highest-stakes surface in the app has no visual treatment, and the record of what an agent did is the least legible thing in the transcript.

- `.ds-approval-card` (`_agentic_terminal.tcss:75-79`) is the design system's approval treatment and is applied by **nothing**; `#chat-approval-card` has zero CSS rules. The card renders as plain body text.
- The row spends 26 + 14 + 14 = **54 fixed cells** on controls, leaving header and arguments to split the remainder -- roughly 13 cells each at 80 columns. Arguments are additionally capped at `_ARGS_SUMMARY_LIMIT = 80` chars of compact JSON.
- The tool trace renders `dim italic` muted (`_agentic_terminal.tcss:3118`) and transcript rows are `can_focus = False` (`console_transcript.py:373`), so it cannot even be selected by keyboard.
- The only explanatory tooltip on the card sits on a non-focusable `Static` -- mouse-only, on a keyboard-first product.

Decision taken: **adopt** `tool_message_widgets.py` as the shared tool renderer. It is the file named for this job and the Console never touches it (sole caller `ccp_message_manager.py:156`). Leaving a decoy by that name costs the next reader an hour.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The approval card carries its design-system treatment rather than rendering as body text
- [ ] #2 Arguments get full width at every supported terminal size
- [ ] #3 The tool trace renders at normal weight and its row is focusable
- [ ] #4 tool_message_widgets.py is the shared renderer used by both Console and CCP, or is deleted -- no decoy remains
- [ ] #5 Any why-affordance lives on a focusable element or in the row text, never a tooltip alone
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Direction for the row: put decision controls on their own line beneath the identity/argument block rather than competing horizontally.

Keep `redact_mapping` on every path including any expanded view -- argument redaction is why the card can show payloads at all.
<!-- SECTION:NOTES:END -->
