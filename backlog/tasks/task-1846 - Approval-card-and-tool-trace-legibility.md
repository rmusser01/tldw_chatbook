---
id: TASK-1846
title: 'Approval card is unstyled and the tool trace is the faintest text on screen'
status: In Progress
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
- [x] #1 The approval card carries its design-system treatment rather than rendering as body text
- [ ] #2 Arguments get full width at every supported terminal size
- [x] #3 The tool trace renders at normal weight and its row is focusable
- [x] #4 RESOLVED BY A THIRD ROUTE: neither shared nor deleted. `tool_message_widgets.py` is documented as CCP-owned at the top of the file, because adopting one renderer would break either the Console's row model (`ConsoleTranscriptMessage(Static)` selection/jump-pill behaviour) or its markup-OFF escaping contract. The decoy was the generic NAME reading as "tool rendering lives here"; the docstring now points Console work at `format_agent_step_marker`
- [x] #5 Satisfied: the state itself is row TEXT (`(high risk)`/`(definition changed)` badges via `_REASON_SUFFIXES`, and `needs decision · ` via `NEEDS_DECISION_PREFIX`); the tooltip only elaborates, so no meaning is tooltip-only
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Direction for the row: put decision controls on their own line beneath the identity/argument block rather than competing horizontally.

Keep `redact_mapping` on every path including any expanded view -- argument redaction is why the card can show payloads at all.
**AC #2 remains open** -- the approval row's information budget (roughly 54 fixed cells of controls crowding the argument text) needs the controls moved to their own line. Deferred pending live verification at several terminal widths, because the 32-row compact contract makes a naive reflow a regression at small heights. Left In Progress.
<!-- SECTION:NOTES:END -->

## Progress (2026-08-01)

<!-- SECTION:PROGRESS:BEGIN -->
**Done:** `.ds-approval-card` now applied via `DEFAULT_CLASSES` (it was defined in
the design system and applied by nothing, so the card rendered as body text);
tool-trace rows no longer `dim` — they carry `$ds-text-primary` with italic kept
as the quiet distinguishing mark.

**`tool_message_widgets.py`: adopt was REJECTED on evidence, not skipped.** The
decision was "adopt", but the two renderers are structurally incompatible:

1. `ToolResultMessage` extends `ChatMessage`; the Console transcript is built
   from `ConsoleTranscriptMessage(Static)` rows whose selection, `sync_message`
   and jump-pill behaviour depend on that widget type.
2. CCP formats with Rich MARKUP (`[bold green]`, `[red]`); the Console's marker
   text is deliberately markup-OFF — `format_agent_step_marker` documents that
   escaping for a parser that never runs left literal backslashes in markers.

Sharing one renderer would break one surface's row model or the other's escaping
contract. Instead the module docstring now names its sole consumer and says
plainly that the Console does not and cannot use it, pointing changers at
`format_agent_step_marker`/`console_transcript.py` — which removes the decoy
without a risky refactor.

**Still open:** the row's information budget (26+14+14 = 54 fixed cells for
controls) and moving decision controls onto their own line so arguments get full
width. Deferred as a layout change wanting live verification at several terminal
widths.
<!-- SECTION:PROGRESS:END -->
