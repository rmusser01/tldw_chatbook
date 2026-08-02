---
id: TASK-1846
title: 'Approval card is unstyled and the tool trace is the faintest text on screen'
status: Done
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
- [x] #2 Arguments get full width at every supported terminal size
- [x] #3 The tool trace renders at normal weight and its row is focusable
- [x] #4 RESOLVED BY A THIRD ROUTE: neither shared nor deleted. `tool_message_widgets.py` is documented as CCP-owned at the top of the file, because adopting one renderer would break either the Console's row model (`ConsoleTranscriptMessage(Static)` selection/jump-pill behaviour) or its markup-OFF escaping contract. The decoy was the generic NAME reading as "tool rendering lives here"; the docstring now points Console work at `format_agent_step_marker`
- [x] #5 Satisfied: the state itself is row TEXT (`(high risk)`/`(definition changed)` badges via `_REASON_SUFFIXES`, and `needs decision · ` via `NEEDS_DECISION_PREFIX`); the tooltip only elaborates, so no meaning is tooltip-only
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Direction for the row: put decision controls on their own line beneath the identity/argument block rather than competing horizontally.

Keep `redact_mapping` on every path including any expanded view -- argument redaction is why the card can show payloads at all.
**AC #2 remains open** -- the approval row's information budget (roughly 54 fixed cells of controls crowding the argument text) needs the controls moved to their own line. Deferred pending live verification at several terminal widths, because the 32-row compact contract makes a naive reflow a regression at small heights. Left In Progress.

## AC#2 (the deferred one)

Measured before touching anything, with the SHIPPED stylesheet loaded -- a bare `App` harness measures an unstyled row (header and args both reported the full 80 cells, the Select 1) and would have passed any test written against it:

| cols | row | header | args | select |
|---|---|---|---|---|
| 80 | 74 | 10 | **10** | 26 |
| 120 | 114 | 30 | **30** | 26 |
| 212 | 206 | 76 | **76** | 26 |

Ten cells shows `{"path":"~/` of `{"path":"~/notes/secrets.md"}`. Since TASK-1861 the card offers one decision per TARGET, so telling `spec.md` from `secrets.md` IS the row's job, and at 80 columns it was impossible.

`.approval-row` is now a Vertical of three stacked full-width lines: header, arguments, then `.approval-row-controls`. Args go 10 -> 74 cells at 80 columns.

**The task note's prescription -- controls on their OWN line -- was right, and the "cheaper" variant is wrong.** Keeping the header beside the controls to save a line passed every mounted-widget measurement at 80/120/212 and looked correct. In a real terminal at 120x40 it was broken: the Console's chat pane is only ~52 cells, the 54 fixed cells of controls starve the header to ONE cell, it wraps to nine lines, and the arguments are pushed out of the card entirely -- strictly worse than the layout it replaced, which at least wrapped the args into a narrow column. Measured 46x1 for the header after the correction, 1x9 before.

**The deferral was right.** The extra line moved a pre-existing cliff: on an 80x24 terminal the card is `height: auto` inside a plain Container, so a long batch grew past the viewport and took Submit with it. Five rows already did that on dev (Submit at y=24); four rows would have with this change. `#approval-batch-rows` is now capped at 15 and scrolls, which fixes the older bug rather than inheriting it -- Submit stays visible at 1, 4, 6 and 10 rows.

Three CSS guarantees, each mutation-verified: the row cap (drop it -> 4 rows push Submit off), the headline's `height: auto` (drop it -> the row balloons 5 -> 15, the fr-inside-flex trap this block already documents), and the full-width args. The height rule needed a test written FOR it -- the width and action-bar tests both still passed while the row was 3x too tall.

Live-verified in tmux at 80x24, 120x40 and 212x64 against the real app (batch injected through `ChatScreen.set_task_resume_state`, the same seam production uses; everything below it is the shipped path). At 120x40 the argument now reads `{"path":"~/notes/project-spec.md"}` in full on its own line. **This is why AC#2 was deferred for live checks rather than shipped on green tests** -- the tests were green on a layout that was broken on screen.

Two existing geometry contracts encoded the old one-line layout (`args.x >= header.right`). Updated to assert the guarantee they were protecting -- no widget overlaps another, in 2D -- rather than a left-to-right ordering that no longer describes the card.
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
