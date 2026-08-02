---
id: TASK-1845
title: 'Approval card: Enter approves an unread call, and xN hides what is being approved'
status: To Do
assignee: []
created_date: '2026-08-01 19:30'
labels:
  - console
  - ux
  - agents
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three defects that combine into one risk: it is easy to approve something you have not seen.

1. **Enter approves.** `_DEFAULT_DECISION = "approve_once"` pre-arms the row, and BOTH review entry points focus `#approval-submit` (`console_status_chips.py:489`, `chat_screen.py:17428`). The documented keyboard route ends one keystroke from granting a tool access.
2. **`xN` hides calls.** `_collapse_pending_calls` groups by tool name, keeps the FIRST call's arguments and increments a counter. Three calls to three different targets render as one row showing one target. The user approves three having seen one.
3. **State is colour-only.** `.approval-row.needs-decision` (`_agentic_terminal.tcss:5451`) is border + 10% tint with no text label, against PRODUCT.md's "colour must never be the only carrier of meaning".

Plus: `ChatApprovalCard` declares no `BINDINGS` at all, so every decision on a keyboard-first product is a Tab walk through six controls.

Priority is raised by the product's actual threat model: tools are how agents reach the outside world, so the approval card is the egress boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No keystroke sequence shorter than two commits an approval from a fresh card
- [ ] #2 Every distinct argument set behind a collapsed xN row is visible without a mouse
- [ ] #3 Every row state is readable in monochrome
- [ ] #4 The card offers direct keyboard decisions rather than a six-control Tab walk
- [ ] #5 Tests cover the focus-landing target and the multi-argument disclosure
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Keep `_DEFAULT_DECISION` -- a blank Select breaks `allow_blank=False` and the bulk-assign path. Change the FOCUS target instead: land on the row's Select, never on the commit control.

Decision taken: keep one verdict per tool name and disclose every argument set (option a). Re-keying verdicts per call id was considered and deferred -- it touches the provider, the gate, the round-trip and every approval test, and adds prompt volume for internal reads that are assumed permitted. Revisit it scoped to outside-world tools only, where per-target granularity buys real security.

Any new keybinding must route through `_submit_fast_decision`, not post `ApprovalDecided` directly, or it reopens the stale-round hole the `round_id` round-trip and fast-button membership guard exist to close.
<!-- SECTION:NOTES:END -->
