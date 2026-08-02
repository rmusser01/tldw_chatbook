---
id: TASK-1845
title: 'Approval card: Enter approves an unread call, and xN hides what is being approved'
status: Done
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
- [x] #1 No keystroke sequence shorter than two commits an approval from a fresh card
- [x] #2 Every distinct argument set behind a collapsed xN row is visible without a mouse
- [x] #3 Every row state is readable in monochrome
- [x] #4 The card offers direct keyboard decisions rather than a six-control Tab walk
- [x] #5 Tests cover the focus-landing target and the multi-argument disclosure
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Keep `_DEFAULT_DECISION` -- a blank Select breaks `allow_blank=False` and the bulk-assign path. Change the FOCUS target instead: land on the row's Select, never on the commit control.

Decision taken initially: keep one verdict per tool name and disclose every argument set (option a), deferring the per-call re-key.

**That deferral was retired in this same PR.** Verdicts are now keyed per `call_id`, so two reads of two files are two decisions -- the user can allow `spec.md` and refuse `secrets.md`. The re-key is ADDITIVE, not a replacement: the runtime resolves `call_id` first and falls back to name, because the name path is load-bearing in two places -- `MCPToolProvider.apply_batch_decisions` emits name-keyed verdicts, and the fence path (`agent_runtime._fence_call`) builds ToolCalls with NO call_id at all, so a name-keyed verdict must still stop every matching call or the MCP gate silently opens. The card therefore groups per call id where one exists and still collapses by name where none does: splitting id-less calls into separate rows would offer a decision the runtime cannot honour.

Trap found while wiring it: BOTH downstream consumers are name-keyed by contract -- `apply_batch_decisions` takes llm_names, and `builtin_gate.stamp` records a grant against a tool NAME because a session/always grant is per tool, not per call. With the card emitting call-id keys they received nothing: MCP got `{}` and no grant was ever stamped, silently. The entire 85-test approval suite passed with that break in place.

Second trap, on AC #2: the first fix aggregated `all_arguments` and taught the summariser to render it, but `set_batch` still passed `entry["arguments"]` -- the first call's payload -- so the branch never ran in production and the row still showed one target out of three. The test that "covered" it called the helper with a collapsed entry, a shape production never builds. Both shapes are now separate functions (`_summarize_arguments` = one payload, `_summarize_row_arguments` = a collapsed entry), and the regression test drives the mounted widget instead of the helper.

Any new keybinding must route through `_submit_fast_decision`, not post `ApprovalDecided` directly, or it reopens the stale-round hole the `round_id` round-trip and fast-button membership guard exist to close.
<!-- SECTION:NOTES:END -->
