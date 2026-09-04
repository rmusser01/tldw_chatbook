---
id: TASK-24415
title: >-
  Console composer draft invisible at narrow widths when a send-disabled reason
  shows
status: Done
assignee:
  - '@zcode'
created_date: '2026-08-29'
updated_date: '2026-08-29'
labels:
  - console
  - defect
  - ux
priority: high
dependencies: []
---

## Description (the why)

A user reported the Console `/` command trigger as "funky in a bad way". Live
verification (tmux, real app, isolated scratch profile, 2026-08-29) traced the
worst component to the composer row layout, not the popup logic: at **≤90
columns**, whenever `#console-send-disabled-reason` is rendered — missing API
key, unfinished provider setup, **or an active run streaming** — the row's
fixed-width furniture (label ≈20 cells + reason strip up to
`SEND_REASON_MAX_WIDTH = 52` + Send + Dictate) consumes the whole row and the
`1fr` draft collapses to **zero columns**: no draft text, no caret, no
placeholder. The user types (including `/` to trigger commands) and sees
nothing while the suggestion popup filters against invisible input. At 95
columns the draft silently truncates mid-word with no ellipsis.

The width sweep (120→110→100→95→90→85→80): visible at ≥100 (cramped),
truncated at 95, invisible at ≤90 with the 42-cell "Send blocked — add an API
key to continue" copy. The reason strip's stylesheet ellipsis only engages
*above* its 52-cell cap — below it the strip renders whole and never cedes
space, the opposite of the code comment's claim ("narrower composers
ellipsize … while the `1fr` draft yields the space" —
`console_composer_bar.py:397-402`). The `1fr` draft is what yields, to nothing.

This is not limited to unconfigured setups:
`build_console_disabled_reason` (`Chat/console_display_state.py`) also emits a
reason while a run is active, so a fully-configured user on a narrow terminal
(or a tmux split pane, ≈80 cols) goes blind in the composer whenever a response
is streaming.

Found during the same review as TASK-24416 (popup etiquette) and TASK-24417
(hands-free Escape ordering).

## Acceptance Criteria

- [x] With a send-disabled reason present and the terminal 80 columns wide,
      the composer draft renders the typed text and the caret — the reason
      strip yields space instead of the draft collapsing to zero columns.
- [x] The reason strip ellipsizes when horizontal space is tight, and hides
      entirely below a legible budget (~12 cells); it renders whole copy
      only when the row can spare its full width beyond the enforced
      32-cell draft floor (measured live: ellipsized at 120 app columns,
      hidden at 80, whole at ~160+). The Send tooltip always carries the
      full reason.
- [x] A regression test asserts the *geometry* at narrow width — the draft's
      rendered strip has non-zero width with a reason present — not just a
      `.value` check (see `backlog/docs/lessons-testing-evidence.md`).
- [x] Live verification in a real terminal at 80 columns with a blocked
      provider: typing `/` and filter text is visible in the composer.

## Implementation Plan

ADR required: no
ADR path: N/A
Reason: layout bug fix inside the existing ConsoleComposerBar widget; no
storage, contract, or boundary decision.

1. Reproduce in the pilot harness at 80 columns with the reason strip
   populated; assert the draft strip's current zero-width (RED).
2. Fix the row layout so the reason strip cannot starve the `1fr` draft:
   give the draft a minimum share (e.g. `min-width` + the strip
   `text-overflow: ellipsis` below content width), or relocate the reason off
   the draft row at narrow widths — whichever matches the existing composer
   layout conventions.
3. Re-run the targeted composer/popup tests; re-verify live at 80 cols via
   the tmux recipe.

## Implementation Notes

Fixed 2026-08-29, TDD (RED reproduced the bug in the harness: draft laid out
at **2 columns** at 80-column app width with the idle reason present; GREEN
after the fix; 217 neighboring composer/popup tests green; live tmux
verification at 80 cols shows `/` + caret + filtering popup, and at 120 cols
an ellipsized strip beside a full-floor draft).

- **Approach**: the reason strip's static 52-cell auto width became a
  live-width-derived cap — `_send_reason_width_cap()` on
  `ConsoleComposerBar` computes `row_width − LEFT_CLUSTER_WIDTH(18) −
  actions_row_width − DRAFT_MIN_RENDER_WIDTH(32)`; below
  `SEND_REASON_MIN_LEGIBLE_WIDTH(12)` the strip hides (display:none, content
  retained) because the Send tooltip carries the same reason.
  `_sync_send_disabled_reason` applies it; `on_resize` re-derives it so a
  shrink retracts the strip rather than the draft.
- **Why not a static `min_width` on the draft**: with fixed/auto siblings,
  Textual clamps the fr child to its min and then overflows the container —
  at narrow widths that pushes the actions row (Send/Dictate) off-screen,
  trading an invisible draft for unreachable buttons. The Python-side clamp
  matches the composer's existing dynamic-width pattern
  (`_set_actions_row_width`).
- The TASK-2154.14 constants comment promised "the draft keeps its 32-cell
  floor" in arithmetic only; `DRAFT_MIN_RENDER_WIDTH = 32` now states it in
  layout terms the strip budget enforces.
- Files: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
  (constants + `_send_reason_width_cap` + sync/resize hooks),
  `Tests/UI/test_console_composer_reason_width.py` (3 geometry tests:
  narrow, wide-with-cap, resize-retract).
- Known same-class case, out of scope here: `#console-voice-status`
  (dictation chip, up to 53 cells) can starve the draft the same way during
  dictation at narrow widths; its full-width mode already force-hides the
  reason strip. File separately if it bites.
- ADR: not required (layout bug fix within an existing widget; no contract
  or boundary decision).
