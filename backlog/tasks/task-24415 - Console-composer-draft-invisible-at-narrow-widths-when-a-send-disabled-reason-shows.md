---
id: TASK-24415
title: >-
  Console composer draft invisible at narrow widths when a send-disabled reason
  shows
status: In Progress
assignee:
  - @zcode
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

- [ ] With a send-disabled reason present and the terminal 80 columns wide,
      the composer draft renders the typed text and the caret — the reason
      strip yields space instead of the draft collapsing to zero columns.
- [ ] The reason strip ellipsizes when horizontal space is tight (below its
      content width), and still renders whole copy at wide widths (≥120).
- [ ] A regression test asserts the *geometry* at narrow width — the draft's
      rendered strip has non-zero width with a reason present — not just a
      `.value` check (see `backlog/docs/lessons-testing-evidence.md`).
- [ ] Live verification in a real terminal at 80 columns with a blocked
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

(added after implementation)
