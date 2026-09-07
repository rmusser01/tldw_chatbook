---
id: TASK-1801
title: 'Disabled control labels are unreadable at ~1.1:1 contrast'
status: Done
assignee: []
created_date: '2026-08-01 13:20'
labels:
  - console
  - ux
  - accessibility
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live verification of the temporary-conversations work (2026-08-01) measured a disabled control's label at foreground `rgb(31,31,31)` on background `rgb(3,3,3)` — roughly 1.1:1 contrast, effectively unreadable without hovering for the tooltip.

This is a general Console convention, not one feature's bug, but it undercuts that feature specifically. Temporary conversations communicate **every** restriction through a disabled control that states its reason: Generate Image, Save Chatbook, and six save-as sinks all render disabled-with-a-reason rather than being hidden, on the explicit principle that a user who cannot find an action assumes the app is broken, while one who sees it greyed out with a reason learns the rule.

That principle fails if the greyed-out label cannot be read. The user gets the worst of both: the action is visibly present, apparently broken, and its explanation is invisible.

A related instance was already fixed in the composer's ☰ menu (a disabled row's reason was tooltip-only and now renders on screen in `$warning`), but the underlying disabled-label styling is unchanged everywhere else.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Disabled control labels meet a stated minimum contrast ratio against their background, and the chosen threshold is recorded in DESIGN.md
- [x] #2 A disabled control's reason is discoverable without hovering, wherever a reason exists
- [x] #3 Disabled still reads as visually distinct from enabled — fixing contrast must not make disabled controls look actionable
- [x] #4 Verified by measuring real rendered colours in a terminal, not by reading token values, since the defect was found by measurement and token names did not reveal it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Measured during the 2026-08-01 live pass on the Console composer's Generate Image row in a temporary chat. The reason text itself was correct and correctly sourced from `EPHEMERAL_BLOCKED_ACTIONS`; only its legibility failed.

Check `.console-action-disabled` and the `$ds-*` disabled tokens in `tldw_chatbook/css/components/_agentic_terminal.tcss`. Never hand-edit `tldw_chatbook/css/tldw_cli_modular.tcss` — regenerate it with `python tldw_chatbook/css/build_css.py`.
<!-- SECTION:NOTES:END -->

## Progress (2026-08-01)

<!-- SECTION:PROGRESS:BEGIN -->
**Composer menu rows: FIXED** — 1.25:1 -> 4.80:1, measured in the running app.
Enabled rows stay at 12.63:1 so the states remain obviously distinct (AC#3).
Threshold and mechanism recorded in DESIGN.md (AC#1); AC#2 was already met by
the on-screen reason lines added alongside PR #1181.

**Root cause:** two dimmers compound, neither visible in the stylesheet — the
theme's `text-disabled: auto 38%` (~3.4:1 alone) plus Textual's
`Button:disabled` `text-style: bold dim` and `color: auto 50%`. All 58 shipped
themes fall below 3:1, including `high_contrast_yellow_black`.

**Two traps, both found by measuring rather than reading the cascade:**
- `text-style: none` does NOT clear Textual's dim; a declared colour still
  renders at ~half strength.
- A screen's `DEFAULT_CSS` cannot override `Button`'s — same tier, Button wins.
  Disabled overrides must go in the app stylesheet.

**Workbench action bar: FIXED** — 1.45:1 -> 6.74:1. It stacked THREE dimmers
(`$ds-text-disabled` alpha + `.is-disabled { opacity: 0.55 }` + Textual's
`bold dim`/`color: auto 50%`). The owning class is `is-disabled`, NOT
`console-action-disabled` — which is exactly why the earlier speculative edit
measured nothing and was reverted. The test now pins the class.

**Both surfaces measured in a running terminal (AC#4):**

| surface | before | after |
| --- | --- | --- |
| composer menu rows | 1.25:1 | 4.80:1 |
| Workbench action bar | 1.45:1 | 6.74:1 |

Enabled controls measure 10.6-12.6:1, so disabled stays plainly distinct (AC#3).
<!-- SECTION:PROGRESS:END -->
