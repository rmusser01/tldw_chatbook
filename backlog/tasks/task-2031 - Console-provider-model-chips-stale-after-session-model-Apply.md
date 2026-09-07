---
id: TASK-2031
title: 'Console provider/model chips stale after session model Apply'
status: Done
assignee: []
created_date: '2026-08-03 00:45'
labels:
  - console
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in TASK-1980 live UAT. Changing provider/model via the Provider chip's
session model modal and pressing Apply updates the SESSION (the next run
uses the new provider — verified against a local stub endpoint) but the
status chips keep showing the old provider/model until a session/tab switch
forces a refresh. The user watches "Provider: Anthropic" while the run is
actually served by Custom — the chips' whole purpose (PR #1153) inverted.

The left-rail Model section DOES show the new values immediately; only the
status chip row misses the poke after Apply.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After Apply in the session model modal, the Provider/Model chips reflect the new values without switching sessions
- [x] #2 A test pins the chip refresh on the Apply path
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Closed as not-a-bug, with the contract pinned by a new test.**

A full-flow harness test (`Tests/UI/test_console_model_apply_chips.py`)
drives the real popover-apply path — `_apply_console_model_popover_result`
→ `_replace_active_console_session_settings` → the tick's
`_sync_console_control_bar` — and the provider chip refreshes WITHOUT any
session switch. It passes against unmodified production code.

Re-examining the TASK-1980 UAT evidence with that in hand: every "stale"
chip capture was taken while the popover was still open or immediately
after a MISSED Apply click (the UAT's own tmux-driving defect: `grep -bo`
returns byte offsets, and multibyte glyphs shifted the click columns, so
several Apply presses never landed). The first capture after a VERIFIED
successful Apply (modal actually closed) showed the new provider
correctly, mid-run, no tab switch. The original filing conflated
"Apply didn't fire" with "chips didn't refresh".

AC#1 holds on production code (proven by the new test); AC#2's test now
exists and stays as a regression pin. No production change.
<!-- SECTION:NOTES:END -->
