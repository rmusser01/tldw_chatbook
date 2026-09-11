---
id: TASK-32332
title: >-
  Staged source status text is user-facing copy
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review C3. Staged-source rows display the normalized CSS class name as the status word -- 'ready'/'running'/'blocked'/'muted' (console_staged_context.py ~15-19, 100-105); 'muted' is a developer catch-all. Map statuses to sentence-case user copy and split 'muted' into real reasons where the state knows them.

Filed from the 2026-09-10 Console rail UX review (review item C3).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend the pinned legacy-row test to sentence-case copy + Off catch-all (RED). 2. Map class tokens to user copy in the widget. 3. Re-run suites.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Status words render in user-facing sentence-case copy agreed with the status vocabulary
- [x] #2 The catch-all bucket no longer shows 'muted'; states that cannot be classified show a plain-language fallback
- [x] #3 CSS status classes are unchanged (styling keeps working)
- [x] #4 Widget tests assert the copy mapping for every status bucket
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**Approach.** The legacy `rows` fallback path rendered the normalized CSS
class token ("ready"/"running"/"blocked"/"muted") as the status TEXT. Added
a `_STATUS_CLASS_COPY` map (Ready/Retrieving/Blocked/Off) used for the
Static's text while the class stays the styling seam. The unclassifiable
catch-all now says "Off" instead of the developer token "muted". Production
constructors use `from_live_work`/`empty()` (already sentence-case
"Ready · …" primary rows); this fixes the hand-built/legacy shape.

**ADR check.** Not required — copy-only change inside one widget.
Linked ADR: N/A.

**Modified.** `tldw_chatbook/Widgets/Console/console_staged_context.py`,
`Tests/UI/test_console_staged_context.py` (pinned status contract updated,
added the muted catch-all case). Verified: 50 passed across the tray +
evidence-strip suites.

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

SPLIT on dev: primary path renders 'Ready - title' readable copy (console_staged_context.py:97-104, console_display_state.py:826-853). Legacy 'rows' fallback still renders the raw class token and it is pinned by Tests/UI/test_console_staged_context.py:205-232. Scope: fix legacy fallback copy + update the pinned test.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
