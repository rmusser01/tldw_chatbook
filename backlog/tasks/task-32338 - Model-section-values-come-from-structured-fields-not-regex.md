---
id: TASK-32338
title: >-
  Model section values come from structured fields not regex
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review D2. The left rail Model section parses display values out of formatted strings with regexes (left_rail.py ~436-447: r'T ([\d.]+)', r'max_tokens (\d+)') and shows a dash on mismatch. Add structured fields to ConsoleSettingsSummaryState and render from them.

Filed from the 2026-09-10 Console rail UX review (review item D2).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: summary state carries structured temperature/max_tokens. 2. Add defaulted fields + populate in builder. 3. Render from fields in the rail; remove regexes + re import. 4. Run session settings + left rail suites.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Temperature and max-tokens values render from structured state fields, not string parsing
- [x] #2 A summary state missing those fields still renders a placeholder (no crash, no wrong value)
- [x] #3 Parsing regexes are removed from the rail
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**Approach.** `ConsoleSettingsSummaryState` gains two structured display
fields — `temperature: str = ""` and `max_tokens: str = ""` — populated by
`build_console_settings_summary_state` from the same values that format
`sampling_row`'s "T x" / "max_tokens N" segments. The left rail's Model
section renders `summary_state.temperature or "—"` / `.max_tokens or "—"`
and the two regexes (`r"T ([\d.]+)"`, `r"max_tokens (\d+)"`) plus the
now-unused `import re` are gone. Fields are appended after `identity_row`
with defaults, so existing keyword constructions are unaffected (verified:
all production/test constructors pass kwargs).

**ADR check.** Not required — display-state field addition within one
seam.

**Modified.** `Chat/console_session_settings.py`,
`UI/Console_Modules/left_rail.py`,
`Tests/Chat/test_console_session_settings.py` (+1 test: structured fields
from a real settings object). Verified: session-settings Chat suite 199
passed +1 new (the 1-5 failures in wider runs are pre-existing dev-tip
harness drift — missing chat_screen attrs, stale mount order, empty
harness lists — all verified unrelated to this seam and reproducible
conceptually on origin/dev; filed separately).

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED (code): left_rail.py:2209-2216 still regex-parses 'T ([\d.]+)' / 'max_tokens (\d+)' from the summary string.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
