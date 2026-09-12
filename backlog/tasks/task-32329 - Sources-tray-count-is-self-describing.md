---
id: TASK-32329
title: >-
  Sources tray count is self-describing
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
UX review B5. The staged-context tray header shows a bare digit (str(state.source_count), console_staged_context.py ~58-67) with no unit; zero renders as '0' beside the word Sources. Render a self-describing count with a real empty word.

Filed from the 2026-09-10 Console rail UX review (review item B5).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update the zero-count pinned test to expect 'none' (RED). 2. Render word-form zero in the tray header. 3. Re-run the tray + evidence-strip suites.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The tray count renders 'none' (word form) when zero sources are staged, and the plain digit otherwise (title already names the noun)
- [x] #2 Existing sync_state fingerprinting still updates the count in place
- [x] #3 Widget tests assert the wording for 0, 1, and N sources
<!-- AC:END -->
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**Approach.** The tray header's count Static now renders the word "none"
when zero sources are staged and the plain digit otherwise — the title
("Sources — next send") already names the noun, so the fix is only the
zero case that read like a broken counter.

**ADR check.** Not required — copy-only change, no boundary moved.
Linked ADR: N/A.

**Modified.** `tldw_chatbook/Widgets/Console/console_staged_context.py`,
`Tests/UI/test_console_staged_context.py` (zero-count contract updated).
Verified: `pytest Tests/UI/test_console_staged_context.py
Tests/UI/test_console_staged_evidence_strip.py` — 50 passed.

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED LIVE: header now reads 'Sources - next send' + bare digit '0'. Rescope: zero should read 'none' (word form only where it clarifies); title already carries the noun.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
