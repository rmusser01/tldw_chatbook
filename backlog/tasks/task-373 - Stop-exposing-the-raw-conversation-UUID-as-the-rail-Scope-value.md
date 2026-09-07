---
id: TASK-373
title: Stop exposing the raw conversation UUID as the rail Scope value
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After resuming a conversation the rail Session block reads 'Scope d1ebe478-c825-46b6-83b3-d5901d7bb3a1' (wrapped mid-token). The UUID has no user meaning, duplicates nothing useful, and consumes two lines of premium rail space.

Also observed independently in J6 keyboard-only/small-terminal as `j6-scope-raw-uuid`: Rail 'Scope' field displays the raw conversation UUID wrapped mid-token.

**Repro:** Resume any saved conversation and read the rail Session section's Scope row.

**Verifier note:** Code-verified: build_console_workspace_state sets scope_label = str(current_conversation or '') — the raw conversation UUID (display_state.py:282) — rendered as the 'Scope' label/value pair (console_workspace_context.py:581-588), wrapping across two rail lines as screenshotted. The setup-state-callout ledger item blesses label/value rows but not raw-identifier content; no task covers humanizing it. Straightforward jargon-leak polish fix.

**Source:** Console UX expert review 2026-07-20 (finding j2-scope-uuid-exposed, j6-scope-raw-uuid; P3, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J2 returning power user journey. Evidence: `j2-08-after-switcher-select.png`, `j2-24-resumed-long.png`, `j2-57-mangle-trialB.png`, `j6-a18-select-user.png`, `j6-a32-post-shift-enter.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Human-readable scope ('This conversation') or omit
- [x] #2 Keep identifiers in a debug/details view
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`build_console_workspace_state` set `scope_label = str(current_conversation or "")`
— the raw conversation UUID, wrapped mid-token across two rail lines. Now the
label is "This conversation" (or "" when none), and the raw id is carried in a
new `scope_detail` field surfaced only as the Scope row's hover tooltip
("Conversation id: <uuid>") — human-readable primary row, identifier in a details
view. RED->GREEN `test_console_scope_shows_readable_label_not_raw_uuid`; 63 tests
green. (Also resolves the Scope-UUID part of task-387.)
<!-- SECTION:NOTES:END -->
