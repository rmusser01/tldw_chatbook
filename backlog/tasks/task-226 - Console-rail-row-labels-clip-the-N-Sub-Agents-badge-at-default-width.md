---
id: TASK-226
title: 'Console rail row labels clip the [N Sub-Agents] badge at default width'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-14 03:33'
updated_date: '2026-07-16 16:18'
labels:
  - console
  - agents
  - ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The agent-runtime live gate (Docs/superpowers/qa/agent-runtime-2026-07/) showed the conversation-row badge renders as '[1' at the rail's default width — the same truncation every row label already has (titles clip at ~20 chars). Pre-existing display constraint, surfaced by the new badge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The [N Sub-Agents] badge is fully visible on conversation rows at the default rail width,Row titles degrade gracefully (ellipsis) without swallowing trailing badges
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace format_console_conversation_row_label + _conversation_button + _compose_conversation_browser_row to find where the badge is appended relative to the row's title/secondary lines.
2. Root-cause: the badge is appended to the END of the row's already-composed multi-line text, sharing the LAST line with the unbounded secondary-detail text (workspace - status - age); a long secondary line pushes the trailing badge past the row's rendered width and Textual Buttons clip (no wrap), cutting the badge.
3. Fix: give the badge its own dedicated trailing line (format_console_conversation_row_label appends "\n[dim]...[/dim]" instead of "  [dim]...[/dim]"), decoupling its visibility from title/secondary length.
4. Grow the row + star button height to 3 lines only when a badge will render (_conversation_row_render_height), and sum per-row height in _conversation_browser_list_height so badge rows reserve their extra line without disturbing badge-less rows.
5. RED-verify new tests against the unmodified source via git stash, then restore the fix and confirm GREEN; run the full requested regression suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: format_console_conversation_row_label appended the badge to the
END of the row's already-composed multi-line text (title line + secondary
detail line), so the badge shared the LAST line with the unbounded
secondary-detail text (workspace label - status - age). Textual Buttons
clip on the right edge (no wrap), so whenever that combined line exceeded
the row's rendered width, the badge -- sitting at the tail -- was the first
thing cut (observed as a bare "[1" in the agent-runtime live gate).

Fix: give the badge its own dedicated trailing line
(format_console_conversation_row_label now appends "\n[dim]...[/dim]"
instead of "  [dim]...[/dim]"), decoupling its visibility from how long the
title or secondary line happen to be -- the badge's own line is always
~14-16 chars regardless. New _conversation_row_render_height(subagent_count)
grows the row button (and matching star button) to 3 lines only when a
badge will render; _conversation_browser_list_height sums real per-row
height via a new _conversation_browser_rows_height helper instead of a flat
per-row constant, so badge rows reserve their extra line without disturbing
badge-less rows. Legacy (workspace-membership) rows are unaffected --
that call site never passes subagent_count.

4 new tests in Tests/UI/test_console_agent_rail.py, RED-verified via git
stash against the unmodified source (3 of 4 failed as expected), then
GREEN after restoring the fix. Full requested regression suites: 347
passed / 1 pre-existing failure out of 348 (test_console_workspace_context_
syncs_active_conversation_marker, confirmed pre-existing via the same
git-stash isolation -- fails identically without this change). No CSS
touched. Full report: .superpowers/sdd/task-226-report.md.
<!-- SECTION:NOTES:END -->
