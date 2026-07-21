---
id: TASK-374
title: Rebalance rail rows - 17-char title truncation vs full-line boilerplate second lines
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In a 48-cell-wide rail, titles render as 'Websocket reconne...', 'Compare SQLite FT...', 'Meeting notes 202...' (17 chars + ellipsis, well short of the ~30 chars the width allows), because ~10 cells right of the title are reserved (star toggle + padding). Meanwhile all 11 rows repeat the same subtitle text, differing only in the age digits — so half the section's vertical space carries almost no information while the distinguishing part of the titles is cut. Long titles ('Long conversation...') become indistinguishable from any other 'Long conversa...' item.

**Repro:** Open Console with the 12 seeded conversations at 2050x1240 and read the rail Chats section.

**Verifier note:** Code-verified: _MAX_CONVERSATION_ROW_TITLE=20 is a fixed constant (console_workspace_context.py:52,988-993) regardless of available rail width, and every row repeats 'workspace_label - detail - age' as the second line (line 924-929). Real density/recognition critique, but downgraded to P3: full titles are available via row tooltips (tooltip_label carries untruncated title) and Ctrl+K search; nothing is blocked. Not covered by any ledger item (micro-polish-186 fixed different rail nits).

**Source:** Console UX expert review 2026-07-20 (finding j2-rail-title-truncation-and-boilerplate; P3, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J2 returning power user journey. Evidence: `j2-02-initial.png`, `j2-20-boot2-rail.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Titles get the available width (25-35 chars) with tighter right-side controls
- [ ] #2 Subtitle should compress to just the differentiator (age, or state only when not the default)
<!-- AC:END -->
