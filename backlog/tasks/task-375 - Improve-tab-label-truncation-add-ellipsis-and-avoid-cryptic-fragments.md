---
id: TASK-375
title: Improve tab-label truncation - add ellipsis and avoid cryptic fragments
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resumed conversations open tabs labelled with ~9-14 characters of the title and no truncation marker: 'Long conversation about embeddings and vector stores in local RAG' becomes just 'Long'; 'Terraform state migration help' becomes 'Terraform'. Two conversations sharing a first word would be indistinguishable in the strip; the label doesn't even hint it is truncated.

**Repro:** Resume a conversation with a long title; read its tab label in the strip.

**Verifier note:** Code-verified rendering defect: _display_title truncates to 19 chars and appends '...' (console_session_surface.py:32,138-144), but the fixed 21-cell button (CONSOLE_SESSION_TAB_WIDTH) word-wraps the label at ~16 usable cells and height-1 shows only the first line — i.e. the first word ('Long', 'Terraform'), never the ellipsis. So the intended truncation marker exists in code and is defeated by Button word-wrap. Not covered by tab-strip-symbols (glyphs only) or auto-title-30ch (session titles). Tooltips carry the full title, keeping this P3.

**Source:** Console UX expert review 2026-07-20 (finding j2-tab-labels-cryptic-fragments; P3, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J2 returning power user journey. Evidence: `j2-12-reply-later.png`, `j2-08-after-switcher-select.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Truncated tab labels always show a visible truncation mark ('…') so a fragment is never mistaken for the full title
- [ ] #2 Labels are wide enough (or middle-truncated) to preserve distinguishing words between similarly-named conversations
<!-- AC:END -->
