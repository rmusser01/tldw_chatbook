---
id: TASK-15478
title: STTS audiobook paste box queries a switch that is never composed
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - bug
  - stts
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the latency audit: `UI/STTS_Window.py:555-565` handles TextArea.Changed by materializing the full text then querying `#auto-chapters-switch` — an id that is composed nowhere in the repo (4 query sites at `:388/:443/:528/:561`, zero compose sites) — so the handler raises NoMatches on every keystroke. If the switch were restored as-is, the design would run `ChapterDetector.detect_chapters` over the entire pasted book plus a notify toast per keystroke (`:630-670`).

Decide: restore the switch with detection moved to Submit or a debounced worker, or remove the dead queries. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Typing in the audiobook paste box raises no exceptions (evidence)
- [ ] #2 If chapter detection is kept, it runs off the keystroke path (Submit or debounced worker)
- [ ] #3 The chosen behavior is covered by a test
<!-- AC:END -->
