---
id: TASK-41527
title: "User Guide accuracy sweep: Library Tab claim and remaining dated changelog prose"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- docs
priority: low
references:
- Docs/User_Guide/library.md
- Docs/User_Guide
dependencies:
- TASK-41101
- TASK-41147
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the docs sweep (TASK-41101, round 15). (1) `library.md` says "Tab stays on this screen"; the old stamp said Tab walks the top tab bar, so check the claim from a cold landing and correct the page or the screen. (2) Dated changelog prose remains in `notes.md`, `file-notes.md`, `agent-runs-and-tools.md`, `settings` and a `collections.md` blockquote; it belongs in the task notes. The small body corrections are in TASK-41147.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Tab claim is checked live from a cold landing and the page matches
- [ ] #2 The listed pages have no dated changelog prose left, found by a grep recorded in the notes
<!-- AC:END -->
