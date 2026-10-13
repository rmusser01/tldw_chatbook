---
id: TASK-34938
title: Debounce-the-file-notes-search-and-cache-its-navigator-lookups
status: To Do
assignee: []
created_date: '2026-10-13 14:57'
labels:
  - performance
  - ui-perf
  - wave-4-blocking-rebuilds
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The file-notes search box has NO debounce (library_file_notes_workspace.py:6960): every keystroke runs 5 query_one DOM walks in _sync_navigator_mode, spawns an exclusive worker, issues an FTS replica.search (or, replica-less, re-casefolds EVERY entry per keystroke), and rebuilds a 100-row result tree. Every comparable search box in the repo is debounced (media 0.3s, artifacts 0.2s, emoji 0.3s) — this one was missed, and it is the highest frequency-x-cost rebuild path remaining.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Input.Changed debounced (repo-standard 0.2-0.3s) with in-flight generation guard,Navigator query_one lookups cached as widget refs (no DOM walks per keystroke),Replica-less fallback no longer re-casefolds all entries per keystroke (pre-folded index)
<!-- AC:END -->
