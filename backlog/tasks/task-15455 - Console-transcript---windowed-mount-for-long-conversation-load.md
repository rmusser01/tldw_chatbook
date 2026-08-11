---
id: TASK-15455
title: Console transcript: windowed mount for long-conversation load
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: session resume loads the entire persisted tree (`depth_cap=10_000`) and the first `refresh_messages` mounts every row via individual awaited `mount()` calls — no batching, no windowing (`Widgets/Console/console_transcript.py:2283-2301`; old rows likewise removed one awaited `remove()` at a time). Height-watermark pruning runs only after first layout, so a long conversation pays full mount plus full-history Markdown parse (one Textual Markdown widget per assistant row, one child widget per markdown block) before anything is trimmed — and up to the 12k-20k-line watermarks stay mounted permanently, which also inflates every reconcile pass (task-15453) and layout.

Fix direction: mount a tail-first window (bottom N lines) and hydrate scrollback lazily on scroll; batch mounts. This is structural — stability first: anchor()/tail-follow semantics (`:1295/:1344/:1399-1408`), selection, pruning, and branch navigation must be pinned by tests before the windowing lands. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Loading a 500+-message conversation mounts only the visible tail window initially (evidence + session-switch latency before/after)
- [ ] #2 Scrollback hydrates on demand without breaking anchor/tail-follow, selection, or branch navigation (tests)
- [ ] #3 Prune watermarks still bound total mounted height
<!-- AC:END -->
