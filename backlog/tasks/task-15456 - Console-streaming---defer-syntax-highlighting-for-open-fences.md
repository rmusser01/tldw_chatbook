---
id: TASK-15456
title: Console streaming: defer syntax highlighting for open fences
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
From the audit: Textual's `Markdown.append` re-parses only from the last completed top-level block, so a reply that is one long code fence re-parses the whole fence and re-runs Pygments over the entire fence-so-far on every 0.2 s tick, synchronously on the event loop (`textual/widgets/_markdown.py:1445-1509`, `MarkdownFence` highlight at `:895-901`); a growing paragraph is a remove+remount per tick. Multi-block prose is genuinely O(delta) — the worst case is exactly the long-code-block replies this audience produces.

Fix direction: throttle fence-interior appends (e.g. plain-text tail while the fence is open, highlight at fence close, or a slower cadence for fence-interior deltas). Keep final rendered output byte-identical at stream end. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Streaming a long single code fence no longer re-highlights the full fence every tick (evidence)
- [ ] #2 Final rendered message identical to today's output at stream end (test)
- [ ] #3 No behavior change for multi-block prose streaming
<!-- AC:END -->
