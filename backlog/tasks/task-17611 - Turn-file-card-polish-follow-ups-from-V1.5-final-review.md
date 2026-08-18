---
id: TASK-17611
title: 'Turn-file-card polish follow-ups from V1.5 final review'
status: To Do
assignee: []
created_date: '2026-08-17 17:19'
labels: [console, ux-polish]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Five parked minors from the turn-file-card annotate feature's V1.5 final review
(`feat/console-turn-file-annotate`), each a real but low-priority polish item deliberately not
folded into the final-review fix wave (which scoped itself to correctness/safety fixes and doc
honesty). Filed together as one follow-up task since each is small and independent; none blocks
the others.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `ConsoleTurnFileCard`'s hunk blocks have an honest ceiling: a file with an unusually large number of hunks stops mounting one block per hunk past a cap and instead shows a "… N more hunks" tail, rather than mounting an unbounded number of blocks.
- [ ] #2 The header's expand/collapse-all toggle button label derives from the live DOM state at every render (not just at toggle-press time), so it never shows a stale "expand all" chevron/tooltip after the user has manually expanded some rows individually.
- [ ] #3 The card's ✎ (note)/✕ (delete)/📝 (delivery disclosure) glyphs are all routed through `resolve_glyph` for terminal-fallback safety, matching the card's existing chevron glyphs.
- [ ] #4 The bundle-attach and diff-feedback-attach loops in `run_reply` (`Chat/console_agent_bridge.py` or wherever `run_reply` lives) share one "append to the last user message" helper instead of two near-duplicate inline loops.
- [ ] #5 `middle_elide_path` budgets in terminal display CELLS (accounting for double-width characters) rather than raw `len()` characters, so a path containing wide characters elides to the correct visual width instead of overflowing or under-filling the row.
<!-- AC:END -->
