---
id: TASK-397
title: Command palette fast Down+Enter can dismiss without running the command
status: To Do
assignee: []
created_date: '2026-07-20 18:45'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed during live TUI verification (2026-07-20, tmux-driven session): open the command palette (Ctrl+P), type a query that still has SEVERAL matching commands (e.g. "logs"), then press Down and Enter in quick succession (~1s apart) — the palette closed without running the highlighted command and the app stayed on the current screen. Retyping a narrower query that left exactly ONE match, then Down+Enter, worked reliably every time. Likely a race between the palette's async result refresh and selection state (Textual's built-in CommandPalette), but worth reproducing under pilot control to determine whether it is upstream Textual behavior or something in our providers (e.g. commands being re-yielded and resetting the highlight). Fast keyboard users will hit this as "the palette ate my command".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reproduced (or ruled out) under a pilot-driven test: type a multi-hit query, Down+Enter while results are still refreshing
- [ ] #2 If ours: highlighted command runs even when selection races the result refresh; if upstream: issue filed/linked and any feasible mitigation noted on this task
<!-- AC:END -->
