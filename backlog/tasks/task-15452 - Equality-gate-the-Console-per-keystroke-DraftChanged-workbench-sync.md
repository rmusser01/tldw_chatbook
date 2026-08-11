---
id: TASK-15452
title: Equality-gate the Console per-keystroke DraftChanged workbench sync
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
Verified first-hand: the `ConsoleComposerBar.DraftChanged` handler (`chat_screen.py:18904`) routes through `_sync_console_workbench_actions_from_draft` (`:18143`), which rebuilds and pushes Workbench state with no comparison against `_last_console_workbench_state` — bypassing the guard the main sync path has at `:18039-18041`. Per printable keystroke this produces ~12 layout-invalidating `Static.update()` calls across DestinationHeader/ModeStrip/CommandStrip/RecoveryCallout plus two scheduled `sort_children` calls; Textual's `sort_children` bumps the DOM version up the ancestor chain even when the order is unchanged, invalidating the screen-wide `query_one` LRU cache so the next keystroke's ~15 screen-rooted queries become full DOM walks over the largest tree in the app. The same path also rebuilds provider/readiness state 7-8 times per keystroke.

Fix direction: route the draft path through the same state-equality guard as `:18039`; add `state == self.state -> return` early-outs inside the four Workbench widget sync methods (`UI/Workbench/workbench_widgets.py`); skip `sort_children` when the desired order already matches; build provider selection state once per keystroke and pass it down. Stability constraint: the slash-command popup open/close and first-run-guidance dismissal ride this handler — pin their behavior with tests before landing the gate. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No Workbench widget writes occur when the derived state is unchanged (evidence)
- [ ] #2 sort_children is not invoked when child order already matches (evidence), and the query_one cache survives idle keystrokes
- [ ] #3 Slash popup open/close and guidance dismissal behavior unchanged (tests pinned before the gate)
- [ ] #4 Per-keystroke handler cost measured before/after and recorded
<!-- AC:END -->
