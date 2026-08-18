---
id: task-18156
title: Console keyboard text selection phase 5
status: In Progress
assignee: ['@Robert']
created_date: '2026-08-18'
labels: [console, selection, keyboard]
dependencies: []
priority: high
---

## Description (the why)

Text selection in the Console transcript is mouse-only; the selection menu's
actions (quote, side chat, review feedback) are unreachable by keyboard.
Phase 5 of the console-selection program adds a vim-style single-row
keyboard selection mode that drives the SAME SelectionManager as the mouse,
so both inputs converge on identical selection state and the entire action
pipeline is shared. Spec:
Docs/superpowers/specs/2026-08-18-console-keyboard-selection-and-note-management-design.md
(Part 1). Plan: Docs/superpowers/plans/2026-08-18-console-keyboard-selection.md.

## Acceptance Criteria (the what)

- [ ] `s` enters selection mode only on the j/k-selected message when its row supports selection; ineligible rows toast and do not enter the mode
- [ ] Motions per row kind: h/l/w/b/0/$ char motions on plain and markdown rows, j/k line motions on all kinds, `o` swaps anchor and active end; char motions inert on diff rows; 1-unit selection floor
- [ ] Enter opens the identical selection menu (same anchoring/clamp path, same feedback availability and run gating) as a mouse release
- [ ] Esc layering: first Esc exits the mode keeping message selection; second Esc clears message selection
- [ ] The in-mode hint advertises exactly the keys the active row kind honors; the static footer gains only `s`
- [ ] Keyboard finish drains the release-click suppression tokens so the next genuine row click is not eaten
- [ ] Tests green across the selection suites; ruff clean on touched files
- [ ] Selection menu offers Create note for any selection; a note titled from the selection's first line is durably created with quote + provenance content; failures toast and never block
- [ ] Docs updated: user guide keyboard section, ADR-068 amendment, spec §42 amendment note
