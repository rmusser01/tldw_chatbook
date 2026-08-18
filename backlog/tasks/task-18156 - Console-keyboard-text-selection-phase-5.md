---
id: task-18156
title: Console keyboard text selection phase 5
status: Done
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

- [x] `s` enters selection mode only on the j/k-selected message when its row supports selection; ineligible rows toast and do not enter the mode
- [x] Motions per row kind: h/l/w/b/0/$ char motions on plain and markdown rows, j/k line motions on all kinds, `o` swaps anchor and active end; char motions inert on diff rows; 1-unit selection floor
- [x] Enter opens the identical selection menu (same anchoring/clamp path, same feedback availability and run gating) as a mouse release
- [x] Esc layering: first Esc exits the mode keeping message selection; second Esc clears message selection
- [x] The in-mode hint advertises exactly the keys the active row kind honors; the static footer gains only `s`
- [x] Keyboard finish drains the release-click suppression tokens so the next genuine row click is not eaten
- [x] Tests green across the selection suites; ruff clean on touched files
- [x] Selection menu offers Create note for any selection; a note titled from the selection's first line is durably created with quote + provenance content; failures toast and never block
- [x] Docs updated: user guide keyboard section, ADR-068 amendment, spec §42 amendment note

## Implementation Notes

Vim-style single-row keyboard selection driving the SAME SelectionManager
as the mouse (begin/extend re-anchoring per motion; `_active_selection_row`
resolves through manager state, so a bypass path would produce a menu of
no-ops). Enter replays the mouse-release path by posting
TranscriptTextSelected with row-region coordinates, draining both
release-click suppression tokens (keyboard has no release Click). Full
printable+enter/up/down consumption while armed (Task-2 review found the
nav-key fall-through desync live); Esc layers mode-exit before
message-selection clear. Motions: h/l/w/b/0/$ chars (plain+markdown — its
live-spike char-range storage superseded the line-granularity wording),
j/k lines everywhere, `o` swaps ends (mid-text spans unreachable without
it); diff rows j/k/o only; 1-unit floor.

Task 6 (maintainer request, mid-execution): fourth base menu action
"Create note" — 48-char first-line title, quote + provenance content,
off-thread write through the store's persistence DB, never-raises toasts.

Evidence: TDD throughout with per-task independent reviews (2 Important
findings fixed with RED-first tests: nav-key desync; malformed task file);
keyboard suite 19, menu 45, e2e journeys incl. unmocked SQLite note
round-trip; live tmux verification: s→hint→w/w/l→Enter→menu→Create note→
note row in the real DB with derived title+provenance; two-stage Esc
confirmed live. Known dev-baseline failures excluded by clean-HEAD/dev
comparison. Docs: user guide, ADR-068 amendment 5, both spec amendments.
