---
id: TASK-32610
title: >-
  Library Notes: the lasting-sync setup pane ships a garbled Obsidian sentence,
  a false scroll hint and a stale Resolution history line
status: To Do
assignee: []
created_date: '2026-09-15 06:39'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A P2 and assessor B D4, personas Jordan and Alex, Obsidian workflow. Heuristic 10 setter -- the one thing that kept Help and documentation from rising.

What happened, three strings on the Keep-a-folder-synced panes.
1. The toggle that decides how a whole vault is read explains itself as, verbatim: 'Turning it off lasts until you quit Chatbook: a vault is offered it again, on, on the next start.' (A cap 48, B cap 31).
2. 'More below — scroll.' is printed above 21 blank rows on the same pane (A cap 48 tail).
3. After activation the pane still reads 'Resolution history unavailable — it starts after this root is activated' with the control disabled (A cap 50); in the sync review the same control is a bare circle glyph with no reason at all (B cap 32, D19) -- the pattern the screen follows correctly for Sort, Server notes and Commit.

Attribution: string 1 is a WAVE-4 REGRESSION, PROVEN. git show 5fd502dbac of Widgets/Library/library_notes_add_from_files_canvas.py has zero occurrences of the sentence and no notes-sync-obsidian checkbox at all: the toggle is new in commit 29cd22159e (task-32535, PR #2679), source lines 425-426. String 2 is INFERRED as consequential -- the pane grew with that toggle. String 3 is PROVEN from the capture; task-32549 gave the disabled controls their reasons but not this recomputation.

Related open task: 32587 owns the behaviour half (the toggle choice does not survive a quit because it has nowhere durable to live). This task is the copy and the state, not the persistence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Obsidian toggle's help text is a sentence a reader can parse, and it states the same scope the behaviour has
- [ ] #2 The scroll hint renders only when the scroll container actually overflows
- [ ] #3 The Resolution history line is recomputed on activation, and wherever the control is disabled it carries its reason on screen rather than a bare glyph
- [ ] #4 Every user-visible string added by wave 4 to these panes is re-read against the shipped render, and the guide's matching paragraphs agree
<!-- AC:END -->
