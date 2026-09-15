---
id: TASK-32612
title: >-
  Library Notes: the Add from files chooser offers two relationships for three
  worlds and never names the structural difference
status: Done
assignee: []
created_date: '2026-09-15 06:40'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A P2 and A P1 (consolidated), persona Jordan, Obsidian workflow. This is the exact decision point for 'I have a vault'.

What happened (A cap 17). The chooser stacks three near-identical header lines -- 'Add files to Library notes.' / 'Choose how files should relate to Library notes.' / 'Choose the relationship before selecting a file or folder.' -- over two options, across a 190x40 empty stage. It never mentions Folder files, the option that actually matches 'edit my vault where it is', even though the strip above it offers exactly that third world. No back control is rendered on the canvas either; only Escape works.

And the consequence the two options do not state: Import once reproduced the vault's tree (vault/Archive, vault/Daily, vault/Projects -- A cap 23) while lasting sync put all 54 notes FLAT under one PowerVault folder (A cap 51). That is the single largest consequence of the choice made here, and neither sentence mentions it.

Cause PROVEN by capture. The structural half is already owned and proven blocked by open task 32586 (create_folder refuses a manual child of a subtree holding a managed placement, so a synced note cannot keep the vault's tree until lasting sync gets its own folder-creation path). This task is the decision point's copy, not the hierarchy work.

Docs contradicted: notes.md says the bar below the chooser 'holds only ‹ Notes'. Nothing is rendered there -- a claim introduced with wave 4's back-cue unification (task-32553, PR #2685).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The chooser carries one heading, not three
- [x] #2 Each option names its outcome, including that one keeps your folder structure and the other collects the notes in one managed folder
- [x] #3 The chooser points at Folder files as the third answer to the same question
- [x] #4 The documented back control is rendered, or the guide stops promising it
<!-- AC:END -->

## Implementation Plan

1. Re-derive the three header lines and the missing back control live on the
   canvas under production CSS.
2. Collapse the stack to one heading; give the status line a state rather than
   a restated question.
3. Extend both option sentences with the structural consequence, and add a
   pointer at Folder files.
4. Settle AC#4 on the real Library screen, not the canvas in isolation.

## Implementation Notes

**What changed.** The chooser asks its question once. The body's third header
sentence ("Choose the relationship before selecting a file or folder.") is
gone and the choose-phase `status_line` says where the chooser stands
("Nothing chosen yet.") instead of restating the heading. Both option
sentences now carry the consequence neither named -- Import once "reproducing
your folder structure as Library folders", Keep a folder synced "every note is
collected in one managed Library folder rather than your folder structure" --
and a third Static (`#notes-add-folder-files-pointer`) names Folder files as
the answer that is on neither button, pointing at the source strip above the
canvas rather than duplicating the mode.

**AC#4: the finding did not reproduce.** `#notes-sync-back` IS composed for
the choose phase and IS painted -- measured under both production stylesheets
at 190x40, and now pinned on the REAL Library screen
(`test_the_chooser_renders_its_documented_back_control`) rather than on the
canvas alone, because the canvas-only pin is exactly what would have missed a
screen-level clipping. notes.md's promise is therefore kept, not withdrawn,
and the guide now says so explicitly.

**The fence.** task-32627 idea 2 (one screen, three outcomes) was NOT built;
this is the existing chooser naming the difference. What I think would still
confuse a first-timer, as evidence for that deferred idea:

1. *The third world is a sentence, not a control.* Import once and Keep a
   folder synced are buttons; Folder files is prose telling you to go
   somewhere else. A reader who has already decided "edit my vault where it
   is" has to leave this screen to act on the only sentence that matches them.
2. *"Import once" still reads like a file copy.* The clause I added says the
   folder STRUCTURE is reproduced, which invites "so are my files copied
   somewhere?" -- the honest answer (the CONTENT is copied into the Library
   database; the files are read and left alone) is a sentence the chooser
   still does not have room for.
3. *The two structure clauses are asymmetric in a way the reader must invert.*
   One says what you keep, the other what you lose. A three-column table with
   one row per consequence would be read; three prose sentences are skimmed.
4. *Nothing says which is reversible.* "Import once ... ends" and "lasting
   connection" imply it, but a first-timer's real question at an
   irreversible-feeling choice is "what happens if I pick wrong", and neither
   sentence answers it.

**Not relitigated.** Import once still adopts rather than copies the vault;
which three worlds exist is unchanged.

**Dependency to keep in step.** The "collected in one managed Library folder"
clause is true *today* because lasting sync is flat (task-32586 owns the
structural half and is blocked). When 32586 lands, that sentence and its pin
(`test_each_relationship_names_what_happens_to_the_folder_structure`) change
with it; a comment at the compose site says so.

**Files.** `Widgets/Library/library_notes_add_from_files_canvas.py`,
`Library/library_notes_lasting_sync_state.py`,
`Tests/UI/test_library_notes_w5_import_preview.py`,
`Tests/UI/test_library_notes_wave_import_ux.py` (task-32256's completeness pin
re-pointed at the new sentences -- the assertion is unchanged in substance),
`Docs/User_Guide/library/notes.md`.

**The task-32586 coupling is now recorded where a 32586 reader will look**
(review round 1): it is **AC#5 on task-32586** itself, naming this sentence,
its compose site and the pin that guards it. It was previously flagged only
in a code comment and in these notes, so discovery was red test -> read test
-> read source comment.
