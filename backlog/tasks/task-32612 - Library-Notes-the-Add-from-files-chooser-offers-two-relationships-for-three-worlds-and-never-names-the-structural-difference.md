---
id: TASK-32612
title: >-
  Library Notes: the Add from files chooser offers two relationships for three
  worlds and never names the structural difference
status: To Do
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
- [ ] #1 The chooser carries one heading, not three
- [ ] #2 Each option names its outcome, including that one keeps your folder structure and the other collects the notes in one managed folder
- [ ] #3 The chooser points at Folder files as the third answer to the same question
- [ ] #4 The documented back control is rendered, or the guide stops promising it
<!-- AC:END -->
