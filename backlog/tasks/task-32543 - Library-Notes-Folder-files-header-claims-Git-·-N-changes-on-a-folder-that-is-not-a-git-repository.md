---
id: TASK-32543
title: >-
  Library Notes: Folder files header claims "Git · N change(s)" on a folder that
  is not a git repository
status: To Do
assignee: []
created_date: '2026-09-13 06:46'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), both assessors saw the string, persona solo operator, Folder files workflow. A P2 #8.

**What happened.** Fresh vault fixture with no `.git` (`git rev-parse` fails). After one edit the header reads "Folder files · Folder: vault · Git · 1 change" at 100x30 (A 57) and at 235x52 (B 40, 50). The solo operator reads "Git" as "this folder is version-controlled". Captures: A 57; B 40, 50.

**Cause.** INFERRED: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py:230-232` builds `f"Git · {git_changes} {change_word}"` from the session change count with no repository gate in the renderer; the git failure/uncertain/running branches above it only fire when a git probe ran. Docs contradicted: file-notes.md's header is "Linked · Local folder: <folder>" — the suffix is undocumented and false.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The "Git · …" suffix renders only after a successful repository check
- [ ] #2 A non-repository folder with session changes reads "N session change(s)" (or nothing), never "Git"
- [ ] #3 A test renders the header for a non-repository folder with one change and asserts "Git" is absent; file-notes.md documents the suffix
<!-- AC:END -->
