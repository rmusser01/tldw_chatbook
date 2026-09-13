---
id: TASK-32559
title: 'Library Notes: critique #3 improvement ideas (decide before wave 4)'
status: To Do
assignee: []
created_date: '2026-09-13 06:48'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor A §13, ten improvement opportunities beyond the filed fixes. One task so the decision is taken once; adopt / defer / reject per idea, then file the adopted ones. Sizes S/M/L are A's.

1. **One review renderer for both import paths** — path · what happens · where, Obsidian toggle, grouped outcomes — reused by Keep a folder synced. Alex / solo operator. **M** (the sync-review task filed alongside is the defect; this is the structural answer)
2. **Sync activation as a What / Where / Impact / Recovery review**, copied from Session Git: "60 notes will be created in Vault; nothing on disk changes until you edit a note here; Undo available 30 days". Solo operator. **M**
3. **Readiness-independent Console hand-off**: stage the note, land in Console, let Console's own empty state say "add a model". Researcher/student. **S** (one of the two remedies in the Use-in-Console task)
4. **Local time everywhere** and a single status authority in the editor. Jordan. **S** (the duplicated meta line is riders 32513 / 32514; the clock is its own task)
5. **Preview as a mode, not a widget**: Tab leaves it; `p` / `i` / `e` switch Preview / Info / Edit from anywhere in the editor. Sam, Alex. **S**
6. **Obsidian-aware editor rendering**: fold the `(note://…)` suffix in Edit and show `[[Daily/2026-09-07|yesterday's meeting]]` only — the task-32263 decision keeps the stored form, this is display only; render `> [!warning]` callouts in Preview. Jordan. **M**
7. **Source-truth badge on every note header**: "Library note" / "Synced · Vault (writes to disk on save)" / "File · vault/Daily/…" — the three worlds legible from inside a note. Solo operator. **S**
8. **Capture-from-Console entry on the Notes list** ("New from last Console reply") so the return leg task-32146 shipped in Console (#2659) is discoverable from Notes. Researcher/student. **M**
9. **Structured-source fingerprinting** so a second import of notes.csv / meta.yaml is an "Unchanged repeat", plus an opt-in "Import as notes?" for .csv/.yaml/.txt inside a vault. Riley, Jordan. **M** (the fingerprinting half is the CSV re-import task; the opt-in is the idea)
10. **Toolbar diet**: New · Find · Add from files… · More ▾ (Sort, Select, Export, folders, sync) — four visible choices, the rest disclosed. Everyone. **S**

A's provocative questions, for the same decision: why two importers for one folder that disagree about what a vault is; whether Notes is really "handing off" if Use in Console fails closed without a model; who reads "durable receipt recorded"; whether the UTC clock is a bug or a policy; why sync activation does not get Session Git's review treatment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the ten ideas has a recorded decision: adopt (with a filed task id), defer (with the condition), or reject (with the reason)
- [ ] #2 The adopted ideas are sequenced relative to the critique-3 defect tasks before wave 4 starts
<!-- AC:END -->
