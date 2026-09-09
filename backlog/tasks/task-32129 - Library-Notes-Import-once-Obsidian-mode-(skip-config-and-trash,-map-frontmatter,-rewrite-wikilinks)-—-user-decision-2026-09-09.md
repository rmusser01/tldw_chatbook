---
id: TASK-32129
title: >-
  Library Notes Import once: Obsidian mode (skip config and trash, map frontmatter, rewrite wikilinks) — user decision 2026-09-09
status: To Do
assignee: []
created_date: '2026-09-08 21:39'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - import
  - obsidian
  - feature
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The user chose 'Adopt (Obsidian mode)' over a format-agnostic copy. Today (PROVEN): discovery walks `.obsidian/` and `.trash/` (no ignore rule in note_import_discovery.py); `.json` is a supported type so five config files land in 'Failed (5) · could not be imported safely'; `_parse_text` never parses YAML frontmatter (title is the first '# ' heading or the file stem) so `tags:` produce zero keywords and the block stays in the body; `Templates/Daily.md` becomes a note titled `{{date:YYYY-MM-DD}}`; `[[wikilinks]]` and `[[link|alias]]` stay literal even though every target was imported in the same batch. Import once remains byte-safe on disk (verified by hash). Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When `.obsidian/` exists at the selected root, the review shows a default-on 'Obsidian vault' toggle and states what it will do
- [ ] #2 With the toggle on, `.obsidian/`, `.trash/` and `Templates/` are listed as skipped with a plain reason, never as Failed
- [ ] #3 YAML frontmatter is parsed: title to the note title, tags to keywords, aliases kept in Info, and the block is removed from the body
- [ ] #4 `[[wikilink]]` and `[[link|alias]]` targets created in the same batch become note links; unresolved links stay as text
- [ ] #5 Template placeholders such as `{{date}}` never become a note title
- [ ] #6 The review shows the resulting titles and keywords before import; the vault on disk stays byte-identical
- [ ] #7 Covered by tests with a fixture vault carrying each of the above
<!-- AC:END -->
