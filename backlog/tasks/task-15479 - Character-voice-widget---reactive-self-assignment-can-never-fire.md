---
id: TASK-15479
title: Character voice widget: reactive self-assignment can never fire
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - bug
  - tts
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the latency audit: `Widgets/TTS/character_voice_widget.py:455/:465` uses `self.characters = self.characters` to trigger a refresh after add/remove — but assigning the same object compares equal and the reactive has no `always_update`, so the watcher never runs and the table never refreshes.

Fix direction: `mutate_reactive`, assign a new list, or `always_update=True`; add the missing refresh test; grep for other instances of the dead pattern. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Adding and removing a character updates the table (test)
- [ ] #2 No other self-assignment reactive triggers remain (grep evidence in notes)
<!-- AC:END -->
