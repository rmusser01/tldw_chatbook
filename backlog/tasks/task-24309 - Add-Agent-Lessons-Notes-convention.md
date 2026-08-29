---
id: TASK-24309
title: Add the Agent Lessons Notes convention
status: To Do
assignee: []
created_date: '2026-08-29'
labels:
  - notes
  - agents
  - knowledge
priority: high
dependencies:
  - TASK-24307
  - TASK-24308
documentation:
  - Docs/superpowers/specs/2026-08-29-agent-lessons-notes-organization-sync-design.md
  - Docs/superpowers/plans/2026-08-29-agent-lessons-convention.md
  - backlog/decisions/102-portable-notes-organization-and-agent-lessons.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a user-owned `Agent_Lessons` Notes convention that guides permitted agents to find and record verified reusable solutions, including failed attempts and why they failed, without turning note content into trusted instructions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Synchronized and permanently local-only profiles expose one conventional root `Agent_Lessons` folder after their applicable readiness boundary without recreating a folder the user renamed or deleted
- [ ] #2 Agent Lessons discovery is governed by the spelling-exact `agent-lesson` keyword and remains correct after folder rename, movement, deletion, marker removal, and case-fold collision review
- [ ] #3 Primary agents and ordinary subagents receive search/save guidance only for Notes capabilities they actually hold, and save-only agents are not told to bypass search-first behavior
- [ ] #4 Newly generated lessons use one-note-per-lesson structure with applicability, symptoms, root cause, verified solution, failed attempts and why, verification evidence, caveats, and related public note IDs
- [ ] #5 Retrieved lessons remain labeled untrusted tool-result data and cannot grant permission, authorize commands, expand scope, or enter trusted runtime or project instructions
- [ ] #6 High-confidence credential formats are rejected without logging content, while long hashes, error IDs, and clearly fake examples remain saveable
- [ ] #7 Two-device seed races converge only for untouched empty coordinator seeds; edited, acknowledged, differently spelled, or otherwise used candidates require explicit review
- [ ] #8 An end-to-end test proves Agent A can record a verified resolution with failed attempts and Agent B can discover and safely apply it through Notes search
<!-- AC:END -->
