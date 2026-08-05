---
id: TASK-2451
title: Make the default assistant an editable sample persona
status: To Do
assignee: []
created_date: '2026-08-05 04:48'
updated_date: '2026-08-05 04:49'
labels: []
dependencies:
  - TASK-2450
  - TASK-951
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The voice-profiles expansion design (2026-08-04) deferred an idea raised while scoping per-character voice settings: today's seeded 'Default Assistant' is a fixed character with no persona-side identity of its own. Owner ruling 1 on that spec explicitly separated this from the voice work ('a separate follow-up task, not this feature') and asked for it to be filed noting ADR-037's constraint. This task is that filing: give the default assistant an editable Persona presence so a new user has something to customize on day one, without letting that Persona acquire the character's own live identity or its TTS assignment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A new user has an editable Persona (name, description, and the other Persona-domain fields) available out of the box, seeded from or alongside the default assistant character, without any manual setup
- [ ] #2 Editing that Persona never mutates the seeded character card, and the Persona is never treated as the character's live identity (ADR-037: a Persona's origin-character provenance is not a live character identity)
- [ ] #3 The sample Persona does not inherit, read, or expose the default assistant character's TTS assignment; Persona-side voice identity remains out of scope here per ADR-037's characters-remain-the-voice-bearing-entity boundary
- [ ] #4 Existing installs that already have Persona records are unaffected; the sample Persona is additive, not a migration that alters or removes anything a user already created
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Filed per owner ruling 1 in Docs/superpowers/specs/2026-08-04-voice-profiles-expansion-design.md section 4.4: 'the default assistant becomes an editable sample persona' idea, explicitly scoped OUT of the voice-profiles slice work and left as its own follow-up. ADR-037 (backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md) governs the boundary this task must respect: a Persona may retain an origin-character snapshot but that provenance is not a live character identity and a Persona does not inherit a character TTS assignment. Full Persona runtime parity (assignment UI, speech runtime, authority/provenance work) is tracked separately as TASK-617 and its five subtasks -- this task is scoped to the sample-persona UX only, not to closing 617.
<!-- SECTION:NOTES:END -->
