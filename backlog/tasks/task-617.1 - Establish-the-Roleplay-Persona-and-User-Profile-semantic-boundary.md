---
id: TASK-617.1
title: Establish the Roleplay Persona and User Profile semantic boundary
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-28 18:47'
updated_date: '2026-07-28 19:18'
labels:
  - roleplay
  - personas
  - identity
dependencies: []
references:
  - TASK-551
  - >-
    backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md
documentation:
  - >-
    Docs/superpowers/specs/2026-07-28-tts-character-identity-persona-separation-design.md
parent_task_id: TASK-617
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct the first bounded slice of Roleplay persona parity by treating Personas as assistant-side profiles and reserving User Profiles for authenticated human accounts. Remove the obsolete Persona-as-human behavior and naming without introducing persona chat authority, speech snapshots, TTS assignment, or managed audio.cpp behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Roleplay workbench exposes Personas with mode id personas and assistant-profile copy; Persona-facing runtime APIs widgets messages events and tests use Persona names with no UserProfile compatibility aliases.
- [ ] #2 Persona server request DTOs match tldw_server exactly; local-only description and freeform personality traits remain supported by local persistence but are never sent to the server.
- [ ] #3 Persona PATCH requests serialize only fields explicitly supplied and preserve explicit null values.
- [ ] #4 Existing local Persona JSON records and unknown or legacy fields remain readable and are preserved across ordinary updates; the on-disk filename remains unchanged.
- [ ] #5 Persona selection no longer exposes Set as my name Chatting as an active-human marker or any other Persona-as-human control.
- [ ] #6 Character and Persona preview or handoff paths render the human placeholder as the neutral literal User and never resolve a Persona into the user slot.
- [ ] #7 The obsolete active-user-profile implementation is removed while character_defaults.active_user_profile remains inert and is neither read written cleared nor repaired.
- [ ] #8 Console and chat identity surfaces remove the Persona-as-human chip row and As label; generic character and Persona assistant summaries use Assistant Character and Persona labels respectively.
- [ ] #9 Legacy persona_label and user_profile_label session keys are accepted only as ignored input and are not emitted in newly serialized Console settings; no stored settings or transcript migration is performed.
- [ ] #10 Focused schema service local-persistence Textual preview handoff and Console regression tests pass, and no persona chat-authority TTS assignment speech-snapshot or managed audio.cpp behavior is added.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add strict local Persona mutation DTOs without removing live APIs, with exact-field and extra-rejection tests.
2. Remove Persona-as-human preview handoff inspector and active-user-profile behavior while preserving the legacy config value byte-for-byte and keeping {{user}} as User.
3. Rename the Personas workbench mode entity widgets events editor and copy, and make local-only editor fields source-aware.
4. In one coordinated green change, narrow server DTOs, preserve explicit-null PATCH semantics, rename every service/pager/caller to Persona terminology, and prove lossless local JSON reads and merges.
5. Remove Persona-as-human Console settings/chips/rows and render one assistant identity label as Assistant, Character, or Persona from already-available presentation values.
6. Run focused, baseline-replay, TTS/profile/application, broad, import, static, terminology, and executable scope gates; request independent code review before recording evidence and marking Done.

Full plan: Docs/superpowers/plans/2026-07-28-persona-user-profile-semantic-boundary.md
ADR required: yes
ADR path: backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md
Reason: ADR-037 already governs the Persona/User Profile boundary and Slice 3A.1 compatibility rules; no new ADR, schema migration, store, dependency, TTS boundary, or process-runtime decision is introduced.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 All acceptance criteria are checked and concise Implementation Notes record the delivered behavior and any deviations.
- [ ] #2 Focused schema service persistence Textual preview handoff Console and global-TTS regression tests pass; inherited failures are identified separately.
- [ ] #3 Task-scoped format lint typing compile and git diff checks pass or exact unchanged repository baselines are documented.
- [ ] #4 ADR-037 the approved Slice 3A design and relevant user or developer documentation remain current.
- [ ] #5 Independent code and scope review has no unresolved Critical Important or Minor finding.
- [ ] #6 No character-authority conversation-migration speech-snapshot TTS-assignment Persona-runtime or managed-audio.cpp work enters this task.
<!-- DOD:END -->
