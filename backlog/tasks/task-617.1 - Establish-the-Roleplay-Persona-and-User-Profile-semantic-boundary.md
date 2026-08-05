---
id: TASK-617.1
title: Establish the Roleplay Persona and User Profile semantic boundary
status: Done
assignee:
  - '@codex'
created_date: '2026-07-28 18:47'
updated_date: '2026-07-29 13:48'
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
- [x] #1 The Roleplay workbench exposes Personas with mode id personas and assistant-profile copy; Persona-facing runtime APIs widgets messages events and tests use Persona names with no UserProfile compatibility aliases.
- [x] #2 Persona server request DTOs match tldw_server exactly; local-only description and freeform personality traits remain supported by local persistence but are never sent to the server.
- [x] #3 Persona PATCH requests serialize only fields explicitly supplied and preserve explicit null values.
- [x] #4 Existing local Persona JSON records and unknown or legacy fields remain readable and are preserved across ordinary updates; the on-disk filename remains unchanged.
- [x] #5 Persona selection no longer exposes Set as my name Chatting as an active-human marker or any other Persona-as-human control.
- [x] #6 Character and Persona preview or handoff paths render the human placeholder as the neutral literal User and never resolve a Persona into the user slot.
- [x] #7 The obsolete active-user-profile implementation is removed while character_defaults.active_user_profile remains inert and is neither read written cleared nor repaired.
- [x] #8 Console and chat identity surfaces remove the Persona-as-human chip row and As label; generic character and Persona assistant summaries use Assistant Character and Persona labels respectively.
- [x] #9 Legacy persona_label and user_profile_label session keys are accepted only as ignored input and are not emitted in newly serialized Console settings; no stored settings or transcript migration is performed.
- [x] #10 Focused schema service local-persistence Textual preview handoff and Console regression tests pass, and no persona chat-authority TTS assignment speech-snapshot or managed audio.cpp behavior is added.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Delivered the approved Slice 3A.1 semantic boundary: Personas are assistant-side
profiles throughout the Roleplay workbench and runtime surfaces, while genuine
authenticated-account and RAG User Profile APIs remain intact. Preview and
handoff paths use the neutral literal `User`; Persona-as-human resolution,
actions, labels, markers, and compatibility aliases were removed.

Server Persona DTOs now mirror the server contract exactly, with strict
source-aware local mutation DTOs retaining local description and freeform
personality fields. PATCH serialization distinguishes omission from explicit
null, and ordinary local updates preserve unknown or legacy JSON fields and the
existing on-disk filename. All callable Persona surfaces use hard Persona
terminology.

Console and Chat now render one shared assistant identity as Assistant,
Character, or Persona. Legacy `persona_label` and `user_profile_label` values
are ignored only at restore and are not emitted; no settings or transcript
migration occurs. Assistant chip text and its separate Textual tooltip render
user-controlled labels literally, while existing retrieval-scope behavior is
unchanged.

Final validation on rebased HEAD `96947071` over `origin/dev` `a2947be90`:

- The expanded focused gate passed 1217 tests with 1 documented tooltip
  follow-up skip and 7 known failures deselected.
- The undeselected replay produced 7 failures and 1217 passes; all seven exact
  nodes and failure shapes reproduced on the current base.
- TTS/profile/application regressions passed 464 tests; Console dictation
  overlap passed 142; the post-review Console gate passed 506 with the 6
  current-base Console failures deselected; tooltip/scope coverage passed 40.
- Task-scoped formatting, compile, diff, terminology, legacy-key, allowlist,
  and deferred-scope guards are clean. The one changed-file Ruff E702 and the
  exact-six-file mypy baseline of 81 errors across 4 files match the current
  base and introduce no task-only diagnostic.
- The pre-final-rebase broad recovery run was still globally red, but every
  terminal red case was reproduced or classified as base, order, or
  environment behavior; it found no branch-only red case and is not a claim
  that the repository-wide suite is green.

ADR-037 remains the governing decision; no new ADR, migration, store,
dependency, or runtime boundary was required. No managed audio.cpp
launch/supervision, Persona chat authority, speech snapshot, persistent TTS
assignment, character authority, or conversation migration entered this
slice.

There was no material implementation-plan deviation. Final independent code
and scope review was APPROVED with no unresolved finding after adding the
narrow literal-tooltip regression and hardening fix.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 All acceptance criteria are checked and concise Implementation Notes record the delivered behavior and any deviations.
- [x] #2 Focused schema service persistence Textual preview handoff Console and global-TTS regression tests pass; inherited failures are identified separately.
- [x] #3 Task-scoped format lint typing compile and git diff checks pass or exact unchanged repository baselines are documented.
- [x] #4 ADR-037 the approved Slice 3A design and relevant user or developer documentation remain current.
- [x] #5 Independent code and scope review has no unresolved Critical Important or Minor finding.
- [x] #6 No character-authority conversation-migration speech-snapshot TTS-assignment Persona-runtime or managed-audio.cpp work enters this task.
<!-- DOD:END -->
