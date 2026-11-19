---
id: TASK-617
title: >-
  Bring Roleplay personas to parity with tldw_server's personas module
status: Done
assignee: []
created_date: '2026-07-24 14:45'
labels:
  - roleplay
  - personas
  - parity
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
chatbook's persona concept is currently wrong: it treats a persona as "who you are" (the user's own identity) and the preview renders the user as the literal "User". Per the app author, personas in chatbook are supposed to be the SAME concept as tldw_server's personas module — a persona is a CHARACTER-side assistant identity, not the user.

In tldw_server (`tldw_Server_API/app/core/Persona/`, `app/api/v1/endpoints/persona.py`, `app/api/v1/schemas/persona.py`, `app/core/Chat/chat_service.py:450`/`:474`), a persona profile is derived from a character (`origin_character_id`/`origin_character_name`) and projects into a chat as the assistant (`assistant_kind=persona`, `assistant_id`, `persona_memory_mode`), with exemplar prompt assembly (`app/core/Persona/exemplar_prompt_assembly.py`) feeding the system prompt. Correct macro semantics: `{{user}}` = the human app user; `{{persona}}`/`{{char}}`/`{{character}}` = the character/persona. chatbook's `replace_placeholders` (`Character_Chat/Character_Chat_Lib.py:404`) already maps `{{user}}`→user and `{{char}}`→char but lacks `{{character}}`/`{{persona}}` aliases and is fed the wrong "who you are" model upstream.

The goal is parity with tldw_server's personas module. The server module is large (~13k lines) and includes browser/voice/visual layers (persona buddy avatars, visual packs, renderers, Persona Live voice/websocket/wake) that have no terminal analog; scope for chatbook should center on the portable behavioral core (persona = character-derived assistant, chat projection, exemplars, memory mode, macros), with the visual/live layers out of scope unless separately decided. This is large enough to need decomposition into sub-tasks during planning.

Supersedes the archived TASK-442, which had the concept inverted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A persona in chatbook is modeled as a character-side assistant identity that matches tldw_server's persona profile concept (derived from / linked to a character), not as the user's own identity
- [x] #2 A persona projects into a chat as the assistant (system prompt / exemplar assembly), mirroring tldw_server's persona-backed chat behavior
- [x] #3 Macro substitution matches tldw_server: `{{user}}` resolves to the human user; `{{persona}}`/`{{char}}`/`{{character}}` resolve to the character/persona
- [x] #4 A persona can be selected/active and persists across sessions, consistent with tldw_server's persona selection semantics
- [x] #5 The "persona = who you are" framing is removed from the Roleplay/Settings surfaces and copy
<!-- AC:END -->


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: DONE** — evidence verified against origin/dev (a3c47c6c72).

Persona parity landed: Character_Chat/local_character_persona_service.py models character-side assistant personas (assistant_kind in {'character','persona'}, persona_memory_mode, exemplar assembly — commits 1921dea238 'Add local persona profile parity' and cd8bf5c004 'Add local persona exemplar parity'); macro semantics fixed via task-442 ('{{character}}' and '{{persona}}' are character-side aliases, Character_Chat_Lib.py:553); persona selection persists (Tests/Character_Chat/test_character_persona_scope_service.py); no 'persona = who you are' copy remains in tldw_chatbook.
