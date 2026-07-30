---
id: TASK-1530
title: 'Character chat sends {{user}}/{{char}} macros raw to the provider'
status: Done
assignee: []
created_date: '2026-07-30 15:30'
labels: [bug, roleplay, console, P2]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verified live against origin/dev @ c329def0d (2026-07-30, stub provider
capturing payloads): Roleplay → Start Chat → Console with a real V2+V3 card.
The card's system prompt (description + scenario + post-history) IS correctly
assembled and sent every turn, but 12 `{{user}}` and 7 `{{char}}` macro
occurrences reach the model verbatim — no substitution. Same raw macros were
observed on the Roleplay preview path.

`replace_placeholders` exists (Character_Chat_Lib) and dev's preview controller
uses it for greeting DISPLAY, but no outbound request path applies it to the
system prompt or history.

Terminology guard (author-confirmed): `{{user}}` = the human app user's
name/persona; `{{char}}`/`{{character}}` = the character. Cards are written
expecting SillyTavern-style substitution; sending them raw weakens persona
adherence and leaks template syntax to the model.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

1. Extract the Start Chat card->prompt build into module-level `_character_session_prompt_seed` (chat_screen.py) and apply `replace_placeholders` to the joined system prompt (greeting already substituted); "User" as user name matches existing Personas display substitution.
2. Extract preview prompt build into module-level `build_preview_system_prompt` (personas_preview_controller.py) with the same substitution.
3. TDD: new Tests/UI/test_character_session_prompt_seed.py + builder tests in Tests/UI/test_personas_preview.py; watch macro assertions fail before wiring substitution.
4. Live stub-provider payload re-verification (no raw macros outbound).

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Outbound character-chat requests (Console session and Roleplay preview) contain no unsubstituted {{user}}/{{char}}/{{character}} macros.
- [x] #2 {{user}} resolves to the user's name/persona and {{char}} to the character name, matching the greeting-display substitution.
- [x] #3 Regression tests assert substitution at the prompt-assembly seams that feed provider payloads (AC reworded: substitution happens at session/preview prompt build; payload delivery of those strings is covered by existing system-prompt payload tests and was live-verified end to end).
<!-- AC:END -->

## Implementation Notes

Extracted the two prompt builds into pure, tested helpers and applied
`replace_placeholders` (char name + "User", matching the existing
greeting-display substitution) to the joined card fields:

- `_character_session_prompt_seed` (chat_screen.py, module level) now feeds
  Start Chat's Console session settings with macros resolved; behaviour was
  extracted as a pure refactor first so the macro assertions failed RED
  before the substitution landed.
- `build_preview_system_prompt` (personas_preview_controller.py, module
  level) does the same for the Roleplay preview; `system_prompt()` delegates
  to it (single consumer: the `_run_reply` messages build).

Live-verified against the stub provider: Console turn 1/turn 2 and a preview
send all left ZERO raw macros in captured payloads (previously 12 {{user}} +
7 {{char}} per send). Existing sessions created before the fix keep their
stored raw-macro system prompts; re-running Start Chat reseeds cleanly.

Tests: Tests/UI/test_character_session_prompt_seed.py (new),
build_preview_system_prompt cases in Tests/UI/test_personas_preview.py.
Full runs: test_console_chat_controller (139 passed), adjacent suites
(319 passed), test_personas_preview (39 passed).
