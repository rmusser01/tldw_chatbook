---
id: TASK-34868
title: >-
  Persona per-send system-prompt re-expansion does not reach the production send
  path
status: Done
assignee: []
created_date: '2026-09-12 19:23'
updated_date: '2026-09-12 21:30'
labels:
  - console
  - persona
  - roleplay
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to task-32481 (persona session identity) found during its live smoke test; linked to task-32483 (fixed on the same branch/PR). Commit 7ec9bbda60 added per-send template re-expansion to ConsoleChatController._provider_selection_for_session but the live send path never calls it: ChatScreen wires turn_context_provider=_build_console_turn_execution_context whose selection comes from _build_console_provider_selection_uncached which copies settings.system_prompt verbatim (chat_screen.py ~5112). Production correctness today rests entirely on eager projection rematerialization (identity setters and the global-name refresh) which the smoke test verified live in both directions; there is no known production-reachable stale path but any future identity mutation that bypasses the setters would silently send a stale prompt and the tip commit's branch only exercises bare-controller/unit-test paths. Fix: route the production selection through the same per-send identity resolution (single shared resolver for persona and character parity) or make an explicit recorded decision to rely on rematerialization and drop the dead branch. To be fixed on branch feat/custom-endpoint-registry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Production send path resolves the per-send identity expansion for named persona sessions with a trusted template;Character sessions get the same per-send resolution (parity);Template-less or name-unresolved persona sessions keep the settings-derived prompt;Existing task-32481 unit tests keep passing;New tests cover the production-path resolution including per-chat override and cleared override;Live headless verification: override set mid-session then send then wire carries the override expansion without relying on the settings projection
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extract the controller's per-send identity expansion into a shared pure resolver in console_roleplay_identity.py (persona/character parity)\n2. Route the production path (ChatScreen._build_console_provider_selection_uncached) through it for named persona/character sessions with a trusted template; everything else keeps the settings prompt\n3. Refactor the controller's _resolved_system_prompt tail onto the same resolver (behavior-preserving)\n4. Tests: production-path resolution for override, cleared override, template-less, name-unresolved, character parity; mutation-check by reverting the screen fix\n5. Live headless verification: override set mid-session direct on the store object, then send, then wire carries the override expansion
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Chose the shared-resolver route over dropping the branch: new pure helper resolve_send_system_prompt in console_roleplay_identity.py (blank name/template -> fallback; else expand_character_template against effective_user_display_name(override, global_default)) gives persona and character sessions one per-send expansion. ChatScreen._build_console_provider_selection_uncached — the production turn-context path — now applies it via replace() when the target session is assistant_kind persona/character, reading assistant_name/persona_system_template or character_name/character_system_template plus user_display_name_override, with the global display-name read guarded (except -> 'User'); anything else keeps the settings-derived prompt. The controller's _resolved_system_prompt tail was refactored onto the same resolver (behavior-preserving; provider-continuation suite 41 passed). Tests: 5 new cases in Tests/Chat/test_console_turn_execution_context.py driving a fake screen through the production builder — per-chat override ('Guide Wren as Archivist.'), cleared override falls back to the global name, template-less persona keeps settings prompt, name-unresolved persona keeps settings prompt, character parity; 3 of the file's screen-selection tests fail when the screen fix is reverted (mutation-checked). Full file 19/19. Live headless verification: persona session created through the real chat-now handoff, override 'SecondVerifier' mutated directly on the frozen store session (settings projection verified stale at 'Guide SmokeVerifier with care.'), and the wire system prompt started with 'You are Archivist, keeper of the archive. Guide SecondVerifier with care.' Probe file removed after the run. ADR required: no — completes task-32481's design inside ADR-149 (persona session identity), no new boundary. Files: tldw_chatbook/Chat/console_roleplay_identity.py, tldw_chatbook/Chat/console_chat_controller.py, tldw_chatbook/UI/Screens/chat_screen.py, Tests/Chat/test_console_turn_execution_context.py.
<!-- SECTION:NOTES:END -->
