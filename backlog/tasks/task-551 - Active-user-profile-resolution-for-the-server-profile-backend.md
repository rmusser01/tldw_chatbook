---
id: TASK-551
title: Active user profile resolution for the server profile backend
status: Done
assignee: []
created_date: '2026-07-24 20:30'
updated_date: '2026-07-25 06:35'
labels:
  - roleplay
  - enhancement
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from Qodo review of PR #849 (task-442). `resolve_active_user_profile_name`
validates the config pointer against the SYNC local file-backed profile service
(`app.local_character_persona_service`) — by design (the task-442 spec scoped
resolution to "config + one file-backed profile read", and validation is what
makes a dangling pointer resolve to None instead of substituting a stale name).
When the user-profile backend mode is "server" and the active profile exists
only server-side, the pointer fails local validation and `{{user}}` silently
falls back to its site default. Making it lenient (trust the pointer when
validation can't see it) would break the deliberate dangling→None semantics,
so this needs the real fix: an ASYNC resolver that validates against
`character_persona_scope_service.list_user_profiles(mode=...)`, used at the
async substitution sites (`chat_screen._start_character_console_session`,
`chat_events` display path, the preview controller's async flows).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 #1 With the server profile backend active, a server-side active profile's name substitutes into {{user}} at the three task-442 send surfaces
- [x] #2 #2 A dangling pointer (profile deleted on the resolving backend) still resolves to no-active (never a stale name, never an error)
- [x] #3 #3 With no pointer set, behavior is byte-identical (the task-442 twins keep passing)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write three failing server-mode site tests (one per substitution site), siblings of the existing task-442 twins in the same files, arranged with a server-backend scope service + server mode instead of the local persona service; confirm all three FAIL against the unswapped sites while the task-442 twins stay green.
2. Swap chat_screen._start_character_console_session's resolver call to the async resolver, mode sourced from resolve_runtime_backend_mode(app_instance).
3. Swap chat_events.display_conversation_in_chat_tab_ui's resolver call the same way, mode sourced from resolve_runtime_backend_mode(app), preserving the trailing USERS_NAME fallback byte-exact.
4. Swap the preview controller's site: add an async _resolve_user_name_async() that reads the workbench's own mode (persona_handler.current_mode(), guarded, default local), give _load_greetings a sentinel-defaulted user_name kwarg so the sync _active_user_name() fallback stays intact for any other caller, and thread the resolved name through reset_for_character/handle_character_loaded.
5. Re-run the three new tests plus the resolver unit tests and the task-442 twins; confirm all green.
6. Update this task file and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Swapped the three task-442 {{user}} substitution sites onto the task-2 async resolver (`resolve_active_user_profile_name_async` / `resolve_runtime_backend_mode`, `Character_Chat/active_user_profile.py`), so a server-backend active user profile now substitutes correctly instead of silently falling back (Qodo #849-4).

Approach:
- chat_screen._start_character_console_session and chat_events.display_conversation_in_chat_tab_ui both swap their sync resolve_active_user_profile_name(local_service) call for `await resolve_active_user_profile_name_async(character_persona_scope_service, mode=resolve_runtime_backend_mode(app), local_service=local_character_persona_service)`; both use the app-authoritative runtime source. chat_events preserves the trailing `or app.app_config.get("USERS_NAME", "User")` fallback byte-exact.
- personas_preview_controller gets a new async `_resolve_user_name_async()` that reads the WORKBENCH's own mode (`persona_handler.current_mode()`, guarded, default "local") rather than the app-level source -- the preview validates against the backend it's actually browsing. `_load_greetings` gained a sentinel-defaulted `user_name` kwarg (`_UNSET` module constant) so it still falls back to the sync `_active_user_name()` for any caller that doesn't pass one; `reset_for_character` and `handle_character_loaded` (both already async) now await the resolver and pass the result in. `_active_user_name()` itself is untouched -- it stays as the sync fallback default.
- Each site keeps a separate mode-source rule per the design's "one precedence rule per surface": chat surfaces use the app-authoritative runtime source, the workbench preview uses the workbench mode.

Tests: added one new server-mode test per site, siblings of the existing task-442 twins, arranged with a fake async scope service + server mode instead of the (now-None) local persona service:
- Tests/UI/test_chat_first_handoffs.py::test_native_start_chat_greeting_uses_server_backend_profile
- Tests/Event_Handlers/Chat_Events/test_chat_events.py::test_display_conversation_substitutes_server_backend_profile
- Tests/UI/test_personas_preview.py::test_greeting_renders_server_backend_profile_name (exercises the real site via `reset_for_character`, not the sync `_load_greetings` short-circuit, so the assertion actually depends on the swap)
All three were verified RED (falling back to "User"/local-only resolution) against the unswapped sites before implementing the swap, then GREEN after. All existing task-442 twins and the resolver unit tests stayed green throughout, untouched.

Modified files: tldw_chatbook/UI/Screens/chat_screen.py, tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py, tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py, Tests/UI/test_chat_first_handoffs.py, Tests/Event_Handlers/Chat_Events/test_chat_events.py, Tests/UI/test_personas_preview.py.
<!-- SECTION:NOTES:END -->
