---
id: TASK-551
title: Active user profile resolution for the server profile backend
status: To Do
assignee: []
created_date: '2026-07-24 20:30'
labels:
  - roleplay
  - enhancement
dependencies: []
---
## Description

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

## Acceptance Criteria

- [ ] #1 With the server profile backend active, a server-side active profile's name substitutes into {{user}} at the three task-442 send surfaces
- [ ] #2 A dangling pointer (profile deleted on the resolving backend) still resolves to no-active (never a stale name, never an error)
- [ ] #3 With no pointer set, behavior is byte-identical (the task-442 twins keep passing)
