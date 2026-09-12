---
id: TASK-32481
title: 'Console persona session identity: label parity and dynamic identity macros'
status: Done
assignee: []
created_date: '2026-09-11'
labels:
  - console
  - personas
  - roleplay
dependencies: []
references:
  - backlog/decisions/149-console-persona-session-identity.md
  - >-
    Docs/superpowers/specs/2026-09-11-console-persona-session-identity-design.md
  - >-
    Docs/superpowers/plans/2026-09-11-task-32481-console-persona-session-identity.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A Personas-screen "Chat now" on a persona currently stages the card as context
in a generic Console chat: the transcript keeps showing "Assistant" and no
template macros expand. Character cards already bind a real session identity
with a named transcript label and dynamic macro expansion. This task brings
persona cards to parity, locally and against a connected server, per the
approved spec and ADR-149, so the Console speaker label always reflects the
session's current assistant identity whatever its kind.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona "Chat now" (local and server backends) creates a persona-bound native Console session whose transcript labels assistant rows with the persona's name instead of "Assistant".
- [x] #2 Persona system-prompt macros (`{{user}}`/`{{random_user}}`/`<USER>` and `{{char}}`/`{{character}}`/`{{persona}}`/`<CHAR>`) expand with the effective user display name and persona name at session seed, per send, and after identity-name changes (single non-recursive pass, per ADR-046).
- [x] #3 Resumed persona conversations — including legacy ones persisted with `assistant_kind="persona"` — re-resolve the persona name from `assistant_id`; resolution failure degrades honestly to "Assistant".
- [x] #4 No database schema migration; the trusted persona system template persists as an optional `persona_system_template` key under the existing version-1 `console_roleplay_context` metadata envelope.
- [x] #5 Character behavior is unchanged; a character swap onto a persona session clears persona identity fields, and at most one of `character_name`/`assistant_name` is ever set.
- [x] #6 Persona sessions never store `assistant_authority_id` (ADR-037 authority-free rule); server persona fetches still use the existing target-freshness fencing.
- [x] #7 Targeted tests covering all changed behavior pass.
<!-- AC:END -->

## Implementation Notes

Implemented persona session identity for the native Console as a kind-keyed
persona branch on the existing ADR-046 character-identity machinery, so
persona cards reach label parity with character cards without a parallel
stack.

- **Approach:** `ConsoleChatSession` carries `assistant_name` /
  `persona_system_template` alongside the character identity fields, with a
  one-name invariant (at most one of `character_name` / `assistant_name` set,
  keyed by `assistant_kind`) enforced on identity mutation and on
  screen-state restore. Persona "Chat now" builds a persona-bound native
  session through the same handoff/seed path as characters
  (`seed_persona_roleplay`), and the send path re-expands the trusted
  template per turn via the kind-keyed `_resolved_system_prompt` branch, so
  macro expansion always uses the live user display name.
- **Authority-free per ADR-037:** persona sessions never store
  `assistant_authority_id`; server persona fetches reuse the existing
  target-freshness fencing rather than a new authority channel.
- **Resume:** the persona name is re-resolved from `assistant_id` at resume
  (including legacy `assistant_kind="persona"` conversations); a failed
  lookup degrades honestly to the generic "Assistant" label.
- **Persistence:** no schema migration — the trusted persona template rides
  as an optional `persona_system_template` key in the existing version-1
  `console_roleplay_context` metadata envelope, and as a
  `persona_system_template` field in serialized screen state.
- **Modified files:** `Chat/console_chat_store.py` (session fields, persona
  seed/swap/clear operations, envelope persistence),
  `Chat/console_roleplay_identity.py` (shared expansion/presentation),
  `Chat/console_chat_controller.py` (kind-keyed `_resolved_system_prompt`),
  `UI/Console_Modules/session.py` (handoff starter, resume re-resolution,
  screen-state round trip), plus targeted tests under `Tests/Chat/` and
  `Tests/UI/`.
