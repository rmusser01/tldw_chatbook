---
id: TASK-32481
title: 'Console persona session identity: label parity and dynamic identity macros'
status: To Do
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
- [ ] #1 Persona "Chat now" (local and server backends) creates a persona-bound native Console session whose transcript labels assistant rows with the persona's name instead of "Assistant".
- [ ] #2 Persona system-prompt macros (`{{user}}`/`{{random_user}}`/`<USER>` and `{{char}}`/`{{character}}`/`{{persona}}`/`<CHAR>`) expand with the effective user display name and persona name at session seed, per send, and after identity-name changes (single non-recursive pass, per ADR-046).
- [ ] #3 Resumed persona conversations — including legacy ones persisted with `assistant_kind="persona"` — re-resolve the persona name from `assistant_id`; resolution failure degrades honestly to "Assistant".
- [ ] #4 No database schema migration; the trusted persona system template persists as an optional `persona_system_template` key under the existing version-1 `console_roleplay_context` metadata envelope.
- [ ] #5 Character behavior is unchanged; a character swap onto a persona session clears persona identity fields, and at most one of `character_name`/`assistant_name` is ever set.
- [ ] #6 Persona sessions never store `assistant_authority_id` (ADR-037 authority-free rule); server persona fetches still use the existing target-freshness fencing.
- [ ] #7 Targeted tests covering all changed behavior pass.
<!-- AC:END -->
