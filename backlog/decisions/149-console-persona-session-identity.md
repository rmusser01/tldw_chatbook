# ADR-149: Console persona sessions bind a kind-generic assistant display identity

Status: Accepted
Date: 2026-09-11
Related Spec: [Console Persona Session Identity Design](../../Docs/superpowers/specs/2026-09-11-console-persona-session-identity-design.md)
Extends: ADR-037, ADR-046

## Decision

A Personas-screen "Chat now" on a persona creates a persona-bound native
Console session, paralleling character sessions. `ConsoleChatSession` gains
two nullable fields — `assistant_name` (the persona-kind display name) and
`persona_system_template` (the trusted system-prompt source) — rather than
overloading the character fields. At most one of `character_name` /
`assistant_name` is ever non-blank; every identity write (persona bind,
character swap, resume-clear) blanks the other side, and
`settings.character_label` is cleared whenever a persona identity applies.

The transcript's shared presentation resolver gains a persona branch: a
persona session with a non-blank `assistant_name` renders the persona name as
the speaker label with the standard assistant styling; roleplay styling
remains character-gated. Persona system prompts expand through the existing
single-pass identity-template expander at seed, per send, and on identity
change, per ADR-046's source-and-projection model. Personas have no greeting
field, so no greeting is seeded.

No database migration is introduced. The persona display name is never
persisted; it is re-resolved at resume from the persisted `assistant_id` via
the mode-routed persona service, mirroring character-name resolution and
covering legacy persona conversations. The trusted template persists as one
optional `persona_system_template` key under the existing version-1
`console_roleplay_context` metadata envelope, whose per-key fail-soft parsing
keeps old and new builds interoperable in both directions.

Server persona handoffs reuse the existing server-target freshness fencing
(capture context, verify currency, fetch, re-verify) but, per ADR-037,
persona conversations remain authority-free: no authority ID is resolved or
stored, and failed fencing or fetch falls back to the existing staged-context
path. The Console identity picker remains character-only; persona sessions
originate from the Personas screen.

## Context

Persona "Chat now" handoffs currently fail the character-only identity
extraction and land as staged context in a generic chat, so the transcript
shows "Assistant", no macros expand, and the persona is not the session's
identity. The character machinery this mirrors — display labeling, macro
expansion, re-materialization, template provenance — was built under ADR-046
but is gated on `assistant_kind == "character"`. ADR-046 also centralizes
label resolution in one presentation resolver, so one persona branch
propagates to transcript rows, Markdown rows, copy/export/save surfaces, and
model-context assembly. ADR-037 fixes personas as assistant-side profiles and
rules their conversations authority-free, which this decision preserves while
still supporting server-backed personas.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Overload `character_name` / `character_system_template` for personas | The fields would lie about their contents, and every character-only reader (avatar, rail, chip, speech) would need an audit against persona leakage. |
| Resolve the display name live by ID at render time | Synchronous service lookups in the render path; breaks the deliberately pure presentation resolver and ADR-046's stored-projection model. |
| Persist the persona name in a new conversations column | Unnecessary: the character pattern of re-resolving the name from the persisted ID at resume already works and avoids a schema migration. |
| Persist the template under a version-2 roleplay envelope | The strict version check would make old builds drop the entire envelope, including `user_name_override`; an optional version-1 key is backward and forward compatible. |
| Re-fetch the persona's current system prompt at resume | Persona edits would silently rewrite existing chats, breaking snapshot parity with character sessions. |
| Resolve/store `assistant_authority_id` for server personas | Contradicts ADR-037's authority-free persona rule. |
| Add personas to the Console identity picker | Deferred by owner decision; persona sessions originate from the Personas screen only. |

## Consequences

- Persona identity becomes a first-class native-Console session kind with
  transcript labeling, control-chip labeling (the existing
  `resolve_assistant_identity_label` persona branch lights up unchanged), and
  dynamic macro expansion.
- The session dataclass grows two nullable fields; session snapshots and
  screen-state serialization carry them.
- Legacy persona conversations gain the persona label on resume without
  migration.
- Character behavior is structurally unchanged: `_is_named_character_session`
  stays character-only, shared materialization machinery uses a combined
  named-identity gate, and the character handoff path is tried before the
  persona path with the staged-context fallback preserved.
- A persona renamed after binding shows its fresh name on next resume;
  deleted or unreachable personas degrade honestly to "Assistant" while the
  session stays usable.
- The persona fallback path continues to stage card bodies with raw,
  unexpanded macros; staged context is evidence, not identity.

## Verification Consequences

Unit tests cover the presentation persona branch, handoff extraction
accept/reject, macro expansion both directions with single-pass safety,
identity-gate separation, the one-name-set invariant on swap, envelope
round-trip with and without the new key, and resume name resolution for
local hit/miss and server mode. Store/controller-level tests cover
handoff-to-persona-session end to end and re-materialization on
user-display-name change. Only tests related to the changed files and
behavior are part of the implementation gate.

## Links

- [Console Persona Session Identity Design](../../Docs/superpowers/specs/2026-09-11-console-persona-session-identity-design.md)
- [ADR-037: Persona/User Profile separation and authority-scoped assistant identity](037-roleplay-assistant-identity-and-persona-user-profile-separation.md)
- [ADR-046: Human chat display identity and template provenance](046-roleplay-chat-display-identity-and-template-provenance.md)
