# Console Persona Session Identity Design

Status: Approved design, pre-planning
Date: 2026-09-11
Extends: ADR-037, ADR-046
ADR required: yes — `backlog/decisions/149-console-persona-session-identity.md` (to be created before implementation, linked from the Backlog task and implementation plan)

## Summary

A Personas-screen "Chat now" on a persona currently stages the card as
context in a generic Console chat; the transcript keeps showing "Assistant"
and no template macros expand. Character cards already bind a real session
identity: the transcript shows the character's name and `{{user}}`-family
macros expand dynamically. This design brings persona cards to parity: a
persona "Chat now" creates a persona-bound native Console session whose
transcript label is the persona's name, whose system prompt is materialized
through the existing single-pass macro expander, and whose identity survives
persistence and resume. The speaker label always reflects the session's
current assistant identity, whatever its kind.

## Context

Current state, verified against the code:

- The transcript speaker label is resolved in
  `Chat/console_roleplay_identity.py::resolve_console_message_presentation`:
  assistant rows show the character name only when
  `assistant_kind == "character"` and `character_name` is non-blank;
  otherwise the literal "Assistant".
- Character sessions bind through
  `UI/Console_Modules/session.py::_start_character_console_session` from a
  Personas-screen handoff (`source="personas"`, `item_type="character-card"`,
  `metadata.intent="start_chat"`, `metadata.selected_kind="character"`).
  Persona handoffs (`item_type="persona-card"`, `selected_kind="persona"`)
  fail that extraction and fall through to
  `ChatScreen._stage_handoff_as_console_live_work`, which stages the card as
  evidence with the suggested prompt "Respond as \<name\>." — no bound
  identity, no macro expansion.
- Macro expansion exists as
  `Chat/console_roleplay_identity.py::expand_character_template`: one
  non-recursive pass; `{{user}}` / `{{random_user}}` / `<USER>` map to the
  effective user display name; `{{char}}` / `{{character}}` / `{{persona}}` /
  `<CHAR>` map to the assistant-side name. Character sessions expand at seed,
  re-expand per send (`console_chat_controller.py::_resolved_system_prompt`),
  and re-materialize on identity change
  (`console_chat_store.py::_materialize_roleplay_projections`, gated by
  `_is_named_character_session` and fenced by `identity_revision`).
- The effective user display name resolves per ADR-046: per-conversation
  override -> `[chat_defaults].user_display_name` -> "User"
  (`effective_user_display_name`).
- Persona profiles (`CharacterPersonaScopeService.get_persona_profile`,
  mode-routed local/server) carry `name`, `system_prompt` (optional),
  `description`, `personality_traits`. They have no greeting field, so there
  is no persona greeting seeding.
- The conversations table persists `assistant_kind`, `assistant_id`,
  `character_id`, and `assistant_authority_id`. There is no assistant-name
  column; character names are re-resolved at resume from `character_id`
  (`_resolve_resumed_character_name`).
- The versioned `console_roleplay_context` metadata envelope
  (`Chat/console_roleplay_metadata.py`, version 1) persists
  `user_name_override` and `character_system_template` with per-key fail-soft
  parsing.
- The control chip already has a persona branch:
  `resolve_assistant_identity_label` renders `Persona: <name>` when given
  `assistant_kind="persona"` and an `assistant_name`
  (`Chat/console_display_state.py`), and
  `ChatScreen._build_console_control_state` already passes
  `assistant_name=getattr(active_session, "assistant_name", None)` — always
  `None` today because the field does not exist.
- ADR-037 decides that persona and generic conversations remain
  authority-free: persona sessions never store `assistant_authority_id`.
- ADR-037's server "character authority" trio
  (`capture_character_authority_context`,
  `is_character_authority_context_current`,
  `resolve_character_authority_id`) resolves a server-user authority
  (`server-user-v1:<sha256>`) and is already reused by persona-adjacent code
  (`personas_screen.py` TTS authority). The capture/is-current pair is
  server-target freshness fencing and is kind-agnostic.

## Goals

- The Console transcript speaker label always reflects the session's current
  assistant identity: character name for character sessions, persona name for
  persona sessions, "Assistant" only when no identity is bound.
- Persona "Chat now" from the Personas screen creates a persona-bound native
  Console session, locally and against a connected server.
- Persona system prompts expand `{{user}}`-family macros with the effective
  user display name and `{{char}}`/`{{persona}}`-family macros with the
  persona name, at seed, per send, and on identity change — identical to the
  character behavior required by ADR-046.
- Resumed persona conversations — including legacy persona conversations that
  already persist `assistant_kind="persona"` — render the persona name.
- No database schema migration.

## Non-goals

- The Console identity picker stays character-only. Personas cannot be picked
  or swapped mid-chat from inside the Console; persona sessions originate
  from the Personas screen. (Decided with the owner.)
- No persona greeting seeding; persona profiles have no greeting field.
- No persona avatar/rail art, buddy visuals, or TTS/voice changes.
- The staged-context "Attach" intent is unchanged, and the handoff fallback
  path continues to stage persona card bodies with raw, unexpanded macros —
  staged context is evidence, not identity. This matches character fallback.
- STTS window and TTS readback copy are separate surfaces and are unchanged.
- Persona conversations remain authority-free per ADR-037; no
  `assistant_authority_id` is resolved or stored for them.

## Design

### 1. Session identity model (storage approach B)

`ConsoleChatSession` gains one nullable field:

```python
#: Persona-kind assistant display name. Set only when
#: ``assistant_kind == "persona"``; mutually exclusive with
#: ``character_name``.
assistant_name: str | None = None
```

and one nullable trusted-template field:

```python
#: Trusted persona system-prompt source; materialized into
#: ``settings.system_prompt`` via the identity-template expander.
persona_system_template: str | None = None
```

Rules:

- `assistant_kind == "character"`: display name in `character_name` (unchanged).
- `assistant_kind == "persona"`: display name in `assistant_name`.
- `generic`: neither.
- Invariant: at most one of `character_name` / `assistant_name` is non-blank.
  Every identity write enforces it: persona bind blanks `character_name`,
  `_swap_console_session_character` blanks `assistant_name` and
  `persona_system_template`, and resume-clear blanks in both directions.
  `settings.character_label` is cleared on persona bind and persona resume so
  the chip's character-first precedence can never show a stale character name
  over a live persona.

A single helper — `session_assistant_display_name(session)` in
`Chat/console_roleplay_identity.py` — centralizes the kind-conditional pick
so no other code branches on kind for the display name.

Gate hygiene (review finding): `_is_named_character_session` stays
character-only (it guards greeting-template re-expansion, which personas must
never inherit). A new `_is_named_persona_session` covers the persona branch,
and the shared materialization machinery uses a combined
`_is_named_identity_session`.

### 2. Handoff routing and session creation

A new `_persona_session_identity_from_handoff` in
`UI/Console_Modules/session.py` accepts exactly:

- `payload.source == "personas"`
- `payload.item_type == "persona-card"`
- `metadata.intent == "start_chat"`
- `metadata.selected_kind == "persona"`
- `payload.runtime_backend in {"local", "server"}`
- three-way backend consistency: `source_owner == source_selector_state ==
  metadata.backend == runtime_backend`
- `metadata.selected_target_id == f"{runtime_backend}:persona:{persona_id}"`

It returns `(runtime_backend, persona_id, name_hint, assistant_id)`. Unlike
the character extractor, `selected_record_id` stays an opaque string — no
`int()` coercion, no `_canonical_character_id_text` (persona IDs are
`local-persona-<hex>` or server opaque strings). Anything but a clean match
returns `None`, and the handoff falls through to the existing staged-context
path unchanged.

`_consume_pending_chat_handoff` tries the character starter first, then the
persona starter, then stages as live work — preserving today's fallback
order and semantics for every other payload.

Session creation reuses the character fencing spine rather than copying it
(review finding): the ~150 lines of
`_start_character_console_session` are mostly security-critical fencing. The
implementation extracts a shared fenced-fetch spine parameterized by a
kind-specific fetch and a kind-specific seed:

- Local: fetch via
  `character_persona_scope_service.get_persona_profile(persona_id,
  mode="local")`.
- Server: capture the authority context, verify currency, fetch via
  `get_persona_profile(persona_id, mode="server")`, re-verify currency before
  using the profile. Per ADR-037, personas do **not** resolve or store an
  authority ID; `assistant_authority_id` stays `None`. Failed fencing or a
  failed fetch returns `False` and the caller falls back to staged context
  with a notification — the character path's existing contract.

On success the new session is created with:

- `assistant_kind="persona"`, `assistant_id=persona_id`
- `assistant_name` = profile name through `sanitize_character_display_label`
  (180-character cap, same as characters; persona names may be 200)
- `character_id=None`, `character_name=None`, `assistant_authority_id=None`
- `runtime_backend` from the handoff
- title `Chat with <name>` (`derive_conversation_title` already produces
  this for persona kind)
- `settings.character_label` cleared; `settings.system_prompt` set to the
  materialized projection when the persona has a system prompt

Personas without a system prompt are fully valid: `persona_system_template`
stays `None`, settings keep the Console default, every expansion point
no-ops, and the session is still persona-bound — the name label works
regardless.

### 3. Template expansion

All expansion runs through the existing `expand_character_template`, whose
docstring gains one line declaring it the identity-template expander for both
kinds (the name stays; seven callsites are not worth the churn). For persona
sessions it is called with `character_name=<persona name>`, so
`{{char}}`/`{{character}}`/`{{persona}}`/`<CHAR>` all map to the persona name
and `{{user}}`/`{{random_user}}`/`<USER>` to the effective user display name.
Expansion remains single-pass and non-recursive per ADR-046: a display name
literally containing `{{user}}` renders literally.

Three coordinated expansion points, mirroring characters:

1. **Seed**: `persona_system_template` is stored as the trusted source;
   `settings.system_prompt` receives the materialized projection.
2. **Per send**: `console_chat_controller.py::_resolved_system_prompt` gains
   a persona branch (today it gates character-only), re-expanding from the
   trusted template so a display-name change lands on the next message even
   before re-materialization runs.
3. **Identity change**: the generalized staleness / re-materialization
   machinery re-projects `settings.system_prompt` when the user display name
   or persona name changes, bumping `identity_revision` so mounted
   transcripts repaint. Editing template-derived content clears provenance,
   per ADR-046's existing rule.

### 4. Presentation

`ConsolePresentationContext` gains `assistant_name: str | None = None`;
`ConsoleChatStore.presentation_context()` passes the session's value
through.

`resolve_console_message_presentation` gains a persona branch:
`assistant_kind == "persona"` with a non-blank sanitized `assistant_name`
yields the persona name as `speaker_label`. Styling deliberately stays on the
standard assistant treatment (`console-transcript-message-role-assistant`
row class under role-accent styles, `"assistant"` speaker tone): personas are
not roleplay characters, and the immersive-RP classes remain gated on the
character branch. Because the label rides the presentation `revision_token`,
every surface the shared resolver feeds — plain rows, Markdown rows,
copy/export/save, model-context assembly (ADR-046) — picks up the persona
name from one change.

The control chip needs no change: it already receives
`session.assistant_name` and renders `Persona: <name>` once the field exists.

### 5. Persistence and resume

No schema migration. `assistant_kind` and `assistant_id` already persist on
the conversation row; the persona display name is **never** persisted — it is
re-resolved at resume, exactly the character pattern
(`_resolve_resumed_character_name`). This also means a persona renamed after
binding shows its fresh name on next resume, same as characters.

The one value that must persist is the trusted `persona_system_template`. It
rides the existing `console_roleplay_context` metadata envelope as an
**optional third key under version 1** — not a version bump. The envelope's
parse is per-key fail-soft, so:

- old builds read the envelope and simply never look at the new key
  (backward compatible);
- new builds reading an old envelope see the key absent (forward
  compatible).

A version bump was rejected: the strict `version == 1` check would make old
builds drop the whole envelope including `user_name_override` — an unrelated
regression. `ConsoleRoleplayContext` gains
`persona_system_template: str | None = None`; serialization writes the key
only when set and preserves sibling keys per ADR-046.

Resume (`UI/Console_Modules/workspace.py::_resume_console_workspace_conversation`,
which today handles character name resolution) gains a persona branch:

- `assistant_kind == "persona"` -> resolve the name via
  `get_persona_profile(assistant_id, mode=session.runtime_backend)`;
  set `session.assistant_name` on success, clear it on failure;
  clear `settings.character_label` either way (inherited settings must not
  leak a different identity's label).
- Hydration (`Chat/console_conversation_hydration.py`) additionally reads
  `persona_system_template` from the parsed roleplay envelope onto the
  session, then the existing staleness check re-materializes if the
  projection drifted.

Legacy persona conversations (persisted with `assistant_kind="persona"` by
the pre-native-Console route) get the same branch, so they gain the persona
label on resume with no migration.

### 6. Error handling

- Handoff fetch/fencing failure -> `False` -> staged-context fallback +
  notification. Card content is never dropped.
- Resume name-resolution failure (deleted persona, unreachable server) ->
  `assistant_name=None` -> the transcript honestly falls back to
  "Assistant"; the session stays fully usable.
- Absent/corrupt template key -> envelope fail-soft -> the last materialized
  `settings.system_prompt` keeps serving; no heuristic rewrite (ADR-046).
- `_resolved_system_prompt`'s persona branch gates on non-blank name and
  template, else returns the session prompt unchanged.
- Every display-boundary name passes through
  `sanitize_character_display_label`.

### 7. Testing

Unit tests (mirroring the existing character suites):

- presentation resolver persona branch: label, assistant (not roleplay)
  styling, revision-token contents;
- handoff extraction: accepts a clean persona payload; rejects wrong kind,
  backend mismatch, wrong `selected_target_id`; preserves the opaque string
  ID;
- seed expansion: `{{user}}` and `{{persona}}`/`{{char}}` both directions;
  single-pass safety with a display name containing literal `{{user}}`;
- persona without system prompt: bound session, no template, expansion
  no-ops;
- gate separation: `_is_named_character_session` unchanged for personas,
  `_is_named_identity_session` true for both kinds;
- swap invariant: character swap onto a persona session blanks
  `assistant_name`/`persona_system_template` and vice versa;
- envelope round-trip with and without the persona key; sibling keys
  preserved;
- resume name resolution: local hit, local miss, server-mode call
  parameters.

Store/controller level: handoff-to-persona-session end-to-end (transcript
label, materialized system prompt, per-send re-expansion), and
re-materialization of a persona session on user-display-name change.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Overload `character_name` for personas (approach A) | The field would lie about its contents; every character-only reader (avatar, rail, chip, speech) would need an audit against persona leakage. |
| Resolve the name live by ID at render (approach C) | Puts synchronous service lookups in the render path and breaks the deliberately pure presentation resolver; contradicts ADR-046's stored-projection model. |
| Bump the roleplay envelope to version 2 | The strict version check would make old builds drop the whole envelope, including `user_name_override` — an unrelated regression. An optional v1 key is both backward and forward compatible. |
| Re-fetch the persona's current system prompt at resume instead of persisting the template | Persona edits would silently rewrite existing chats, breaking snapshot parity with character sessions (ADR-046 source-and-projection). |
| Resolve and store `assistant_authority_id` for server personas | Contradicts ADR-037: persona conversations remain authority-free. |
| Add personas to the Console identity picker | Deferred with the owner; persona sessions originate from the Personas screen only. |

## Consequences

- Persona identity becomes a first-class native-Console session kind with
  transcript labeling, chip labeling, and dynamic macro expansion.
- The session dataclass grows two nullable fields; snapshots and screen-state
  serialization carry them.
- Legacy persona conversations improve on resume without migration.
- Server persona binding reuses the existing freshness fencing but stores no
  authority, keeping ADR-037's authority-free persona rule intact.
- The handoff consumption order (character -> persona -> staged) preserves
  every existing fallback behavior.

## Links

- [ADR-037: Persona/User Profile separation and authority-scoped assistant identity](../../backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md)
- [ADR-046: Human chat display identity and template provenance](../../backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md)
- [ADR-149: Console persona sessions bind a kind-generic assistant display identity](../../backlog/decisions/149-console-persona-session-identity.md)
