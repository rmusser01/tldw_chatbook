# TASK-14801: Roleplay chat identity and speaker theming design

- Date: 2026-08-08
- Status: approved in user brainstorming; self-review complete
- Backlog: TASK-14801
- ADR: `backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md`

## Goal

Make native Console character chats read like conversations between named
participants:

1. User and character rows receive distinct, subtle, theme-aware treatments.
2. Assistant transcript labels use the loaded character's name.
3. The human can set a global chat display name and override it per chat.
4. The effective human name expands user-side macros in trusted character
   templates used by the current transcript and model context.

## Existing boundaries

- `ConsoleChatSession` already owns durable assistant and character identity:
  `assistant_kind`, `assistant_id`, `character_id`, and `character_name`.
- `ConsoleSessionSettings.character_label` is a display projection used by
  character handoffs. It is not a human identity slot.
- Transcript role labels are currently derived directly from
  `ConsoleMessageRole` in both `console_transcript.py` and
  `UI/Console_Modules/message.py`.
- Plain transcript rows, Markdown assistant rows, `to_plain_text`, Copy,
  Save As, and Chatbook artifacts have separate presentation paths.
- Character-card prompt and greeting text currently resolves user macros
  eagerly with the literal `User` during handoff.
- Conversation `metadata` is an existing merge-owned JSON object. Message
  `metadata_json` is an existing local-only structured-provenance column.
- ADR-037 reserves Persona for assistant-side profiles. The retired active
  Persona-as-human pointer remains inert.
- `[general].users_name` controls storage-profile paths and is not a cosmetic
  chat label.

## Product behavior

### Effective human identity

The effective name resolves as:

```text
nonblank per-chat override
    else nonblank [chat_defaults].user_display_name
    else "User"
```

The global setting appears as **Default chat display name** in the canonical
Settings surface. The per-chat field appears as **Your name in this chat** in
Console Settings.

The per-chat field shows that blank means inherit. Clearing it immediately
returns the chat to the current global value. Changing the global value updates
open sessions without an override; overridden sessions do not change.

The custom human label applies to native Console user rows in generic and
character sessions. Character-only color treatment and template expansion do
not activate in generic or Persona sessions.

### Assistant identity

For transcript speaker labels:

```text
character session with a name -> character_name
generic or Persona assistant  -> Assistant
system/tool rows               -> existing System/Tool labels
```

Existing Persona identity summaries and chips retain their current
Persona-specific presentation. This task does not add Persona display-name
persistence to native transcript messages.

The loaded character name is session identity, not text copied onto each
assistant message. Character swaps update current assistant labels without
rewriting message roles or ordinary message content.

## Ownership and persistence

### Global default

Add `[chat_defaults].user_display_name = "User"` to the default config and
canonical Settings adapter. This key is display-only. It must not read from,
write to, migrate, or fall back through `[general].users_name`.

### Per-chat override

Add `user_display_name_override: str | None` to `ConsoleChatSession`, not
`ConsoleSessionSettings`. The value describes conversation identity, not a
provider/model generation setting.

For screen-state persistence, serialize and restore the optional override with
the session. For durable conversations, store it under a versioned,
task-owned object in conversation metadata:

```json
{
  "console_roleplay_context": {
    "version": 1,
    "user_name_override": "Captain Rowan",
    "character_system_template": "You are speaking with {{user}}."
  }
}
```

Clearing the override removes `console_roleplay_context` when it has no other
owned fields. Reads accept only an object, version `1`, and a valid string;
invalid data degrades to inheritance without blocking conversation restore.

Writes re-read and merge the current metadata object, preserve every sibling
key, and use the existing optimistic-version check. One bounded retry may
re-read after a conflict. Failure after that keeps the live session value and
shows a warning that the change may not survive reopening.

When an unsaved session first persists, the override is written after the
conversation receives its durable id. Temporary sessions retain it only in
session/screen state until promoted.

## Template source and safe projection

### Why two forms exist

Dynamic renaming requires the original character template. Safe persistence
and sync require ordinary fields to remain immediately readable without local
provenance. Template-derived content therefore has:

- **Source:** the exact trusted character template, retained with explicit
  provenance.
- **Projection:** the currently resolved text stored in the normal message or
  conversation field.

The projection is always safe to render or send by itself. Source is consulted
only when the template kind is recognized and the owning session is still a
character session.

### Character system prompt

At character handoff, retain the exact composed character system-template
source in the task-owned conversation metadata object and store the resolved
projection in the existing `system_prompt` field. The provider controller uses
the current source projection when provenance is present; otherwise it uses
the ordinary stored system prompt unchanged.

Changing the effective name re-materializes the in-memory prompt immediately
and writes the new safe projection when the conversation is durable. Editing
the system prompt through `/system` or Console Settings clears its character-
template source and stores the edit as ordinary content.

### Seeded character greetings

Seed the ordinary assistant message with the resolved greeting projection.
Extend structured local message provenance with a closed template kind and
the exact source, conceptually:

```text
template_kind = "character_greeting"
template_source = "Hello, {{user}}."
```

Only this closed kind participates in dynamic expansion. It remains local-only
under the current message-metadata contract. Sync receives the safe resolved
message content. A remote restore without local provenance therefore displays
the last materialized text rather than a literal macro.

When a user edits a derived greeting, the edit uses the currently resolved
projection, clears both template fields, and becomes ordinary assistant
content. Generated assistant replies never receive template provenance.

### Macro expansion

Introduce one pure, single-pass character-template expander. It recognizes the
existing character aliases and user aliases, but only user aliases vary with
chat display identity:

| Tokens | Value |
| --- | --- |
| `{{user}}`, `{{random_user}}`, `<USER>` | Effective human chat name |
| `{{char}}`, `{{character}}`, `{{persona}}`, `<CHAR>` | Loaded character name |

Replacement values are inserted once and never scanned again. A name such as
`Archivist {{character}}` remains literal. Matching stays case-sensitive for
compatibility with the existing helper.

The character-template source builder must use this tokenizer directly. It
must not preserve user macros by routing sentinel text through the existing
sequential replacement helper, because a sentinel can collide with real card
content and reintroduce recursive replacement behavior.

Manual user messages, user-edited content, generated assistant replies, system
diagnostics, and tool output are never passed through this expander.

## Shared presentation seam

Add one pure presentation model/resolver owned by the Console Chat layer. Its
inputs are the session identity, effective global name, message, and selected
variant. Its outputs include:

- resolved speaker label;
- resolved visible content;
- whether the row is a character-roleplay user or character row;
- a stable identity/template revision token for transcript caching.

The same resolver is consumed by:

- plain `ConsoleTranscriptMessage` rows;
- `ConsoleMarkdownMessage` header and body/footer flow;
- transcript `to_plain_text`;
- Copy and message excerpts;
- Save As Note/Media/Prompt/Chatbook payloads;
- speech snapshots when the spoken content is the visible derived projection;
- provider-context assembly, including seeded-greeting folding;
- context-preview payloads.

Edit actions are the exception: they receive the currently resolved projection
and clear template provenance on save rather than silently preserving a
template the user can no longer see.

This replaces duplicate `role.title()` and raw-content presentation helpers.
Protocol and persistence code may still read raw roles/content where identity
presentation is intentionally irrelevant.

## Transcript rendering

Character-bound sessions apply two additional row classes:

- character-roleplay user row;
- character-roleplay assistant row.

The source component stylesheet uses semantic theme colors, initially:

```text
user row background       -> $primary at a low percentage
character row background  -> $secondary or $accent at a low percentage
user name                 -> stronger $primary + bold
character name            -> stronger $secondary/$accent + bold
body text                 -> normal $text
```

Exact percentages are selected through dark/light compositor checks. The
generated CSS bundle is rebuilt from the source component and never edited by
hand.

Selection, failure, streaming, system, and tool states retain priority over
roleplay backgrounds. Color is supplemental: every row keeps a literal speaker
name. Both plain and full-Markdown assistant rows receive the same outer-row
treatment.

An identity revision participates in the transcript render signature. Name or
character changes update rows in place without remounting the transcript,
losing selection, or disturbing tail-follow state.

## Validation and safe rendering

- Trim leading/trailing whitespace.
- Blank global values resolve to `User`; blank overrides clear/inherit.
- Reject newline and other control characters.
- Permit Unicode names.
- Bound names to 48 terminal cells, not 48 Python code points. Reject longer
  input with inline validation instead of silently truncating the stored name.
- Render the name as literal `Content` or escaped text on markup-enabled
  surfaces. A name containing Rich/Textual markup syntax must not style or
  inject neighboring content.
- The name is user-authored prompt data by design, not a security principal.
  No authorization or storage decisions may depend on it.

## Failure behavior

- Global save failure leaves the previous effective global value active and
  reports the existing Settings save error pattern.
- Per-chat persistence failure keeps the in-memory override and warns that it
  may not survive reopening.
- Template-projection persistence failure keeps the correct live rendering
  and provider payload, while warning once for the initiating identity change.
- Invalid or future-version metadata degrades to global inheritance and safe
  stored projections. It never blocks restoring or sending a conversation.
- A missing character name falls back to `Assistant` and neutral styling; it
  never invents a name from a numeric id.

## Compatibility and sync boundary

- No database migration and no scan/rewrite of existing conversations.
- Existing literal `User` text remains literal.
- Existing unmarked macro text remains literal.
- New template provenance is created only at trusted character handoff.
- Roles and ordinary message content remain compatible with existing provider
  adapters and message persistence.
- Conversation and message Sync v2 contracts are unchanged. Resolved
  projections are safe fallback content; local dynamic provenance and the
  per-chat override are not promised to synchronize in this task.
- Persona, User Profile, character authority, TTS assignment, and
  `[general].users_name` behavior remain unchanged.

## Joined data flow

```text
Settings global name ─┐
                      ├─> effective identity ─> transcript labels/styles
chat override ────────┘                    ├─> copy/export/speech
                                           └─> trusted template projection
character identity ───────────────────────────────┘
                                                        ├─> visible greeting/prompt
template provenance ────────────────────────────────────└─> provider context
```

## Verification strategy

Per the owner directive, run only tests related to touched files and reachable
behavior.

### Pure behavior

- resolution precedence and inheritance;
- validation, Unicode, control characters, and markup literals;
- single-pass macro replacement, including replacement values containing
  macro syntax;
- trusted-template-only expansion;
- metadata parse/degrade/merge/conflict behavior;
- source-and-projection materialization and edit-clears-provenance behavior.

### Joined Console behavior

- real character handoff seeds character name, resolved system prompt,
  resolved greeting, and provenance;
- per-chat rename updates current user labels, seeded greeting, model context,
  context preview, Copy, and plain-text export;
- generic and Persona sessions retain existing assistant identity semantics;
- restore round-trips the override and template source where locally durable;
- a safe projection remains usable when provenance is absent;
- Markdown and plain rows use identical identity labels and outer styling;
- transcript updates preserve row identity, selection, and scroll/tail-follow.

### Visual behavior

Use a harness that loads the real stylesheet plus one scratch-profile live run.
Check dark and light themes, narrow and wide terminals, plain and Markdown
assistant rendering, long Unicode names, selected rows, streaming, failure,
tool, and system rows. Assert compositor-painted text/background behavior, not
only widget mount state. Verify notifications through `app._notifications`,
since screenshots do not include the toast rack in this Textual version.

Mutation checks must prove the new tests go red when:

- the per-chat override is ignored;
- the provider payload uses the stored old projection instead of the current
  effective template projection;
- macro expansion becomes sequential/recursive;
- role labels fall back to `role.title()`;
- provenance is not cleared after an edit;
- roleplay row classes are removed.

## ADR check

ADR required: yes

ADR path:
`backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md`

Reason: the feature introduces a persisted human chat-identity owner, a
source/projection persistence contract, metadata conflict behavior, and a
cross-module presentation/context interface while extending ADR-037's identity
separation.

## Non-goals

- User Profile or Persona selection as the human identity.
- Per-character saved human names.
- Custom user-selected row colors in this tranche.
- Rewriting legacy literal `User` or unmarked macros.
- Expanding macros in manually typed or generated messages.
- Changing provider message roles or API schemas.
- Adding or changing Sync v2 metadata transport.
- Character-card editing, Persona runtime parity, TTS identity, or avatars.
