# ADR-046: Keep human chat display identity separate and preserve safe template projections

Status: Accepted
Date: 2026-08-08
Related Task: [TASK-14801](../tasks/task-14801%20-%20Add-roleplay-chat-identities-and-speaker-theming.md)
Related Spec: [Roleplay Chat Identity and Speaker Theming Design](../../Docs/superpowers/specs/2026-08-08-task-14801-roleplay-chat-identity-design.md)
Extends: ADR-037

## Decision

Chatbook will introduce a display-only human chat identity that is distinct
from authenticated User Profiles, assistant Personas, and the existing
`[general].users_name` storage-profile selector.

The effective human display name resolves in this order:

```text
per-conversation override -> global chat display name -> "User"
```

The global default is stored in
`[chat_defaults].user_display_name`. A per-conversation override is carried
in native Console session state and persisted as one merge-safe key in the
conversation's existing metadata JSON under the versioned
`console_roleplay_context` object. Blank overrides mean "inherit" and are
represented by removing the owned field. No database migration is introduced.

The effective name owns two behaviors:

1. Human transcript labels and related presentation/export surfaces.
2. User-side macro expansion in trusted character-template content.

It does not replace message protocol roles, account identity, data-directory
ownership, or Persona identity. Stored and provider-facing roles remain
`user`, `assistant`, `system`, and `tool`.

Character-template content uses a source-and-projection model. Canonical
message and conversation text keeps a resolved, provider-safe projection.
The original character template plus an explicit template kind is retained
in local provenance metadata. When the effective name changes, Chatbook may
re-materialize only content with that provenance. If provenance is missing,
the safe resolved projection remains authoritative and no heuristic rewrite
occurs.

Macro expansion is single-pass. Recognized macros are replaced without
re-scanning replacement values, so a user or character name containing macro
syntax is rendered literally. Only trusted character-template content expands
`{{user}}`, `{{random_user}}`, and `<USER>`. Manually typed messages and
ordinary generated replies do not.

One shared Console presentation resolver supplies the role label, effective
visible content, and character-roleplay styling state to plain transcript
rows, Markdown rows, copy/export/save surfaces, and model-context assembly.
Transcript styling is a global Appearance preference stored as
`[appearance].console_transcript_style`. Its closed vocabulary is `neutral`,
`role_accents`, and `immersive_rp`, with `role_accents` as the default.
Neutral retains speaker labels without role color. Role accents apply
semantic row tints and stronger speaker-name accents to both generic and
character-bound sessions. Immersive RP retains those cues and additionally
accents user, assistant, and character prose.

### 2026-08-11 amendment: accessible role-accent modes

Speaker treatments use semantic, theme-aware tokens rather than fixed role
colors in renderer code. Dark and light themes may resolve different concrete
accent colors to preserve contrast. Color is never the only identity signal:
speaker labels remain visible in every mode. Failed, selected, system, tool,
code, and link treatments take precedence over immersive prose color where
their operational meaning would otherwise be obscured. A successful
Appearance save publishes a generation signal and mounted Console transcripts
re-resolve their presentation without an application restart.

### 2026-08-26 amendment: durable historical character-name snapshots

`console_roleplay_context` v2 adds an optional `character_name_snapshot`. New
writes use v2; readers accept v1 and v2; versions greater than v2 remain
fail-closed and block merge writes. The snapshot is the character name that
owned the resolved prompt/template projection when the conversation was saved.

A v1 conversation has no historical-name authority. Resume must not fetch or
backfill the current character-card name. Saved resolved `system_prompt`
remains authoritative when provenance or the historical name is absent. The
data remains in the existing merge-safe metadata object; no schema migration
is introduced.

This amendment supports [TASK-22988](../tasks/task-22988%20-%20Resume-prior-character-chats-from-Roleplay.md)
and the approved [Roleplay Resume Prior Character Chat Design](../../Docs/superpowers/specs/2026-08-26-roleplay-resume-prior-character-chat-design.md).

### 2026-09-03 amendment: typed activation and aggregate Roleplay draft veto

[ADR-116](116-character-conversation-navigation-and-local-semantic-search.md)
preserves historical saved display identity and exact ID-only resume while
making cross-surface activation Console-owned, cancellable, and result typed.
Context, `Ctrl+K`, and Roleplay pass an immutable resolved local character key
and exact conversation ID to the canonical Console opener; none reconstructs a
transcript or substitutes the current or same-named card.

The opener returns exactly `OPENED`, `CANCELLED_PRECOMMIT`, `NOT_FOUND`,
`DATA_PROFILE_CHANGED`, `CHARACTER_UNAVAILABLE`, or `FAILED`. Its atomic
`commit_started` acknowledgement is the cancellation linearization point.
Cancellation that wins before it guarantees no Console target, tab, draft, or
focus change. Once commit starts, the caller remains mounted until the exact
destination is current and visible or the opener atomically restores the prior
Console state. Only `OPENED` dismisses directly to Console.

Before a Roleplay deep link changes card selection or unmounts an editor, the
app-owned navigation coordinator captures one aggregate draft snapshot across
form edits, character-visual authoring, shared-Persona visual authoring, Persona
visual authoring, attachments, and every in-flight save owner. Dirty state must
complete Save and continue, Discard and continue, or Stay; save failure or
partial success preserves remaining drafts and blocks navigation. This
amendment is owned by
[TASK-31241](../tasks/task-31241%20-%20Align-character-conversation-navigation-decisions.md).

## Context

The native Console session already persists character identity, including
`assistant_kind`, `character_id`, and `character_name`. The transcript does
not consume that identity: it converts the role enum directly to `User` and
`Assistant` in more than one location. Plain rows, Markdown headers, exports,
and save actions therefore drift if only one renderer is changed.

ADR-037 deliberately removed an earlier Persona-as-human feature. Personas
are assistant-side profiles; reviving their active pointer as the human's
chat name would reintroduce that domain error. The existing
`[general].users_name` value is also unsuitable because it selects the user
data-directory name. Renaming it for roleplay could silently relocate the
application's databases and files.

Character-card macros are currently expanded eagerly while creating the
session prompt and greeting. Eager expansion cannot support a per-chat name
change that updates current character context. Storing only raw templates is
also unsafe: current Chat synchronization transports ordinary message
content, while local message provenance is not part of that payload. A
source-and-projection model preserves dynamic local behavior without allowing
literal macros to leak when provenance or sync context is absent.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Reuse a Persona or active User Profile record as the human identity | Contradicts ADR-037 and risks assigning assistant-side semantics to the human. |
| Reuse `[general].users_name` | That value participates in storage-path resolution; a cosmetic rename could relocate application data. |
| Expand templates once at chat creation | Cannot update current character context when a per-chat override changes. |
| Store raw templates as canonical message and prompt content | Literal macros can surface or reach a provider when provenance is unavailable, especially across sync boundaries. |
| Rewrite every message containing macro-like text | Cannot distinguish trusted templates from manually typed syntax and would mutate ordinary prose. |
| Store a resolved display name on every message | Duplicates session identity and conflicts with retroactive relabeling of current user rows. |
| Reload the current character card whenever the name changes | Character cards may be edited or deleted and would not reproduce the historical template exactly. |

## Consequences

- Global and per-chat chat names become explicit display identity, not account
  or Persona identity.
- Open sessions that inherit the global name update when it changes; sessions
  with an override remain stable.
- Editing template-derived content clears its template provenance and makes
  the edited projection ordinary content.
- Old conversations require no migration. Literal `User`, unmarked macros,
  and other legacy text remain unchanged.
- Per-chat override persistence is local conversation metadata under the
  current Sync v2 contract. The standard resolved content remains safe if the
  metadata does not travel to another device.
- Metadata writes must preserve sibling keys and use the database's optimistic
  concurrency contract. A failed durable write keeps the live session value
  and reports that reopening may restore the previous value.
- Transcript color remains semantic and theme-aware. Speaker labels continue
  to carry identity when color is unavailable, disabled, or indistinguishable.
- Existing installations adopt Role accents through the configuration default;
  users can restore the former neutral presentation from Settings > Appearance.
- No schema migration, new identity table, Persona compatibility alias, or
  message-role change is introduced.

## Verification Consequences

Unit tests must cover resolution precedence, validation, clearing/inheritance,
single-pass macro behavior, template provenance, legacy fallback, and
merge-safe metadata updates. Joined tests must exercise the real character
handoff through transcript presentation and provider-context assembly.

Visual verification must use the real stylesheet or the real app compositor.
Both dark and light themes must show readable user and character rows, with
selected, streaming, failed, system, and tool states retaining precedence.
Speaker labels and any Immersive RP prose accent must reach at least WCAG AA
4.5:1 contrast against the compositor-painted background in both supported
light and dark theme modes.

Only tests related to the files and behavior changed by TASK-14801 are part of
the implementation gate.

## Links

- [ADR-033: Application session state ownership](033-application-session-state-ownership.md)
- [ADR-037: Roleplay assistant identity and Persona/User Profile separation](037-roleplay-assistant-identity-and-persona-user-profile-separation.md)
