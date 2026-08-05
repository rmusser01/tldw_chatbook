# TTS Slice 3A — Character Identity and Persona/User Profile Separation

**Status:** approved by the user on 2026-07-28; adversarial review amendments incorporated
**Date:** 2026-07-28
**Parent design:** [Character TTS Generation Profiles](2026-07-25-character-tts-generation-profiles-design.md)
**Related tasks:** [TASK-617](<../../../backlog/tasks/task-617 - Bring-Roleplay-personas-to-parity-with-tldw_server-personas-module.md>), [TASK-763](<../../../backlog/tasks/task-763 - Add-TTS-generation-profile-domain-and-repository-lifecycle.md>), and [TASK-951](<../../../backlog/tasks/task-951 - Add-audio.cpp-TTS-profile-service-and-STTS-library.md>)
**Canonical ADR:** [ADR-037](../../../backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md)

## Goal

Establish the trustworthy character identity and message-authorship foundation
needed for per-character TTS without yet changing which profile speaks a
message. At the same time, correct the Roleplay workbench's inverted Persona
semantics so Persona records represent assistant-side character profiles rather
than the human user.

This slice deliberately produces two immediately useful outcomes:

- Console **Speak** requests become bound to the exact completed assistant
  message and selected variant the user clicked.
- The Roleplay workbench stops presenting Personas as user identities.

Visible TTS assignment controls and assigned-profile speech remain in Slice 3B,
where those controls will affect speech immediately.

## Context

ADR-028 already requires character TTS assignments to use
`(source, authority_id, character_id)`. Chatbook now has the profile repository
and profile library, but its Console sessions and persisted conversations do
not carry that complete authority. Persisted character identity currently
requires a numeric ID, which cannot represent opaque server character IDs.

The current Console speech event also carries caller-supplied text. A user can
click **Speak**, then switch variants, edit or delete the message, or change the
active session before the asynchronous handler runs. The handler normalizes
text and consumes cooldown before it can prove which message the text came
from.

Separately, the Roleplay workbench calls Persona records "User Profiles",
offers actions such as **Set as my name**, and uses the selected Persona as
`{{user}}`. This conflicts with `tldw_server`, where:

- a User Profile belongs to the authenticated human account; and
- a Persona is an assistant identity that may originate from a character card,
  then evolve independently with its own prompt, state, memory, exemplars,
  policy, setup, and voice defaults.

TASK-617 owns full Persona runtime parity. This slice makes only the semantic
and contract correction needed to stop deepening the current inversion.

## Domain boundaries

### User Profiles

A User Profile represents the human user or authenticated account. Existing
account-profile schemas under `tldw_api/auth_user_schemas.py` remain the
User Profile contract. This slice does not add a Roleplay User Profile editor,
select a human identity, or infer that an authenticated server account is the
user's roleplay identity.

Until a genuine Roleplay User Profile integration is designed, `{{user}}`
uses the existing neutral fallback, **User**.

### Personas

A Persona is an assistant-side character profile. The workbench label is
**Personas**, with the description:

> Personas — assistant profiles for roleplay and chat

`is_active` continues to mean enabled or disabled; it is not the selected
human identity. A Persona's `origin_character_id` and origin snapshot are
provenance, not a live character dependency. A Persona never receives a
`CharacterRef` merely because it originated from a character and never
inherits that character's TTS assignment.

Persona-backed Console runtime, state documents, memory modes, exemplars,
macros for `{{persona}}`/`{{char}}`/`{{character}}`, and Persona-specific voice
behavior remain TASK-617. Until then, a Persona handoff remains a
generic/global-TTS session rather than pretending to be a character-authored
session.

### Persona API models

`/api/v1/persona/*` uses `PersonaProfileCreate`,
`PersonaProfileUpdate`, and `PersonaProfileResponse`. Misleading aliases such
as `UserProfileCreate = PersonaProfileCreate` are removed; account User Profile
types keep their distinct names and module.

The runtime-facing Persona surface uses Persona terminology end to end:
`list_persona_profiles`, `get_persona_profile`, `create_persona_profile`,
`update_persona_profile`, `delete_persona_profile`, and
`restore_persona_profile`; the workbench mode ID is `personas`; and its pager,
widgets, messages, events, and tests use Persona names. There is no temporary
`UserProfile*` alias or method fallback for Persona operations. Compatibility
belongs only at persisted-data boundaries, not in callable domain APIs.

Server-facing Persona DTOs match the server contract exactly. Requests do not
send local-only `description` or free-form `personality_traits` fields that the
server does not persist. Server updates contain only explicitly changed,
supported fields. PATCH serialization uses the changed-field set while
retaining an explicitly supplied `null`; omission means unchanged and `null`
means clear where the server field is nullable. It does not use
`exclude_none=True` to collapse those two meanings. These rules prevent an
update from erasing an unedited `character_card_id`, origin snapshot,
`voice_defaults`, `setup`, state document, or future server field.

Existing local Persona records are preserved. A local update merges only the
edited supported fields into the stored record instead of reconstructing the
record and discarding unknown or legacy extensions. Existing UI must not imply
that a local-only extension will persist to the server. Local mutation input
therefore remains a distinct source-specific contract (or equivalently a
source-specific boundary payload), rather than widening the server wire DTO.
The editor builds the contract for its actual source; server mode does not
offer local-only fields as though the server will save them. Local updates also
distinguish omitted fields from explicit clears.

### Persona-as-user runtime removal

The Persona-backed `active_user_profile` resolver/configuration helper is
removed after its imports are unwired. It is not retained as a dormant
compatibility service: its list contract points at Persona records and would
keep the domain inversion callable. The old
`character_defaults.active_user_profile` value remains untouched in stored
configuration but no runtime code reads, validates, repairs, clears, or writes
it.

Console and shell presentation no longer project a Persona as the human:

- the `You: default` Persona mode/chip and read-only **User Profile** row are
  removed;
- generic sessions use **Assistant: General** when an identity summary is
  required, rather than **As: General**;
- character sessions use **Character: _name_**; and
- an existing Persona-authored session may be identified as
  **Persona: _name-or-ID_**, never **As: _name_**.

Legacy `persona_label` and `user_profile_label` values are ignored during
restore and are not emitted by new Console session settings. Their stored
bytes and historical transcript text are not migrated or deleted.

## Character identity

### Canonical reference

The existing TTS domain value remains authoritative:

```text
CharacterRef = (source, authority_id, character_id)
```

No generic `AssistantRef` or parallel identity hierarchy is introduced.
`source` is exactly `local` or `server`; `authority_id` and `character_id` are
bounded canonical text.

### Local authority

The character database already owns a durable `local_authority_id`. The
database exposes it through a narrow DB-owned accessor. TTS and Console code do
not query the citation/RAG identity table directly and do not derive authority
from a path, process, active database, or display name.

### Server authority

Server authority is a versioned, non-secret canonical encoding of:

1. the configured server target's durable `authority_scope_id`; and
2. the stable authenticated `user.id` returned by
   `GET /api/v1/users/me/profile?sections=identity`.

`authority_scope_id` is a canonical lowercase hyphenated UUIDv4 owned by the
saved `ConfiguredServerTarget`. New targets receive it at creation. A legacy
target receives one through an atomic persisted upgrade on first authority
use; the value must be durably saved before it can participate in authority.
Failure to persist it returns **Server identity unavailable** rather than
using an ephemeral value. Duplicate, malformed, or missing scope IDs fail
closed.

The existing routing `server_id` is not this value: production legacy targets
derive that field from their normalized URL. Routing IDs, base URLs, labels,
and origins therefore never enter the authority frame. Updating mutable
routing details on the same saved target preserves its `authority_scope_id`.
A legacy configuration change that is represented as a newly created target
receives a new scope and does not silently inherit the old target's
assignments.

The exact version-one encoding is:

1. validate `authority_scope_id` as exact canonical lowercase hyphenated
   UUIDv4 ASCII;
2. validate `user.id` as an integer from `1` through `2^63 - 1` and encode its
   canonical base-10 form;
3. define `LP(value)` as its unsigned four-byte big-endian byte length followed
   by its bytes, then construct
   `LP(b"tldw-chatbook.character-authority") + LP(b"1") +
   LP(authority_scope_id_ascii) + LP(user_id_ascii)`; and
4. persist `server-user-v1:` followed by the lowercase hexadecimal SHA-256
   digest of that unambiguous frame.

The result is fixed at 79 ASCII characters, below the assignment store's
256-character authority bound. Length framing prevents component-boundary
ambiguity; the domain separator and visible version prevent collisions with a
future authority construction. The digest is identity scoping, not credential
or secrecy material.

Identity may be cached only within the matching active authenticated server
context and is invalidated when that context changes. An asynchronous lookup
captures the target, `authority_scope_id`, bound client/authentication context,
and context revision before issuing the request. It rechecks the same context
before caching or returning the result; a server switch, account switch, or
credential-context change makes the response stale. Credential rotation causes
a refetch; the same target scope and authenticated `user.id` produce the same
authority. Different users on the same target produce different authorities.

The existing event-scope helper is not assignment authority: its fallback is a
credential fingerprint, which changes when a token rotates. Assignment
authority never uses a URL, normalized origin, API key, token, auth method,
credential source, or credential fingerprint. Server-profile-only authority is
allowed only if a future server contract explicitly guarantees character IDs
are global across principals.

If the identity endpoint is unavailable, malformed, ambiguous, or lacks a
stable user ID, Chatbook returns **Server identity unavailable** for authority
and assignment operations. It does not guess from the currently active server.
The ordinary server chat may still open, resume, and generate text as an
explicitly unscoped server-character session. In Slice 3A, its trusted message
snapshot has no `CharacterRef` and continues through global TTS because
assigned-profile resolution does not yet run. Slice 3B must fail assigned
resolution closed for such an unscoped message unless the user chooses its
separately designed explicit global override.

## Persisted conversation provenance

The main conversation database gains one nullable text column:

```text
assistant_authority_id
```

For a character-backed conversation, the complete identity is:

```text
(runtime_backend, assistant_authority_id, assistant_id)
```

The adjacent fields retain their existing meanings:

- `runtime_backend` is the source, `local` or `server`;
- `assistant_id` is the source-supplied canonical character ID;
- `character_id` remains a local numeric compatibility projection.

Validation is joint:

- Local character conversations require a numeric `character_id`,
  `assistant_id` equal to its canonical decimal form, and a local authority.
- A fully scoped server character conversation keeps `character_id` null,
  allows any valid bounded opaque `assistant_id`, and requires a server
  authority.
- A server character conversation with null authority is a valid degraded or
  legacy record, but cannot produce a `CharacterRef`, create or resolve an
  assignment, or claim character-authored TTS.
- Persona conversations use `assistant_kind = persona` and their Persona ID,
  with no `CharacterRef` or `assistant_authority_id`.
- Generic conversations carry no character authority.

The new column participates in local conversation create, read, update,
normalized validation, application-owned backup, and application-owned
restore beside the existing assistant identity fields.

Current Sync V2 does not transport `assistant_authority_id`. Its server
materializer and Chatbook apply path do not share a conversation-provenance
contract, and a target-local authority scope is not portable merely because an
envelope can carry text. Chatbook therefore does not emit this field into the
current sync contract. A conversation materialized from sync or another import
without locally proven authority remains authority-null and unassignable. It
is never rewritten to the receiver's active source. Cross-device authority
transport requires a separately agreed server/client contract.

### Migration

The migration is non-destructive:

- A legacy local character conversation can be backfilled from the durable
  local authority owned by the same database.
- A legacy server character conversation remains null and unassignable. It is
  never backfilled from whichever server or credential is active during
  migration.
- Existing Persona and generic conversations remain authority-free.

Reopening an unscoped legacy record never silently attaches the current
principal. An explicit future repair may do so after proving the conversation's
origin; otherwise a new source-aware handoff creates a new scoped session.

A source-aware Console session stores the complete identity in memory and
persists it with the conversation. `character_id` may remain as a local
compatibility projection, but it is not identity authority.

## Immutable Console speech snapshots

Clicking **Speak** asks the Console store to issue an immutable
`TTSMessageSpeechSnapshot`. The snapshot contains:

- native session and message IDs;
- optional persisted conversation and message IDs;
- the exact raw visible content;
- the selected variant ID plus a monotonic in-memory speech revision;
- the persisted message version when one exists;
- role and completion state; and
- trusted assistant kind plus a complete `CharacterRef` when the session is
  backed by complete scoped character authority.

The UI does not construct this value from labels, selectors, or the currently
active session. Before any text normalization, cooldown mutation, profile
lookup, or provider action, admission verifies that:

- the message still belongs to the captured session;
- it is the same completed assistant message;
- the selected variant and exact raw content still match; and
- persisted or in-memory authorship still matches the captured authority.

Deleted, edited, incomplete, non-assistant, variant-switched, spoofed, or
authority-mismatched snapshots fail safely. Only after successful validation
does the existing whitespace normalization and length validation run.
A rejected snapshot does not consume the message cooldown.

The Console store increments the speech revision on every content, status,
variant-addition, variant-selection, or variant-content mutation. It is
process-local because an issued snapshot cannot survive process exit. A
persisted message also requires the captured durable row version to remain
current. Editing and then restoring identical text therefore remains stale; raw
text equality alone is never admission authority.

This changes admission intentionally, but not synthesis selection in Slice 3A:
valid snapshots continue through the current global provider, model, voice,
format, speed, and legacy/native routing rules. Assignment resolution begins
in Slice 3B.

Non-Console TTS callers that genuinely lack a Console message use an explicit
trusted global-speech request path; they do not forge a message snapshot.

## Assignment service foundation

The existing profile repository remains the sole assignment store. The profile
service exposes only the minimum source-aware operations needed by Slice 3B:

- create or replace the assignment for one exact `CharacterRef`; and
- detach the assignment for one exact `CharacterRef`.

A new assignment accepts a caller-held `LoadedTTSProfile` and its exact
`repository_generation`, plus the caller-observed current assignment state
(the exact assigned profile ID or an explicit unassigned state); it never
substitutes the repository's current generation or current assignment. The
service first checks that expected generation, validates the exact loaded
profile revision against a fresh authoritative capability observation,
rechecks the expected generation, and passes the expected generation, selected
profile revision, and expected current assignment into repository mutation
admission.

The repository verifies all three expectations inside the final transaction
before inserting or replacing the assignment. A target profile edited after
the caller loaded it is stale, and an assignment changed from the
caller-observed state is a conflict rather than a silent overwrite. Detach
similarly carries the generation and exact profile ID from the caller-held
assignment result. It succeeds idempotently when already absent, but refuses
to remove a different replacement assignment. No assignment revision column
is needed for this compare-and-set contract.

A restore at any point makes the operation stale even if the replacement store
contains the same profile UUID and revision.

The repository's generation, selected-profile revision, expected-current-
assignment, foreign-key, and transaction checks remain final authority. This
does not claim an atomic transaction across the remote capability catalog and
SQLite; later speech admission still revalidates the assigned profile.

Slice 3A adds no hidden selector, feature-gated widget, dormant **Detach**
button, or automatic assignment cleanup. Visible assignment, repair, and
detach controls move together to Slice 3B.

## Atomic delivery

This document is the shared integration contract, not permission for one
omnibus implementation plan or PR. Slice 3A is delivered as four ordered,
independently reviewed increments:

1. **3A.1 — Persona/User Profile semantic boundary.** Correct DTO ownership,
   source-specific non-destructive local/server Persona updates, runtime API
   naming, Roleplay copy/actions, macro fallback, and all legacy
   Persona-as-user projections. It changes no TTS or conversation schema.
2. **3A.2 — Character authority and conversation provenance.** Add the local
   authority accessor, persisted target `authority_scope_id`, exact
   revision-fenced server authority resolver/encoding, one-column local
   migration, and source-aware Console session identity. It changes no Sync V2
   contract, speech-event admission, or assignment repository.
3. **3A.3 — Trusted Console speech snapshots.** Add monotonic message speech
   revisions, app-issued snapshots, validation-before-cooldown, and valid
   global-TTS regression coverage. It performs no assigned-profile resolution.
4. **3A.4 — Assignment mutation service.** Add exact source-aware set/replace
   and detach operations with caller-held lifecycle generation, selected
   profile revision, expected-current-assignment compare-and-set, and
   capability fencing. It adds no UI or speech resolver.

Each increment receives its own atomic Backlog task, implementation plan,
verification evidence, and PR. After this written spec is approved, only 3A.1
transitions to implementation planning; the later increments retain these
approved contracts but do not enter that plan.

## Compatibility and state handling

- Existing Persona records are not renamed, rewritten, or deleted.
- The old `character_defaults.active_user_profile` value remains stored but is
  inert. New runtime code neither reads it as the human identity nor writes it.
- **Set as my name**, **Chatting as**, and active-human Persona markers are
  removed. Persona enabled/disabled state remains.
- Restored Console settings ignore legacy `persona_label` and
  `user_profile_label` projections until a genuine User Profile integration
  owns them. New settings do not emit them. Stored settings and historical
  transcript text are left intact.
- No saved-workbench-mode migration is added; the current restore contract
  already falls back from non-character modes.
- Valid global TTS behavior, profile-library behavior, external audio.cpp
  ownership, and complete-WAV playback remain unchanged.

## Failure, privacy, and observability rules

- Stable identity failure is explicit and recoverable; it never degrades to a
  credential-derived authority.
- A stale speech snapshot reports a bounded user-facing reason and performs no
  provider or assignment work.
- Logs and metrics may record safe outcome codes and source kind. They never
  record message text, tokens, credentials, server origins, routing server IDs,
  raw authority scope IDs/components, or the encoded authority ID.
- The snapshot is ephemeral and is not added to profile storage or portable
  character payloads.
- Persona correction failures do not disable Characters, the profile library,
  or ordinary global speech.

## Verification

Focused deterministic tests cover:

1. Persona and account User Profile types are distinct; Persona endpoints,
   runtime service methods, workbench mode, widgets, events, and tests use only
   Persona terminology.
2. Local Persona edits preserve unknown legacy fields, while server PATCH
   payloads contain only explicitly changed server-supported fields and
   preserve omitted-versus-explicit-null semantics.
3. Roleplay copy, actions, `is_active`, and `{{user}}` fallback reflect the
   corrected semantics.
4. Legacy active-human pointers and restored Persona-derived user labels are
   ignored without deleting stored state or transcript content; Console no
   longer shows `You: default`, a Persona-backed **User Profile** row, or
   **As: General**.
5. Local authority is stable across restart and exposed only through the DB
   accessor.
6. The schema migration backfills eligible local conversations, preserves null
   legacy server provenance, accepts opaque server character IDs, and
   round-trips `assistant_authority_id` through local CRUD and backup/restore.
   Current sync/import paths deliberately materialize unproven provenance as
   null and never substitute the receiver's active authority.
7. Server authority uses a persisted opaque target scope rather than the
   URL-derived routing ID, stays stable across mutable routing details and
   credential rotation, separates two users on one target, rejects stale
   in-flight identity responses after context switches, and fails closed when
   scope persistence or the identity response is unavailable or invalid
   without blocking ordinary server text chat.
8. Speech snapshots reject session switches, edits, deletion, incomplete or
   non-assistant messages, variant changes, and authorship mismatches before
   cooldown or provider work, including edit-then-revert of identical text.
9. An unchanged valid snapshot produces the same global TTS selection and
   complete-WAV flow as before.
10. Assignment mutations are exact to `CharacterRef`, lifecycle-generation
    fenced, selected-profile-revision checked, compare-and-set against the
    caller-observed assignment, capability checked, and never inferred for
    Personas.

Regression coverage includes the existing profile repository/service, Console
session persistence, Persona workbench, Persona API schemas, global
audio.cpp/legacy Console TTS, and application-owned service lifecycle suites.

## Explicitly deferred

- Visible character assignment, repair, and detach UI.
- Assigned-profile request resolution or character-specific speech.
- Automatic speech after roleplay responses.
- Persona-specific TTS or inheritance from a source character.
- Full Persona runtime, memory, exemplar, policy, setup, and macro parity from
  TASK-617.
- A Roleplay User Profile editor or account-to-roleplay identity mapping.
- Cross-device or server-side Sync V2 transport of
  `assistant_authority_id`.
- Character-card TTS portability.
- Managed audio.cpp launch or supervision.
- Any new generic identity abstraction.

## ADR assessment

ADR required: yes

ADR path:
`backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md`

Reason: the four atomic increments collectively change the main conversation
schema and local backup/restore provenance, define durable local/server
authority and authentication boundaries, separate two previously conflated
domain models, establish a cross-module authorship contract, and explicitly
decline to extend the current Sync V2 transport.
