# ADR-037: Separate Persona and User Profile domains and persist authority-scoped assistant identity

Status: Accepted
Date: 2026-07-28
Related Tasks: TASK-617, TASK-763, TASK-951
Extends: ADR-028
Clarifies: ADR-004, ADR-007, ADR-033
Supersedes: N/A

## Decision

Chatbook will treat User Profiles and Personas as separate domains:

- A User Profile represents the human user or authenticated account.
- A Persona represents an assistant-side character profile. It may retain an
  origin-character snapshot, but that provenance is not a live character
  identity and does not inherit a character TTS assignment.

Persona API routes use Persona DTOs rather than aliases named `UserProfile`.
Existing Persona records remain intact, `is_active` remains enabled/disabled,
and the legacy Persona-as-human pointer becomes inert. Full Persona runtime
parity remains TASK-617.

Character-backed conversations will persist their complete TTS authority using
the existing `runtime_backend` and `assistant_id` fields plus one new nullable
`assistant_authority_id` column. Together they encode:

```text
(source, authority_id, character_id)
```

Local authority comes from the durable `local_authority_id` owned by the same
character database. Server authority is a versioned non-secret encoding of
Chatbook's durable server-profile ID plus the stable authenticated `user.id`
returned by `GET /api/v1/users/me/profile?sections=identity`.

Version one validates the canonical server-profile ID and positive integer
user ID, length-frames their UTF-8 bytes with a domain separator and version,
and stores `server-user-v1:` plus the lowercase SHA-256 digest. The fixed
79-character result is deterministic, component-boundary safe, and within the
assignment store's authority bound.

Credentials, credential fingerprints, auth methods, origins, database paths,
display names, and the currently active source are not durable assignment
authority. If stable server identity cannot be established, server assignment
fails closed.

Legacy local character conversations may be backfilled using the authority
owned by their database. Legacy or identity-unavailable server character
conversations may remain valid authority-null records for ordinary text chat,
but cannot produce a `CharacterRef` or participate in assignment resolution.
They are never assigned the currently active server authority. Reopening alone
does not silently repair them.

Console manual speech will use an app-issued immutable message snapshot. Before
cooldown or provider work, Chatbook verifies the exact session, message,
selected variant, monotonic process-local speech revision, persisted row
version when present, raw visible content, completion state, role, and trusted
authorship. Slice 3A leaves provider/profile selection global after successful
validation; assigned-profile resolution and visible assignment controls ship
together in Slice 3B.

Slice 3A is not one implementation PR. Persona semantics, authority and
conversation provenance, trusted speech snapshots, and assignment mutation
service are four ordered atomic tasks and PRs. Only the first advances to
planning after the shared written design is approved.

## Context

ADR-028 requires character assignments to use full source-aware authority.
Bare local IDs collide across character databases, and server character IDs
can collide across authenticated users. Chatbook's existing persisted
conversation validator also requires numeric character IDs, preventing opaque
server IDs.

The current server-event helper can derive a credential fingerprint when a JWT
subject is unavailable. That value is useful for ephemeral event scoping but
violates ADR-028's durable-assignment rule because ordinary credential rotation
changes it. The server's authenticated User Profile endpoint provides the
stable principal contract required here.

The current Roleplay workbench simultaneously treats Persona records as
character-like assistant profiles and as "who I am". That inversion leaks into
schema aliases, `{{user}}`, Console labels, and active-profile actions.
Continuing TTS work on that model would allow a Persona or its source character
to acquire an unintended voice identity.

This ADR is required because the decision changes schema and sync contracts,
authentication-derived authority, cross-module message authorship, and the
long-lived meaning of Persona and User Profile records.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Use bare `character_id` | Collides across local databases and authenticated server principals. |
| Use server origin or active server profile alone | An origin is mutable, and one server can expose per-user character namespaces. |
| Use JWT subject or credential fingerprint fallback | Token forms and fingerprints can change during credential rotation and may not be the server's canonical user identity. |
| Reuse `discovery_owner`, `discovery_entity_id`, `source`, or `external_ref` | Those fields have different ownership and lifecycle semantics; overloading them would make assistant provenance ambiguous. |
| Add a second assistant-identity table | One nullable authority column completes the existing conversation identity without a new join or lifecycle owner. |
| Treat a Persona's origin character as its `CharacterRef` | A Persona evolves independently; this would silently inherit the source character's TTS assignment. |
| Delete legacy Persona-as-user state | Destructive cleanup is unnecessary. Preserving it inert permits future explicit migration. |
| Keep `UserProfile*` aliases for compatibility | The aliases preserve the domain error and make incorrect endpoint payloads type-check. |
| Add hidden assignment controls in Slice 3A | Dormant UI creates untestable behavior. Controls should ship when they affect speech in Slice 3B. |
| Deliver all Slice 3A changes in one PR | Persona semantics, schema/authority, speech admission, and assignment mutation are independently testable and reviewable. |

## Consequences

- The main conversation schema advances and sync payloads carry
  `assistant_authority_id` beside existing assistant provenance.
- Local character conversations retain their numeric `character_id` projection;
  server conversations may use opaque text `assistant_id` values.
- Persona and generic conversations remain authority-free.
- Synced authority remains provenance from the source conversation and is never
  rewritten to match the receiving client's active context.
- Stable server identity requires one bounded authenticated identity lookup per
  active context, with fail-closed behavior when unavailable.
- Identity-unavailable server text chat remains usable, but its authority-null
  conversations cannot create or resolve character assignments.
- Credential rotation for the same server user preserves assignments; different
  users on one server profile cannot collide.
- Old Persona records, active-pointer configuration, saved session state, and
  transcript content are preserved, but Persona-derived human labels are no
  longer projected into the active UI.
- Valid Console speech continues to use global TTS selection in Slice 3A.
  Stale or spoofed requests are newly rejected before cooldown.
- Assignment service foundations can use the existing `CharacterRef` and
  repository. Mutations require the caller-held repository generation; no
  generic assistant-reference framework is introduced.
- Visible assignment controls, assigned-profile speech, Persona-specific
  voices, and managed audio.cpp remain outside this decision's implementation
  slice.

## Rollback plan

- Disable source-aware assignment consumers while retaining the nullable
  conversation provenance column.
- Continue reading legacy conversations and global TTS preferences; do not
  erase authority values or down-migrate the database automatically.
- Keep the legacy Persona pointer and record fields inert rather than restoring
  Persona-as-human behavior.
- If the server identity endpoint is temporarily unusable, leave server
  assignments unavailable rather than substituting credentials or origin.

## Links

- [Approved Slice 3A design](../../Docs/superpowers/specs/2026-07-28-tts-character-identity-persona-separation-design.md)
- [ADR-028 — Character TTS generation profile ownership](028-character-tts-generation-profile-ownership.md)
- [ADR-004 — Personas destination-native workbench](004-personas-destination-native-workbench.md)
- [ADR-007 — Personas workbench route consolidation](007-personas-workbench-route-consolidation.md)
- [ADR-033 — Application session state ownership](033-application-session-state-ownership.md)
