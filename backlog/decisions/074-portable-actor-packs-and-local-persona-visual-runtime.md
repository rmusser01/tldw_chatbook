# ADR-074: Portable Actor Packs and local Persona Visual runtime

Status: Proposed
Date: 2026-08-20
Related Tasks:
[TASK-19053](../tasks/task-19053%20-%20Add-local-Persona-Visual-pack-foundation.md)
[TASK-19054](../tasks/task-19054%20-%20Author-and-import-Persona-Visual-packs.md)
[TASK-19055](../tasks/task-19055%20-%20Add-opt-in-app-wide-floating-Persona-Buddy.md)
[TASK-19056](../tasks/task-19056%20-%20Enable-Shared-Visual-Identity-for-Persona-actors.md)
[TASK-19057](../tasks/task-19057%20-%20Define-and-create-portable-Actor-Packs.md)
[TASK-19058](../tasks/task-19058%20-%20Export-self-contained-Actor-Packs.md)
[TASK-19059](../tasks/task-19059%20-%20Import-review-and-activate-Actor-Packs.md)
Related Spec: [Actor Pack, Persona Buddy, and Streaming Emote Programme Design](../../Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md)
Related Decision: [ADR-067](067-bundled-samira-visual-identity-pack.md)
Supersedes: N/A

## Decision

Chatbook will define `.tldw-actor-pack` as a portable, self-contained envelope
containing exactly one local Character or one local Persona. Shared Visual Identity
and Persona Visual remain separate schemas and runtimes inside that envelope: the
former represents chat expressions, while the latter represents Persona operational
states. Neither runtime projects into the other's state catalog, storage, bindings,
or resolver.

The local Persona Visual runtime will implement the `sprite_frames`
manifest-version-1 contract pinned at server commit
`385afa951922c8a9dc2002c675bb6cad65e4ac23`. Its reserved built-ins are `idle`,
`wake_armed`,
`listening`, `thinking`, `speaking`, `tool_running`, `approval_needed`, `error`, and
`offline`; an activatable pack must resolve `idle`, `listening`, `thinking`,
`speaking`, and `error` directly or through validated fallback chains. Persona Buddy
will follow one explicit profile-local Persona preference and remain opt-in and off
by default. Only local Personas are eligible.

Every portable actor has a globally unique canonical lowercase RFC 4122 UUIDv4 as its
portable identity, independent of local row IDs, names, and content. The profile-local
registry is keyed by `(actor_kind, local_actor_id)` and enforces UUID uniqueness across
both actor kinds; this does not imply cross-installation coordination. Server-backed
Personas cannot receive that identity, drive Buddy, or participate in Actor Pack
workflows in place; the user must first use Save Local Copy.

Actor Pack member paths are canonical lowercase ASCII relative POSIX paths. Exports
remap local names to canonical internal names, include all declared actor, portrait,
visual, license, and provenance data, and are deterministic and self-contained. Thin
or externally referenced exports are invalid.

Persona JSON remains the authority for local Persona records in V1. Actor Pack
operations that also mutate the SQLite portable-identity registry or visual records
will use one purpose-built SQLite write-ahead intent containing bounded old/new
Persona snapshots and authority digests. The coordinator atomically replaces Persona
JSON, then uses one SQLite transaction to atomically write the new registry/visual
rows and change the intent from `prepared` to `committed`. Recovery applies this full
matrix:

- `prepared` + old JSON + old SQLite is a no-op; discard and clean the intent;
- `prepared` + new JSON + old SQLite compensates JSON to the old record, or removes a
  newly created record, then cleans the intent;
- `committed` + new JSON + new SQLite retains the new authority and finishes cleanup;
- old JSON + new SQLite, `committed` + new JSON + old SQLite, any other unexpected
  digest or revision, or any intent/store-state contradiction is quarantined without
  a destructive guess and requires explicit recovery.

Because the SQLite rows and committed status change atomically, `prepared` + new
SQLite is impossible under normal operation and is treated as a contradiction. Intent
payloads are profile-private, bounded to one actor mutation, never logged or exported,
and deleted after successful completion or recovery. This is not a general distributed
transaction framework or a broad Persona-to-SQLite migration.

Import is review-first and makes no live mutation before explicit consent. The exact
portable UUID matrix is:

- no UUID match: Create New preserves the incoming UUID; Create Copy assigns a fresh
  UUID and keeps the incoming UUID only as provenance;
- same-kind exact UUID match: Create Copy or explicitly confirmed Update Existing;
- cross-kind UUID match: reject as an identity conflict.

Hostile archives receive bounded validation and extraction into private staging;
import rejects undeclared files, external references, nested archives, linked entries,
encrypted entries, and duplicate or colliding member paths.

Update Existing preserves any local visual binding whose optional section was omitted
from the archive. Activation revalidates the reviewed actor, UUID, bindings, versions,
and staged filesystem identity; stale or ambiguous state never auto-merges. Recovery
and cleanup operate only on validated, pinned private staging or intent state and fail
closed when authority cannot be proven.

This decision adds no third-party floating-window dependency and no server
implementation. Persona Buddy will use a minimal native Textual 8 window; the pinned
server contract is a compatibility source only.

## Context

ADR-067 established Shared Visual Identity as an immutable expression runtime and
kept Persona Visual Packs separate. The programme now needs a local Persona operational
runtime, an app-wide Buddy, and a portable actor format without weakening that boundary
or confusing local database identity with portable identity.

Local Personas are authoritative in atomically replaced JSON, while portable identity
and both visual systems require SQLite records and profile-owned assets. Actor Pack
create and import therefore need a narrow crash-recovery boundary across existing
stores. Imports also cross an archive trust boundary and must preserve live actor data
until review, validation, and activation all succeed.

## Alternatives Considered

| Alternative | Why rejected |
| --- | --- |
| Merge Shared Visual Identity and Persona Visual | Expressions and operational states have different semantics, actors, bindings, fallbacks, and runtime inputs. One schema would erase the server compatibility boundary established by ADR-067. |
| Thin or reference-only Actor Packs | They are not portable, can expose private paths or remote dependencies, and cannot produce deterministic offline import. |
| Mutate server-backed Personas in place | Chatbook does not own server Persona authority or a server write contract; Save Local Copy creates the required local ownership boundary. |
| Migrate all Persona authority to SQLite as part of Actor Packs | This is unrelated, high-risk scope and would change existing Persona ownership before portability requires it. |
| Add a generic distributed transaction framework | Only bounded Persona JSON plus SQLite actor mutations need coordination; a purpose-built intent is smaller and auditable. |
| Depend on third-party `textual-window` | Its current release targets Textual versions older than Chatbook's Textual 8 runtime and provides much more desktop machinery than Buddy needs. |
| Store Persona runtime sections opaquely or defer the runtime | Actor Packs and Buddy would claim Persona visual portability without being able to validate, resolve, render, or safely activate it. |

## Consequences

- Actor Pack import/export and Buddy operate only on profile-local actors; server
  Personas require an explicit local copy.
- The portable envelope can carry both visual systems without coupling their models or
  runtimes.
- Deterministic, self-contained archives cost more space than thin references but have
  stable trust, replay, and offline behavior.
- Persona mutations spanning JSON and SQLite gain bounded recovery machinery and a
  startup quarantine state for ambiguous authority.
- Unknown required features, unsupported renderers, stale reviews, and unverifiable
  recovery state stop activation rather than being guessed or silently discarded.
- Server ingestion, server sync, a general multi-window system, and third-party window
  integration remain out of scope.

## Links

- [Approved programme design](../../Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md)
- [ADR-067: Bundle Samira through a local Visual Identity bridge](067-bundled-samira-visual-identity-pack.md)
- [TASK-19053: Add local Persona Visual pack foundation](../tasks/task-19053%20-%20Add-local-Persona-Visual-pack-foundation.md)
- [TASK-19054: Author and import Persona Visual packs](../tasks/task-19054%20-%20Author-and-import-Persona-Visual-packs.md)
- [TASK-19055: Add opt-in app-wide floating Persona Buddy](../tasks/task-19055%20-%20Add-opt-in-app-wide-floating-Persona-Buddy.md)
- [TASK-19056: Enable Shared Visual Identity for Persona actors](../tasks/task-19056%20-%20Enable-Shared-Visual-Identity-for-Persona-actors.md)
- [TASK-19057: Define and create portable Actor Packs](../tasks/task-19057%20-%20Define-and-create-portable-Actor-Packs.md)
- [TASK-19058: Export self-contained Actor Packs](../tasks/task-19058%20-%20Export-self-contained-Actor-Packs.md)
- [TASK-19059: Import, review, and activate Actor Packs](../tasks/task-19059%20-%20Import-review-and-activate-Actor-Packs.md)
