# Actor Pack, Persona Buddy, and Streaming Emote Programme Design

Date: 2026-08-20
Status: Approved for spec review
Repository: `tldw_chatbook`
Server reference: `tldw_server` dev commit
`385afa951922c8a9dc2002c675bb6cad65e4ac23`
Related decision: [ADR-067](../../../backlog/decisions/067-bundled-samira-visual-identity-pack.md)

## Summary

Deliver eight bounded capabilities that make actor visuals portable and useful at
runtime:

1. a local Persona Visual Pack foundation compatible with the server's distinct
   Persona runtime;
2. Persona Visual Pack authoring and `.tldw-persona-vpack` import;
3. an opt-in, app-wide floating Persona Buddy;
4. Shared Visual Identity reactions for Persona actors;
5. a new `.tldw-actor-pack` format and New Actor Pack workflow;
6. self-contained Actor Pack export;
7. secure Actor Pack import, review, and activation; and
8. exact server-parity streaming `Emote:` directives and durable final-expression
   metadata.

An Actor Pack contains exactly one Character or one Persona. It is a portable actor
package, not a reusable image library. It may contain the actor record and portrait,
a Shared Visual Identity reaction section, and—for Persona actors—a separate Persona
runtime visual section. The envelope unifies portability while the two visual
runtimes retain separate schemas, storage, bindings, resolvers, and state semantics.

This programme changes only Chatbook. It uses frozen compatibility fixtures from the
pinned server commit; server ingestion of `.tldw-actor-pack` is separate future work.

## User-confirmed product rules

- The portable archive is `.tldw-actor-pack`, schema `tldw.actor-pack/v1`.
- One archive contains exactly one Character or one Persona.
- A minimally valid new actor contains an actor record and portrait. Visual sections
  are optional and may be added later.
- Exports are always self-contained. Thin/reference-only exports are not supported.
- Imports enter review before any live record changes. A UUID match permits an
  explicitly confirmed update; users may instead create a copy.
- Chatbook implements the Persona Visual runtime now rather than storing its section
  opaquely or deferring it.
- Persona Buddy is app-wide, floating, and explicitly enabled. It is off by default.
- Persona Buddy uses the server's `sprite_frames` baseline and required state catalog,
  with safe custom states.
- Persona Buddy follows its explicitly selected local Buddy Persona and uses
  deterministic application-state signals, not prose or mood classification.
- V1 Buddy, Actor Pack create/export, and Actor Pack Update Existing operate on
  profile-local actors only. Server-backed Personas must first be saved as a local
  copy.
- Character reactions match the server's standalone streaming `Emote:` directive
  contract. History restores only the final expression; beat replay is deferred.
- Full-repository pytest is not required. Each task verifies the complete affected
  component surface and its cross-cutting governance gates.

## Goals

- Let users create a pack-ready Character or Persona with a stable portable identity.
- Let users export and import a complete actor without leaking local state or relying
  on external files.
- Preserve server-compatible subcontracts for both Shared Visual Identity reactions
  and Persona Visual runtime packs.
- Give Personas an app-wide animated Buddy driven by trusted application state.
- Give Characters streaming expression changes whose visible and persisted text is
  free of control directives.
- Preserve immutable visual versions, review-before-activation, rollback, and
  historical final-expression resolution.
- Reuse the existing hardened publication, cancellation, invalidation, and asset
  validation patterns rather than create parallel unsafe paths.

## Non-goals

- No server implementation or sync adapter.
- No multi-actor archive or linked companion graph.
- No reusable standalone Actor Pack visual library separate from an actor.
- No merge of Shared Visual Identity and Persona runtime schemas or resolvers.
- No model-directed Persona Buddy states.
- No Persona Visual generation provider or recipe system.
- No historic intra-message emote beat replay.
- No thin archives, external URLs, remote asset fetches, or nested archives.
- No taskbar, window switcher, snapping desktop, maximize system, or general
  multi-window framework.
- No new default Persona and no automatic enabling of Persona Buddy.
- No provider settings, credentials, chats, local database identifiers, private
  paths, or session/UI state in exported archives.
- No new third-party window dependency.

## Architectural decisions

### Separate runtimes, one portable envelope

Shared Visual Identity represents actor expressions used by character/persona chat
and immutable message appearance. Persona Visual represents operational application
states such as listening, tool execution, and approval. They remain separate:

| Concern | Shared Visual Identity | Persona Visual runtime |
| --- | --- | --- |
| Semantic state | reaction/expression | application operational state |
| Actor kinds | Character and Persona | Persona only |
| Runtime input | manual, explicit emote, mood, operational fallback | trusted app lifecycle signals |
| Versioning | immutable expression version | immutable runtime visual version |
| UI | Personas editor, Console portraits | Personas editor, floating Buddy |
| Portable section | `shared-visual-identity/` | `persona-runtime/` |

The Actor Pack manifest references the two typed sections. It never projects their
assets into one common state catalog.

### Portable actor identity

Portable update semantics require identity that is not a name, content digest, or
local database ID. Add a small profile-local registry keyed by
`(actor_kind, local_actor_id)` with a UUID portable identity. The registry:

- is created atomically with New Actor Pack actor creation;
- survives soft deletion and restoration;
- never cascades into chats or visual versions;
- is emitted in the Actor Pack manifest and actor payload;
- is rechecked at import activation; and
- assigns a fresh UUID on Create Copy while retaining the source UUID only as
  provenance.

Existing actors remain valid without registry rows. Export validates the actor and
portrait first, then asks the user to assign and persist a stable portability UUID.
A failed later archive write does not roll back that harmless identity assignment.

V1 portability is deliberately local-only. New Actor Pack creates local actors;
export accepts local Characters and local Personas; import Create New/Create Copy
creates local actors; and Update Existing targets only a local actor of the same kind
with the exact portable UUID. Server-backed Persona rows never receive registry
entries and cannot be exported or updated in place. Their UI offers `Save a local
copy first` rather than attempting a cross-source write.

### ADR boundary

Two new ADRs are required:

1. **Portable Actor Packs and local Persona Visual runtime** — governs the separate
   runtime boundary, local persistence, asset ownership, portable identity, archive
   format, import trust boundary, activation, and rollback.
2. **Durable character emote metadata** — amends ADR-067's session-only message
   boundary for server-parity directives, immutable final-expression references, and
   history reload.

The task files and later implementation plans must link the relevant ADR. ADR numbers
are allocated only when the documents are created, after a fresh all-ref collision
check.

## Persona Visual runtime

### Foundation

Add a local semantic subset of the pinned server Persona Visual contract:

- Persona Visual packs;
- immutable versions;
- version-bound assets;
- active Persona bindings;
- the pinned `sprite_frames` manifest-version-1 contract;
- `sprite_frames` renderer capability; and
- validated profile-owned asset storage.

The required catalog is:

```text
idle
wake_armed
listening
thinking
speaking
tool_running
approval_needed
error
offline
```

Safe custom state slugs are allowed. The nine built-in state identifiers are reserved.
An activatable runtime pack must resolve `idle`, `listening`, `thinking`, `speaking`,
and `error` through direct mappings or validated manifest fallback chains, matching
the pinned server contract. Other built-ins may omit a direct mapping. Runtime misses
follow the manifest fallback chain, then `idle`, then the Persona portrait with an
explicit reason.

`sprite_frames` accepts only `manifest_version: 1` in V1 and supports validated PNG,
JPEG, WebP, and GIF raster assets, timing, static fallback, and reduced-motion
behavior. A manifest-version-2 or non-sprite renderer may be inspected only far
enough to return a stable unsupported-capability result; it cannot be imported,
activated, guessed, or partially rendered.

### Workbench authoring

The Personas Workbench provides:

- all baseline state slots plus safe custom states;
- path-free metadata inventory and validation status;
- one selected lazy preview;
- Replace, Clear, Add Custom State, Save, and Cancel;
- immutable publish-once Save;
- full review-first `.tldw-persona-vpack` import; and
- explicit distinction from legacy expression-set and Actor Pack import actions.

Only the selected preview is decoded. Editing and import use the same validation and
publication boundary. Cancellation drains uncancellable thread work before staging
cleanup. An imported archive never becomes active before review and Save.

### Persona Buddy controller

Persona Buddy is disabled by default. Users enable it through Personas Workbench or
the canonical Settings screen. Geometry, open/collapsed state, and enabled state are
profile-local UI preferences and are never exported.

The app owns one explicit Buddy Persona selection, stored as the profile-local
preference `(source = local, local_persona_id)`. It does not depend on the later Actor
Pack portable-identity registry. Enabling Buddy requires choosing an eligible local
Persona. Workbench highlighting, Console session actors, and server runtime-source
switches never silently replace this selection. Selecting a different Persona is an
explicit `Use for Buddy` action. If the selected Persona is disabled, soft-deleted,
missing, or loses its local binding, the Buddy view hides and the controller reports a
stable unavailable reason while preserving the enabled preference. Restoring or
explicitly replacing that Persona re-resolves the view. With no selection, no floating
view mounts and Settings/Workbench prompts for one.

Server-backed Personas are ineligible in V1. Their Buddy and Actor Pack actions are
disabled with `Save a local copy first`; saving a local copy creates a distinct local
Persona that can then receive a portable identity and Buddy selection.

An app-owned controller stores:

- active Persona identity and binding;
- resolved runtime state and version/asset identity;
- source-scoped state leases;
- generation/currentness token;
- enabled/open/collapsed state; and
- floating-window geometry.

The controller contains no screen or widget reference. Each active primary screen
mounts a lightweight view connected to the controller. Navigation replaces the view
without resetting controller state. Splash, authentication, recovery, and modal
surfaces hide or layer above the Buddy so it cannot intercept their input.

State priority matches the pinned server resolver for trusted built-in signals:

```text
error
approval_needed
time-bounded explicit/custom state lease
authored trigger
tool_running
wake_armed, only while live state is absent or idle
trusted live voice state (idle/listening/thinking/speaking/error)
offline
idle
```

Signals are source-scoped leases, not shared booleans. Releasing one tool, speech, or
approval operation cannot clear a state still held by another operation. Active
Persona replacement, version publication, binding change, and asset change advance
the generation token and fence late decode/render work.

### Floating window

The linked `textual-window` project is an interaction reference only. Its current
release requires Textual `<6`, while Chatbook uses Textual 8.x. Chatbook implements a
minimal native window and imports none of that project's code.

The Buddy window:

- mounts with `position: absolute` and `overlay: screen` so it contributes nothing to
  parent flow or `1fr` resolution;
- starts bottom-right;
- is draggable, resizable, focusable, collapsible, and closable;
- remains within current screen bounds;
- persists geometry and re-clamps it after every viewport change;
- exposes keyboard move, resize, reset, collapse, and close actions without
  shadowing terminal-convention or existing global bindings;
- never steals focus on state changes;
- pauses animation while hidden or collapsed;
- renders a static frame in reduced-motion mode; and
- collapses to a small labelled control when its minimum geometry cannot fit.

It has no taskbar, alt-tab switcher, maximize action, snapping system, context-menu
desktop, or multi-window abstraction.

## Shared Visual Identity for Persona actors

ADR-067's local schema permits `actor_kind = 'persona'`, but current Chatbook behavior
creates, edits, resolves, and renders only Character bindings. A dedicated task must
complete the Persona path before Actor Packs claim support.

The task adds:

- Persona binding creation and replacement;
- Persona-aware actor validation and lifecycle handling;
- Personas Workbench Shared Visual Identity browser/authoring;
- resolver support with the same immutable identity and fallback guarantees;
- cache invalidation after Persona pack publication; and
- tests proving Character behavior is unchanged.

This does not merge Persona runtime states with expressions. It only makes the
already-declared Shared Visual Identity actor kind operational.

## Actor Pack format

### Layout

`.tldw-actor-pack` is a ZIP container with one root manifest:

```text
actor-pack.json
actor/
  actor.json
  portrait.<validated-raster-extension>
shared-visual-identity/       # optional
  manifest.json
  assets/...
persona-runtime/              # optional; Persona only
  manifest.json
  assets/...
licenses/
  ...
```

`actor-pack.json` uses schema `tldw.actor-pack/v1` and records:

- actor kind: `character` or `persona`;
- portable actor UUID;
- actor payload and portrait references;
- typed optional visual sections;
- producer name/version;
- license and provenance declarations;
- required feature identifiers;
- byte count and SHA-256 for every declared file; and
- a content digest over canonical manifest data.

The portrait file is authoritative. `actor/actor.json` contains no duplicate image
bytes. Actor-kind adapters populate canonical local avatar/card fields on import.
Character payloads preserve supported character-card fields; Persona payloads
preserve canonical Persona fields. Imported text is untrusted and rendered as plain
text in review.

A Character archive may contain Shared Visual Identity only. A Persona archive may
contain either or both visual sections. Archives cannot contain local IDs, external
asset references, nested archives, executable content, provider configuration,
credentials, chats, private paths, or session/UI state.

The license value may be explicit `unspecified`; the UI warns rather than inferring
ownership or clearance.

### Canonicalization and versioning

V1 archive member paths are canonical lowercase ASCII relative POSIX paths. Every
segment must match `[a-z0-9][a-z0-9._-]{0,127}`. They may not contain empty parts,
backslashes, drive prefixes, absolute paths, NULs, `.`/`..`, trailing dots/spaces,
platform device names, aliases, uppercase, or non-ASCII characters. Export remaps
local filenames to canonical actor/asset identifiers, so human display names remain
inside JSON rather than archive paths. Import rejects noncanonical paths instead of
normalizing them; this removes Unicode and case-folding differences across filesystems.

Canonical JSON uses UTF-8, sorted object keys, compact separators,
`ensure_ascii = false`, and no trailing newline. The top-level digest excludes its own
field. `actor-pack.json` is not included in its own per-file size/SHA-256 inventory.
The digest covers the remaining canonical top-level manifest plus every declared
payload/portrait/license/section file's canonical path, size, and SHA-256. Export
normalizes ZIP entry order, permissions, and timestamps for deterministic output.

Unknown required features or sections are rejected. V1 does not silently discard and
later re-export content it cannot understand.

## New Actor Pack workflow

The flow reuses canonical Character and Persona editors:

1. User chooses Character or Persona.
2. The canonical editor collects actor fields.
3. A portrait is required before pack-ready Save.
4. One transaction creates the actor and portable-identity registry row.
5. Visual sections may be authored later.
6. Export remains unavailable until the actor-pack validator passes.

This is an actor creation flow, not a parallel editor or an archive writer. ZIP export
belongs to the export task.

## Export

Export captures one consistent authority snapshot:

1. actor record and content revision;
2. portable registry identity;
3. portrait;
4. active Shared Visual Identity binding/version/assets; and
5. active Persona runtime binding/version/assets, when applicable.

Assets load only through existing validated package/profile loaders. Hashing and ZIP
writes stream through bounded buffers. A temporary file is created beside the chosen
destination, flushed/synced where supported, and atomically replaced without
following a substituted destination link.

Before archive publication, export re-reads and compares the complete authority
tuple. Any actor, UUID, binding, version, asset, or source change returns a stable
“actor changed; retry” category instead of producing a mixed snapshot.

Exports are always self-contained. A missing declared asset is an export failure, not
a thin reference.

## Import, review, and activation

### Outer archive limits

The envelope composes two separately bounded visual systems:

- maximum 4,096 entries;
- maximum 768 MiB total uncompressed bytes;
- maximum 50 MiB per member;
- maximum 100:1 per-entry and aggregate decompression ratio;
- no encrypted entries, symlinks, nested archives, duplicate/colliding paths,
  undeclared files, device names, or external references.

Section limits remain stricter:

- Shared Visual Identity: 128 assets, 256 MiB total, 25 MiB per asset;
- Persona runtime: 500 MiB total, 50 MiB per asset, plus renderer/state/frame limits.

The importer preflights free space for archive, extraction, immutable publication,
and bounded overhead. It streams every read/write and rechecks actual written bytes.

### Review flow

Import writes only to a private same-filesystem staging directory before final
consent:

1. validate ZIP metadata and names without extraction;
2. extract only declared files using no-follow operations;
3. validate top-level schema, features, digests, sizes, MIME, decode, actor payload,
   portrait, and section manifests;
4. build a path-free review model with actor fields, visual inventory, license status,
   warnings, and differences from a UUID match;
5. capture the actor revision, portable UUID, both bindings, and both active versions
   represented by the review; and
6. offer the allowed action from this exact matrix:
   - no UUID match: Create New preserves the incoming UUID, while Create Copy assigns
     a fresh UUID and records the incoming UUID only as source provenance;
   - same-kind exact UUID match: Create Copy or explicitly confirmed Update Existing;
   - cross-kind UUID match: reject as an identity conflict.

Create Copy assigns a fresh UUID and retains the source UUID only as provenance.
Update Existing is available only for an exact UUID match. It updates reviewed
portable actor fields and publishes imported visual sections as new immutable local
versions. It never overwrites local database IDs, chats, deletion state, provider
settings, credentials, or session/UI state.

Immediately before activation, the importer revalidates actor revision, UUID, both
bindings, both versions, free-space authority, and exact staged filesystem identity.
Stale review returns to review; it never auto-merges.

Activation commits actor changes, pack/version/assets, and bindings in one database
transaction around the existing staged-publication boundary. Cache/Buddy refresh
occurs only after commit. Failure preserves the prior actor and active versions.
Post-filesystem database failure exposes only an opaque internal cleanup token.

Cancellation drains uncancellable thread work before cleanup. A bounded startup sweep
removes crash-left staging only when its recognized name and pinned filesystem
identity are beneath the private Actor Pack staging root. POSIX uses descriptor-
relative no-follow operations; capability-limited platforms use a fail-closed
verified-path fallback.

### Compatibility surfaces

Existing imports remain separate and honestly labelled:

- **Import Expression Set** → legacy four-state character images;
- **Import Persona Visual Pack** → Persona runtime draft;
- **Import Actor Pack** → complete actor review and activation.

Existing formats are not silently reinterpreted as Actor Packs.

## Server-parity streaming emotes

### Directive contract

Character-chat prompts list only deterministic normalized expression slugs from the
active Shared Visual Identity version and permit standalone directives:

```text
Emote: annoyed
```

The list preserves the active version's stored asset order after normalization and
deduplication, exposes the first 25 states, and appends the exact `(+N more)` suffix
when additional states exist, matching the pinned server. It contains no imported
display labels or arbitrary text.

The streaming parser matches the server contract:

- standalone directive lines only;
- case-insensitive prefix;
- trimmed/lowercased safe slug with internal whitespace normalized;
- maximum 40-character safe state;
- maximum five accepted events;
- consecutive duplicates ignored as events;
- directives in fenced code and inline prose remain visible;
- valid, invalid, duplicate, and over-cap standalone directives are stripped;
- CRLF, arbitrary chunk boundaries, and unterminated final lines work; and
- offsets use JavaScript-compatible UTF-16 units in sanitized visible text.

The parser consumes assistant-visible text deltas only. Reasoning, tool arguments,
tool results, citations, and provider control events never enter it. Its buffer holds
only a bounded possible directive/fence prefix; ordinary long prose streams without
waiting for a newline. Cancellation discards incomplete directive candidates without
leaking them into partial display or persistence.

Python behavior is pinned by a frozen fixture corpus derived from server dev commit
`385afa951922c8a9dc2002c675bb6cad65e4ac23`.

### Live selection

Selection is phase-aware:

```text
manual override, if present
otherwise:
  pre-response       -> operational thinking
  streaming          -> operational speaking until first accepted Emote
  after Emote        -> explicit expression
  complete/no Emote  -> server-compatible heuristic final message expression
  failed             -> existing operational error behavior
  next idle          -> normal pack/default resolution
```

Manual override prevents automatic display changes but does not prevent directive
parsing or metadata persistence. A safe directive with no matching asset is accepted,
suppresses the heuristic, and records an asset-missing fallback while the live
portrait keeps its current/base image.

Persona Buddy is never driven by character emote directives.

### Persistence and history

Assistant metadata stores bounded scalar values:

- final `mood_label`;
- optional heuristic `mood_confidence` and `mood_topic`;
- at most five normalized `{state, at_char}` `emote_events`;
- actor identity;
- resolved pack and immutable version identity;
- expression and asset identity; and
- fallback reason.

Offsets are nondecreasing and within sanitized UTF-16 text length. When explicit
events exist, the last state equals final `mood_label`.

History reload restores the exact final immutable expression/asset when available.
If unavailable, it produces an explicit fallback reason. V1 retains activated
immutable versions and does not introduce physical version garbage collection.
Historic beat replay is deferred.

Parser, resolver, or asset failure never blocks or corrupts the assistant reply.
Diagnostics contain fixed categories and identifiers only—never assistant text,
prompts, paths, raw provider output, archive member names, bytes, or cleanup tokens.

## Floating Buddy and async safety

All DB, archive, hash, image decode, and frame preparation work runs off the Textual
event loop. Cancellation of `to_thread` work is shielded and drained before releasing
same-owner serialization or applying replacement work.

The app owns coordinators whose lifetimes span screen navigation. Controllers retain
immutable arguments and weak or identity-fenced view targets, not screens or widgets.
Every await is followed by authority validation before state or DOM application.

The Buddy view fences:

- active screen instance;
- controller generation;
- active Persona and binding;
- requested runtime state;
- pack/version/asset identity;
- enabled/open/collapsed state; and
- current viewport geometry.

Late views remove only themselves. State changes never mutate a stale or replaced
screen.

## Error and recovery contract

- Missing Persona runtime state: fall back to idle, then portrait.
- Missing Shared Visual Identity expression: use existing deterministic fallback.
- Unsupported renderer or required archive feature: reject before activation.
- Invalid actor payload or portrait: keep import in failed review; create nothing.
- UUID match with changed live actor: mark review stale; do not auto-merge.
- User declines update/navigation: preserve staging and current UI until explicit
  Cancel, or cancel as stated by the dialog.
- Export authority changes: leave destination untouched and ask for retry.
- Import cancellation: drain workers, discard staging, retain current actor/version.
- Database failure after filesystem publication: prior bindings remain active and
  only pinned orphan cleanup is eligible.
- Buddy frame failure: keep current/static fallback and report a path-free category.
- Emote parser/resolver failure: preserve sanitized assistant content and fall back.
- App restart: clear temporary emote/manual state, preserve immutable message metadata,
  portable identities, Buddy preference, and published versions.

## Verification strategy

The user explicitly requested touched/modified-component testing instead of a full
repository suite. Each task runs the complete affected component surface plus relevant
cross-cutting gates.

### Required evidence for every implementation task

- Born-RED tests at the actual seam, followed by GREEN.
- Mutation proof for every new authority, safety, cancellation, and precedence guard.
- Import-provenance assertion proving tests load the assigned worktree.
- Real SQLite repository/migration tests where data changes.
- Scoped Ruff, formatter, compile, and `git diff --check` results.
- Diagnostic inventory and architecture/governance gates when touched.
- Isolated HOME/XDG/config/data roots for app-importing probes and live verification.
- No unrelated full-suite claim.

### Persona Visual and Buddy

- Frozen server manifest, state, renderer, and `.tldw-persona-vpack` fixtures.
- Asset decode, dimension, frame, MIME, path, immutable version, binding, and fallback
  tests.
- State-lease concurrency, active-Persona replacement, screen-navigation, stale
  decode, missing asset, hidden/collapsed animation, and reduced-motion barriers.
- Textual Pilot at 80x24 and normal/wide dimensions.
- Painted-frame assertions, not style-only assertions.
- Real-terminal mouse and keyboard verification for drag, resize, focus, hit testing,
  modal layering, navigation, viewport resize, and geometry restore. Pilot alone does
  not certify real-terminal mouse behavior.
- Geometry tests prove `overlay: screen` contributes no flow/fr budget.
- Impeccable detector after the final visible Buddy UI change.

### Actor Pack portability

- Independent golden Actor Pack fixtures plus real export/import round trips.
- Property/adversarial tests for traversal, links, duplicate and Unicode/case path
  collisions, device names, encryption, nested archives, undeclared files,
  compression bombs, truncation, MIME mismatch, digest mismatch, unsupported required
  features, insufficient disk, and archive replacement races.
- Create, Copy, Update, cancel, stale-review, actor edit, binding change, version
  publication, and crash-left staging cleanup barriers.
- Proof that local IDs, chats, credentials, provider settings, paths, and private
  diagnostics never enter an archive or review copy.
- Deterministic archive digest and byte output on identical inputs.

### Streaming emotes

- Frozen cross-language parser vectors covering arbitrary chunk boundaries, CRLF,
  fenced code, inline prose, invalid states, 40-character boundary, five-event cap,
  duplicates, UTF-16 offsets, long lines, cancellation, and unterminated final lines.
- Streaming, non-streaming, provider/tool interleaving, manual override, missing asset,
  heuristic fallback, persistence, history reload, and resolver failure tests.
- Proof that raw directives never reach rendered text, persisted content, search, or
  exports.
- Proof that history restores final expression only and retains no beat-replay claim.

### Live isolation

Live TUI evidence uses a disposable profile with explicit HOME, XDG config/data/cache,
`TLDW_CONFIG_PATH`, and profile data directory set before importing Chatbook. Real
profile config/data fingerprints are compared before and after. No live probe uses a
scratch pytest file outside `Tests/` without recreating the suite's isolation.

## Task programme and dependencies

### Task 1 — Add local Persona Visual pack foundation

Deliver server-aligned manifest/state models, renderer capabilities, schema,
repository, immutable versions, validated assets, Persona bindings, resolver, and
profile-owned storage. The frozen contract includes all nine built-ins, five required
resolvable states, and `sprite_frames` manifest version 1. No UI authoring or Buddy.

### Task 2 — Author and import Persona Visual packs

Deliver Personas Workbench baseline/custom state editing, one lazy preview,
stage/Save/Cancel, and full review-first `.tldw-persona-vpack` import.

Depends on Task 1.

### Task 3 — Add the opt-in app-wide floating Persona Buddy

Deliver app-owned state coordinator, active-Persona resolution, minimal native
Textual 8 window, `sprite_frames`, deterministic state leases, preferences,
accessibility, responsive geometry, and live verification.

The Buddy follows only the explicit profile-local Buddy Persona preference; ordinary
Workbench, Console, or server-source selection does not retarget it.

Depends on Task 1.

### Task 4 — Enable Shared Visual Identity for Persona actors

Deliver Persona binding, authoring, resolution, fallback, publication invalidation,
and Character non-regression coverage for the existing Shared Visual Identity model.

Depends on the merged TASK-16319/ADR-067 foundation.

### Task 5 — Define and create portable Actor Packs

Deliver `tldw.actor-pack/v1`, portable identity registry, actor-kind adapters,
canonical digest/validation, and New Actor Pack creation through canonical editors.
Only local actors are eligible; server-backed Personas expose Save Local Copy first.

Depends on Tasks 1 and 4.

### Task 6 — Export self-contained Actor Packs

Deliver consistent snapshotting, both visual sections, deterministic streamed ZIP,
portable-ID assignment for eligible existing actors, authority revalidation, and
atomic destination publication.

Depends on Tasks 1, 4, and 5.

### Task 7 — Import, review, and activate Actor Packs

Deliver hostile-archive defense, private staging, path-free review, Create/Copy/Update,
authority revalidation, atomic activation, cleanup, and post-commit cache/Buddy refresh.

Depends on Tasks 1–6.

### Task 8 — Match server streaming emotes and persistence

Deliver exact directive parsing/prompting, live character portrait beats, sanitized
text, heuristic fallback, bounded message metadata, and final-expression history
restore.

Prompting preserves canonical active-version order, exposes 25 states, and reports
the hidden remainder with the pinned `(+N more)` suffix.

Depends only on the merged TASK-16319/ADR-067 foundation and may proceed independently
of Tasks 1–7.

## Backlog filing rules

- File exactly these eight implementation tasks after this design and its ADRs are
  approved.
- Re-sweep every remote ref and worktree immediately before assigning IDs; do not
  trust the Backlog CLI's proposed ID or a maximum carried from this document.
- Re-read each task file after creation because repeated `--ac` values and five-digit
  IDs have known CLI traps.
- Search branches, worktrees, and open/closed PRs again before each task begins.
- Each task receives its own implementation plan only after being moved to In
  Progress.
- Each task links this programme design and the relevant ADR.

## Design review checklist

- [x] One portable archive contains exactly one actor.
- [x] Character and Persona actor kinds are supported.
- [x] Shared Visual Identity and Persona runtime remain semantically separate.
- [x] Persona runtime storage, authoring, rendering, and Buddy surface are explicitly
  tasked.
- [x] Shared Visual Identity Persona support is explicitly tasked.
- [x] Export is self-contained and import is review-first.
- [x] Portable update identity is independent of local IDs, names, and content.
- [x] Archive and filesystem failure boundaries are fail-closed and bounded.
- [x] Floating Buddy is native Textual 8, minimal, opt-in, and app-wide.
- [x] Streaming emotes match the current pinned server behavior.
- [x] Historical beat replay, server work, sync, and provider generation remain out
  of scope.
- [x] Verification is scoped to touched components per user direction.
