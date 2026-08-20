# Actor Pack and Persona Buddy Task Filing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create two governing ADRs and exactly eight collision-free Backlog tasks for the approved Actor Pack, Persona Visual, floating Buddy, Shared Visual Identity, and streaming-emote programme.

**Architecture:** This plan creates planning artifacts only—no production code and no per-task implementation plans. The first ADR owns portable actors, separate visual runtimes, local-only authority, archive security, and Persona JSON/SQLite crash recovery; the second amends ADR-067 for durable server-parity emote metadata. Eight To Do tasks then divide the programme into independently verifiable increments with explicit dependencies.

**Tech Stack:** Markdown, Backlog.md CLI, Git plumbing, repository ADR/task conventions.

---

## Scope and file map

Create or update only these planning artifacts:

- Create: `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md`
- Create: `backlog/decisions/075-durable-character-emote-metadata.md`
- Create: `backlog/tasks/task-19053 - Add-local-Persona-Visual-pack-foundation.md`
- Create: `backlog/tasks/task-19054 - Author-and-import-Persona-Visual-packs.md`
- Create: `backlog/tasks/task-19055 - Add-opt-in-app-wide-floating-Persona-Buddy.md`
- Create: `backlog/tasks/task-19056 - Enable-Shared-Visual-Identity-for-Persona-actors.md`
- Create: `backlog/tasks/task-19057 - Define-and-create-portable-Actor-Packs.md`
- Create: `backlog/tasks/task-19058 - Export-self-contained-Actor-Packs.md`
- Create: `backlog/tasks/task-19059 - Import-review-and-activate-Actor-Packs.md`
- Create: `backlog/tasks/task-19060 - Match-server-streaming-emotes-and-persistence.md`
- Modify: `Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md`
- Modify: this plan only to replace the allocation table placeholders after the final collision sweep.

Do not create production, migration, UI, CSS, fixture, or test files. Do not add an `## Implementation Plan` or `## Implementation Notes` section to a Backlog task while it remains To Do; AGENTS.md requires the task to move In Progress first.

## Final allocation

The initial local/remote sweep on 2026-08-20 after `git fetch --all --prune` reported task ceiling `19048` and ADR ceiling `073`. A follow-up all-reachable-history path scan found task IDs through `19051`, and the registered-worktree scan found `19052`; the task allocation is strictly above `19052`. The Backlog CLI allocator probe proposed unsafe `18912`, so it was removed and not used. The following IDs are allocated strictly above the swept maxima.

| Symbol | Allocated value | Title |
| --- | ---: | --- |
| `ADR_PACK_ID` | `074` | Portable Actor Packs and local Persona Visual runtime |
| `ADR_EMOTE_ID` | `075` | Durable character emote metadata |
| `TASK_PERSONA_FOUNDATION_ID` | `19053` | Add local Persona Visual pack foundation |
| `TASK_PERSONA_AUTHORING_ID` | `19054` | Author and import Persona Visual packs |
| `TASK_BUDDY_ID` | `19055` | Add opt-in app-wide floating Persona Buddy |
| `TASK_PERSONA_SVI_ID` | `19056` | Enable Shared Visual Identity for Persona actors |
| `TASK_ACTOR_FOUNDATION_ID` | `19057` | Define and create portable Actor Packs |
| `TASK_ACTOR_EXPORT_ID` | `19058` | Export self-contained Actor Packs |
| `TASK_ACTOR_IMPORT_ID` | `19059` | Import, review, and activate Actor Packs |
| `TASK_EMOTE_ID` | `19060` | Match server streaming emotes and persistence |

### Task 1: Allocate collision-free ADR and task IDs

**Files:**
- Modify: `Docs/superpowers/plans/2026-08-20-actor-pack-persona-buddy-task-filing-plan.md`
- Inspect: every `backlog/tasks/` and `backlog/decisions/` tree in local heads, remote refs, and worktrees

- [x] **Step 1: Refresh remote refs**

Run:

```bash
git fetch --all --prune
```

Expected: fetch succeeds; no working-tree files change.

- [x] **Step 2: Re-sweep task IDs across every ref and worktree**

Run the numeric, NUL-safe sweep from `backlog/docs/lessons-backlog-hygiene.md`. Include both `refs/remotes` and `refs/heads`, then inspect every path from `git worktree list --porcelain`. Never assign zsh's special lowercase `path` variable.

Expected: one numeric maximum from refs and one from worktrees. Allocate eight consecutive IDs strictly above the larger maximum.

- [x] **Step 3: Re-sweep ADR IDs and existing programme titles**

Run the equivalent NUL-safe scan for `backlog/decisions/<number>-*.md`. Search all ref task filenames and the current tree for `Actor Pack`, `Persona Visual`, `Persona Buddy`, `Shared Visual Identity`, and `streaming emote` duplicates.

Expected: two unused ADR numbers above the larger all-ref/all-worktree ADR ceiling; no task already covers an approved programme increment.

- [x] **Step 4: Probe the Backlog CLI allocator without trusting it**

Create one unmistakable throwaway task with `backlog task create`, read the generated filename, and remove that probe before any real task is filed. If the CLI's proposed ID is not above the swept maximum, keep the swept allocation and use the generated task only as a formatting template.

Expected: no probe file remains in `backlog/tasks/`.

- [x] **Step 5: Record the final allocation**

Replace the provisional values in this plan's allocation table with the freshly
allocated values, then replace every corresponding dynamic ADR and task ID symbol
throughout this plan's paths, dependencies, commands, and link instructions. Use a
whole-document search to prove no allocation symbol remains. Do not carry any “next
safe ID” beyond this filing session.

- [x] **Step 6: Verify allocation uniqueness**

Run whole-repo and all-ref searches for every chosen ADR/task ID before creating files.

Expected: zero pre-existing matches for all ten chosen IDs.

### Task 2: Write the portable Actor Pack and Persona Visual ADR

**Files:**
- Create: `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md`
- Reference: `Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md`
- Reference: `backlog/decisions/067-bundled-samira-visual-identity-pack.md`

- [x] **Step 1: Create the ADR with status Proposed**

Write the canonical ADR headings `Decision`, `Context`, `Alternatives Considered`, `Consequences`, and `Links`. The Decision must pin:

- one `.tldw-actor-pack` contains exactly one local Character or local Persona;
- Shared Visual Identity and Persona Visual remain separate schemas/runtimes inside one portable envelope;
- local Persona Visual uses the pinned server `sprite_frames` manifest-version-1 contract, nine built-ins, and five required resolvable states;
- Buddy follows an explicit local Persona preference and is opt-in/default-off;
- portable identity uses a globally unique canonical UUIDv4 independent of local row IDs;
- server-backed Personas require Save Local Copy before Buddy or Actor Pack use;
- Actor Pack internal paths are canonical lowercase ASCII and exports are self-contained/deterministic;
- Persona JSON remains authoritative in V1; a purpose-built SQLite write-ahead intent coordinates crash-safe Persona JSON plus SQLite registry/visual mutations;
- review-first import, exact UUID choice matrix, omitted-section preservation, and fail-closed recovery are mandatory; and
- no third-party floating-window dependency or server implementation is included.

- [x] **Step 2: Record rejected alternatives**

Include at least: merging the two visual runtimes, thin/reference archives, server-backed in-place writes, broad Persona-to-SQLite migration as incidental scope, a generic distributed transaction framework, third-party `textual-window`, and opaque/deferred Persona runtime storage.

- [x] **Step 3: Link the spec, ADR-067, and Tasks 1–7**

Use the final allocated task IDs. Links must be relative and render without Markdown hard-break trailing spaces.

- [x] **Step 4: Validate the ADR**

Run:

```bash
git diff --check
rg -n "unresolved allocation marker" backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
```

Expected: diff check passes; placeholder search returns no matches.

- [x] **Step 5: Commit the ADR**

```bash
git add backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
git commit -m "docs: decide portable actor pack architecture"
```

### Task 3: Write the durable character emote ADR

**Files:**
- Create: `backlog/decisions/075-durable-character-emote-metadata.md`
- Reference: `backlog/decisions/067-bundled-samira-visual-identity-pack.md`
- Reference: `Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md`

- [x] **Step 1: Create the ADR with status Proposed**

The Decision must pin:

- exact compatibility with `tldw_server` dev commit `385afa951922c8a9dc2002c675bb6cad65e4ac23`;
- standalone, stripped `Emote: <state>` directives with five-event cap and UTF-16 offsets;
- 25-state prompt inventory in stored order with exact ` (+N more)` suffix;
- assistant-visible text only as parser input;
- manual-display override without suppressing parsing/persistence;
- explicit-emote precedence over heuristic fallback;
- bounded durable `mood_label` and `emote_events` metadata referencing immutable visual identity; and
- history restores the final expression only, not historical beat replay.

- [x] **Step 2: Record rejected alternatives**

Include: final-emote-only parsing, tool call first, Persona Buddy state reuse, heuristic plus explicit emote together, raw directive persistence, and beat replay in V1.

- [x] **Step 3: Link ADR-067, the programme spec, and Task 8**

State explicitly that this ADR amends ADR-067's session-only message boundary without merging Persona Visual operational states into character expressions.

- [x] **Step 4: Validate and commit**

Run `git diff --check`, placeholder search, and then:

```bash
git add backlog/decisions/075-durable-character-emote-metadata.md
git commit -m "docs: decide durable character emote metadata"
```

### Task 4: File Task 1 — local Persona Visual foundation

**Files:**
- Create: `backlog/tasks/task-19053 - Add-local-Persona-Visual-pack-foundation.md`

- [x] **Step 1: Create the task through Backlog.md**

Use `backlog task create` with status `To Do`, priority `high`, separate repeated `--ac` flags, and references to the programme spec, the portable-pack ADR, and pinned server commit. If the CLI emits a different unsafe ID, preserve its rendered structure but move the content to the allocated collision-free filename and correct frontmatter before any commit.

- [x] **Step 2: Set the description**

Use this outcome-focused description:

> Add a profile-local Persona Visual runtime compatible with the pinned server's sprite-frame contract so local Personas can own immutable operational-state visuals without merging them into Shared Visual Identity reactions.

- [x] **Step 3: Add acceptance criteria**

Add these as separate criteria:

1. Local persistence supports Persona Visual packs, immutable versions, assets, and one active binding per eligible local Persona without changing existing Persona records.
2. Validation matches the pinned server contract: manifest version 1 for `sprite_frames`, nine reserved built-ins including `wake_armed`, five required resolvable states, bounded safe custom states, fallback chains, frames, regions, timing, authored triggers, validated static fallback selection, and reduced-motion rendering that stops animation.
3. Activatable packs resolve `idle`, `listening`, `thinking`, `speaking`, and `error`; runtime misses fall through validated manifest fallbacks, then `idle`, then Persona portrait with a stable reason.
4. Assets use validated profile-owned storage, MIME/decode/dimension/frame budgets, and immutable full-identity cache keys; public resolver/result objects, user-facing errors, logs, and diagnostic inventory expose stable identifiers and reasons only, never private paths.
5. Repository and publication paths enforce optimistic binding/version authority, rollback, and pinned orphan cleanup, and return stable old/new full identities for later targeted consumer invalidation.
6. Frozen fixtures derived from server commit `385afa...` pin supported and unsupported renderer/manifest behavior.
7. No Workbench authoring UI, floating Buddy, provider generation, or server write path is introduced.
8. Focused migration/repository/validator/asset/resolver/publication tests plus scoped static and diagnostic-governance checks pass in an isolated profile.

- [x] **Step 4: Verify and commit**

Re-read the file directly and through `backlog task list --plain`; confirm eight independent unchecked ACs, empty dependencies, no Implementation Plan/Notes, and exact references. Commit only this task file.

### Task 5: File Task 2 — Persona Visual authoring and import

**Files:**
- Create: `backlog/tasks/task-19054 - Author-and-import-Persona-Visual-packs.md`

- [x] **Step 1: Create the task with dependency on Task 1**

Status `To Do`, priority `high`, dependency `TASK-19053`, references to the spec and portable-pack ADR.

- [x] **Step 2: Set the description**

> Let users review, edit, import, stage, and explicitly publish Persona Visual packs for local Personas while keeping active runtime visuals unchanged until Save.

- [x] **Step 3: Add acceptance criteria**

1. Personas Workbench shows all nine baseline state slots, bounded safe custom states, path-free validation inventory, and one selected lazy preview for an eligible local Persona.
2. Replace, Clear, Add Custom State, and import mutate only an isolated draft; Save revalidates Persona, binding, and draft authority, publishes exactly one immutable version, then invalidates both stable old/new full identities while preserving unrelated cache entries; failed or cancelled publication invalidates nothing; Cancel discards the draft and leaves authoritative metadata unchanged.
3. `.tldw-persona-vpack` import validates full pinned sprite-frame archives into review drafts in bounded private staging and never activates before explicit Save; it rejects traversal, links, nested/encrypted archives, undeclared/external files, duplicate/colliding paths, bomb/budget violations, MIME/digest mismatches, and archive replacement races; failure or cancellation removes only identity-pinned staging and never changes active authority.
4. Unsupported renderer/manifest capabilities, malformed assets, stale Persona/binding/session authority, and import cancellation fail closed without changing the active version.
5. Server-backed Personas show Save Local Copy first; legacy expression-set and Actor Pack import remain separate, honestly labelled actions.
6. Preview inventory/resolve/decode work is screen-owned, serialized across navigation, drained on cancellation, weak-targeted, and fenced after every await.
7. No image-generation provider, recipe workflow, Shared Visual Identity merge, or Buddy window is added.
8. Labelled actions are keyboard-operable, preserve focus, and add no forbidden bindings; compact and normal layouts paint usable controls; untrusted archive text renders as plain text; user-facing errors, logs, and diagnostics remain path-free; focused widget/screen/race/import/publication tests pass, followed by Impeccable review after the final visible change and scoped static/governance gates.

- [x] **Step 4: Verify and commit**

Confirm dependency and eight ACs through direct read plus `backlog task list --plain`; commit only this task file.

### Task 6: File Task 3 — floating Persona Buddy

**Files:**
- Create: `backlog/tasks/task-19055 - Add-opt-in-app-wide-floating-Persona-Buddy.md`

- [x] **Step 1: Create the task with dependency on Task 1**

Status `To Do`, priority `high`, dependency `TASK-19053`, references
to the programme spec and portable-pack ADR.

- [x] **Step 2: Set the description**

> Give users an explicitly enabled, app-wide floating visual companion for one selected local Persona, driven only by trusted application lifecycle state.

- [x] **Step 3: Add acceptance criteria**

1. Buddy is default-off and mounts only after the user explicitly selects an eligible local Persona, persisted profile-locally as `(source = local, local_persona_id)`; Workbench highlight, Console actor, and server-source changes never silently retarget it. If the selected Persona is disabled, soft-deleted, missing, or loses its local binding, the view hides with a stable path-free unavailable reason while the enabled preference remains; restoring or explicitly replacing the Persona re-resolves the view, and no selection mounts nothing.
2. An app-owned controller survives screen navigation without retaining screen/widget references and resolves the pinned state priority, all nine built-ins, source-scoped leases, safe custom triggers, and exact Persona/binding/version identity.
3. A native Textual 8 floating view is bottom-right by default, draggable, resizable, focusable, collapsible, closable, bounded to the viewport, and never steals focus on state changes; it provides keyboard move, resize, reset, collapse, and close actions without shadowing terminal-convention, reserved, or existing global bindings, and collapses to a labelled compact control when its minimum geometry cannot fit.
4. Geometry/enabled/open/collapsed preferences persist profile-locally and are never exported; geometry re-clamps after every viewport change, and splash/auth/recovery/modal surfaces safely hide or cover the Buddy so it cannot intercept input behind them.
5. `sprite_frames` animation pauses while hidden/collapsed, respects reduced motion, and falls back through state, idle, and portrait without blanking the UI; frame and availability failures report stable path-free categories.
6. Same-owner Buddy work is serialized across replacement screens; DB, resolve, decode, and frame-preparation work runs off the event loop, uncancellable work is shielded and drained before releasing serialization, view targets are weak or identity-fenced, and authority is revalidated after every await. Stale work and replaced views cannot repaint or remove the current view.
7. No third-party window dependency, taskbar, snapping desktop, maximize system, model-directed state, or default Persona is introduced.
8. Production-shaped Pilot tests cover normal, wide, and 80x24 layouts, compositor output, and zero flow/`fr` budget; isolated real-terminal verification covers mouse drag/resize, keyboard controls, focus, modal hit testing, navigation, viewport resize, and geometry restore. Impeccable review follows the final visible change; scoped Ruff, format, compile, diff, and static checks pass with mutation evidence for authority, lease, and cancellation guards.

- [x] **Step 4: Verify and commit**

Confirm eight ACs, one dependency, and no Implementation Plan/Notes; commit only this task file.

### Task 7: File Task 4 — Shared Visual Identity for Personas

**Files:**
- Create: `backlog/tasks/task-19056 - Enable-Shared-Visual-Identity-for-Persona-actors.md`

- [x] **Step 1: Create the task against the merged ADR-067 foundation**

Status `To Do`, priority `high`, dependency `TASK-16319`, references to ADR-067, the programme spec, and the portable-pack ADR.

- [x] **Step 2: Set the description**

> Complete the already-declared Persona actor path in Shared Visual Identity so local Personas can own reaction/expression packs without merging those expressions into Persona Buddy operational states.

- [x] **Step 3: Add acceptance criteria**

1. Eligible local Personas, identified by exact local source/id plus current profile/editor revision, can create, replace, clear, publish, and resolve Shared Visual Identity bindings using the existing immutable pack/version model; inactive, disabled, deleted, or missing Personas cannot author, publish, or render. Restore, replacement, or concurrent update re-resolves authority, and stale authority cannot mutate active state.
2. Personas Workbench exposes path-free Shared Visual Identity metadata, lazy selected preview, staged edits, visible manual labels where applicable, Save, and Cancel with full session/actor/binding fences; declining dirty navigation preserves the draft and staging, while accepted navigation or Cancel signals and drains in-flight work, discards only the unpublished candidate/draft, and preserves the active version.
3. Persona resolution uses exact full actor and cache identities, deterministic fallback, targeted actor invalidation, and source-only change detection.
4. Console/persona-chat consumers render the active Persona expression without giving Persona Buddy operational states any reaction semantics.
5. Server-backed Personas require Save Local Copy first; source, session, actor, binding, version, and profile-revision authority is revalidated after every await, and any stale change fails closed without publication or repaint.
6. Existing Character creation, authoring, Console rendering, publication, cache, and four-state operational behavior remain unchanged.
7. No schema/runtime merge with Persona Visual, Actor Pack archive workflow, or server write path is introduced.
8. Labelled actions are keyboard-operable and compact and normal layouts paint usable controls; user-facing errors, logs, and diagnostics remain path-free; focused real SQLite repository/resolver/Workbench/Console/race/invalidation/lifecycle tests pass in an isolated profile, with mutation proof for authority, cancellation, and invalidation guards plus scoped static and ADR-067 architecture/privacy/governance checks.

- [x] **Step 4: Verify and commit**

Confirm dependency on `TASK-16319`, eight ACs, and exact ADR links; commit only this task file.

### Task 8: File Task 5 — Actor Pack format, identity, and creation

**Files:**
- Create: `backlog/tasks/task-19057 - Define-and-create-portable-Actor-Packs.md`

- [x] **Step 1: Create the independent foundation task**

Status `To Do`, priority `high`, no dependency on programme Tasks 1–4. Reference the portable-pack ADR and programme spec.

- [x] **Step 2: Set the description**

> Define a secure, deterministic one-actor portable envelope and let users create pack-ready local Characters or Personas with a required portrait and stable portable identity.

- [x] **Step 3: Add acceptance criteria**

1. `tldw.actor-pack/v1` defines exactly one local Character or Persona, required canonical actor JSON and portrait, optional typed visual sections, license/provenance declarations, required features, and no local IDs or external references.
2. Internal paths, canonical JSON, per-file SHA-256/size inventory, non-self-referential top digest, deterministic ZIP metadata, and all actor/manifest/portrait limits match the approved spec.
3. The profile-local registry is keyed by `(actor_kind, local_actor_id)`, stores a globally unique canonical lowercase RFC 4122 UUIDv4 as portable identity independent of names, content, and local IDs, enforces UUID uniqueness across both actor kinds without claiming cross-install coordination, survives soft deletion/restoration, and records copy provenance without reusing the source UUID.
4. New Actor Pack uses the canonical local Character/Persona editors, admits only one operation at a time and rejects duplicate submits, requires a portrait, and fences source, editor, and portrait authority. Cancellation or declined navigation during portrait or commit work signals and drains owned work and leaves no actor, registry row, intent, or staged portrait; success creates only the actor plus portable identity, without writing an archive or requiring visual sections.
5. Server-backed Personas cannot receive portable registry rows and expose Save Local Copy first.
6. Persona actor/registry changes use a bounded profile-private intent, durably written before the atomic Persona JSON replace and one SQLite registry commit; ordinary errors compensate, including atomically removing a newly created Persona. Before affected Personas or Actor Pack surfaces become available, startup recovery idempotently cleans up old-JSON/old-SQLite no-ops, compensates new-JSON/old-SQLite state, retains committed new JSON and finishes cleanup, and quarantines any unexpected authority—including old-JSON/new-SQLite as third authority—until explicit recovery. Intents are never logged or exported; Character changes remain one SQLite transaction.
7. Unknown required features, malformed/colliding paths, invalid actor kinds/payloads/portraits, concurrent registry assignment or UUID collision, and stale profile, editor, or portrait authority fail closed with no partial actor, registry row, intent, staged portrait, or other residue.
8. This task is scoped to Actor Pack format, schema, canonicalization, digest, and pure-validator contracts plus actor creation and the Persona cross-store coordinator; export writer, import reader, extraction, staging, review, and activation implementation are absent and reserved for TASK-19058 and TASK-19059. Verification includes born-RED→GREEN tests, mutation proof for authority, safety, cancellation, and recovery guards, assigned-worktree provenance, real SQLite migration and crash-recovery tests in an isolated profile, and scoped Ruff, format, compile, diff, diagnostic, privacy, architecture, and governance checks.

- [x] **Step 4: Verify and commit**

Confirm empty dependencies, eight ACs, and no premature export/import implementation plan; commit only this task file.

### Task 9: File Task 6 — Actor Pack export

**Files:**
- Create: `backlog/tasks/task-19058 - Export-self-contained-Actor-Packs.md`

- [x] **Step 1: Create the task with Tasks 1, 4, and 5 dependencies**

Status `To Do`, priority `high`, dependencies `TASK-19053`, `TASK-19056`, and `TASK-19057`.
Add references to the programme spec and portable-pack ADR.

- [x] **Step 2: Set the description**

> Let users export one eligible local Character or Persona as a deterministic, self-contained Actor Pack whose actor, portrait, and active visual versions come from one consistent authority snapshot.

- [x] **Step 3: Add acceptance criteria**

1. Export validates an eligible local Character or Persona and its portrait before assigning a missing portable UUID; one-time assignment is durable and harmless and remains assigned if later archive writing or publication fails, while server-backed Personas remain disabled.
2. The snapshot captures and, after every await and immediately before publication, revalidates exact local source/profile identity, actor revision, portable UUID, portrait, active visual bindings/versions/assets, canonical content digests, and pinned source filesystem identity.
3. Every included visual section is self-contained and preserves its typed manifest/license/provenance; a missing declared asset fails rather than emitting a thin reference.
4. Export consumes TASK-19057 canonical JSON, canonical path, inventory, `actor-pack.json` self-exclusion, and non-self-referential digest contracts; output uses `ZIP_STORED`, fixed metadata/order, bounded streaming, and byte-identical bytes for identical canonical inputs, with archive, hash, decode, and file work off the event loop.
5. Publication uses a same-directory temporary file, file fsync then atomic replacement then parent-directory fsync where supported, no-follow pinned identities, and a capability-limited verified fail-closed fallback; cancellation shields and drains uncancellable work before cleanup or serialization release, removes only the owned temporary file, and leaves the destination untouched on stale authority, failure, or cancellation.
6. Local IDs, chats, deletion state, provider settings, credentials, paths, session/UI preferences, and private diagnostics never enter the archive.
7. Real export-to-independent-pure-validator/readback round trips, without import activation, cover minimal actor+portrait, Character, Persona, and both-visual-section exports alongside independent golden deterministic byte and digest oracles.
8. Verification includes born-RED-to-GREEN evidence, mutation proof for authority, path, cancellation, and privacy guards, assigned-worktree provenance, isolated HOME/XDG/config/data roots, focused race/package/licence/privacy tests, scoped Ruff/format/compile/diff checks, and diagnostic/privacy/architecture/governance gates.

- [x] **Step 4: Verify and commit**

Confirm the three exact dependencies and eight ACs; commit only this task file.

### Task 10: File Task 7 — Actor Pack import and activation

**Files:**
- Create: `backlog/tasks/task-19059 - Import-review-and-activate-Actor-Packs.md`

- [x] **Step 1: Create the task with Tasks 1, 3, 4, 5, and 6 dependencies**

Status `To Do`, priority `high`, dependencies `TASK-19053`, `TASK-19055`, `TASK-19056`, `TASK-19057`, and `TASK-19058`.
Add references to the programme spec and portable-pack ADR.

- [x] **Step 2: Set the description**

> Let users safely inspect and activate an untrusted Actor Pack as a new local actor, a copy, or an explicitly confirmed update without risking existing actor data or visual bindings.

- [x] **Step 3: Add acceptance criteria**

1. Import enforces all outer/member/section budgets, canonical paths, declared-file and digest integrity, MIME/decode limits, and free-space preflight before extraction into pinned private staging; symlinks, hardlinks, other linked entries, encryption, nesting, devices, external references, undeclared files, and duplicate, Unicode, case, device, or alias path collisions are rejected.
2. Review remains path-free and shows actor fields, portrait, visual inventory, license/provenance, warnings, UUID match, differences, and the exact effect of every activation choice; all untrusted actor, license, provenance, and archive text renders as plain text, and review actions are labelled, keyboard-operable, focus-safe, usable in compact and normal layouts, and bind no forbidden terminal-convention, reserved, or global keys.
3. With no UUID match, Create New preserves the incoming UUID and Create Copy assigns a fresh UUID; with a same-kind exact match, Create Copy or explicitly confirmed Update Existing is offered; cross-kind reuse is rejected.
4. Update Existing changes only reviewed portable actor fields and present visual sections; every omitted optional section visibly preserves its current local binding.
5. Review snapshots the profile and source identity, actor kind and revision, portable UUID and registry row, both bindings and active versions, staged-file inode/digest identity, and free-space authority; all are revalidated immediately before activation, and any delete/recreate or revision ABA returns to review without auto-merge.
6. Character activation is transactional; Persona activation consumes the cross-store coordinator; failure/cancellation preserves prior actor/bindings, drains workers, and exposes only opaque pinned cleanup eligibility.
7. After commit, affected-only invalidation and refresh run independently for Shared Visual Identity caches, Persona runtime, mounted Buddy, and authoritative review/editor consumers; one consumer failure reports a fixed path-free category without suppressing the others or rolling back the committed activation.
8. Verification includes born-RED-to-GREEN evidence; mutation proof for authority, archive, cancellation, cleanup, and invalidation-isolation guards; real SQLite migration and crash-recovery tests; assigned-worktree provenance; isolated HOME/XDG/config/data roots; independent golden round trips and adversarial traversal/link/collision/bomb/truncation/digest/MIME/disk/race/crash/cleanup tests; Pilot coverage at normal and 80x24 geometry plus isolated real-terminal keyboard confirm/cancel/focus checks; Impeccable review after the final visible UI change; and scoped Ruff, format, compile, diff, diagnostic, privacy, architecture, and governance gates.

- [x] **Step 4: Verify and commit**

Confirm five exact dependencies and eight ACs; commit only this task file.

### Task 11: File Task 8 — server-parity streaming emotes

**Files:**
- Create: `backlog/tasks/task-19060 - Match-server-streaming-emotes-and-persistence.md`

- [x] **Step 1: Create the independent emote task**

Status `To Do`, priority `high`, dependency `TASK-16319`, references to ADR-067, the durable-emote ADR, the programme spec, and pinned server commit.

- [x] **Step 2: Set the description**

> Match the server's explicit streaming character-emote behavior so reaction directives drive live portraits while remaining absent from visible and persisted assistant text, with durable final-expression history restore.

- [x] **Step 3: Add acceptance criteria**

1. Streaming and non-streaming character responses parse only assistant-visible text lines outside fenced code that match the standalone, case-insensitive `Emote: <state>` form; accepted states are trimmed and lowercased with internal whitespace replaced by hyphens, match `[a-z0-9][a-z0-9_-]{0,39}`, are capped at five events with consecutive accepted duplicates suppressed, and use pinned CRLF, arbitrary-chunk, unterminated-final-line, and JavaScript-compatible UTF-16 offset behavior. The stream buffer retains only a bounded possible directive/fence prefix, ordinary long prose is released immediately without waiting for a newline, and cancellation discards incomplete candidates with zero rendered or persisted leakage.
2. Valid, invalid, consecutive-duplicate, and over-cap standalone directive lines never reach rendered text, persisted content, search, or exports, while fenced directives and inline prose remain visible.
3. Character prompts deterministically project safe slugs from canonical expression keys in the active Shared Visual Identity version; invalid, ambiguous, colliding, and non-round-tripping projections are omitted, remaining slugs are deduplicated in first stored asset order, the first 25 are exposed with the exact ` (+N more)` suffix when states remain, and imported labels or arbitrary display text are excluded.
4. Live portrait precedence remains manual override, then operational thinking/speaking until the first accepted explicit event, then explicit expression; every accepted explicit event updates the live expression immediately in stream order, and the last accepted state becomes the persisted final expression and mood label; manual display choice suppresses automatic display changes but not parsing or persistence, the heuristic runs only when no explicit event exists, and an accepted state with no asset keeps the current or base portrait with a stable fallback reason.
5. Assistant metadata durably stores bounded final mood fields, at most five normalized `{state, at_char}` events, actor identity, immutable pack/version/expression/asset identity, and fallback reason; every offset is a nonnegative integer, offsets are nondecreasing and bounded by sanitized-text length in JavaScript-compatible UTF-16 units, references are immutable profile-local identities rather than server IDs, and no sync or server transport is authorized. Outside the bounded event records, durable visual metadata is bounded scalar metadata only; it excludes raw directives, assistant text, prompts, provider payloads, local paths, and manual overrides, and malformed metadata fails soft on load.
6. Activated immutable visual versions are retained with no physical version garbage collection; history restores only the exact final immutable expression when available, reports a deterministic fallback otherwise, and never replays historical intra-message beats.
7. Reasoning, tool arguments or results, citations, provider controls, Persona Buddy, and raw non-visible inputs never enter directive parsing or state control; parser, resolver, and asset failures never block or corrupt the sanitized assistant reply and produce a deterministic fixed-category fallback; diagnostics use fixed categories and identifiers and exclude assistant content, prompts, local paths, raw provider output, archive member names, bytes, and cleanup tokens.
8. Frozen cross-language vectors and focused streaming, non-streaming, provider-tool, manual, missing-asset, persistence, history, failure, and real SQLite repository and migration tests covering durable fields and reload provide born-RED-to-GREEN evidence, mutation proof for authority, precedence, cancellation, and persistence guards, assigned-worktree provenance, isolated HOME/XDG/config/data roots, and scoped Ruff, format, compile, diff, diagnostic, privacy, architecture, and governance gates.

- [x] **Step 4: Verify and commit**

Confirm dependency on `TASK-16319`, eight ACs, and exact durable-emote ADR link; commit only this task file.

### Task 12: Link, validate, and commit programme filing

**Files:**
- Modify: `Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md`
- Modify: the two new ADRs if final task links require correction
- Inspect: all eight new task files

- [x] **Step 1: Mark the programme spec approved and add exact ADR/task links**

Change `Status: Approved for spec review` to `Status: Approved`. Replace the allocation-language ADR placeholders with links to both created ADRs and add a compact task table containing all eight final IDs/titles/dependencies.

- [x] **Step 2: Re-read every task from source**

Because the current Backlog CLI cannot reliably address five-digit IDs, use direct file reads as authority and use `backlog task list --plain` only as a secondary parse check. Confirm each file has the correct title, To Do status, priority, dependencies, description, separate unchecked ACs, references, and no Implementation Plan/Notes.

- [x] **Step 3: Run collision and dangling-placeholder gates**

Re-fetch and repeat the bounded, path-aware all-reachable-ref and registered-worktree
ID scan. Run whole-repo searches for all ten IDs, duplicate titles, and unresolved
allocation markers. If another ref or worktree claimed an allocated path, stop and
report the collision; do not renumber independently.

- [x] **Step 4: Run Markdown and Backlog integrity checks**

Run:

```bash
git diff --check
backlog task list --plain
git status --short
```

Expected: whitespace check passes. Direct reads remain authoritative for all eight
five-digit tasks; if `backlog task list --plain` omits those rows, record that known
CLI limitation instead of treating it as missing source data. Status contains only
the plan and approved spec modified by this final filing step.

- [x] **Step 5: Self-review scope and dependencies**

Verify there is no production code, test, CSS, generated manifest, task status change, assignee claim, implementation plan, or implementation note. Verify Task 5 can proceed independently; Tasks 2/3 depend on Task 1; Task 4 and Task 8 depend on TASK-16319; Task 6 depends on Tasks 1/4/5; Task 7 depends on Tasks 1/3/4/5/6.

- [x] **Step 6: Commit final links and plan record**

```bash
git add Docs/superpowers/plans/2026-08-20-actor-pack-persona-buddy-task-filing-plan.md Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md backlog/decisions backlog/tasks
git diff --cached --check
git commit -m "docs: file actor pack and Persona Buddy programme"
```

Expected: commit succeeds and the worktree is clean.
