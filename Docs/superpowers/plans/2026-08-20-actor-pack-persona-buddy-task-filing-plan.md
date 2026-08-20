# Actor Pack and Persona Buddy Task Filing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create two governing ADRs and exactly eight collision-free Backlog tasks for the approved Actor Pack, Persona Visual, floating Buddy, Shared Visual Identity, and streaming-emote programme.

**Architecture:** This plan creates planning artifacts only—no production code and no per-task implementation plans. The first ADR owns portable actors, separate visual runtimes, local-only authority, archive security, and Persona JSON/SQLite crash recovery; the second amends ADR-067 for durable server-parity emote metadata. Eight To Do tasks then divide the programme into independently verifiable increments with explicit dependencies.

**Tech Stack:** Markdown, Backlog.md CLI, Git plumbing, repository ADR/task conventions.

---

## Scope and file map

Create or update only these planning artifacts:

- Create: `backlog/decisions/<ADR_PACK_ID>-portable-actor-packs-and-local-persona-visual-runtime.md`
- Create: `backlog/decisions/<ADR_EMOTE_ID>-durable-character-emote-metadata.md`
- Create: `backlog/tasks/task-<TASK_PERSONA_FOUNDATION_ID> - Add-local-Persona-Visual-pack-foundation.md`
- Create: `backlog/tasks/task-<TASK_PERSONA_AUTHORING_ID> - Author-and-import-Persona-Visual-packs.md`
- Create: `backlog/tasks/task-<TASK_BUDDY_ID> - Add-opt-in-app-wide-floating-Persona-Buddy.md`
- Create: `backlog/tasks/task-<TASK_PERSONA_SVI_ID> - Enable-Shared-Visual-Identity-for-Persona-actors.md`
- Create: `backlog/tasks/task-<TASK_ACTOR_FOUNDATION_ID> - Define-and-create-portable-Actor-Packs.md`
- Create: `backlog/tasks/task-<TASK_ACTOR_EXPORT_ID> - Export-self-contained-Actor-Packs.md`
- Create: `backlog/tasks/task-<TASK_ACTOR_IMPORT_ID> - Import-review-and-activate-Actor-Packs.md`
- Create: `backlog/tasks/task-<TASK_EMOTE_ID> - Match-server-streaming-emotes-and-persistence.md`
- Modify: `Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md`
- Modify: this plan only to replace the allocation table placeholders after the final collision sweep.

Do not create production, migration, UI, CSS, fixture, or test files. Do not add an `## Implementation Plan` or `## Implementation Notes` section to a Backlog task while it remains To Do; AGENTS.md requires the task to move In Progress first.

## Provisional allocation observed during planning

The 2026-08-20 all-ref/all-worktree sweep observed task ceiling `19021` and ADR ceiling `073`. These values are evidence only, not reservations. Task and ADR IDs must be re-swept and allocated in Task 1 immediately before files are created.

| Symbol | Provisional value | Title |
| --- | ---: | --- |
| `ADR_PACK_ID` | `074` | Portable Actor Packs and local Persona Visual runtime |
| `ADR_EMOTE_ID` | `075` | Durable character emote metadata |
| `TASK_PERSONA_FOUNDATION_ID` | `19022` | Add local Persona Visual pack foundation |
| `TASK_PERSONA_AUTHORING_ID` | `19023` | Author and import Persona Visual packs |
| `TASK_BUDDY_ID` | `19024` | Add opt-in app-wide floating Persona Buddy |
| `TASK_PERSONA_SVI_ID` | `19025` | Enable Shared Visual Identity for Persona actors |
| `TASK_ACTOR_FOUNDATION_ID` | `19026` | Define and create portable Actor Packs |
| `TASK_ACTOR_EXPORT_ID` | `19027` | Export self-contained Actor Packs |
| `TASK_ACTOR_IMPORT_ID` | `19028` | Import, review, and activate Actor Packs |
| `TASK_EMOTE_ID` | `19029` | Match server streaming emotes and persistence |

### Task 1: Allocate collision-free ADR and task IDs

**Files:**
- Modify: `Docs/superpowers/plans/2026-08-20-actor-pack-persona-buddy-task-filing-plan.md`
- Inspect: every `backlog/tasks/` and `backlog/decisions/` tree in local heads, remote refs, and worktrees

- [ ] **Step 1: Refresh remote refs**

Run:

```bash
git fetch --all --prune
```

Expected: fetch succeeds; no working-tree files change.

- [ ] **Step 2: Re-sweep task IDs across every ref and worktree**

Run the numeric, NUL-safe sweep from `backlog/docs/lessons-backlog-hygiene.md`. Include both `refs/remotes` and `refs/heads`, then inspect every path from `git worktree list --porcelain`. Never assign zsh's special lowercase `path` variable.

Expected: one numeric maximum from refs and one from worktrees. Allocate eight consecutive IDs strictly above the larger maximum.

- [ ] **Step 3: Re-sweep ADR IDs and existing programme titles**

Run the equivalent NUL-safe scan for `backlog/decisions/<number>-*.md`. Search all ref task filenames and the current tree for `Actor Pack`, `Persona Visual`, `Persona Buddy`, `Shared Visual Identity`, and `streaming emote` duplicates.

Expected: two unused ADR numbers above the larger all-ref/all-worktree ADR ceiling; no task already covers an approved programme increment.

- [ ] **Step 4: Probe the Backlog CLI allocator without trusting it**

Create one unmistakable throwaway task with `backlog task create`, read the generated filename, and remove that probe before any real task is filed. If the CLI's proposed ID is not above the swept maximum, keep the swept allocation and use the generated task only as a formatting template.

Expected: no probe file remains in `backlog/tasks/`.

- [ ] **Step 5: Record the final allocation**

Replace the provisional values in this plan's allocation table with the freshly
allocated values, then replace every corresponding `<ADR_*_ID>` and `<TASK_*_ID>`
symbol throughout this plan's paths, dependencies, commands, and link instructions.
Use whole-document searches to prove no dynamic ID symbol remains. Do not carry any
“next safe ID” beyond this filing session.

- [ ] **Step 6: Verify allocation uniqueness**

Run whole-repo and all-ref searches for every chosen ADR/task ID before creating files.

Expected: zero pre-existing matches for all ten chosen IDs.

### Task 2: Write the portable Actor Pack and Persona Visual ADR

**Files:**
- Create: `backlog/decisions/<ADR_PACK_ID>-portable-actor-packs-and-local-persona-visual-runtime.md`
- Reference: `Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md`
- Reference: `backlog/decisions/067-bundled-samira-visual-identity-pack.md`

- [ ] **Step 1: Create the ADR with status Proposed**

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

- [ ] **Step 2: Record rejected alternatives**

Include at least: merging the two visual runtimes, thin/reference archives, server-backed in-place writes, broad Persona-to-SQLite migration as incidental scope, a generic distributed transaction framework, third-party `textual-window`, and opaque/deferred Persona runtime storage.

- [ ] **Step 3: Link the spec, ADR-067, and Tasks 1–7**

Use the final allocated task IDs. Links must be relative and render without Markdown hard-break trailing spaces.

- [ ] **Step 4: Validate the ADR**

Run:

```bash
git diff --check
rg -n "TBD|TODO|PLACEHOLDER|<ADR_|<TASK_" backlog/decisions/<ADR_PACK_ID>-portable-actor-packs-and-local-persona-visual-runtime.md
```

Expected: diff check passes; placeholder search returns no matches.

- [ ] **Step 5: Commit the ADR**

```bash
git add backlog/decisions/<ADR_PACK_ID>-portable-actor-packs-and-local-persona-visual-runtime.md
git commit -m "docs: decide portable actor pack architecture"
```

### Task 3: Write the durable character emote ADR

**Files:**
- Create: `backlog/decisions/<ADR_EMOTE_ID>-durable-character-emote-metadata.md`
- Reference: `backlog/decisions/067-bundled-samira-visual-identity-pack.md`
- Reference: `Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md`

- [ ] **Step 1: Create the ADR with status Proposed**

The Decision must pin:

- exact compatibility with `tldw_server` dev commit `385afa951922c8a9dc2002c675bb6cad65e4ac23`;
- standalone, stripped `Emote: <state>` directives with five-event cap and UTF-16 offsets;
- 25-state prompt inventory in stored order with exact `(+N more)` suffix;
- assistant-visible text only as parser input;
- manual-display override without suppressing parsing/persistence;
- explicit-emote precedence over heuristic fallback;
- bounded durable `mood_label` and `emote_events` metadata referencing immutable visual identity; and
- history restores the final expression only, not historical beat replay.

- [ ] **Step 2: Record rejected alternatives**

Include: final-emote-only parsing, tool call first, Persona Buddy state reuse, heuristic plus explicit emote together, raw directive persistence, and beat replay in V1.

- [ ] **Step 3: Link ADR-067, the programme spec, and Task 8**

State explicitly that this ADR amends ADR-067's session-only message boundary without merging Persona Visual operational states into character expressions.

- [ ] **Step 4: Validate and commit**

Run `git diff --check`, placeholder search, and then:

```bash
git add backlog/decisions/<ADR_EMOTE_ID>-durable-character-emote-metadata.md
git commit -m "docs: decide durable character emote metadata"
```

### Task 4: File Task 1 — local Persona Visual foundation

**Files:**
- Create: `backlog/tasks/task-<TASK_PERSONA_FOUNDATION_ID> - Add-local-Persona-Visual-pack-foundation.md`

- [ ] **Step 1: Create the task through Backlog.md**

Use `backlog task create` with status `To Do`, priority `high`, separate repeated `--ac` flags, and references to the programme spec, the portable-pack ADR, and pinned server commit. If the CLI emits a different unsafe ID, preserve its rendered structure but move the content to the allocated collision-free filename and correct frontmatter before any commit.

- [ ] **Step 2: Set the description**

Use this outcome-focused description:

> Add a profile-local Persona Visual runtime compatible with the pinned server's sprite-frame contract so local Personas can own immutable operational-state visuals without merging them into Shared Visual Identity reactions.

- [ ] **Step 3: Add acceptance criteria**

Add these as separate criteria:

1. Local persistence supports Persona Visual packs, immutable versions, assets, and one active binding per eligible local Persona without changing existing Persona records.
2. Validation matches the pinned server contract: manifest version 1 for `sprite_frames`, nine reserved built-ins including `wake_armed`, five required resolvable states, bounded safe custom states, fallback chains, frames, regions, timing, and authored triggers.
3. Activatable packs resolve `idle`, `listening`, `thinking`, `speaking`, and `error`; runtime misses fall through validated manifest fallbacks, then `idle`, then Persona portrait with a stable reason.
4. Assets use validated profile-owned storage, MIME/decode/dimension/frame budgets, immutable full-identity cache keys, and never publish beneath package resources or expose private paths.
5. Repository and publication paths enforce optimistic binding/version authority, rollback, cancellation drain, orphan cleanup, and source-only cache invalidation.
6. Frozen fixtures derived from server commit `385afa...` pin supported and unsupported renderer/manifest behavior.
7. No Workbench authoring UI, floating Buddy, provider generation, or server write path is introduced.
8. Focused migration/repository/validator/asset/resolver/publication tests plus scoped static and diagnostic-governance checks pass in an isolated profile.

- [ ] **Step 4: Verify and commit**

Re-read the file directly and through `backlog task list --plain`; confirm eight independent unchecked ACs, empty dependencies, no Implementation Plan/Notes, and exact references. Commit only this task file.

### Task 5: File Task 2 — Persona Visual authoring and import

**Files:**
- Create: `backlog/tasks/task-<TASK_PERSONA_AUTHORING_ID> - Author-and-import-Persona-Visual-packs.md`

- [ ] **Step 1: Create the task with dependency on Task 1**

Status `To Do`, priority `high`, dependency `<TASK_PERSONA_FOUNDATION_ID>`, references to the spec and portable-pack ADR.

- [ ] **Step 2: Set the description**

> Let users review, edit, import, stage, and explicitly publish Persona Visual packs for local Personas while keeping active runtime visuals unchanged until Save.

- [ ] **Step 3: Add acceptance criteria**

1. Personas Workbench shows all nine baseline state slots, bounded safe custom states, path-free validation inventory, and one selected lazy preview for an eligible local Persona.
2. Replace, Clear, Add Custom State, Save, and Cancel are staged-only; Save publishes one immutable version and Cancel restores the exact authoritative metadata.
3. `.tldw-persona-vpack` import validates the full pinned sprite-frame archive into a review draft and never activates before explicit Save.
4. Unsupported renderer/manifest capabilities, malformed assets, stale Persona/binding/session authority, and import cancellation fail closed without changing the active version.
5. Server-backed Personas show Save Local Copy first; legacy expression-set and Actor Pack import remain separate, honestly labelled actions.
6. Preview inventory/resolve/decode work is screen-owned, serialized across navigation, drained on cancellation, weak-targeted, and fenced after every await.
7. No image-generation provider, recipe workflow, Shared Visual Identity merge, or Buddy window is added.
8. Focused widget/screen/race/import/publication tests, compact and normal geometry checks, Impeccable review after final visible change, and scoped static/governance gates pass.

- [ ] **Step 4: Verify and commit**

Confirm dependency and eight ACs through direct read plus `backlog task list --plain`; commit only this task file.

### Task 6: File Task 3 — floating Persona Buddy

**Files:**
- Create: `backlog/tasks/task-<TASK_BUDDY_ID> - Add-opt-in-app-wide-floating-Persona-Buddy.md`

- [ ] **Step 1: Create the task with dependency on Task 1**

Status `To Do`, priority `high`, dependency `<TASK_PERSONA_FOUNDATION_ID>`, references
to the programme spec and portable-pack ADR.

- [ ] **Step 2: Set the description**

> Give users an explicitly enabled, app-wide floating visual companion for one selected local Persona, driven only by trusted application lifecycle state.

- [ ] **Step 3: Add acceptance criteria**

1. Buddy is default-off and mounts only after the user explicitly selects an eligible local Persona; Workbench highlight, Console actor, and server-source changes never silently retarget it.
2. An app-owned controller survives screen navigation without retaining screen/widget references and resolves the pinned state priority, all nine built-ins, source-scoped leases, safe custom triggers, and exact Persona/binding/version identity.
3. A native Textual 8 floating view is bottom-right by default, draggable, resizable, focusable, collapsible, closable, bounded to the viewport, keyboard-operable, and never steals focus on state changes.
4. Geometry/enabled/open/collapsed preferences persist profile-locally, re-clamp on resize, and are never exported; splash/auth/recovery/modal surfaces safely hide or cover the Buddy.
5. `sprite_frames` animation pauses while hidden/collapsed, respects reduced motion, and falls back through state, idle, and portrait without blanking the UI.
6. Same-app work is serialized across replacement screens; stale decode/render results and replaced views cannot repaint or remove the current view.
7. No third-party window dependency, taskbar, snapping desktop, maximize system, model-directed state, or default Persona is introduced.
8. Production-shaped Pilot tests cover normal and 80x24 layouts, compositor output and flow-budget isolation; isolated real-terminal verification covers mouse drag/resize, keyboard controls, focus, modal layering, navigation, and geometry restore.

- [ ] **Step 4: Verify and commit**

Confirm eight ACs, one dependency, and no Implementation Plan/Notes; commit only this task file.

### Task 7: File Task 4 — Shared Visual Identity for Personas

**Files:**
- Create: `backlog/tasks/task-<TASK_PERSONA_SVI_ID> - Enable-Shared-Visual-Identity-for-Persona-actors.md`

- [ ] **Step 1: Create the task against the merged ADR-067 foundation**

Status `To Do`, priority `high`, dependency `TASK-16319`, references to ADR-067, the programme spec, and the portable-pack ADR.

- [ ] **Step 2: Set the description**

> Complete the already-declared Persona actor path in Shared Visual Identity so local Personas can own reaction/expression packs without merging those expressions into Persona Buddy operational states.

- [ ] **Step 3: Add acceptance criteria**

1. Eligible local Personas can create, replace, clear, publish, and resolve Shared Visual Identity bindings using the existing immutable pack/version model.
2. Personas Workbench exposes path-free Shared Visual Identity metadata, lazy selected preview, staged edits, visible manual labels where applicable, Save, and Cancel with full session/actor/binding fences.
3. Persona resolution uses exact full actor and cache identities, deterministic fallback, targeted actor invalidation, and source-only change detection.
4. Console/persona-chat consumers render the active Persona expression without giving Persona Buddy operational states any reaction semantics.
5. Server-backed Personas require Save Local Copy first; stale source/session/actor/binding/version changes cannot publish or repaint.
6. Existing Character creation, authoring, Console rendering, publication, cache, and four-state operational behavior remain unchanged.
7. No schema/runtime merge with Persona Visual, Actor Pack archive workflow, or server write path is introduced.
8. Focused repository/resolver/Workbench/Console/race/invalidation tests plus ADR-067 architecture/privacy/governance gates pass.

- [ ] **Step 4: Verify and commit**

Confirm dependency on `TASK-16319`, eight ACs, and exact ADR links; commit only this task file.

### Task 8: File Task 5 — Actor Pack format, identity, and creation

**Files:**
- Create: `backlog/tasks/task-<TASK_ACTOR_FOUNDATION_ID> - Define-and-create-portable-Actor-Packs.md`

- [ ] **Step 1: Create the independent foundation task**

Status `To Do`, priority `high`, no dependency on programme Tasks 1–4. Reference the portable-pack ADR and programme spec.

- [ ] **Step 2: Set the description**

> Define a secure, deterministic one-actor portable envelope and let users create pack-ready local Characters or Personas with a required portrait and stable portable identity.

- [ ] **Step 3: Add acceptance criteria**

1. `tldw.actor-pack/v1` defines exactly one local Character or Persona, required canonical actor JSON and portrait, optional typed visual sections, license/provenance declarations, required features, and no local IDs or external references.
2. Internal paths, canonical JSON, per-file SHA-256/size inventory, non-self-referential top digest, deterministic ZIP metadata, and all actor/manifest/portrait limits match the approved spec.
3. A profile-local registry assigns globally unique canonical UUIDv4 identities independent of names/content/local IDs, survives soft deletion/restoration, and records copy provenance without reusing the source UUID.
4. New Actor Pack reuses canonical local Character/Persona editors, requires a portrait, and creates the actor plus portable identity without writing an archive or requiring visual sections.
5. Server-backed Personas cannot receive portable registry rows and expose Save Local Copy first.
6. Persona actor/registry changes use the purpose-built write-ahead intent, atomic JSON replace, SQLite commit, compensation, startup recovery, and quarantine-on-third-authority protocol; Character changes remain one SQLite transaction.
7. Unknown required features, malformed/colliding paths, invalid actor kinds/payloads/portraits, duplicate UUIDs, and stale authority fail closed without a partial actor.
8. Focused schema/registry/validator/editor/cross-store crash-recovery tests and scoped static/privacy gates pass; export and import UI are absent.

- [ ] **Step 4: Verify and commit**

Confirm empty dependencies, eight ACs, and no premature export/import implementation plan; commit only this task file.

### Task 9: File Task 6 — Actor Pack export

**Files:**
- Create: `backlog/tasks/task-<TASK_ACTOR_EXPORT_ID> - Export-self-contained-Actor-Packs.md`

- [ ] **Step 1: Create the task with Tasks 1, 4, and 5 dependencies**

Status `To Do`, priority `high`, dependencies `<TASK_PERSONA_FOUNDATION_ID>`, `<TASK_PERSONA_SVI_ID>`, and `<TASK_ACTOR_FOUNDATION_ID>`.
Add references to the programme spec and portable-pack ADR.

- [ ] **Step 2: Set the description**

> Let users export one eligible local Character or Persona as a deterministic, self-contained Actor Pack whose actor, portrait, and active visual versions come from one consistent authority snapshot.

- [ ] **Step 3: Add acceptance criteria**

1. Export supports eligible local Characters and Personas and offers one-time portable UUID assignment for eligible existing actors without one; server-backed Personas remain disabled.
2. The snapshot captures actor revision, UUID, portrait, active Shared Visual Identity, and active Persona Visual binding/version/assets where applicable, then revalidates the complete authority before publication.
3. Every included visual section is self-contained and preserves its typed manifest/license/provenance; a missing declared asset fails rather than emitting a thin reference.
4. Archive output uses canonical lowercase-ASCII paths, `ZIP_STORED`, fixed metadata/order, bounded streaming reads/writes, and byte-identical output for identical canonical inputs.
5. Destination publication uses a same-directory temporary file, flush/sync where supported, no-follow identity checks, and atomic replacement; stale authority or failure leaves the destination untouched.
6. Local IDs, chats, deletion state, provider settings, credentials, paths, session/UI preferences, and private diagnostics never enter the archive.
7. Character-only, Persona-only, both-visual-section, and minimal actor+portrait exports validate against independent golden fixtures and deterministic digest/byte oracles.
8. Focused export/authority-race/path-substitution/privacy/package tests and scoped static/governance gates pass.

- [ ] **Step 4: Verify and commit**

Confirm the three exact dependencies and eight ACs; commit only this task file.

### Task 10: File Task 7 — Actor Pack import and activation

**Files:**
- Create: `backlog/tasks/task-<TASK_ACTOR_IMPORT_ID> - Import-review-and-activate-Actor-Packs.md`

- [ ] **Step 1: Create the task with Tasks 1, 3, 4, 5, and 6 dependencies**

Status `To Do`, priority `high`, dependencies `<TASK_PERSONA_FOUNDATION_ID>`, `<TASK_BUDDY_ID>`, `<TASK_PERSONA_SVI_ID>`, `<TASK_ACTOR_FOUNDATION_ID>`, and `<TASK_ACTOR_EXPORT_ID>`.
Add references to the programme spec and portable-pack ADR.

- [ ] **Step 2: Set the description**

> Let users safely inspect and activate an untrusted Actor Pack as a new local actor, a copy, or an explicitly confirmed update without risking existing actor data or visual bindings.

- [ ] **Step 3: Add acceptance criteria**

1. Import enforces all outer/member/section budgets, canonical paths, declared-file and digest integrity, MIME/decode limits, no links/encryption/nesting/devices/external references, and free-space preflight before private staging.
2. Review is path-free and shows actor fields, portrait, visual inventory, license/provenance, warnings, UUID match, differences, and the exact effect of every activation choice.
3. With no UUID match, Create New preserves the incoming UUID and Create Copy assigns a fresh UUID; with a same-kind exact match, Create Copy or explicitly confirmed Update Existing is offered; cross-kind reuse is rejected.
4. Update Existing changes only reviewed portable actor fields and present visual sections; every omitted optional section visibly preserves its current local binding.
5. Immediately before activation, actor/UUID/binding/version/staged-filesystem/free-space authority is revalidated; stale review returns to review and never auto-merges.
6. Character activation is transactional; Persona activation consumes the cross-store coordinator; failure/cancellation preserves prior actor/bindings, drains workers, and exposes only opaque pinned cleanup eligibility.
7. Successful activation invalidates affected Shared Visual Identity caches, Persona runtime, and mounted Buddy only after commit, then refreshes authoritative review/editor state.
8. Independent golden round trips and adversarial traversal/link/collision/bomb/truncation/digest/MIME/disk/race/crash/cleanup tests plus scoped static/privacy/governance gates pass.

- [ ] **Step 4: Verify and commit**

Confirm five exact dependencies and eight ACs; commit only this task file.

### Task 11: File Task 8 — server-parity streaming emotes

**Files:**
- Create: `backlog/tasks/task-<TASK_EMOTE_ID> - Match-server-streaming-emotes-and-persistence.md`

- [ ] **Step 1: Create the independent emote task**

Status `To Do`, priority `high`, dependency `TASK-16319`, references to ADR-067, the durable-emote ADR, the programme spec, and pinned server commit.

- [ ] **Step 2: Set the description**

> Match the server's explicit streaming character-emote behavior so reaction directives drive live portraits while remaining absent from visible and persisted assistant text, with durable final-expression history restore.

- [ ] **Step 3: Add acceptance criteria**

1. Streaming and non-streaming character responses parse only standalone case-insensitive `Emote:` lines from assistant-visible text, with exact safe-slug normalization, five-event cap, duplicate handling, fences, CRLF, chunk, and unterminated-line behavior from the pinned server.
2. Valid, invalid, duplicate, and over-cap directive lines never reach rendered text, persisted content, search, or exports; inline prose and fenced code remain visible.
3. Prompting lists normalized active-version states in stored order, exposes the first 25, and uses the exact `(+N more)` suffix without imported labels/text.
4. Live portrait precedence is manual override, then operational thinking/speaking
until the first accepted explicit event; every accepted event updates the live
expression, the last accepted event becomes the persisted final expression, and the
heuristic runs only when no explicit event exists. Missing assets retain the
current/base portrait with a stable reason.
5. Assistant metadata durably stores bounded final mood fields, at most five UTF-16-offset events, actor identity, immutable pack/version/expression/asset identity, and fallback reason; malformed metadata fails soft on load.
6. History restores only the exact final immutable expression when available, reports deterministic fallback otherwise, and does not replay historical beats.
7. Reasoning, tool arguments/results, citations, provider controls, Persona Buddy, and raw assistant content never enter directive diagnostics or state control.
8. Frozen cross-language vectors plus streaming/non-streaming/provider-tool/manual/missing-asset/persistence/history/failure tests and scoped static/privacy/governance gates pass.

- [ ] **Step 4: Verify and commit**

Confirm dependency on `TASK-16319`, eight ACs, and exact durable-emote ADR link; commit only this task file.

### Task 12: Link, validate, and commit programme filing

**Files:**
- Modify: `Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md`
- Modify: the two new ADRs if final task links require correction
- Inspect: all eight new task files

- [ ] **Step 1: Mark the programme spec approved and add exact ADR/task links**

Change `Status: Approved for spec review` to `Status: Approved`. Replace the allocation-language ADR placeholders with links to both created ADRs and add a compact task table containing all eight final IDs/titles/dependencies.

- [ ] **Step 2: Re-read every task from source**

Because the current Backlog CLI cannot reliably address five-digit IDs, use direct file reads as authority and use `backlog task list --plain` only as a secondary parse check. Confirm each file has the correct title, To Do status, priority, dependencies, description, separate unchecked ACs, references, and no Implementation Plan/Notes.

- [ ] **Step 3: Run collision and dangling-placeholder gates**

Re-fetch and repeat the all-ref/all-worktree ID scan. Run whole-repo searches for all ten IDs, duplicate titles, `TBD`, `TODO`, `PLACEHOLDER`, `<ADR_`, and `<TASK_`. If another ref claimed an allocated ID, renumber this unimplemented planning slice above the new maximum and update every whole-repo reference before commit.

- [ ] **Step 4: Run Markdown and Backlog integrity checks**

Run:

```bash
git diff --check
backlog task list --plain
git status --short
```

Expected: whitespace check passes; all eight tasks appear once as To Do; status contains only the plan, approved spec, two ADRs, and eight task files created by this plan.

- [ ] **Step 5: Self-review scope and dependencies**

Verify there is no production code, test, CSS, generated manifest, task status change, assignee claim, implementation plan, or implementation note. Verify Task 5 can proceed independently; Tasks 2/3 depend on Task 1; Task 4 and Task 8 depend on TASK-16319; Task 6 depends on Tasks 1/4/5; Task 7 depends on Tasks 1/3/4/5/6.

- [ ] **Step 6: Commit final links and plan record**

```bash
git add Docs/superpowers/plans/2026-08-20-actor-pack-persona-buddy-task-filing-plan.md Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md backlog/decisions backlog/tasks
git diff --cached --check
git commit -m "docs: file actor pack and Persona Buddy programme"
```

Expected: commit succeeds and the worktree is clean.
