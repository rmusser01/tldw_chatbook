# Portable Actor Pack Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define the bounded `tldw.actor-pack/v1` contract and let users create one pack-ready local Character or Persona with a required portrait and stable portable UUID, including crash-safe Persona JSON/SQLite coordination.

**Architecture:** Keep the archive contract pure and transport-agnostic in a new `Actor_Packs` package; TASK-19057 validates in-memory documents but does not read, write, extract, review, or activate ZIP archives. Add one ChaChaNotes migration for the cross-kind portable registry and bounded Persona mutation intents. Character creation and registry assignment share one SQLite transaction. Persona creation uses the existing JSON authority plus a purpose-built prepared/committed intent and idempotent startup recovery. The Workbench adds a distinct `New Actor Pack` entry that reuses the existing Character/Persona editors; Character portrait bytes stay in the Character card, while Persona creation selects the incumbent canonical `character_card_id` portrait link.

**Tech Stack:** Python 3.12, stdlib `json`/`hashlib`/`uuid`/`sqlite3`, Pydantic, Pillow through existing image validators, Textual 8, pytest, Ruff.

**ADR required:** no

**ADR path:** `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md`

**Reason:** ADR-074 already governs the portable UUID registry, Actor Pack schema, Persona JSON/SQLite intent boundary, recovery matrix, and separate visual-runtime sections. This task directly implements that decision without introducing a new architectural choice.

## Scope guard

- Implement format/schema/canonicalization/digest/pure-validator contracts, portable identity persistence, pack-ready actor creation, and Persona recovery.
- Declare Shared Visual Identity and Persona Visual sections only as typed references. Reuse their existing validators where a section document is supplied.
- Do not implement ZIP export, ZIP import, extraction, private import staging, review, Update Existing, or activation. Those belong to TASK-19058 and TASK-19059.
- Do not add a dependency or a second Character/Persona editor.

## Task 1: Freeze the pure Actor Pack contract

**Files:**

- Create: `tldw_chatbook/Actor_Packs/__init__.py`
- Create: `tldw_chatbook/Actor_Packs/contracts.py`
- Create: `Tests/Actor_Packs/test_actor_pack_contracts.py`
- Create: `Tests/Actor_Packs/conftest.py`

- [ ] Write `test_actor_pack_contracts.py` first with named RED cases for `tldw.actor-pack/v1`, exactly one actor, canonical lowercase POSIX paths, actor-kind/section compatibility, required-file inventory, portrait and actor limits, unknown required features, forbidden local IDs/external references, and exact scalar/type bounds.
- [ ] Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Actor_Packs/test_actor_pack_contracts.py`; verify collection fails only because `tldw_chatbook.Actor_Packs` does not exist.
- [ ] Implement frozen/slotted records and one public `validate_actor_pack_document(manifest, files)` boundary. Keep the public result path-free and immutable; return fixed-category `ActorPackValidationError` values without embedding actor text, paths from the host, bytes, or provider data.
- [ ] Re-run the file and confirm GREEN.
- [ ] Add RED adversarial cases for `..`, absolute/backslash/drive paths, empty segments, uppercase/non-ASCII, trailing dot/space, Windows device aliases including extensions, duplicate/case/Unicode collisions, undeclared files, unsupported sections, Character Persona-runtime sections, and manifest self-inventory.
- [ ] Implement only the shared path/inventory predicates needed to turn those cases GREEN.
- [ ] Mutation-check path grammar, actor-kind compatibility, file declaration equality, and unknown-required-feature guards; restore after each named test fails.
- [ ] Commit: `feat: define portable Actor Pack contracts`

## Task 2: Pin canonical JSON, inventories, and non-self-referential digest

**Files:**

- Modify: `tldw_chatbook/Actor_Packs/contracts.py`
- Modify: `Tests/Actor_Packs/test_actor_pack_contracts.py`
- Create: `Tests/Actor_Packs/fixtures/minimal-character-v1/actor-pack.json`
- Create: `Tests/Actor_Packs/fixtures/minimal-character-v1/actor/actor.json`
- Create: `Tests/Actor_Packs/fixtures/minimal-character-v1/actor/portrait.png`

- [ ] Add born-RED tests for UTF-8 canonical JSON (`sort_keys=True`, compact separators, `ensure_ascii=False`, no newline), canonical actor payload projection, per-file SHA-256/size, `actor-pack.json` self-exclusion, a content digest that excludes only its own field, and immutable export metadata constants for `ZIP_STORED`, canonical path order, creator system, flags, regular-file permissions, and the DOS timestamp `1980-01-01 00:00:00`.
- [ ] Run the focused file and read the expected missing-function/assertion failures.
- [ ] Implement `canonical_json_bytes`, actor-kind adapters, `build_file_inventory`, `actor_pack_content_digest`, and the deterministic ZIP metadata contract using only stdlib primitives. Do not add a ZIP writer. Reject booleans-as-integers, non-finite numbers, recursive/deep inputs, unknown fields, and oversized JSON before canonicalization.
- [ ] Re-run GREEN, then add an independent golden oracle whose expected canonical bytes and hashes are literals rather than values computed through production helpers.
- [ ] Add RED property/adversarial cases for order independence, Unicode preservation, recursion/node/string budgets, digest mismatch, portrait MIME/decode/dimension/byte limits, and prohibited payload keys (`id`, local record IDs, chats, credentials, provider settings, paths, UI/session state).
- [ ] Reuse existing raster validation/Pillow admission instead of writing a second decoder; make the focused cases GREEN.
- [ ] Mutation-check self-exclusion, digest-field exclusion, portrait admission, and forbidden-field projection.
- [ ] Commit: `feat: validate canonical Actor Pack documents`

## Task 3: Add portable identity and intent persistence

**Files:**

- Create: `tldw_chatbook/DB/migrations/chachanotes_v44_to_v45_actor_packs.sql`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Create: `tldw_chatbook/Actor_Packs/repository.py`
- Create: `Tests/ChaChaNotesDB/test_actor_pack_migration.py`
- Create: `Tests/Actor_Packs/test_actor_pack_repository.py`
- Modify: `pyproject.toml`
- Modify: `MANIFEST.in`
- Modify: `Packaging/check_manifest.py`
- Modify: `Tests/Packaging/test_installed_distribution.py`

- [ ] Write the v44 historical migration RED first: current schema remains v44 and registry/intent tables do not exist.
- [ ] Add v45 DDL for `(actor_kind, local_actor_id)` registry identity, one cross-kind unique canonical UUID, optional copy provenance that cannot equal the assigned UUID, and bounded prepared/committed/quarantined Persona intents. Do not add foreign-key cascades into actors, chats, or visual versions.
- [ ] Register the migration in the migration dispatcher and all four packaging inventories. Test end-state against `_CURRENT_SCHEMA_VERSION`, not literal current-version assumptions in unrelated migrations.
- [ ] Re-run migration RED to GREEN; run the complete affected ChaChaNotes migration component.
- [ ] Write repository REDs for exact RFC 4122 lowercase UUIDv4 generation/admission, cross-kind uniqueness, duplicate/concurrent assignment under `BEGIN IMMEDIATE`, stable lookup across actor soft delete/restore, server Persona refusal, fresh UUID plus source provenance for copies, and fixed path-free corruption/write/read categories.
- [ ] Implement a small `ActorPackRepository` whose writes own or explicitly receive one reserved SQLite transaction; reject ambiguous nesting rather than silently splitting atomicity.
- [ ] Add repository REDs for durable prepared intent, atomic registry plus committed-state transition, exact bounded snapshot/digest decoding, cleanup, list-for-recovery, quarantine, SQLite corruption, and transaction rollback.
- [ ] Implement those methods without a general transaction framework or export/import APIs.
- [ ] Mutation-check cross-kind uniqueness, exact UUID version/variant, server-source refusal, transaction ownership, intent bounds, and state transitions.
- [ ] Commit: `feat: persist portable actor identities`

## Task 4: Implement the Persona JSON/SQLite coordinator and startup recovery

**Files:**

- Create: `tldw_chatbook/Actor_Packs/persona_coordinator.py`
- Modify: `tldw_chatbook/Character_Chat/local_character_persona_service.py`
- Modify: `tldw_chatbook/app.py`
- Create: `Tests/Actor_Packs/test_persona_actor_pack_coordinator.py`
- Modify: `Tests/Character_Chat/test_local_character_persona_service.py`
- Modify: `Tests/UI/test_console_runtime_ownership.py`

- [ ] Add a born-RED real-file/real-SQLite creation test proving the intent is durably `prepared` before Persona JSON replacement, registry insertion and `committed` status share one SQLite transaction, success removes the intent, and the service cache reflects the committed Persona.
- [ ] Add a package-private Local Persona store seam that snapshots one bounded canonical profile plus whole-store authority digest, atomically inserts/replaces/removes only that profile, and reloads the incumbent service cache. It must preserve all unrelated Persona store sections and never expose/log raw intent snapshots.
- [ ] Implement `PersonaActorPackCoordinator.create_persona(...)` with exact profile/source/revision/portrait authority and a narrow injected cancellation signal checked before each irreversible boundary.
- [ ] Add born-RED ordinary-failure tests at prepared-write, JSON replace, SQLite commit, and intent cleanup boundaries; prove compensation returns old JSON/old SQLite and removes only owned residue.
- [ ] Add born-RED startup matrix tests for prepared+old+old cleanup, prepared+new+old compensation/removal, committed+new+new retention/cleanup, and quarantine of old+new, prepared+new-SQLite, contradictory committed, digest/revision mismatch, malformed/broad intent, and concurrent Persona changes.
- [ ] Implement idempotent recovery before Personas or Actor Pack surfaces are wired. A contradiction returns one stable blocked category and makes no destructive guess.
- [ ] Add deterministic cancellation barriers before JSON replacement and before SQLite commit; cancel, signal, shield/drain, and assert no Persona/registry/intent/residue. Add a post-commit control proving committed success is preserved rather than deleted.
- [ ] Mutation-check each recovery branch, cancellation fence, exact authority digest, and startup ordering.
- [ ] Commit: `feat: coordinate Persona Actor Pack creation`

## Task 5: Add pack-ready Character creation as one SQLite transaction

**Files:**

- Create: `tldw_chatbook/Actor_Packs/creation.py`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Modify: `tldw_chatbook/Character_Chat/local_character_persona_service.py`
- Create: `Tests/Actor_Packs/test_actor_pack_creation.py`

- [ ] Write RED tests for one Character request with valid portrait producing exactly one character plus one registry row in one transaction; validation, UUID collision, duplicate submit, stale editor/portrait authority, database failure, and cancellation before commit must leave neither row.
- [ ] Extract the existing Character insert body into a connection-owned helper while keeping `add_character_card` behavior unchanged. Do not duplicate the Character schema or bypass existing validation/triggers.
- [ ] Implement `ActorPackCreationService.create_character` and `.create_persona` as the single non-UI entry points. Admit one operation at a time, snapshot immutable inputs, validate portrait before mutation, and return frozen path-free `{actor_kind, local_actor_id, portable_uuid}`.
- [ ] For Persona portrait authority, require an exact eligible local Character card id/revision/image SHA snapshot; pass `character_card_id` through the canonical Persona profile schema. Server-backed Personas return the exact fixed feedback `Save a local copy first` and never reach the registry.
- [ ] Add RED/GREEN controls proving no archive, visual pack/version/binding, chat, or provider row is written.
- [ ] Mutation-check transaction sharing, portrait requirement, source gate, exact portrait revision/SHA, duplicate-operation rejection, and cancellation-before-commit.
- [ ] Commit: `feat: create pack-ready local actors`

## Task 6: Expose New Actor Pack through the canonical Workbench editors

**Files:**

- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/UI/test_personas_workbench.py`
- Create: `Tests/UI/test_actor_pack_creation_workflow.py`
- Modify generated CSS only if the canonical builder produces a reviewed change.

- [ ] Immediately before UI edits, run the Impeccable context command once for the Personas Workbench, read the selected playbook and `reference/craft-floor.md`, and record but do not auto-fix unrelated drift.
- [ ] Write Pilot REDs under the real bundled stylesheet for an explicit `New Actor Pack` action in Character and Persona modes, exact local/server eligibility, exact `Save a local copy first` feedback, canonical editor reuse, visible required-portrait status, keyboard/focus order, and usable 80x24 plus wide layouts.
- [ ] Implement one distinct library action that enters the existing editor with an immutable pack-create session token. Character mode reuses the existing avatar upload. Persona mode shows a labelled local portrait-Character selector populated from eligible local cards; ordinary New remains unchanged.
- [ ] Write RED save tests for one-operation admission, duplicate clicks, source/editor/portrait generation fences before and after every await, dirty-navigation decline, accepted navigation cancellation, and Cancel. Drive deterministic barriers during portrait validation and commit; assert the worker is signalled/drained before serialization release and no residue remains.
- [ ] Route pack-mode Save through the app-owned `ActorPackCreationService`; on durable success refresh/select the new actor and show its portable UUID metadata. Do not write an archive or mount visual authoring as part of the operation.
- [ ] Add painted compositor assertions for required portrait/error/success states and plain-text/path-free errors. Add server Persona and ordinary create non-regression tests.
- [ ] Run the canonical CSS builder if BUNDLED_CSS changes, inspect every generated output, and stage only actual generator-owned diffs. Run the Impeccable detector once after the final visible change; fix only findings caused by this task.
- [ ] Mutation-check source, editor-session, portrait revision/SHA, duplicate-submit, cancellation/drain, and stale-view reconciliation guards.
- [ ] Commit: `feat: add New Actor Pack workflow`

## Task 7: Architecture, privacy, recovery, and closeout evidence

**Files:**

- Create: `Tests/Architecture/test_actor_pack_boundary.py`
- Modify: `backlog/tasks/task-19057 - Define-and-create-portable-Actor-Packs.md`
- Modify lessons only if a new incident genuinely generalizes.

- [ ] Add a born-RED architecture guard proving `Actor_Packs.contracts` and pure validators import no UI, archive writer/reader, network, provider, Buddy, or server transport modules; restore the temporary forbidden import and prove GREEN.
- [ ] Add scope guards proving TASK-19057 defines no ZIP open/write/extract/activation/review surface and no Shared Visual Identity/Persona Visual merge.
- [ ] Run import provenance first and retain its exact assigned-worktree path.
- [ ] Under one isolated HOME/XDG/config/data root created before Python import, run real v44→v45 migration, Character creation, Persona create/recovery, and Workbench Pilot evidence. Do not launch against the real profile while the schema bump is unmerged.
- [ ] Run complete affected components: `Tests/Actor_Packs/`, the v45 migration/repository paths, full local Character/Persona service, focused Workbench creation, affected app ownership, and packaging migration inventory.
- [ ] Run scoped Ruff, formatter, `py_compile`/`compileall`, `git diff --check origin/dev...HEAD`, diagnostic inventory, privacy/database inventory, architecture/governance, and licence checks. Compare any failure with the exact same command on the pinned base SHA before calling it pre-existing.
- [ ] Perform a local specification/correctness review and a separate ponytail review; resolve all Critical/Important findings and delete only complexity that has a simpler equivalent without weakening a boundary.
- [ ] Update all eight ACs to checked, add concise Implementation Notes with RED/GREEN/mutation/static evidence and deviations, and set TASK-19057 Done only when every scoped gate is green. If any scoped gate remains red, leave it In Progress and record the exact blocker.
- [ ] Commit: `docs: close portable Actor Pack foundation task`
