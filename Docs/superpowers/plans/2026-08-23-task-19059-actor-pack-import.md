# Actor Pack Import, Review, and Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely stage, review, and activate one untrusted `.tldw-actor-pack` as a new local actor, a copy, or an explicitly confirmed update without exposing paths or weakening existing actor and visual authority.

**Architecture:** Add one synchronous hostile-archive importer that produces an immutable path-free review backed by a pinned private staging lease, and one activation service that consumes only a still-current review. Character writes share one owned SQLite transaction; Persona writes extend ADR-074's bounded JSON/SQLite coordinator. A small Textual review modal and app-owned controller own consent, cancellation, cleanup, and isolated post-commit refresh while existing Shared Visual Identity and Persona Visual validators/publication seams remain authoritative.

**Tech Stack:** Python 3.11+, using only syntax and stdlib APIs available on the supported Python 3.11 floor; stdlib `dataclasses`/`hashlib`/`json`/`os`/`pathlib`/`shutil`/`stat`/`threading`/`zipfile`, Pillow, SQLite, existing Actor Pack/Visual Identity/Persona Visual contracts and repositories, Textual 8, pytest, Ruff.

**ADR required:** no

**ADR path:** `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md`

**Reason:** ADR-074 already fixes review-first import, hostile-archive staging, the UUID action matrix, omitted-section preservation, Character transaction ownership, Persona coordination, cleanup, and post-commit refresh isolation. This task implements that decision without changing the boundary.

---

## Scope guard

- Import only the V1 self-contained format already defined by `Actor_Packs.contracts`; legacy expression-set and Persona Visual imports remain separate.
- Keep untrusted paths, member names, actor text, bytes, and cleanup capabilities out of exceptions, logs, reprs, notifications, and diagnostic payloads.
- Reuse canonical Actor Pack, Shared Visual Identity, Persona Visual, portrait, asset-loader, repository, and Persona-coordinator contracts. Do not add a generic archive framework, distributed transaction abstraction, server transport, archive-level clear action, or version garbage collection.
- A review is immutable consent evidence, not a mutable draft. Any profile, actor, UUID, binding/version, staged inode/digest, or disk-authority change returns to review.
- All blocking archive, image, filesystem, and database work runs off the Textual event loop. Cancellation is signalled and drained before lease cleanup.

## Task 1: Validate and stage hostile Actor Pack archives

**Files:**

- Create: `tldw_chatbook/Actor_Packs/importer.py`
- Modify: `tldw_chatbook/Actor_Packs/__init__.py`
- Create: `Tests/Actor_Packs/test_actor_pack_import.py`
- Create: `Tests/Actor_Packs/fixtures/import-golden/README.md`

- [ ] **Step 1: Write the born-RED archive admission tests**

  Import the independent TASK-19058 golden Character and Persona archives and add adversarial cases for entry/outer/member/section budgets, compression ratios, truncation, encrypted/nested/device/linked entries, absolute/backslash/dot paths, undeclared files, duplicate raw names, Unicode/case/device/alias collisions, malformed ZIP metadata, noncanonical JSON, inventory/content digests, unknown required features/sections, MIME/decode/pixel limits, section limits, and insufficient free space. Assert fixed path-free categories and a zero-mutation database fingerprint.

- [ ] **Step 2: Run RED and prove the fixture is live**

  Run: `env PYTHONPATH=<worktree> <venv-python> -m pytest Tests/Actor_Packs/test_actor_pack_import.py -q -k 'archive or collision or budget or digest or disk'`

  Expected: collection fails because `Actor_Packs.importer` does not exist. Mutate one golden member and independently prove the fixture oracle rejects it.

- [ ] **Step 3: Implement the minimum immutable review contract**

  Add frozen/slotted path-free records for actor fields, portrait metadata, section inventory, UUID match/differences, warnings, allowed actions, and review authority. Keep staged root/name/secret, source identity, exact file identities/digests, and cleanup capability private and excluded from repr.

  ```python
  class ActorPackImportService:
      def inspect_archive(
          self,
          archive_path: os.PathLike[str] | str,
          *,
          cancel_requested: Callable[[], bool] = lambda: False,
      ) -> ActorPackImportReview: ...

      def read_portrait_preview(
          self,
          review: ActorPackImportReview,
      ) -> ActorPackPortraitPreview: ...

      def cleanup_review(self, review: ActorPackImportReview) -> bool: ...
  ```

  `ActorPackPortraitPreview` carries bounded MIME/dimensions plus bytes excluded from
  repr. The accessor first authenticates and revalidates the exact review lease and
  portrait inode/digest; UI never receives a host path or general staging handle.

- [ ] **Step 4: Pin the source and validate metadata before extraction**

  Open one absolute regular single-link source with no-follow semantics, bind its pathname/open-handle identity and digest, and stream it into the ZIP reader under the archive-size cap. Validate every `ZipInfo` before creating a candidate: only regular files, canonical lowercase ASCII POSIX names, exact canonical order, no encryption/nesting/links/devices/unsupported mode bits, no raw or normalized collision, and bounded counts/sizes/ratios. Accept and validate TASK-19058's canonical regular-file creator/permission attributes rather than rejecting all external attributes.

  Bounded-read and canonical-parse `actor-pack.json` immediately after metadata admission. Validate its schema, known required features/section kinds, top-level size, canonical bytes, inventory shape, and content digest before comparing declared-member equality or creating/extracting a candidate.

- [ ] **Step 5: Extract only declared bytes into private pinned staging**

  Compare the validated root inventory with the regular ZIP-member set and reject any missing or undeclared member. Preflight archive + staging + immutable-publication + fixed overhead free space. Create an application-owned `0700` staging root and one random `0700` candidate with an authenticated marker. Stream declared members through descriptor-relative `O_NOFOLLOW|O_EXCL` writes, enforce actual byte caps/digests while writing, fsync, and reopen every staged file to freeze device/inode/mode/link/size/mtime plus digest identity.

- [ ] **Step 6: Reuse typed validators and build a path-free review**

  Parse bounded canonical actor/section JSON, call `validate_actor_pack_document` with the already validated root, `validate_actor_portrait`, existing Shared Visual Identity validation, and existing Persona Visual manifest/asset loaders against staged bytes. Validate actor kind/fields, portrait MIME/decode, license/provenance scalar bounds, and section-specific budgets. Recheck source and candidate identity after validation, then return only sanitized review data.

- [ ] **Step 7: Implement fail-closed cleanup and bounded startup sweep**

  Delete only the exact marker-authenticated, inode-pinned candidate beneath the private staging root using descriptor-relative no-follow operations. Return an opaque cleanup capability when cleanup cannot be proven. Sweep only recognized stale candidate names whose markers and pinned identities validate; leave ambiguous entries untouched.

- [ ] **Step 8: Run GREEN, mutate every archive guard, and commit**

  Temporarily disable each collision/link/ratio/digest/decode/free-space/source-identity guard and require a named test to fail. Restore, run the full import file, and commit `feat: stage Actor Pack imports securely`.

## Task 2: Snapshot UUID matches, differences, and activation authority

**Files:**

- Modify: `tldw_chatbook/Actor_Packs/importer.py`
- Modify: `tldw_chatbook/Actor_Packs/repository.py`
- Modify: `Tests/Actor_Packs/test_actor_pack_import.py`
- Modify: `Tests/Actor_Packs/test_actor_pack_repository.py`

- [ ] **Step 1: Write REDs for the exact action matrix**

  Cover no UUID match (`create_new`, `create_copy`), same-kind exact match (`create_copy`, `update_existing`), cross-kind reuse rejection, corrupt/duplicate registry rows, deleted/recreated actors, and UUID registry ABA. Assert Create New preserves the incoming UUID, Create Copy allocates a fresh UUID with source provenance, and Update Existing is absent without an exact same-kind match.

- [ ] **Step 2: Add one repository lookup by portable UUID**

  Implement a bounded `get_identity_by_portable_uuid()` using the existing global uniqueness constraint and decoder. Return no actor content; reject corrupt/multiple rows with fixed categories.

- [ ] **Step 3: Freeze complete review authority and exact differences**

  At the end of inspection, snapshot profile/source generation, actor kind/revision/content digest or absence, exact portable registry row/version, Shared Visual Identity binding/pack/version identity, Persona Visual binding/pack/version identity, staged file identities/digests, and disk-space authority. Project actor differences only over portable fields; label every omitted optional section `Not included — existing visuals will be preserved` for Update Existing.

- [ ] **Step 4: Add a single revalidation choke point**

  `revalidate_review(review, action)` must recompute every independently mutable authority input immediately before any effect. Any delete/recreate, revision/content ABA, registry/binding/version change, staged inode/content change, source/profile change, or reduced free-space authority raises `actor_pack_import_review_stale`; it never merges or silently updates consent.

- [ ] **Step 5: Run GREEN, representation/ABA mutation tests, and commit**

  Exercise composed/decomposed Unicode and exact actor representations, interleave writes at the guard/transaction boundary, weaken each tuple component, and require the matching test to red. Commit `feat: bind Actor Pack reviews to authority`.

## Task 3: Activate Characters in one SQLite transaction

**Files:**

- Create: `tldw_chatbook/Actor_Packs/activation.py`
- Modify: `tldw_chatbook/Actor_Packs/repository.py`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Modify: `tldw_chatbook/DB/VisualIdentity_DB.py`
- Modify: `tldw_chatbook/Persona_Visual/repository.py`
- Create: `Tests/Actor_Packs/test_actor_pack_activation.py`

- [ ] **Step 1: Write Character activation REDs**

  Cover create-new UUID preservation, copy UUID/provenance, same-kind update, portable-field-only updates, portrait replacement, omitted-section preservation, present-section immutable publication, name conflict, stale review, cancellation before commit, injected failure at every write, and no partial actor/identity/pack/version/asset/binding rows.

- [ ] **Step 2: Add narrow caller-owned transaction seams**

  Reuse `CharactersRAGDB._insert_character_card_in_transaction` and add the minimum optimistic in-transaction Character update seam. Add internal Visual Identity and Persona Visual activation seams that require exactly one manager-owned outer transaction and share their current validation/CAS logic; public repository methods continue owning their transactions.

- [ ] **Step 3: Implement Character activation**

  `ActorPackActivationService.activate(review, action, ...)` revalidates once before staging publication and again while the owned immediate transaction holds its write reservation. Insert/update only reviewed portable actor fields and portrait, assign/preserve UUID per action, publish only present visual sections, and preserve omitted bindings byte-for-byte. All DB effects commit or roll back together.

- [ ] **Step 4: Fence filesystem publication around the transaction**

  Publish imported immutable section files to profile-owned storage through existing no-follow publication rules. Record only opaque cleanup eligibility if filesystem publication wins but the DB transaction fails; never expose a path or delete unproven state. Verify retries are idempotent and never bind partial files.

- [ ] **Step 5: Run real-SQLite rollback/recovery and mutation tests**

  Use a seeded on-disk database in isolated HOME/XDG/data roots. Crash/fail before and after publication/transaction commit, reopen, and compare complete actor/identity/visual graphs. Mutate the outer transaction requirement, optimistic revision, omitted-section branch, and final staged identity guard; each named test must fail.

- [ ] **Step 6: Commit**

  Commit `feat: activate Character Actor Packs transactionally`.

## Task 4: Extend Persona coordination for import activation

**Files:**

- Modify: `tldw_chatbook/Actor_Packs/persona_coordinator.py`
- Modify: `tldw_chatbook/Actor_Packs/repository.py`
- Modify: `tldw_chatbook/Actor_Packs/activation.py`
- Modify: `Tests/Actor_Packs/test_persona_actor_pack_coordinator.py`
- Modify: `Tests/Actor_Packs/test_actor_pack_activation.py`

- [ ] **Step 1: Write Persona create/copy/update and recovery REDs**

  Cover present-only profile updates, fresh UUID/source provenance, existing exact UUID, omitted Shared Visual Identity/Persona Visual preservation, both sections present, cancellation/failure at prepared/profile-replaced/SQLite-committed/cleanup phases, and every ADR-074 recovery/quarantine state with visual rows included.

- [ ] **Step 2: Extend the bounded intent payload**

  Add only the canonical imported visual write descriptions and old/new binding authority needed to replay/compensate one Persona import. Keep JSON bounded, profile-private, redacted, and absent from repr/logs/export. Extend the existing migration only if storage columns are unavoidable; otherwise reuse the intent JSON contract to avoid a schema bump.

- [ ] **Step 3: Commit Persona JSON plus all SQLite rows together**

  Reuse `_actor_pack_plan_persona_profile`, atomically replace Persona JSON, then in one SQLite transaction assign/preserve identity, publish present visual versions/bindings, preserve omitted bindings, and mark the intent committed. The in-transaction authority guard compares the exact reviewed profile/store/registry/binding/version/staged tuples.

- [ ] **Step 4: Extend recovery without destructive guesses**

  Recover old JSON+old SQLite, new JSON+old SQLite, and committed new JSON+new SQLite exactly as ADR-074 specifies, now including visual state. Quarantine contradictions and expose only opaque intent identifiers.

- [ ] **Step 5: Run crash matrix, mutation tests, and commit**

  Prove each transition with an actual on-disk reopen; mutate compensation, state marking, omitted binding preservation, and quarantine guards. Commit `feat: coordinate Persona Actor Pack activation`.

## Task 5: Own cancellation, cleanup, and post-commit invalidation

**Files:**

- Create: `tldw_chatbook/Actor_Packs/import_controller.py`
- Modify: `tldw_chatbook/Actor_Packs/__init__.py`
- Create: `Tests/Actor_Packs/test_actor_pack_import_controller.py`

- [ ] **Step 1: Write controller REDs**

  Cover one active operation, profile invalidation, inspect/activate cancellation, repeated caller cancellation, shutdown drain, commit-wins versus cancel-wins barriers, stale operation tokens, review cleanup after terminal settlement, and path-free outcomes.

- [ ] **Step 2: Implement one app-owned off-loop controller**

  Mirror the proven export controller: immutable requests, one active operation, shared `threading.Event`, explicit executor task, shield-and-drain waits, and shutdown/profile invalidation. Keep the review lease alive until all uncancellable thread work settles.

- [ ] **Step 3: Isolate post-commit consumers**

  After a committed activation, invoke affected-only callbacks independently for Shared Visual Identity cache, Persona Visual runtime, mounted Buddy, and authoritative review/editor consumers. Collect fixed categories such as `actor_pack_import_refresh_shared_visual_failed`; one failure cannot suppress later callbacks, roll back the commit, or include paths/content.

- [ ] **Step 4: Run interleaving and invalidation-isolation mutation tests**

  Force cancel-before-lock, commit-before-cancel, cancel-after-commit-before-refresh, and every callback failure position. Remove shielding/event propagation/one callback and require a RED. Commit `feat: coordinate Actor Pack imports`.

## Task 6: Add the path-free Textual review flow

**Files:**

- Create: `tldw_chatbook/Widgets/Persona_Widgets/actor_pack_import_review.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `tldw_chatbook/app.py`
- Create: `Tests/UI/test_actor_pack_import_workflow.py`
- Create: `Tests/UI/test_actor_pack_import_review.py`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Load the Impeccable craft floor, then write Pilot REDs**

  Before the first visible edit, read `.agents/skills/impeccable/reference/craft-floor.md`. Add normal and 80x24 Pilot tests for the labelled `Import Actor Pack` action, `.tldw-actor-pack` picker, loading/cancel/error/review states, actor fields, portrait metadata/preview, visual inventory, license/provenance/warnings, UUID match/differences, omitted-section preservation copy, action availability, explicit Update confirmation, focus restoration, compact scrolling, and plain-text rendering of Rich/Textual markup payloads.

- [ ] **Step 2: Add a dedicated action and immutable message**

  Keep legacy `Import` as Character-card import. Add a separately labelled `Import Actor Pack` toolbar action and `ActorPackImportRequested` message with no path or actor content. Do not bind a new screen key; rely on normal focus/Enter/Space operation.

- [ ] **Step 3: Build one compact review modal**

  Compose stable labelled sections in a `VerticalScroll`: identity/match, actor details, portrait, included visuals, license/provenance, warnings/differences, and exact effect. Obtain portrait bytes only through the review-lease-authorized `read_portrait_preview` accessor and mount an in-memory preview without retaining a path. Render untrusted text with markup disabled/escaped. Use dimension-stable primary/secondary buttons and explicit disabled/recovery copy; focus Create New/Copy when safe and require a separate confirmation modal for Update Existing.

- [ ] **Step 4: Wire the screen as orchestration only**

  The screen captures profile/generation/session authority, opens the picker, starts inspection through the app-owned controller, presents the immutable review, submits the selected action, handles stale-review return, drains cancellation on navigation/unmount, restores focus, and refreshes the affected library/inspector only after commit. Add app startup recovery and shutdown drain for the import service/controller.

- [ ] **Step 5: Run Pilot, compact-geometry, keybinding, and markup mutation tests**

  Assert every action is compositor-visible as well as focusable, no forbidden/reserved/global binding is added, and untrusted `[bold]`, links, ANSI, control characters, long words, RTL, and Unicode remain inert plain text. Mutation-check the Update confirmation and omitted-section effect copy.

- [ ] **Step 6: Run one bounded Impeccable pass and commit**

  After the final visible change, run `node .agents/skills/impeccable/scripts/detect.mjs --json` once over the changed UI/CSS targets, fix all valid findings in one batch, rerun focused Pilot tests, and commit `feat: review and activate Actor Packs from Workbench`.

## Task 7: Independent round trips, terminal evidence, and closeout

**Files:**

- Modify: `Tests/Actor_Packs/fixtures/import-golden/README.md`
- Modify: `Tests/Architecture/test_actor_pack_boundary.py`
- Modify: `Tests/Architecture/test_actor_pack_privacy.py`
- Modify: `backlog/tasks/task-19059 - Import-review-and-activate-Actor-Packs.md`
- Modify only if a generalizable incident occurred: `backlog/docs/lessons-*.md`

- [ ] **Step 1: Prove independent golden and real export/import round trips**

  Parse committed golden archives with an oracle that uses stdlib `zipfile`/`json`/`hashlib` and direct SQLite reads, not production import helpers. Export and import minimal Character/Persona, Shared-only, Persona-only, and both-section actors; compare portable fields, UUID policy, portrait bytes, visual manifests/assets, and omitted-binding preservation.

- [ ] **Step 2: Run adversarial, provenance, isolation, and privacy gates**

  Run all Actor Pack, affected repository/migration/recovery, UI, architecture, packaging, provenance, diagnostics, and privacy tests under one temporary HOME/XDG/config/data root established before interpreter import. Assert importer/activation modules do not import Textual/server boundaries and public output contains no local IDs, paths, bytes, member names, secrets, cleanup capabilities, provider data, or actor content beyond the explicit review model.

- [ ] **Step 3: Run isolated real-terminal keyboard verification**

  Launch with explicit scratch HOME/XDG/config/data and worktree `PYTHONPATH`; leave stderr attached. Verify open/picker cancel, review Tab/Shift+Tab reachability, Create/Copy confirmation, Update confirm/cancel, stale return-to-review, and focus restoration at normal and 80x24 geometry. Capture only path-free terminal evidence and restore/compare a decoy real-profile fingerprint.

- [ ] **Step 4: Run static and repository gates**

  Run scoped Ruff, formatter check, compileall, `git diff --check`, generated diagnostic inventory, privacy/architecture/governance gates, migration packaging gates if applicable, and the exact affected regression set. Record exact pass/fail/skip counts; do not claim a full suite unless one ran.

- [ ] **Step 5: Complete backlog hygiene and commit**

  Check all eight acceptance criteria, add concise Implementation Notes with the ADR link, verification evidence, modified boundaries, trade-offs, and any deviations. Add a lesson only if this task produced a concrete reusable incident. Set TASK-19059 Done, self-review the diff, and commit `docs: complete Actor Pack import task`.
