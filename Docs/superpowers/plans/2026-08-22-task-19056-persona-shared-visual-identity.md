# Persona Shared Visual Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan sequentially. Every production behavior starts with `superpowers:test-driven-development`; use `superpowers:verification-before-completion` before each commit and completion claim.

**Goal:** Make the already-declared `actor_kind = "persona"` Shared Visual Identity path operational for eligible local Personas, including immutable authoring/publication, deterministic resolution, Workbench controls, and Console rendering, while keeping Persona Visual/Buddy operational states completely separate.

**Architecture:** Retain one Shared Visual Identity model and repository. Add a small local-Persona authority adapter that snapshots the exact local source, Persona ID/revision/eligibility, and linked Character portrait identity. Refactor the existing Character-only resolver and candidate publisher around actor-generic pack selection plus actor-specific fallback/authority; do not duplicate the repository, schema, manifest validator, asset loader, or publication filesystem state machine. Personas Workbench owns draft/UI lifetime and passes a read-only exact Persona guard into the existing in-transaction publication guard. Console extends its existing actor-scoped cache/controller seam to local Persona sessions; Persona Buddy and Persona Visual receive no Shared Visual Identity expressions or state mappings.

**Tech Stack:** Python 3.11+, Textual 8, SQLite, existing `VisualIdentityRepository`, Pillow/Rich Pixels, local Persona JSON service, pytest/Pilot.

**ADR required:** no

**ADR path:** Existing [ADR-067](../../../backlog/decisions/067-bundled-samira-visual-identity-pack.md) and [ADR-074](../../../backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md)

**Reason:** ADR-067 already declares Persona bindings and the immutable Shared Visual Identity contract; ADR-074 explicitly requires Shared Visual Identity expressions to remain separate from Persona Visual/Buddy operational states. This task implements those accepted boundaries without a new architectural decision.

---

## Fixed contracts and file map

- `tldw_chatbook/DB/VisualIdentity_DB.py` is already actor-generic. Change it only if a born-RED exact-CAS test proves the existing `activate_pack`/`publish_version` guard cannot express required Persona authority.
- `tldw_chatbook/Character_Chat/visual_identity.py` remains the single manifest, asset, immutable version, resolver, candidate, publication, and cleanup engine. Extract only bounded actor-specific helpers; do not create a second Persona visual-identity runtime.
- Create `tldw_chatbook/Character_Chat/persona_visual_identity.py` for exact local-Persona authority capture/revalidation and linked Character portrait loading. It may depend on the local Persona/Character service contract, but never on Textual, Persona Buddy, or Persona Visual.
- `tldw_chatbook/Widgets/Persona_Widgets/personas_visual_identity_pack_widget.py` remains the path-free Shared Visual Identity browser. Add actor-mode copy/allowed actions rather than clone the widget.
- `tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py` mounts the Shared Visual Identity browser beside (not inside) `PersonasPersonaVisualPackWidget`; their IDs, messages, state vocabularies, drafts, and Save/Cancel ownership remain distinct.
- `tldw_chatbook/UI/Screens/personas_screen.py` owns weak editor/browser references, source/profile/editor generations, one unpublished candidate, cancellation/drain, and post-publication targeted invalidation.
- `tldw_chatbook/UI/Console_Modules/session.py`, `character.py`, and `wiring.py` keep the existing actor-scoped reaction/cache controller. Extend actor scope and display naming to eligible local Persona sessions; do not route expressions into `PersonaBuddyConsoleAdapter` or `PersonaBuddyController`.
- Server-backed Personas always show exactly `Save a local copy first`; they never read/write local bindings in place.
- Existing Character resolver order, legacy-expression fallback, four operational states, authoring, publication, cache keys, and Console behavior are non-regression contracts.
- No Actor Pack archive, Persona Visual schema/runtime, Buddy state, server write/sync, or new image-generation provider work is authorized.
- User instruction: run the complete touched/modified component surface and governance gates, not full-repository pytest.

## Task 1: Freeze local Persona authority and actor-generic resolution

**Files:**

- Create: `tldw_chatbook/Character_Chat/persona_visual_identity.py`
- Modify: `tldw_chatbook/Character_Chat/visual_identity.py`
- Create: `Tests/Character_Chat/test_persona_visual_identity_resolution.py`
- Modify: `Tests/Character_Chat/test_visual_identity_resolution.py`

- [ ] **Step 1: Write authority-adapter REDs**

  Add named tests:

  - `test_capture_requires_exact_local_persona_id_revision_and_active_state`
  - `test_capture_rejects_deleted_disabled_missing_and_server_records`
  - `test_linked_character_portrait_is_bounded_and_path_free`
  - `test_authority_revalidation_detects_revision_aba_and_linked_portrait_change`

  Pin frozen/slotted public values with no record dictionaries or paths in repr:

  ```python
  authority = capture_local_persona_visual_identity(service, "p-1")
  assert authority.source == "local"
  assert authority.persona_id == "p-1"
  assert authority.persona_revision == 4
  assert authority.portrait is not None
  assert local_persona_visual_identity_is_current(service, authority) is True
  ```

- [ ] **Step 2: Run the authority RED**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Character_Chat/test_persona_visual_identity_resolution.py -k 'capture or authority or portrait'
  ```

  Expected: collection fails because `persona_visual_identity` does not exist.

- [ ] **Step 3: Implement the minimal authority adapter**

  Implement exact-type validation for source/ID/revision/deleted/active fields. Reuse the linked local Character-card portrait contract (ID, Character revision, bounded bytes, MIME, SHA-256) without importing Persona Buddy. Revalidation reloads the same Persona and linked Character and compares the complete frozen authority; it never logs record contents, paths, or bytes.

- [ ] **Step 4: Write actor-generic resolver REDs**

  Add:

  - `test_persona_bound_pack_resolves_manual_requested_default_and_neutral_order`
  - `test_persona_missing_pack_asset_falls_back_to_linked_portrait`
  - `test_persona_unavailable_or_changed_authority_returns_actor_unavailable`
  - `test_persona_cache_identity_contains_full_actor_binding_version_asset_and_portrait_identity`
  - `test_persona_resolution_does_not_read_character_legacy_expression_rows`
  - `test_character_resolution_cache_and_fallback_contract_is_byte_unchanged`

  The public Persona entry point is synchronous for `to_thread` callers:

  ```python
  result = resolve_persona_visual_identity(
      db,
      local_service,
      persona_id="p-1",
      requested_state="speaking",
      manual_expression_key=None,
      user_data_dir=profile_root,
  )
  assert result.actor_kind == "persona"
  assert "persona_revision=4" in result.cache_identity
  ```

- [ ] **Step 5: Run the resolver RED**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Character_Chat/test_persona_visual_identity_resolution.py Tests/Character_Chat/test_visual_identity_resolution.py
  ```

  Expected: Persona cases fail on missing resolver/Character-only rejection; all pre-existing Character cases remain green.

- [ ] **Step 6: Refactor the resolver without changing Character behavior**

  Extract the bounded active-pack candidate query so both actor kinds use the same binding/version/assets/fallback walk. Character keeps legacy-expression and card-portrait fallback. Persona uses no legacy-expression table and falls back only to its exact linked portrait, then placeholder. Persona results include exact source, Persona revision, linked portrait identity, binding ID/version, pack ID/revision, immutable version ID/number/manifest digest, selected asset ID/digest, request/manual/resolved keys, and fallback source in `cache_identity`. Revalidate Persona authority after pack/asset reads before returning bytes.

- [ ] **Step 7: Run GREEN and mutations**

  Run Step 5. Then independently remove the post-read Persona revalidation, omit binding version from the cache identity, and permit a Character legacy row for Persona; each named test must fail before restoration.

- [ ] **Step 8: Commit Task 1**

  ```bash
  git add tldw_chatbook/Character_Chat/persona_visual_identity.py tldw_chatbook/Character_Chat/visual_identity.py Tests/Character_Chat/test_persona_visual_identity_resolution.py Tests/Character_Chat/test_visual_identity_resolution.py
  git commit -m "feat: resolve Persona Shared Visual Identity"
  ```

## Task 2: Support unbound Persona creation and authority-fenced publication

**Files:**

- Modify: `tldw_chatbook/Character_Chat/visual_identity.py`
- Modify: `tldw_chatbook/Character_Chat/persona_visual_identity.py`
- Modify only if RED requires: `tldw_chatbook/DB/VisualIdentity_DB.py`
- Create: `Tests/Character_Chat/test_persona_visual_identity_publication.py`
- Modify: `Tests/Character_Chat/test_visual_identity_publication.py`
- Modify: `Tests/ChaChaNotesDB/test_visual_identity_repository.py`

- [ ] **Step 1: Write candidate REDs**

  Add:

  - `test_unbound_local_persona_can_create_empty_canonical_candidate`
  - `test_bound_persona_candidate_snapshots_exact_binding_and_actor_authority`
  - `test_persona_candidate_rejects_server_deleted_disabled_missing_and_stale_actor`
  - `test_candidate_replace_clear_and_cancel_leave_active_version_unchanged`

  Extend `VisualIdentityCandidate` only as needed to represent `old_* = None` for a new binding and a frozen Persona authority token. Preserve the existing Character constructor behavior and errors.

- [ ] **Step 2: Run candidate REDs**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Character_Chat/test_persona_visual_identity_publication.py -k 'candidate or replace or clear or cancel'
  ```

  Expected: Persona candidate creation is rejected as `visual_identity_actor_kind_invalid`.

- [ ] **Step 3: Implement actor-generic candidates**

  Bound Persona candidates clone the active immutable graph exactly. Unbound Persona candidates start from the canonical approved expression metadata with no asset bytes and can stage replacements; publication requires a valid default/neutral resolution and at least one validated asset. Do not invent a Persona-only manifest or state catalog. Candidate methods remain in-memory and cancellation discards only unpublished state.

- [ ] **Step 4: Write publication/CAS REDs**

  Add real-SQLite tests:

  - `test_unbound_persona_publish_creates_one_pack_version_assets_and_binding`
  - `test_bound_persona_publish_appends_immutable_version_and_preserves_old_rows`
  - `test_persona_revision_source_binding_version_and_portrait_change_fail_closed`
  - `test_persona_authority_is_rechecked_inside_reserved_sqlite_transaction`
  - `test_failed_cancelled_or_stale_publish_keeps_prior_binding_and_cleans_owned_staging`
  - `test_character_publication_contract_remains_unchanged`

  Use an injected read-only actor guard alongside the existing filesystem `publication_guard`. The final repository guard must check both while the SQLite write reservation is held.

- [ ] **Step 5: Run publication REDs**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Character_Chat/test_persona_visual_identity_publication.py Tests/Character_Chat/test_visual_identity_publication.py Tests/ChaChaNotesDB/test_visual_identity_repository.py
  ```

  Expected: new Persona cases fail; existing Character/repository cases pass.

- [ ] **Step 6: Implement publication through the existing state machine**

  For an unbound Persona call `VisualIdentityRepository.activate_pack` with no expected binding and the combined actor/filesystem guard. For a bound candidate keep the existing copy-on-write/single-binding decisions and exact binding/version CAS. Run the optional actor guard early and again inside the repository transaction; false/raise maps to a fixed `visual_identity_actor_changed` category. Preserve owned staging cleanup/capability behavior and immutable old versions.

- [ ] **Step 7: Run GREEN and mutations**

  Run Step 5. Mutate away the in-transaction actor guard, permit publication after Persona revision ABA, and overwrite rather than append an immutable version; each dedicated test must fail before restoration.

- [ ] **Step 8: Commit Task 2**

  ```bash
  git add tldw_chatbook/Character_Chat/visual_identity.py tldw_chatbook/Character_Chat/persona_visual_identity.py tldw_chatbook/DB/VisualIdentity_DB.py Tests/Character_Chat/test_persona_visual_identity_publication.py Tests/Character_Chat/test_visual_identity_publication.py Tests/ChaChaNotesDB/test_visual_identity_repository.py Docs/superpowers/plans/2026-08-22-task-19056-persona-shared-visual-identity.md
  git commit -m "feat: publish Persona expression packs"
  ```

## Task 3: Add the Persona Workbench Shared Visual Identity editor

**Files:**

- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_visual_identity_pack_widget.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Create: `Tests/UI/test_personas_persona_visual_identity_pack.py`

- [ ] **Step 1: Write widget/eligibility REDs**

  Add:

  - `test_persona_editor_keeps_shared_identity_and_persona_visual_as_separate_sections`
  - `test_local_persona_shows_path_free_metadata_lazy_preview_and_manual_labels`
  - `test_unbound_local_persona_offers_create_replace_clear_save_cancel`
  - `test_server_persona_disables_shared_identity_with_save_local_copy_first`
  - `test_normal_and_80x24_compact_layout_paints_labelled_focusable_actions`

  Reuse the existing widget/messages. Add an actor mode only for truthful copy and action availability; do not clone DOM IDs or introduce Persona Visual state keys.

- [ ] **Step 2: Run widget REDs**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_personas_persona_visual_identity_pack.py Tests/UI/test_personas_visual_identity_pack.py
  ```

  Expected: Persona editor has no Shared Visual Identity section.

- [ ] **Step 3: Mount the distinct Workbench section**

  Add a separate Shared Visual Identity holder below the profile fields and before/after the existing Persona Visual section with explicit titles (`Shared Visual Identity reactions` versus `Persona Visual operational states`). Keep selected-only preview decode and plain/path-free labels. No implicit terminal-convention bindings.

- [ ] **Step 4: Write screen authority/cancellation REDs**

  Add barrier tests:

  - `test_persona_metadata_stale_after_profile_read_does_not_mount`
  - `test_persona_preview_stale_after_resolve_or_decode_does_not_paint`
  - `test_persona_replace_clear_and_save_revalidate_source_session_actor_binding_version_and_profile_revision_after_every_await`
  - `test_declined_dirty_navigation_preserves_draft_and_staging`
  - `test_accepted_navigation_and_cancel_signal_and_drain_before_discard`
  - `test_failed_or_cancelled_save_preserves_active_version_and_draft_truthfully`
  - `test_successful_save_invalidates_only_exact_actor_old_and_new_identity`

- [ ] **Step 5: Run screen REDs**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_personas_persona_visual_identity_pack.py Tests/UI/test_personas_workbench.py -k 'persona and (visual_identity or shared_visual or dirty_navigation)'
  ```

- [ ] **Step 6: Implement one Persona editor-owned candidate lifetime**

  Capture weak editor/browser references, exact local service/DB identity, source, Persona ID/revision, screen/editor-session generations, and binding/pack/version identities. After each `get_persona_profile`, graph read, resolve/decode, picker result, staging operation, publication, cleanup, and invalidation await, compare the complete snapshot. Use the incumbent `_drain_to_thread`/`_drain_async` behavior; register work before awaiting, signal cancellation, shield/drain uncancellable work, and release serialization only after drain. Accepted navigation or Cancel discards only the unpublished candidate; failed publication keeps the authoritative active version.

- [ ] **Step 7: Run GREEN, Pilot, and mutations**

  Run Steps 2 and 5 at 80x24 and a normal/wide Pilot size. Remove one await fence, skip drain, allow server source, and invalidate all actor caches one at a time; each named test must fail before restoration.

- [ ] **Step 8: Commit Task 3**

  ```bash
  git add tldw_chatbook/Widgets/Persona_Widgets/personas_visual_identity_pack_widget.py tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_persona_visual_identity_pack.py Docs/superpowers/plans/2026-08-22-task-19056-persona-shared-visual-identity.md
  git commit -m "feat: author Persona expression packs"
  ```

## Task 4: Render Persona expressions in Console without Buddy coupling

**Files:**

- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `tldw_chatbook/UI/Console_Modules/character.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify only if required by RED: `tldw_chatbook/UI/Screens/chat_screen.py`
- Create: `Tests/UI/test_console_persona_visual_identity.py`
- Modify: `Tests/UI/test_console_character_avatar.py`
- Modify: `Tests/UI/test_console_reaction_picker.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`

- [ ] **Step 1: Write local Persona session/scope REDs**

  Add:

  - `test_local_persona_start_chat_creates_exact_persona_session_authority`
  - `test_server_persona_start_chat_never_claims_local_visual_identity`
  - `test_current_visual_identity_scope_returns_local_persona_id_without_integer_coercion`
  - `test_persona_replacement_clears_only_prior_session_manual_reaction`

  Preserve the existing Character handoff/session path byte-for-behavior. If native Persona Start Chat is not required to make an existing Persona session reachable, keep session creation out of scope and construct the existing `assistant_kind="persona"` path in tests instead; decide from the born-RED product path, not by assumption.

- [ ] **Step 2: Write Console render/cache REDs**

  Add:

  - `test_console_persona_expression_resolves_and_paints_active_asset`
  - `test_console_persona_manual_reaction_is_session_and_actor_scoped`
  - `test_persona_publication_invalidates_only_matching_actor_cache`
  - `test_persona_source_revision_binding_or_asset_change_drops_stale_decode`
  - `test_persona_expression_never_changes_persona_buddy_state_or_lease_maps`
  - `test_character_console_avatar_and_four_operational_states_remain_unchanged`

- [ ] **Step 3: Run Console REDs**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_console_persona_visual_identity.py Tests/UI/test_console_character_avatar.py Tests/UI/test_console_reaction_picker.py Tests/UI/test_console_native_chat_flow.py -k 'visual_identity or reaction or avatar or persona_start_chat'
  ```

- [ ] **Step 4: Extend the actor-scoped controller minimally**

  Make `_current_visual_identity_actor_scope()` return `(session_id, actor_kind, actor_id)` for exact eligible local Character or Persona sessions. Remove `int(actor_scope[2])` assumptions from `ConsoleCharacterController`; display specs carry generic `actor_kind`/`actor_id`. Resolve Persona through the new Persona entry point off-loop. Retain complete `resolution.cache_identity`, perform the existing second resolution after decode, and fence current session/source/actor/manual/state before DOM mutation. Targeted invalidation continues matching `actor_kind` and `actor_id` tokens only.

- [ ] **Step 5: Prove runtime separation**

  Add an architecture/static test that `Character_Chat.visual_identity`, the new Persona authority adapter, and Console reaction paths do not import `Persona_Buddy`, `Persona_Visual`, Actor Pack, or server write modules. Add behavioral proof that selecting/manual-changing a Shared Visual Identity expression leaves Buddy controller state/leases/generation unchanged.

- [ ] **Step 6: Run GREEN and mutations**

  Run Step 3. Mutate away actor kind in cache invalidation, coerce Persona IDs to integers, accept server Persona scope, and route a reaction to the Buddy sink; each dedicated test must fail before restoration.

- [ ] **Step 7: Commit Task 4**

  ```bash
  git add tldw_chatbook/UI/Console_Modules/session.py tldw_chatbook/UI/Console_Modules/character.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_persona_visual_identity.py Tests/UI/test_console_character_avatar.py Tests/UI/test_console_reaction_picker.py Tests/UI/test_console_native_chat_flow.py
  git commit -m "feat: render Persona expressions in Console"
  ```

## Task 5: Lifecycle, navigation, and Character non-regression

**Files:**

- Modify: `Tests/Character_Chat/test_persona_visual_identity_resolution.py`
- Modify: `Tests/Character_Chat/test_persona_visual_identity_publication.py`
- Modify: `Tests/UI/test_personas_persona_visual_identity_pack.py`
- Modify: `Tests/UI/test_console_persona_visual_identity.py`
- Modify: `Tests/Character_Chat/test_visual_identity_lifecycle.py`
- Create: `Tests/Architecture/test_persona_shared_visual_identity_boundary.py`

- [ ] **Step 1: Add lifecycle/race REDs**

  Cover disabled/delete/restore/missing, profile revision ABA, local Persona replacement, source local→server→local ABA, binding replacement, concurrent publication, old editor completion after navigation, duplicate submit, repeated outer cancellation, and app/screen teardown before cleanup. Assert stable path-free categories and no mutation/repaint on stale authority.

- [ ] **Step 2: Run lifecycle REDs**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Character_Chat/test_persona_visual_identity_resolution.py Tests/Character_Chat/test_persona_visual_identity_publication.py Tests/UI/test_personas_persona_visual_identity_pack.py Tests/UI/test_console_persona_visual_identity.py Tests/Character_Chat/test_visual_identity_lifecycle.py
  ```

- [ ] **Step 3: Apply only proven lifecycle fixes**

  Use monotonic screen/editor operation generations and exact immutable actor/binding/version/cache tuples. Never add polling, a generic coordinator framework, or cross-runtime invalidation. Restore keeps the prior binding dormant and re-resolves; explicit binding replacement advances exact identity; late views remove/update only themselves.

- [ ] **Step 4: Add architecture/privacy and Character equivalence gates**

  The architecture test pins no Persona Visual/Buddy/Actor Pack/server imports or state-key mapping. Character tests compare resolver/publication/cache/four-state outputs against pre-task fixtures. Added diagnostics contain fixed categories/IDs only; scan added lines for Persona content, prompts, local paths, raw bytes, and cleanup tokens.

- [ ] **Step 5: Run mutations**

  Individually weaken source, Persona revision, editor generation, binding version, post-decode identity, and affected-only invalidation guards. Each corresponding test must fail before restoration.

- [ ] **Step 6: Commit Task 5**

  ```bash
  git add Tests/Character_Chat/test_persona_visual_identity_resolution.py Tests/Character_Chat/test_persona_visual_identity_publication.py Tests/UI/test_personas_persona_visual_identity_pack.py Tests/UI/test_console_persona_visual_identity.py Tests/Character_Chat/test_visual_identity_lifecycle.py Tests/Architecture/test_persona_shared_visual_identity_boundary.py
  git commit -m "test: prove Persona expression lifecycle safety"
  ```

## Task 6: Final isolated verification and task closeout

**Files:**

- Modify: `backlog/tasks/task-19056 - Enable-Shared-Visual-Identity-for-Persona-actors.md`
- Modify only when this task produced a genuinely reusable incident: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`

- [ ] **Step 1: Record assigned-worktree provenance before imports**

  ```bash
  SCRATCH_ROOT="$(mktemp -d /private/tmp/tldw-task19056.XXXXXX)"
  env HOME="$SCRATCH_ROOT/home" XDG_CONFIG_HOME="$SCRATCH_ROOT/config" XDG_DATA_HOME="$SCRATCH_ROOT/data" XDG_CACHE_HOME="$SCRATCH_ROOT/cache" TLDW_CONFIG_PATH="$SCRATCH_ROOT/config/tldw_cli/config.toml" TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/test_probe_import_provenance.py Tests/Architecture/test_persona_shared_visual_identity_boundary.py
  ```

  Expected: imports resolve beneath `.worktrees/task-19056-persona-svi`; architecture passes.

- [ ] **Step 2: Run the complete affected component gate in the same isolated environment**

  ```bash
  env HOME="$SCRATCH_ROOT/home" XDG_CONFIG_HOME="$SCRATCH_ROOT/config" XDG_DATA_HOME="$SCRATCH_ROOT/data" XDG_CACHE_HOME="$SCRATCH_ROOT/cache" TLDW_CONFIG_PATH="$SCRATCH_ROOT/config/tldw_cli/config.toml" TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/ChaChaNotesDB/test_visual_identity_repository.py Tests/Character_Chat/test_visual_identity_resolution.py Tests/Character_Chat/test_visual_identity_publication.py Tests/Character_Chat/test_visual_identity_lifecycle.py Tests/Character_Chat/test_persona_visual_identity_resolution.py Tests/Character_Chat/test_persona_visual_identity_publication.py Tests/UI/test_personas_visual_identity_pack.py Tests/UI/test_personas_persona_visual_identity_pack.py Tests/UI/test_personas_workbench.py Tests/UI/test_console_persona_visual_identity.py Tests/UI/test_console_character_avatar.py Tests/UI/test_console_reaction_picker.py Tests/UI/test_console_native_chat_flow.py Tests/Architecture/test_persona_shared_visual_identity_boundary.py
  ```

  Expected: all affected tests pass. Do not claim or run full-repository pytest.

- [ ] **Step 3: Run static, generated-inventory, and governance gates**

  Build a reviewed list of every changed Python path, then run Ruff check, Ruff format check, and `py_compile`/`compileall` on every changed production/test module. Also run:

  ```bash
  git diff --check origin/dev...HEAD
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Architecture/test_persona_shared_visual_identity_boundary.py Tests/Architecture/test_persistent_diagnostic_inventory.py
  rg -n "Persona_Buddy|Persona_Visual|actor-pack|server.*write" tldw_chatbook/Character_Chat/persona_visual_identity.py tldw_chatbook/Character_Chat/visual_identity.py
  ```

  Review diagnostic-inventory changes semantically; regenerate only if this branch truly changes the inventory. Verify no CSS bundle changes unless visible widget CSS changed; if it did, use the canonical builder and review all generated outputs.

- [ ] **Step 4: Run live/Pilot evidence only under isolation**

  Use production CSS and real SQLite in Pilot at 80x24 and normal/wide sizes. Prove metadata, selected preview, Replace/Clear/Save/Cancel focus and paint, source feedback, Console Persona expression paint, Character non-regression, and navigation cancellation. If an ad hoc interpreter or real TUI probe is needed, set the Step 1 environment before the first Chatbook import and fingerprint the real profile before/after.

- [ ] **Step 5: Self-review scope and evidence**

  Confirm no Persona Visual/Buddy state mapping, Actor Pack archive, server write, schema migration, new provider, unrelated formatting, or default behavior entered the diff. Confirm all eight task ACs have direct evidence. If any scoped gate fails, leave TASK-19056 In Progress and record the exact baseline/branch attribution; do not check ACs or mark Done.

- [ ] **Step 6: Close the Backlog task only after all gates pass**

  Use Backlog CLI first to add concise Implementation Notes/status, then verify the rendered task. Check all eight ACs, retain this plan link, record ADR-067/ADR-074, affected files, RED→GREEN/mutation evidence, exact scoped commands/results, and any truthful deviations. Then:

  ```bash
  backlog task edit 19056 -s Done
  backlog task 19056 --plain
  git diff --check
  git add 'backlog/tasks/task-19056 - Enable-Shared-Visual-Identity-for-Persona-actors.md'
  git commit -m "docs: close Persona Shared Visual Identity task"
  ```
