# Self-contained Actor Pack Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export one eligible local Character or Persona as a deterministic, self-contained `.tldw-actor-pack` assembled from one exact actor, portrait, portable-identity, and active-visual authority snapshot.

**Architecture:** Reuse TASK-19057's canonical document, inventory, digest, portrait, and ZIP metadata contracts. Add one synchronous export service for exact snapshotting and archive construction, one narrow filesystem publisher for no-follow atomic replacement, and one app-owned async Workbench operation that runs the synchronous boundary off-loop and revalidates its screen/profile/actor authority after every await. Shared Visual Identity and Persona Visual remain typed, independently validated sections; export copies their active immutable manifests/assets but does not merge their runtimes or introduce import activation.

**Tech Stack:** Python 3.12, stdlib `dataclasses`/`hashlib`/`json`/`os`/`pathlib`/`tempfile`/`zipfile`, existing ChaChaNotes/Visual Identity/Persona Visual repositories and confined asset loaders, Textual 8, pytest, Ruff.

**ADR required:** no

**ADR path:** `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md`

**Reason:** ADR-074 already fixes the local-only eligibility, portable UUID, self-contained typed visual sections, deterministic archive, snapshot authority, and publication semantics. This task introduces no new long-lived boundary.

---

## Scope guard

- Export only local Characters and local Personas. Server-backed Personas remain ineligible and receive the exact `Save a local copy first` guidance.
- Reuse `tldw_chatbook.Actor_Packs.contracts`; do not duplicate canonical JSON, path, inventory, digest, portrait, or ZIP metadata logic.
- Include only the selected actor, its authoritative portrait, and currently active immutable visual sections. Do not export chats, local record IDs, deleted state, provider/config/session/UI data, private diagnostics, or host paths.
- Do not implement ZIP import, extraction, review, Create Copy, Update Existing, activation, server transport, or a reusable visual library. Those remain TASK-19059 or explicit non-goals.
- Prefer two focused production modules (`export.py`, `publication.py`) over a framework of adapters/factories. Use injected callables only at existing authority/loader seams that tests must deterministically race.

## Task 1: Capture one exact eligible actor snapshot

**Files:**

- Create: `tldw_chatbook/Actor_Packs/export.py`
- Modify: `tldw_chatbook/Actor_Packs/__init__.py`
- Create: `Tests/Actor_Packs/test_actor_pack_export.py`

- [x] **Step 1: Write the actor snapshot REDs**

  Add named tests for local Character and Persona capture, deleted/missing actors,
  server Persona refusal, inactive-but-local Persona eligibility, portrait validation
  before UUID assignment, existing UUID reuse, and missing registry-row eligibility.

- [x] **Step 2: Run the RED**

  Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Actor_Packs/test_actor_pack_export.py -k 'snapshot or identity or eligibility'`

  Expected: collection fails because `tldw_chatbook.Actor_Packs.export` does not exist.

- [x] **Step 3: Implement the minimum immutable boundary**

  Add frozen/slotted `ActorPackExportSnapshot`, `ActorPackExportResult`, and fixed-category `ActorPackExportError`. The synchronous service must validate actor and portrait before calling `ActorPackRepository.assign_identity`, then freeze actor kind/id/revision, portable UUID/version, portrait bytes/digest/MIME, profile/source identity, and canonical actor payload. Public reprs omit actor text, bytes, paths, and local ids where they are not required for authority.

  Public seam:

  ```python
  class ActorPackExportService:
      def capture_snapshot(
          self,
          actor_kind: str,
          local_actor_id: str,
          *,
          source: str,
      ) -> ActorPackExportSnapshot: ...
  ```

- [x] **Step 4: Re-read after UUID assignment before freezing**

  Assignment is a durable prerequisite, not snapshot publication. Immediately after
  an existing/new UUID is returned, re-read the complete actor and portrait authority
  and compare source/profile, actor revision/content digest, portrait identity/digest,
  and UUID/version. Freeze only the post-assignment values. A deterministic phase
  barrier must prove an actor/portrait mutation between initial admission and identity
  assignment fails with `actor_pack_export_authority_changed` rather than producing a
  mixed snapshot. Task 2 extends this same post-assignment reread to both active visual
  graphs before the final complete snapshot is returned.

- [x] **Step 5: Run GREEN and adjacent repository tests**

  Run the focused export file plus `Tests/Actor_Packs/test_actor_pack_repository.py` and confirm all pass.

- [x] **Step 6: Mutation-check eligibility and assignment order**

  Temporarily move UUID assignment before portrait admission and weaken the local-source guard; confirm the named tests fail, then restore.

- [x] **Step 7: Commit**

  Commit: `feat: capture Actor Pack export snapshots`

## Task 2: Materialize self-contained visual sections

**Files:**

- Modify: `tldw_chatbook/Actor_Packs/export.py`
- Modify: `Tests/Actor_Packs/test_actor_pack_export.py`
- Modify: `tldw_chatbook/Persona_Visual/repository.py`
- Test: `Tests/Persona_Visual/test_persona_visual_repository.py`
- Test: `Tests/ChaChaNotesDB/test_visual_identity_repository.py`

- [x] **Step 1: Write visual-section REDs**

  Cover Character Shared Visual Identity, Persona Shared Visual Identity, Persona Visual, Persona with both sections, no-section actor, immutable active-version selection, missing declared assets, malformed stored manifests, and source-file substitution/content ABA between snapshot and final revalidation.

- [x] **Step 2: Run the RED and inspect exact missing seam**

  Run the visual subset of `test_actor_pack_export.py`; expect missing section capture/materialization failures only.

- [x] **Step 3: Add the missing bounded Persona Visual export read seam**

  `PersonaVisualRepository.get_active_persona_pack()` intentionally hides
  `source_context_json` and asset storage keys from runtime consumers. Add one narrow
  `get_active_persona_pack_for_export()` read contract that returns the same exact
  active identity plus validated bounded licence/provenance scalar metadata and
  immutable asset records with their private storage keys hidden from repr. It must
  use one repository read transaction, reject corrupt stored context/storage values
  with fixed categories, and expose no host path, bytes, local profile path, or server
  identifier. Prove the ordinary runtime graph remains unchanged and path-free.

- [x] **Step 4: Reuse existing repositories/loaders**

  Read active graphs through `VisualIdentityRepository.get_active_actor_pack` and
  `PersonaVisualRepository.get_active_persona_pack_for_export`. Copy manifest,
  licence/provenance, and every declared asset through their existing confined
  profile/package loaders. Canonically remap archive names; never expose storage keys
  or host paths in the manifest or public error.

  For Persona Visual, consume only the new bounded export graph and load each asset
  through `load_persona_visual_asset` using the exact identity/asset/storage tuple.
  For Shared Visual Identity, parse the repository version manifest through the
  existing manifest validator and load every exact asset through
  `load_visual_identity_asset`; do not treat raw repository dictionaries as trusted.

- [x] **Step 5: Freeze complete authority**

  Store the full graph/version/binding identities, canonical manifest digest, per-asset immutable record/digest/size, and pinned source filesystem identity needed for a final exact reread. Any missing asset fails `actor_pack_export_asset_unavailable`; no thin section is emitted.

  After all assets are loaded, re-read actor/portrait/portable identity and both active
  visual graphs together and compare them with the post-assignment capture before
  returning the complete snapshot. This is the snapshot's consistency point; a
  visual publication or actor edit at any phase fails closed.

- [x] **Step 6: Run GREEN and mutation checks**

  Remove one asset, change an active binding/version, swap a source inode, and mutate same-inode content behind a deterministic phase barrier. Each must fail without returning archive bytes.

- [x] **Step 7: Commit**

  Commit: `feat: export Actor Pack visual sections`

## Task 3: Build byte-identical archives and independent readback

**Files:**

- Modify: `tldw_chatbook/Actor_Packs/export.py`
- Modify: `Tests/Actor_Packs/test_actor_pack_export.py`
- Create: `Tests/Actor_Packs/fixtures/export-golden/README.md`
- Create: `Tests/Actor_Packs/fixtures/export-golden/minimal-character.tldw-actor-pack`
- Create: `Tests/Actor_Packs/fixtures/export-golden/minimal-persona.tldw-actor-pack`

- [x] **Step 1: Write deterministic archive REDs**

  Assert exact `ZIP_STORED`, canonical member order, frozen timestamps/creator/flags/permissions, root manifest last construction with its required self-exclusions, bounded chunked writes, and byte-identical output for identical canonical snapshots.

- [x] **Step 2: Run RED**

  Expect the archive-build API and golden fixtures to be absent.

- [x] **Step 3: Implement with stdlib `zipfile` only**

  Build the files mapping through `canonicalize_actor_payload`, `build_file_inventory`, and `actor_pack_content_digest`; validate with `validate_actor_pack_document` before writing. Create each `ZipInfo` from the TASK-19057 constants and stream member data in a fixed bounded chunk size. Do not add compression or a new dependency.

  ```python
  def write_actor_pack_archive(snapshot: ActorPackExportSnapshot, sink: BinaryIO) -> str:
      """Write deterministic bytes and return the archive SHA-256."""
  ```

- [x] **Step 4: Add an independent readback oracle**

  The test oracle must use stdlib `zipfile`/`json`/`hashlib` directly, not production validation helpers, to check entry metadata, declared bytes, per-file digests, content digest, and absence of forbidden actor/private fields.

- [x] **Step 5: Prove round trips without import activation**

  Cover minimal actor+portrait, Character, Persona, Shared Visual Identity only, Persona Visual only, and both visual sections. Confirm no actor/database/profile mutation occurs during readback.

  Also force archive validation/writing to fail after a newly assigned portable UUID
  and prove the UUID remains durably assigned while no destination/archive is
  published.

- [x] **Step 6: Mutation-check determinism/privacy and commit**

  Change order/timestamp/self-inventory/digest exclusion and inject a forbidden local field; each independent oracle must fail. Restore and commit: `feat: write deterministic Actor Pack archives`.

## Task 4: Publish atomically with pinned cleanup authority

**Files:**

- Create: `tldw_chatbook/Actor_Packs/publication.py`
- Create: `Tests/Actor_Packs/test_actor_pack_export_publication.py`
- Modify: `tldw_chatbook/Actor_Packs/export.py`
- Modify: `tldw_chatbook/Actor_Packs/__init__.py`

- [x] **Step 1: Write publication REDs**

  Cover same-directory owned temporary creation, destination symlink/substitution,
  expected destination nonexistence or exact no-follow identity, existing destination
  preservation before commit, source/profile/actor/UUID/binding/version/asset
  authority changes, file fsync before replace, parent fsync after replace where
  supported, pre-commit failure/cancellation cleanup, and a platform capability
  fallback that refuses unverifiable publication.

- [x] **Step 2: Run RED**

  Expect missing `publish_actor_pack` and fixed-category publication error.

- [x] **Step 3: Implement the narrow publisher**

  Capture the destination contract when the user confirms the picker: either the name
  must remain absent or its exact no-follow identity must remain unchanged. Open the
  destination parent and temporary file without following links where supported,
  write+flush+fsync, invoke the final complete snapshot and destination revalidation,
  and check cancellation immediately before `os.replace`.

  `os.replace` is the publication commit point. All ordinary failure/stale/cancel paths
  occur before it and leave the destination untouched. After it, defer cancellation
  until the parent-directory fsync attempt completes; never remove or roll back the
  committed destination. An unexpected supported parent-fsync failure returns a
  fixed `actor_pack_export_durability_uncertain` result with `committed=True`, so the
  caller cannot misreport an untouched destination or retry blindly. Unsupported
  directory fsync is an explicitly tested platform capability outcome, not a generic
  swallowed error. Cleanup may remove only the exact owned pre-commit temporary
  inode/name; any ambiguous fallback fails closed.

- [x] **Step 4: Run GREEN under real filesystem races**

  Use deterministic barriers for destination link swap, parent replacement, temporary substitution, source inode/content ABA, and authority revision changes.

- [x] **Step 5: Mutation-check ordering and cleanup identity**

  Move revalidation before the final await/barrier, omit file fsync, weaken expected
  destination/temp identity, treat post-replace fsync failure as uncommitted, or delete
  the committed destination; each named test must fail.

- [x] **Step 6: Commit**

  Commit: `feat: publish Actor Packs atomically`

## Task 5: Add the app-owned asynchronous export operation

**Files:**

- Modify: `tldw_chatbook/app.py`
- Create: `tldw_chatbook/Actor_Packs/controller.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Create: `Tests/Actor_Packs/test_actor_pack_export_controller.py`
- Create: `Tests/UI/test_actor_pack_export_workflow.py`
- Create: `Tests/UI/test_actor_pack_export_ownership.py`

- [x] **Step 1: Write async ownership REDs**

  Pin one app-owned export operation registry/controller, no duplicate submit,
  archive/hash/decode/file work off-loop, post-await fences for
  profile/source/actor/revision/UUID/visual/destination authority, cancellation
  signaling, shielded repeated-cancellation drain, navigation cancellation without
  transferring ownership to a screen, and shutdown before profile/DB teardown.

- [x] **Step 2: Run RED**

  Expect the app export owner and Workbench operation seam to be absent.

- [x] **Step 3: Wire the existing synchronous services once**

  Construct `ActorPackExportController` after Actor Pack/visual repositories are
  available. The controller owns the operation task, cancellation event, destination
  contract, result ledger, same-owner serialization, and `shutdown()`; it registers
  the inner task before its first await and shields/drains through repeated outer
  cancellation before releasing serialization. The Workbench submits one immutable
  request and operation token, requests cancellation on navigation/replacement, and
  applies a returned result only after exact screen/session/profile/selection fences.
  The controller retains no screen, widget, callback that closes over a screen, or
  user/model content beyond the bounded actor/export snapshot required by the active
  operation.

- [x] **Step 4: Run GREEN and lifecycle mutations**

  Test navigation, profile switch, actor edit, visual publication, destination change,
  repeated cancellation, late result delivery to a replacement screen, and app
  shutdown through deterministic barriers. Mutate each controller and UI-result fence
  plus the drain; the corresponding test must fail.

- [x] **Step 5: Commit**

  Commit: `feat: coordinate Actor Pack exports`

## Task 6: Expose one eligible Workbench action

**Files:**

- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `Tests/UI/test_personas_inspector_pane.py`
- Modify: `Tests/UI/test_actor_pack_export_workflow.py`

- [x] **Step 1: Write Inspector/Pilot REDs**

  Assert a labelled `Export Actor Pack` action for an eligible local Character or
  Persona, disabled only for missing/deleted/corrupt selections, enabled for inactive
  local Personas and actors without registry rows, exact Save Local Copy guidance for
  server Personas, no highlight retargeting across await, no forbidden/reserved
  keybinding, and usable labelled/focusable controls at normal and 80x24 geometry.

- [x] **Step 2: Run RED**

  Expect the message/action/button and handler to be absent.

- [x] **Step 3: Implement the smallest UI seam**

  Add one frozen/slotted request message and one Inspector button. Reuse the existing
  export destination picker, but add Actor Pack-specific suffix normalization: append
  `.tldw-actor-pack` exactly once, preserve an exact existing suffix
  case-insensitively, reject a directory target, and capture absent-or-exact existing
  destination identity only after explicit overwrite confirmation. The handler
  captures exact selection/session/profile authority before opening the picker and
  revalidates it before submitting the immutable controller request.

- [x] **Step 4: Run GREEN**

  Exercise success, cancel, failure, server source, actor-selection ABA, navigation, and duplicate-submit behavior. User-facing failures and logs remain fixed-category/path-free; success may name only the user-selected destination basename.

- [x] **Step 5: Run Impeccable once after the final visible change**

  Run the project Impeccable detector on the Inspector/Personas screen once; address scoped findings before finalizing and retain its exact evidence.

- [x] **Step 6: Commit**

  Commit: `feat: export Actor Packs from Workbench`

## Task 7: Closeout and verification

**Files:**

- Modify: `Tests/Architecture/test_actor_pack_boundary.py`
- Modify: `backlog/tasks/task-19058 - Export-self-contained-Actor-Packs.md`
- Modify if this task produces a genuinely reusable incident: `backlog/docs/lessons-testing-evidence.md`

- [x] **Step 1: Extend architecture/privacy guards**

  Assert the pure Actor Pack export modules import no Textual/UI/server/import-activation boundary; UI owns only orchestration. Assert archives and stable errors contain no local IDs, chats, credentials, provider settings, host paths, session/UI preferences, cleanup tokens, or private diagnostics.

- [x] **Step 2: Run focused component gates under isolated roots**

  Establish one temporary HOME/XDG/config/data root before interpreter start. Run all `Tests/Actor_Packs`, the Actor Pack migration/repository tests, affected Shared Visual Identity and Persona Visual repository/loader tests, the focused Workbench/Pilot tests, `Tests/test_probe_import_provenance.py`, packaging tests, and Actor Pack architecture/privacy tests. Record exact pass/fail/skip counts; do not claim a full suite.

- [x] **Step 3: Run independent golden/readback and real SQLite gates**

  Export Character, Persona, and both-visual-section fixtures from real SQLite/profile state; read them through the independent oracle. Repeat identical export and compare exact bytes/SHA-256. Verify authority/cancellation/privacy mutations remain discriminating.

- [x] **Step 4: Run static/governance gates**

  Run scoped Ruff check and formatter check on every changed Python file, `py_compile`/`compileall` on changed production modules, packaging/diagnostic/privacy/architecture/governance gates, placeholder scans, and `git diff --check`.

- [x] **Step 5: Self-review and close the task**

  Review the diff against all eight ACs and ADR-074. Add concise Implementation Notes, check every AC only after its evidence is green, set TASK-19058 to Done with Backlog CLI, verify `backlog task 19058 --plain`, and commit: `docs: complete Actor Pack export task`.
