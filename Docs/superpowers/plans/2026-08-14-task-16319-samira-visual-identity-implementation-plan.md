# Samira Built-in Visual Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Samira “Sammy” Vadem as an included, editable V2 character card with a verified 31-image reaction pack, server-aligned local persistence and resolution, a Console reaction picker, and copy-on-write pack authoring in Personas—without changing the default assistant or Persona.

**Architecture:** Add one additive ChaChaNotes migration and a focused VisualIdentityRepository over the existing database transaction API. Keep normalization, manifest/resource validation, seeding, resolution, and immutable publication in one bounded Character_Chat.visual_identity service module; keep Textual presentation in two focused widgets. Built-in assets are immutable importlib.resources, user edits live under the active profile, and the current three-BLOB legacy path remains the fallback for unbound characters.

**Tech Stack:** Python 3.11, SQLite/FTS5, Pillow, importlib.resources, Textual 8.x, asyncio workers, pytest, setuptools wheel/sdist packaging, Backlog.md.

---

## Governing contracts and execution rules

- Approved design: 'Docs/superpowers/specs/2026-08-14-task-16319-samira-builtin-visual-identity-pack-design.md'
- Accepted architecture: 'backlog/decisions/067-bundled-samira-visual-identity-pack.md'
- Backlog umbrella: 'backlog/tasks/task-16319 - Bundle-Samira-character-and-full-Visual-Identity-reaction-pack.md'
- Child delivery order: TASK-16319.1 → TASK-16319.2 → TASK-16319.3.
- Server reference: tldw_server commit 385afa951922c8a9dc2002c675bb6cad65e4ac23.
- Clean implementation branch/worktree: codex/task-16319-samira-visual-identity in .worktrees/task-16319-samira-visual-identity.
- Baseline evidence already recorded: the focused DB/character/Console/Personas/packaging suite passes with 327 passed and 2 warnings.
- Do not modify the supplied source files in the long-lived worktree. Read them as inputs and create packaged derivatives only in this clean worktree.
- Do not introduce a dependency, duplicate schema initialization in a service, or grow ChaChaNotes_DB.py with Visual Identity CRUD.
- Use apply_patch for source edits. Preserve unrelated user changes.
- At the start of each child task, use Backlog.md to set it In Progress, assign the current implementer, and add that child’s portion of this plan before writing code. Only mark a child Done after all acceptance criteria, notes, tests, static checks, docs, self-review, and ADR links are complete.
- Keep TASK-16319 open until all three children and end-to-end validation are complete.

## Canonical inventory frozen by the plan

The implementation must use this exact ordered inventory:

~~~python
SAMIRA_REACTION_LABELS = (
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "neutral",
    "optimism", "pride", "realization", "relief", "remorse", "sadness",
    "surprise", "thinking", "speaking", "error",
)

SAMIRA_EXPRESSION_KEYS = {
    "anger": "angry",
    "confusion": "confused",
    "excitement": "excited",
    "joy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprised",
    "thinking": "thinking",
    "speaking": "custom:speaking",
    "error": "custom:error",
    # Every remaining label maps to f"custom:{label}".
}
~~~

Operational requests map only at the resolver boundary:

~~~python
OPERATIONAL_EXPRESSION_KEYS = {
    "idle": "neutral",
    "thinking": "thinking",
    "speaking": "custom:speaking",
    "error": "custom:error",
}
~~~

The complete packaged directory is:

~~~text
tldw_chatbook/assets/characters/samira/
├── ASSET_LICENSE.md
├── Samira.character.json
├── Sammy.png
├── visual_identity_pack.json
└── expressions/
    └── <exactly the 31 labels above>.webp
~~~

## TASK-16319.1 — Assets and local Visual Identity persistence

### Task 1: Claim TASK-16319.1 and record its implementation plan

**Files:**
- Modify: 'backlog/tasks/task-16319.1 - Add-Samira-assets-and-local-Visual-Identity-persistence.md'

- [ ] Run 'backlog task edit 16319.1 -a @<current-implementer> -s "In Progress"'.
- [ ] Add an Implementation Plan containing Tasks 2–7 below with 'backlog task edit 16319.1 --plan "..."'.
- [ ] Re-open it with 'backlog task 16319.1 --plain' and verify status, assignee, dependencies, spec link, and ADR-067 link.
- [ ] Record: ADR required: yes; ADR path: backlog/decisions/067-bundled-samira-visual-identity-pack.md; this directly implements the accepted ADR rather than creating another one.

### Task 2: Add the v37→v38 Visual Identity schema migration

**Files:**
- Create: 'tldw_chatbook/DB/migrations/chachanotes_v37_to_v38_visual_identity.sql'
- Modify: 'tldw_chatbook/DB/ChaChaNotes_DB.py'
- Create: 'Tests/ChaChaNotesDB/test_visual_identity_migration.py'
- Modify: 'Tests/Packaging/test_installed_distribution.py'
- Modify: 'pyproject.toml'

- [ ] Write a failing migration test that initializes a v37 fixture, upgrades it, and asserts schema version 38 plus the four tables, shared server field names, checks, indexes, and foreign keys.
- [ ] Assert that visual_identity_assets.pack_version_id is NOT NULL, pack_id retains the pinned server's nullable shape, draft_id is absent, no draft table exists, and the active-binding partial unique index rejects a second active binding for the same owner/actor.
- [ ] Add a fresh-database test proving the production constructor applies the migration and leaves character_expression_images unchanged.
- [ ] Run:

  ~~~bash
  ../../.venv/bin/python -m pytest -q Tests/ChaChaNotesDB/test_visual_identity_migration.py
  ~~~

  Expected: FAIL because version 38 and the tables do not exist.

- [ ] Add the SQL semantic subset, preserving the pinned server vocabulary:

  ~~~sql
  CREATE TABLE visual_identity_packs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      owner_user_id INTEGER NOT NULL,
      title TEXT NOT NULL,
      description TEXT NOT NULL DEFAULT '',
      status TEXT NOT NULL DEFAULT 'active'
          CHECK(status IN ('active', 'archived', 'deleted')),
      active_version_id INTEGER REFERENCES visual_identity_pack_versions(id),
      default_expression_key TEXT NOT NULL DEFAULT 'neutral',
      source_kind TEXT NOT NULL DEFAULT 'manual',
      source_context_json TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      version INTEGER NOT NULL DEFAULT 1
  );

  CREATE TABLE visual_identity_pack_versions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      pack_id INTEGER NOT NULL REFERENCES visual_identity_packs(id),
      owner_user_id INTEGER NOT NULL,
      version_number INTEGER NOT NULL,
      default_expression_key TEXT NOT NULL DEFAULT 'neutral',
      manifest_json TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(pack_id, version_number)
  );

  CREATE TABLE visual_identity_assets (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      owner_user_id INTEGER NOT NULL,
      pack_id INTEGER REFERENCES visual_identity_packs(id),
      pack_version_id INTEGER NOT NULL REFERENCES visual_identity_pack_versions(id),
      expression_key TEXT NOT NULL,
      original_expression_key TEXT NOT NULL DEFAULT '',
      display_label TEXT NOT NULL DEFAULT '',
      source_filename TEXT NOT NULL,
      storage_relpath TEXT NOT NULL,
      content_type TEXT NOT NULL,
      bytes INTEGER NOT NULL CHECK(bytes > 0),
      sha256 TEXT NOT NULL,
      width INTEGER NOT NULL CHECK(width > 0),
      height INTEGER NOT NULL CHECK(height > 0),
      source_context_json TEXT NOT NULL DEFAULT '{}',
      is_animated INTEGER NOT NULL DEFAULT 0 CHECK(is_animated IN (0, 1)),
      frame_count INTEGER,
      duration_ms INTEGER,
      preview_relpath TEXT,
      deleted INTEGER NOT NULL DEFAULT 0 CHECK(deleted IN (0, 1)),
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE visual_identity_bindings (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      owner_user_id INTEGER NOT NULL,
      actor_kind TEXT NOT NULL CHECK(actor_kind IN ('character', 'persona')),
      actor_id TEXT NOT NULL,
      pack_id INTEGER NOT NULL REFERENCES visual_identity_packs(id),
      active_version_id INTEGER NOT NULL REFERENCES visual_identity_pack_versions(id),
      status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'deleted')),
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      version INTEGER NOT NULL DEFAULT 1
  );
  ~~~

- [ ] Add server-aligned owner/status and asset lookup indexes and the partial unique active-binding index. Do not add draft/job/idempotency tables or columns.
- [ ] Bump _CURRENT_SCHEMA_VERSION to 38, add _migrate_from_v37_to_v38, and register 37: self._migrate_from_v37_to_v38 in the migration map. Load SQL through the existing package-resource migration mechanism.
- [ ] Add the SQL path to explicit tldw_chatbook.DB package data and RUNTIME_MIGRATION_PATHS.
- [ ] Re-run the focused test. Expected: PASS.
- [ ] Run the existing migration/package checks:

  ~~~bash
  ../../.venv/bin/python -m pytest -q Tests/ChaChaNotesDB Tests/Packaging/test_installed_distribution.py -k 'migration or schema'
  git diff --check
  ~~~

- [ ] Mutation-check by making pack_version_id nullable or removing the unique active-binding index; prove the test fails, then restore.
- [ ] Commit: 'feat: add local visual identity schema'.

### Task 3: Implement the minimal Visual Identity repository

**Files:**
- Create: 'tldw_chatbook/DB/VisualIdentity_DB.py'
- Create: 'Tests/ChaChaNotesDB/test_visual_identity_repository.py'

- [ ] Write failing real-SQLite tests for only these bounded operations: find built-in pack including tombstones; fetch active binding/version/assets; create a complete pack/version/assets/binding atomically; publish one immutable user version and update pack/binding atomically; archive/delete pack; tombstone binding; list version assets in label order.
- [ ] Add integrity tests rejecting an asset whose pack_version_id belongs to a different non-null pack_id, a pack.active_version_id belonging to another pack, and a binding whose active_version_id belongs to a different pack. SQLite foreign keys do not express these same-pack invariants, so enforce them in the atomic repository methods.
- [ ] Assert local owner sentinel 0 is represented by a named LOCAL_OWNER_ID constant and documented as local-only.
- [ ] Run the repository test. Expected: FAIL because the module is absent.
- [ ] Implement VisualIdentityRepository over CharactersRAGDB.transaction() and execute_query(). It assumes the migration ran and contains no CREATE TABLE or ALTER TABLE.
- [ ] Keep multi-row transaction ownership explicit. Use one repository method per atomic activation/publication boundary rather than composing independently committing calls.
- [ ] Use parameterized SQL and plain dict rows. Do not add an ORM, generic repository base, or dynamic schema bootstrap.
- [ ] Re-run repository and migration tests. Expected: PASS.
- [ ] Force the 31st asset insert to fail and assert no pack/version/binding survives.
- [ ] Commit: 'feat: add visual identity repository'.

### Task 4: Freeze normalization, manifest validation, and resource loading

**Files:**
- Create: 'tldw_chatbook/Character_Chat/visual_identity.py'
- Create: 'Tests/Character_Chat/test_visual_identity_contract.py'
- Create: 'Tests/Character_Chat/test_visual_identity_assets.py'

- [ ] Write a frozen fixture matrix covering canonical slots, every pinned alias, custom keys, punctuation, paths, empty/non-string values, display labels, and filenames.
- [ ] Capture expected results from server commit 385afa...; never import the sibling server at runtime or from installed tests.
- [ ] Write failing general manifest tests for unique keys, safe relative POSIX paths, canonical digest, SHA/bytes/dimensions/frames, and license shape. Add Samira-bundled-pack validation for the exact 31 labels/mappings, 1 MiB per reaction, 16 MiB reaction total, 20 MiB directory total, and AGPL-3.0-or-later; user-owned versions may contain a validated subset after explicit Clear.
- [ ] Write failing loader tests proving built-ins use importlib.resources.files("tldw_chatbook"), user files remain below injected get_user_data_dir()/visual_identities, unsafe paths are rejected before reads, source_kind is obtained through asset→version→pack, and logs do not leak raw paths.
- [ ] Run the two focused tests. Expected: FAIL because the module is absent.
- [ ] Copy the pinned normalization constants and normalize_expression_key, normalize_expression_filename, is_custom_expression_key, display_label_for_expression_key, and _sanitize_expression_token byte-for-byte into visual_identity.py.
- [ ] Add Samira-specific mappings outside the copied block. Do not alter generic aliases for excitement, sadness, confusion, or surprise.
- [ ] Define frozen, slotted dataclasses for manifest assets, validated manifests, loaded bytes, and later resolution results. Avoid Pydantic because this is a fixed internal package contract and no new dependency/abstraction is needed.
- [ ] Implement canonical digest bytes using json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False), assets ordered by original label, with digest and generation-only notes excluded.
- [ ] Implement one loader with injected package/user roots. Validation paths hash/decode the complete candidate; runtime paths verify/load only the selected asset and fall through on errors.
- [ ] Re-run focused tests. Expected: PASS.
- [ ] Mutation-check digest exclusion, path confinement, and one explicit Samira mapping; restore after each expected failure.
- [ ] Commit: 'feat: add visual identity asset contract'.

### Task 5: Produce and package Samira’s canonical card and 31 reactions

**Files:**
- Create: 'tldw_chatbook/assets/characters/samira/Samira.character.json'
- Create: 'tldw_chatbook/assets/characters/samira/Sammy.png'
- Create: 'tldw_chatbook/assets/characters/samira/visual_identity_pack.json'
- Create: 'tldw_chatbook/assets/characters/samira/ASSET_LICENSE.md'
- Create: 'tldw_chatbook/assets/characters/samira/expressions/*.webp' (exactly 31)
- Modify: 'pyproject.toml'
- Modify: 'Tests/Character_Chat/test_visual_identity_assets.py'
- Modify: 'Tests/Character_Chat/test_character_chat.py'

- [ ] Read the approved source inputs from these exact handoff paths in the long-lived worktree, copy them into a private temporary production directory, and verify:
  - '/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Sammy.png' — file SHA-256 0b86569c3f419836a8e867b035136195a95345b2704ffc28d640849629905bed; decoded RGB pixel SHA-256 77452a48101e437834dedaa09ec5121d524c39ea9a13b02f87a158af80d3185f
  - '/Users/macbook-dev/Documents/GitHub/tldw_chatbook/Samira_Sammy_Vadem_v1.character.json' — file SHA-256 ed5bb3f55ca27cd571dd9f462a9d55f56902fa80d15d8390c6bb0f49235fccc0
- [ ] Stop if either source differs.
- [ ] Use the imagegen skill. Inspect the portrait with view_image first; make every non-neutral reaction an independent identity-preserving edit from that original, never reaction-on-reaction.
- [ ] Make exactly 30 independent image edits—one for every non-neutral label—with one generation call/output per label. Every prompt preserves exact adult identity, skin/face proportions/asymmetry, low page-ribbon knot, bookmark pin, rectangular/index-tab earring, temple line, dark editorial/futuristic uniform, crop, angle, lighting, and circular orange archive interface on black. Change only expression plus slight head/shoulder posture according to the approved spec row.
- [ ] Every prompt prohibits other people, obscuring hands, new accessories, text, logo, watermark, symbols, theatrical/cartoon/emoji cues, flirtation, romance, and smugness. Desire means focused yearning toward an objective; love means warm nonromantic trust.
- [ ] Produce the 31st asset, neutral.webp, deterministically from the original portrait rather than through image generation. Convert it and the 30 accepted edits to metadata-free, sRGB-compatible, single-frame 1024×1024 high-quality WebP with Pillow. Do not use image-processing code to invent reactions.
- [ ] Inspect each accepted file with view_image and review a temporary 31-image contact sheet. Do not package intermediates/contact sheet.
- [ ] Reject identity drift, expression ambiguity, changed outfit/composition, added marks/accessories, text/watermark, romance/flirtation, caricature, or unreadable avatar-size expression.
- [ ] Sanitize the V2 card: creator=tldw_chatbook; public included-demo notes; remove vademhq extensions/private flags; add tldw/builtin_id=samira, tldw/license=AGPL-3.0-or-later, and tldw/visual_identity_pack_id=tldw.builtin.samira.reactions; move approved role/personality/nature metadata under tldw/*; retain surname Vadem.
- [ ] Re-embed the sanitized JSON object in Sammy.png using the existing base64 chara convention from export_character_card_to_png. Preserve decoded 1254×1254 RGB pixels exactly.
- [ ] Build visual_identity_pack.json with the exact digest, server commit, generation provenance, visual direction, dimensions, bytes, hashes, and frame data.
- [ ] Write ASSET_LICENSE.md with AGPL-3.0-or-later, portrait source reference, generation tool/date, and exact inventory; make no unsupported authorship claim.
- [ ] Add explicit package-data patterns under tldw_chatbook for the four top-level files and expressions/*.webp.
- [ ] Extend tests for exact/no-extra inventory, all WebP properties and budgets, JSON/PNG V2 equivalence, manifest validity, license, and zero case-insensitive VademHQ/private keys in all packaged text/binary/metadata. Prove PNG pixel identity inside source-independent CI with sha256(Image.open(packaged_png).convert("RGB").tobytes()) == 77452a48101e437834dedaa09ec5121d524c39ea9a13b02f87a158af80d3185f; do not reference the external handoff path from tests.
- [ ] Run focused asset/card tests and git diff --check. Expected: PASS.
- [ ] Commit: 'feat: bundle Samira reaction pack'.

### Task 6: Implement create-only lifecycle seeding

**Files:**
- Modify: 'tldw_chatbook/Character_Chat/visual_identity.py'
- Modify: 'tldw_chatbook/config.py'
- Modify: 'tldw_chatbook/app.py'
- Create: 'Tests/Character_Chat/test_visual_identity_lifecycle.py'

- [ ] Write failing real-database tests for fresh install, v37 upgrade, repeated startup, name collision/suffixing, renamed/edited built-in, soft-delete/restore, explicit binding tombstone, deleted/archived pack, partial pack failure/retry, and valid user-owned fork binding.
- [ ] Spy on resource reads/hashes and prove a healthy built-in binding or valid user fork exits after DB preflight without opening any of the 31 reactions.
- [ ] Prove card creation survives pack activation failure, emits one bounded warning, and next startup retries only the absent non-tombstoned pack.
- [ ] Test the exact production helper signatures for both eager and lazy DB startup. Do not use mocks with invented signatures.
- [ ] Run lifecycle tests. Expected: FAIL because seeding is absent.
- [ ] Implement ensure_builtin_samira(db, package_root=None, user_data_dir=None) with the spec state machine: stable-ID/tombstone preflight; healthy/custom terminal return; no resurrection; full validation only for first/eligible repair; deterministic collision name; preserve existing card fields/ID; atomic pack activation after full validation.
- [ ] Put stable pack ID/content digest in source_context_json and identify the character through tldw/builtin_id, never display name or byte equality.
- [ ] Keep bindings when a character is soft-deleted; resolver availability makes them dormant.
- [ ] Factor one small config helper so initialize_all_databases() and get_chachanotes_db_lazy() seed after schema/FTS initialization. Do not seed every arbitrary CharactersRAGDB constructor.
- [ ] Ensure an app dependency-injected NotesService/profile DB passes through the same idempotent helper once, with no parallel seed implementation.
- [ ] Re-run focused tests. Expected: PASS.
- [ ] Mutation-check the preflight early return and stable-ID lookup; restore after expected failures.
- [ ] Run an isolated live constructor with temporary redirected data_dir, never the real profile. Assert one Samira, one binding, one version, 31 assets, successful first edit, and healthy FTS.
- [ ] Commit: 'feat: seed built-in Samira safely'.

### Task 7: Prove distribution contents and close TASK-16319.1

**Files:**
- Modify: 'Tests/Packaging/test_installed_distribution.py'
- Create: 'Tests/Packaging/test_samira_distribution.py' only if it keeps the existing probe readable
- Modify: 'backlog/tasks/task-16319.1 - Add-Samira-assets-and-local-Visual-Identity-persistence.md'

- [ ] Build wheel/sdist from a temporary source copy and assert the exact Samira inventory and v37→v38 migration in both.
- [ ] Install the wheel into an isolated target without editable-source shadowing. Also unpack the sdist, build its wheel in isolation, and run the same installed probe against that result. Validate/read resources, initialize a fresh redirected profile, and resolve Samira without writes beneath either installed package root.
- [ ] Enforce per-file/aggregate budgets from installed artifacts.
- [ ] Negative-check removal of one reaction/migration and prove precise test failure.
- [ ] Run all TASK-16319.1 DB/contract/asset/lifecycle/packaging tests, git diff --check, and formatter/linter for changed Python.
- [ ] Self-review against five acceptance criteria and ADR-067. Confirm no source-tree assumptions, package writes, new dependency, default change, VademHQ, or server-ID claim.
- [ ] Check all child criteria, add Implementation Notes with approach/files/trade-offs/tests/ADR, and set TASK-16319.1 Done.
- [ ] Commit: 'test: verify installed Samira assets'.

## TASK-16319.2 — Resolver and Console reaction picker

### Task 8: Claim TASK-16319.2 and define the structured resolver

**Files:**
- Modify: 'backlog/tasks/task-16319.2 - Resolve-Visual-Identity-reactions-in-Console.md'
- Modify: 'tldw_chatbook/Character_Chat/visual_identity.py'
- Create: 'Tests/Character_Chat/test_visual_identity_resolution.py'

- [ ] Set TASK-16319.2 In Progress, assign implementer, add Tasks 8–11 to its Backlog plan, and link ADR-067.
- [ ] Write failing real-DB tests for manual → operational → pack default → neutral → legacy BLOB → card portrait → placeholder.
- [ ] Cover all four operational states, unknown/missing keys, corrupt/missing files, inactive/deleted binding/pack, soft-deleted actor, and legacy-only characters.
- [ ] Freeze machine-oriented resolution_source/fallback_reason values before implementation.
- [ ] Run resolution tests. Expected: FAIL.
- [ ] Add a frozen, slotted VisualIdentityResolution with actor, requested/manual/resolved keys, pack/version/asset IDs, storage source/path, content type/animation, resolution source, fallback reason, and cache_identity.
- [ ] Resolve with one bounded active binding/version/assets query and read legacy/card only if needed. Validate/load one selected asset, never all 31.
- [ ] Pack cache identity includes actor, manual/requested key, source, pack/version/asset IDs, and digest. Legacy/card identity includes stable content digest or row version.
- [ ] Corrupt/missing bytes log stable IDs/categories and fall through without raw paths.
- [ ] Re-run tests. Expected: PASS.
- [ ] Mutation-check fallback order and pack version in cache_identity; restore.
- [ ] Commit: 'feat: resolve visual identity reactions'.

### Task 9: Build the keyboard-operable Console reaction picker

**Files:**
- Create: 'tldw_chatbook/Widgets/Console/console_reaction_picker_modal.py'
- Modify: 'tldw_chatbook/UI/Console_Modules/left_rail.py'
- Create: 'Tests/UI/test_console_reaction_picker.py'

- [ ] Write failing pilot tests for the Reaction… button, 31-label filter, Up/Down/Enter/Escape, explicit Clear, visible text label/count, and focus.
- [ ] Add geometry assertions at 80×24 and normal size; preview hides before filter/list/actions at narrow size.
- [ ] Add a lazy-preview spy: opening/selection decodes at most the selected image, never all 31.
- [ ] Run tests. Expected: FAIL.
- [ ] Implement a small immutable metadata-only option model. Modal receives options, not DB/repository.
- [ ] Follow existing Console modal patterns and emit ReactionPreviewRequested, ReactionSelected, and ReactionCleared messages; ChatScreen owns async bytes and session state.
- [ ] Add Reaction… and active manual-label text to ConsoleLeftRail Character section. Post a rail message upward without importing ChatScreen.
- [ ] Add no global/screen hotkey or footer hint.
- [ ] Re-run tests. Expected: PASS.
- [ ] Mutation-check eager preview and restore.
- [ ] Commit: 'feat: add Console reaction picker'.

### Task 10: Wire session-local overrides and race-safe avatar refresh

**Files:**
- Modify: 'tldw_chatbook/UI/Screens/chat_screen.py'
- Modify: 'tldw_chatbook/UI/Console_Modules/session.py'
- Modify: 'Tests/UI/test_console_character_avatar.py'
- Modify: 'Tests/UI/test_console_reaction_picker.py'

- [ ] Extend failing tests for select/Clear, actor replacement, session disposal, restart non-persistence, invalid selection preserving prior override, and reaction-enable behavior.
- [ ] Add deterministic reordered-await tests: load A, select/publish B, complete B then A, and assert A never overwrites B.
- [ ] Prove cache misses for manual/requested key, actor, fallback source, version, asset, and digest changes.
- [ ] Run focused UI tests. Expected: FAIL because cache/scope is still (character_id, state).
- [ ] Store manual keys in memory keyed by (session_id, actor_kind, actor_id); never persist.
- [ ] Query option metadata off-thread; lazy-load highlighted preview; validate selection before replacing prior override; Clear deletes it.
- [ ] Clear old actor override on actor replacement and all session keys after successful close.
- [ ] Replace _fetch_expression_image_bytes with structured resolver/load off-thread and key decode cache by resolution.cache_identity.
- [ ] After every await, recompute/recheck session, actor, manual/requested key, and full resolved identity before applying pixels.
- [ ] Preserve console_expression_state.py’s four operational transitions; map only at resolver input.
- [ ] Keep the runtime cache bounded with the existing policy or one small fixed maximum.
- [ ] Re-run tests. Expected: PASS.
- [ ] Mutation-check post-await fence and version key; restore.
- [ ] Commit: 'feat: use visual identity in Console'.

### Task 11: Live-verify Console and close TASK-16319.2

**Files:**
- Modify: 'backlog/tasks/task-16319.2 - Resolve-Visual-Identity-reactions-in-Console.md'

- [ ] Run focused resolver/Console tests, git diff --check, formatter/linter.
- [ ] Launch real TUI with temporary redirected profile. Select Samira, filter/keyboard-select/Clear, observe manual label, then verify all four operational mappings.
- [ ] Verify 80×24 and normal layouts, one lazy preview, and no out-of-bounds controls.
- [ ] Induce a missing reaction in isolated test data and verify graceful legacy/card fallback without raw paths.
- [ ] Self-review: no classifier, /emote, durable replay, extra hotkey, or Persona Visual Pack merge.
- [ ] Check criteria, add Implementation Notes/live evidence/ADR-067, and set TASK-16319.2 Done.
- [ ] Commit: 'test: verify Console reaction selection'.

## TASK-16319.3 — Personas browsing and copy-on-write authoring

### Task 12: Claim TASK-16319.3 and build the lazy pack browser

**Files:**
- Modify: 'backlog/tasks/task-16319.3 - Add-Visual-Identity-pack-browsing-and-copy-on-write-authoring.md'
- Create: 'tldw_chatbook/Widgets/Persona_Widgets/personas_visual_identity_pack_widget.py'
- Modify: 'tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py'
- Modify: 'tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py'
- Create: 'Tests/UI/test_personas_visual_identity_pack.py'
- Modify: 'Tests/UI/test_personas_expression_slots.py'

- [ ] Set TASK-16319.3 In Progress, assign implementer, add Tasks 12–15 to its plan, and link ADR-067.
- [ ] Write failing tests proving bound packs use the browser while unbound characters retain existing thinking/speaking/error controls.
- [ ] Cover filtering 31 labels, current/total, exact text label, internal key diagnostic, one selected preview, built-in copy-on-write notice, dirty summary, and Replace/Generate/Clear/Save.
- [ ] Add lazy-decode and 80×24/normal geometry tests.
- [ ] Run tests. Expected: FAIL.
- [ ] Implement a focused metadata/staging-status widget emitting typed messages; PersonasScreen owns DB/files/provider work.
- [ ] Mount only for active character bindings. Preserve legacy controls exactly otherwise.
- [ ] Hide preview before labels/list/actions at narrow sizes.
- [ ] Re-run tests. Expected: PASS.
- [ ] Run CSS builder and Tests/UI/test_css_build_integrity.py.
- [ ] Commit: 'feat: browse visual identity packs'.

### Task 13: Implement same-filesystem copy-on-write publication

**Files:**
- Modify: 'tldw_chatbook/Character_Chat/visual_identity.py'
- Modify: 'tldw_chatbook/DB/VisualIdentity_DB.py'
- Create: 'Tests/Character_Chat/test_visual_identity_publication.py'

- [ ] Write failing real-fs/real-DB tests: staging-only replace/clear/generate; built-in copy-on-write; first Save creates one user pack/version and changes only actor binding; later Save appends one version; a cleared expression is absent from the new immutable version and resolves through the documented fallback chain; cancel/provider/upload/validation/permission failure preserves active version; DB failure leaves only unreferenced cleanup candidate; package resources never open for write.
- [ ] Inject user root and os.replace seam; assert same-filesystem private staging via secure_private_directory(..., application_owned=True).
- [ ] Run tests. Expected: FAIL.
- [ ] Implement one candidate overlay for staged replacements/removals on the active immutable version. Copy retained/replaced bytes lazily into a private sibling staging directory and validate the complete resulting manifest/assets before publication. Only the bundled Samira version is required to contain all 31 assets; user-owned immutable versions may intentionally omit a cleared expression.
- [ ] Built-in first edit creates source_kind=manual profile-owned pack/version; never mutate built-in rows/files.
- [ ] Publish files with os.replace(staging_dir, final_version_dir), then insert version/assets and switch pack/binding in one DB transaction. On DB failure keep old binding active and leave only a known unreferenced directory.
- [ ] Never delete active version directories; cleanup is bounded to validated profile-owned roots.
- [ ] Expose a small publication event/callback with actor and old/new identity for targeted Console invalidation.
- [ ] Re-run tests. Expected: PASS.
- [ ] Mutation-check package-write guard, atomic replace, single-version guarantee; restore.
- [ ] Commit: 'feat: publish immutable reaction packs'.

### Task 14: Wire Personas replacement, generation, cancellation, and Save

**Files:**
- Modify: 'tldw_chatbook/Character_Chat/expression_generation.py'
- Modify: 'tldw_chatbook/UI/Screens/personas_screen.py'
- Modify: 'tldw_chatbook/Widgets/Persona_Widgets/personas_visual_identity_pack_widget.py'
- Modify: 'Tests/Character_Chat/test_expression_generation.py'
- Modify: 'Tests/UI/test_personas_visual_identity_pack.py'
- Modify: 'Tests/UI/test_personas_expression_generate.py'
- Modify: 'Tests/UI/test_console_character_avatar.py'

- [ ] Write failing prompt tests for compose_visual_identity_prompt sharing only the existing identity/style base and accepting label + visual direction. Keep EXPRESSION_PROMPT_STATES unchanged.
- [ ] Write failing screen tests for Replace/Generate/Clear/dirty/Save/Cancel/cache invalidation; staging cannot change active version.
- [ ] Generate All tests: confirmation says 31 provider calls; decline makes zero; accepted concurrency never exceeds named limit 3; every call passes the canonical/current identity as a ResolvedReferenceImage and never another generated reaction; unsupported-reference backends fail before partial publication; blocking calls run off the event loop; cancellation stops scheduling, sets the shared threading.Event, cancels or awaits in-flight work, rejects stale results, discards the candidate, and publishes nothing; one failed call publishes nothing; success stages all 31 and one Save creates one version.
- [ ] Run focused tests. Expected: FAIL.
- [ ] Refactor only shared prompt-prefix code and add compose_visual_identity_prompt; do not expand the legacy operational constant.
- [ ] Reuse build_request → run_generation; add no provider abstraction. Construct one ResolvedReferenceImage from the canonical/current identity bytes/path and pass it to every Generate and Generate All request. Surface the existing unsupported-reference capability error rather than silently generating without identity control.
- [ ] Use asyncio.Semaphore(3) plus Python 3.11 structured cancellation. Invoke blocking run_generation() with asyncio.to_thread() while holding the semaphore. Give every request the same caller-owned threading.Event cancel_event, set it on cancellation, and check cancellation/editor session tokens before scheduling and before accepting each result.
- [ ] Route upload/generated bytes through the candidate validator. Clear stages removal of the selected expression while retaining every other active asset; Save may publish that intentional omission, and runtime resolution must fall through deterministically for the missing key.
- [ ] Save calls publication once, refreshes binding/version, and invalidates the actor’s Console caches before presenting it.
- [ ] Preserve existing editor generation/inflight guards across navigation and actor replacement.
- [ ] Re-run tests. Expected: PASS.
- [ ] Mutation-check concurrency, cancellation fence, and publish-on-Save; restore.
- [ ] Rebuild CSS if changed and run integrity.
- [ ] Commit: 'feat: author reaction pack versions'.

### Task 15: End-to-end verification and backlog closeout

**Files:**
- Modify: 'backlog/tasks/task-16319.3 - Add-Visual-Identity-pack-browsing-and-copy-on-write-authoring.md'
- Modify: 'backlog/tasks/task-16319 - Bundle-Samira-character-and-full-Visual-Identity-reaction-pack.md'
- Modify: user docs only where shipped UI needs discoverability
- Modify: 'backlog/docs/lessons-*.md' only for incident-backed reusable learning

- [ ] Run every new/changed DB, character, Console, Personas, CSS, and packaging test listed above.
- [ ] Run the full suite:

  ~~~bash
  ../../.venv/bin/python -m pytest
  ~~~

  Compare any failure with the same command on fresh origin/dev; fix regressions and record true baseline failures.

- [ ] Run git diff --check plus configured formatter/linter/static checks.
- [ ] Rebuild wheel/sdist, inspect archives, isolated-install the wheel, seed a fresh redirected profile, and resolve installed assets after all UI changes.
- [ ] Real TUI UAT with temporary profile: one searchable Samira and unchanged default; filter all 31 in Personas; stage/cancel leaves version unchanged; save one edit creates one private version and Console refreshes; Generate All confirms 31 calls and can cancel; select/Clear Console reactions; four operational states; 80×24 and normal layouts.
- [ ] Reinspect every final image and contact sheet for identity/expression/composition/content problems. Confirm nonromantic desire/love.
- [ ] Search the packaged directory and metadata case-insensitively for VademHQ: zero matches. Confirm every license is AGPL-3.0-or-later.
- [ ] Self-review against spec/ADR/security/sync boundary and reject scope creep into default Persona, classification, durable replay, server APIs/drafts/jobs/idempotency, or Persona Visual Packs.
- [ ] Check TASK-16319.3 criteria, add Implementation Notes/evidence/ADR, and mark it Done.
- [ ] Only then check umbrella TASK-16319 criteria, add notes linking all child outcomes and ADR-067, and mark it Done.
- [ ] Commit: 'docs: complete Samira visual identity delivery'.

## Final branch handoff

- [ ] Invoke superpowers:verification-before-completion before success claims and quote fresh evidence.
- [ ] Invoke superpowers:requesting-code-review for correctness/security/scope review.
- [ ] Address findings through receiving-code-review and rerun affected/full verification.
- [ ] Invoke superpowers:finishing-a-development-branch only after all tasks are Done and tests pass.

## ADR check

ADR required: yes

ADR path: 'backlog/decisions/067-bundled-samira-visual-identity-pack.md'

Reason: The feature adds local schema and immutable version/binding contracts, package-versus-profile asset ownership and publication, a future server-sync seam, and a narrow amendment to a prior public-release boundary. ADR-067 is accepted and sufficient; create no duplicate ADR unless implementation discovers a materially different architectural decision.
