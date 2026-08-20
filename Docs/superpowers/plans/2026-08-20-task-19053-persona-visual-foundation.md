# TASK-19053 Persona Visual Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a profile-local, immutable Persona Visual runtime that validates and resolves the pinned server `sprite_frames` manifest-version-1 contract without coupling it to Shared Visual Identity or adding UI.

**Architecture:** A new `Persona_Visual` package owns the server-aligned manifest model, asset boundary, resolver, and publication workflow. A separate SQLite schema and repository own packs, immutable versions, version-bound assets, and one optimistic active binding per local Persona; Persona JSON remains unchanged and caller-supplied eligibility/revision authority is rechecked before publication. Asset paths never cross public result/error boundaries, and publication follows the existing profile-private stage → pinned atomic replace → SQLite activation → identity-scoped cleanup pattern.

**Tech Stack:** Python 3.11, dataclasses, Pillow, SQLite/FTS migration framework, pathlib/os descriptor APIs, pytest, Ruff.

**ADR required:** no

**ADR path:** `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md`

**Reason:** ADR-074 already fixes the separate runtime, storage ownership, manifest compatibility, Persona authority, and no-server-write boundaries. This task implements that decision without introducing a new architecture choice.

---

## File map

- Create `tldw_chatbook/Persona_Visual/__init__.py`: small public export surface.
- Create `tldw_chatbook/Persona_Visual/contracts.py`: pinned capabilities, immutable state/manifest dataclasses, fallback traversal, and static/reduced-motion selection.
- Create `tldw_chatbook/Persona_Visual/validation.py`: strict JSON parsing and pinned manifest validation.
- Create `tldw_chatbook/Persona_Visual/assets.py`: image inspection, immutable asset metadata, confined profile-owned reads, digest/decode/dimension/frame budgets.
- Create `tldw_chatbook/Persona_Visual/repository.py`: Persona Visual SQLite graph reads/writes, optimistic binding/version authority, stable graph identities.
- Create `tldw_chatbook/Persona_Visual/runtime.py`: path-free resolution, selected asset loading, portrait fallback, full-identity cache keys.
- Create `tldw_chatbook/Persona_Visual/publication.py`: validated staging, atomic publication, rollback, and identity-pinned orphan cleanup.
- Create `tldw_chatbook/DB/migrations/chachanotes_v40_to_v41_persona_visual.sql`: separate Persona Visual schema.
- Modify `tldw_chatbook/DB/ChaChaNotes_DB.py`: register schema version 41 and the V40→V41 migration.
- Modify `Tests/ChaChaNotesDB/historical_bootstrap.py`: declare V41 artifacts only if the historical fixture helper requires an explicit drop map.
- Create `Tests/ChaChaNotesDB/test_persona_visual_migration.py`: real V40 migration and schema constraints.
- Create `Tests/Persona_Visual/test_persona_visual_contract.py`: pinned server vectors and unsupported-capability cases.
- Create `Tests/Persona_Visual/test_persona_visual_repository.py`: immutable graph, binding authority, rollback, identities.
- Create `Tests/Persona_Visual/test_persona_visual_assets.py`: validation, confinement, budgets, privacy.
- Create `Tests/Persona_Visual/test_persona_visual_runtime.py`: fallback and cache-identity resolution.
- Create `Tests/Persona_Visual/test_persona_visual_publication.py`: atomic publication, stale authority, cancellation-neutral failure, orphan cleanup.
- Create `Tests/Architecture/test_persona_visual_runtime_boundary.py`: separate-runtime, no-UI, no-server-write, and path-free public-contract governance.
- Modify `backlog/tasks/task-19053 - Add-local-Persona-Visual-pack-foundation.md`: plan, checked acceptance criteria, and concise implementation notes only after verification.

## Frozen compatibility boundary

The tests must freeze only the local semantic subset from `tldw_server` commit `385afa951922c8a9dc2002c675bb6cad65e4ac23`:

```python
PERSONA_VISUAL_STATES = frozenset(
    {
        "idle",
        "wake_armed",
        "listening",
        "thinking",
        "speaking",
        "tool_running",
        "approval_needed",
        "error",
        "offline",
    }
)
REQUIRED_PERSONA_VISUAL_STATES = frozenset(
    {"idle", "listening", "thinking", "speaking", "error"}
)
SPRITE_FRAMES_RENDERER = "sprite_frames"
SPRITE_FRAMES_MANIFEST_VERSION = 1
MAX_FRAMES_PER_ANIMATION = 240
MAX_CUSTOM_VISUAL_STATES = 256
MAX_AUTHORED_TRIGGERS = 512
MAX_FALLBACK_DEPTH = 8
MAX_PERSONA_VISUAL_ASSETS = 256
MAX_PERSONA_VISUAL_TOTAL_BYTES = 100 * 1024 * 1024
MAX_PERSONA_VISUAL_IMAGE_DIMENSION = 4096
MIN_FRAME_DURATION_MS = 16
MAX_FRAME_DURATION_MS = 30_000
MIN_TRIGGER_DURATION_MS = 100
MAX_TRIGGER_DURATION_MS = 30_000
```

V1 accepts at most 256 PNG/JPEG/WebP/GIF assets, at most 100 MiB total, and images up to 4096×4096, matching the pinned renderer capability. Manifest version 2 and non-`sprite_frames` renderers produce a stable unsupported-capability result; they are not guessed, activated, imported, or partially rendered.

### Task 1: Freeze and implement the manifest contract

**Files:**
- Create: `tldw_chatbook/Persona_Visual/__init__.py`
- Create: `tldw_chatbook/Persona_Visual/contracts.py`
- Create: `tldw_chatbook/Persona_Visual/validation.py`
- Create: `Tests/Persona_Visual/test_persona_visual_contract.py`

- [ ] **Step 1: Write born-RED contract tests**

Cover all nine reserved states; five required states; safe custom-state grammar `^[a-z][a-z0-9_.:-]{0,95}$`; unsafe prefix/secret-marker rejection; reserved-name collision; `state_catalog` kinds plus label/description/tag bounds; legacy `asset_ids` normalization; animation frames; `frame_rate`; alignment; per-frame duration/region bounds; `preview_frame` and `preview_asset_id` static selection; fallback cycles/depth; authored-trigger nonempty `match`, exact `live_state`/`tool_category`/`mcp_runtime`/`tool_name` sources, target state, duration, and priority; reduced-motion selection; and stable unsupported renderer/version results.

```python
def test_pinned_sprite_frames_manifest_resolves_required_states() -> None:
    manifest = validate_persona_visual_manifest(PINNED_VALID_MANIFEST, KNOWN_ASSETS)
    assert manifest.renderer_type == "sprite_frames"
    assert manifest.manifest_version == 1
    assert all(resolve_manifest_state(manifest, state) for state in REQUIRED_STATES)


@pytest.mark.parametrize(
    ("renderer_type", "manifest_version"),
    [("live2d", 2), ("sprite_frames", 2)],
)
def test_unsupported_capability_is_stable_and_not_activatable(
    renderer_type: str, manifest_version: int
) -> None:
    result = inspect_persona_visual_capability(renderer_type, manifest_version)
    assert result.supported is False
    assert result.reason == "persona_visual_capability_unsupported"
```

- [ ] **Step 2: Run the focused contract file and verify RED**

Run:

```bash
TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Persona_Visual/test_persona_visual_contract.py
```

Expected: collection fails because `tldw_chatbook.Persona_Visual` does not exist.

- [ ] **Step 3: Implement the minimal immutable contract**

Use frozen/slotted dataclasses for capability, frames, animations, triggers, manifest, and state selection in `contracts.py`. Put strict JSON parsing and validation in `validation.py`, including duplicate-key and non-standard-number rejection. Validate known asset IDs but do not read files in either module. Keep diagnostics as stable categories rather than interpolated values.

- [ ] **Step 4: Run the focused contract file and verify GREEN**

Expected: all contract cases pass.

- [ ] **Step 5: Commit the contract slice**

```bash
git add tldw_chatbook/Persona_Visual/__init__.py tldw_chatbook/Persona_Visual/contracts.py tldw_chatbook/Persona_Visual/validation.py Tests/Persona_Visual/test_persona_visual_contract.py
git commit -m "feat: add Persona Visual manifest contract"
```

### Task 2: Add the separate SQLite schema and repository

**Files:**
- Create: `tldw_chatbook/DB/migrations/chachanotes_v40_to_v41_persona_visual.sql`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Modify if required: `Tests/ChaChaNotesDB/historical_bootstrap.py`
- Create: `tldw_chatbook/Persona_Visual/repository.py`
- Create: `Tests/ChaChaNotesDB/test_persona_visual_migration.py`
- Create: `Tests/Persona_Visual/test_persona_visual_repository.py`

- [ ] **Step 1: Write the real V40 migration RED tests**

Bootstrap a genuinely historical V40 DB through production migrations, then open it at V41. Assert these separate tables and indexes exist:

```text
persona_visual_packs
persona_visual_pack_versions
persona_visual_assets
persona_visual_bindings
idx_persona_visual_bindings_persona_active
idx_persona_visual_assets_version_key
```

Assert one active binding per Persona, immutable `(pack_id, version_number)`, foreign-key relationships, and `db_schema_version == 41`. Assert no Persona JSON or Shared Visual Identity table is changed.

- [ ] **Step 2: Run migration tests and verify RED**

```bash
TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/ChaChaNotesDB/test_persona_visual_migration.py
```

Expected: schema remains V40 and the new tables are absent.

- [ ] **Step 3: Implement V40→V41**

Use the current complete-statement migration runner pattern. The schema stores no private absolute path and no Persona payload:

```text
packs: id, title, status, active_version_id, source_kind/context, version
versions: id, pack_id, version_number, renderer_type, manifest_version,
          manifest_json, manifest_sha256, storage_relpath
assets: id, pack_id, pack_version_id, asset_key, role, storage_relpath,
        mime_type, bytes, sha256, width, height, frame_count, duration_ms
bindings: id, persona_id, persona_revision, pack_id, active_version_id,
          status, version
```

- [ ] **Step 4: Write repository RED tests**

Cover create/activate, publish immutable next version, one active Persona binding, inactive/deleted binding behavior, stable `PersonaVisualIdentity(persona_id, persona_revision, binding_id, binding_version, pack_id, pack_revision, pack_version_id, version_number, manifest_sha256)`, optimistic Persona revision and pack/binding/version CAS, cross-pack rejection, transaction rollback, and absence of Shared Visual Identity writes.

- [ ] **Step 5: Run repository tests and verify RED**

Expected: repository import fails.

- [ ] **Step 6: Implement the minimal repository**

`PersonaVisualRepository` accepts an open `CharactersRAGDB`, never initializes schema itself, and exposes only graph-level methods needed by this task:

```python
get_active_persona_pack(persona_id: str) -> PersonaVisualGraph | None
activate_new_pack(..., expected_persona_revision: int, authority_guard: Callable[[], bool])
publish_version(..., expected_identity: PersonaVisualIdentity, authority_guard: Callable[[], bool])
archive_binding(..., expected_identity: PersonaVisualIdentity)
```

The final authority guard runs inside the SQLite write transaction immediately before graph activation.

- [ ] **Step 7: Run migration + repository tests and verify GREEN**

```bash
TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/ChaChaNotesDB/test_persona_visual_migration.py Tests/Persona_Visual/test_persona_visual_repository.py
```

- [ ] **Step 8: Run the complete touched ChaChaNotes migration surface**

```bash
TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/ChaChaNotesDB
```

Expected: touched DB migration suite passes. This is the broadest DB gate; do not run the repository-wide full suite.

- [ ] **Step 9: Commit the persistence slice**

```bash
git add tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/DB/migrations/chachanotes_v40_to_v41_persona_visual.sql tldw_chatbook/Persona_Visual/repository.py Tests/ChaChaNotesDB/test_persona_visual_migration.py Tests/ChaChaNotesDB/historical_bootstrap.py Tests/Persona_Visual/test_persona_visual_repository.py
git commit -m "feat: persist immutable Persona Visual packs"
```

Stage `historical_bootstrap.py` only if it actually changed.

### Task 3: Validate and load profile-owned assets

**Files:**
- Create: `tldw_chatbook/Persona_Visual/assets.py`
- Create: `Tests/Persona_Visual/test_persona_visual_assets.py`

- [ ] **Step 1: Write asset-boundary RED tests**

Cover PNG/JPEG/WebP/GIF MIME/decoder agreement; 256-file and 100-MiB pack limits; 4096 dimension; 240 manifest frames; bounded selected-frame decoding; lowercase SHA-256; positive metadata; safe relative POSIX storage keys; no `..`, absolute, drive/device, NUL, symlink, or directory alias traversal; bounded reads; inode swap failure; missing/digest/decode mismatch; and exception/log privacy.

- [ ] **Step 2: Run focused assets tests and verify RED**

```bash
TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Persona_Visual/test_persona_visual_assets.py
```

- [ ] **Step 3: Implement the minimal asset boundary**

Reuse standard library descriptor APIs and Pillow. Do not refactor Shared Visual Identity. Return immutable `PersonaVisualAsset` metadata and loaded bytes; keep absolute paths local to private helpers. Derive cache inputs only from `persona_id`, binding/pack/version/asset IDs, SHA-256, requested state, and rendering mode.

- [ ] **Step 4: Run asset + contract tests and verify GREEN**

```bash
TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Persona_Visual/test_persona_visual_assets.py Tests/Persona_Visual/test_persona_visual_contract.py
```

- [ ] **Step 5: Commit the asset slice**

```bash
git add tldw_chatbook/Persona_Visual/assets.py Tests/Persona_Visual/test_persona_visual_assets.py
git commit -m "feat: validate Persona Visual assets"
```

### Task 4: Resolve operational states with path-free identities

**Files:**
- Create: `tldw_chatbook/Persona_Visual/runtime.py`
- Create: `Tests/Persona_Visual/test_persona_visual_runtime.py`

- [ ] **Step 1: Write resolver RED tests**

Cover direct state, custom state, multi-hop fallback, missing requested state → `idle`, unusable requested animation → healthy manifest fallback, unusable fallback → healthy `idle`, missing/invalid idle asset → supplied Persona portrait, no portrait → stable unavailable result, fallback-cycle storage corruption, animated frames, reduced-motion static selection, preview-frame/preview-asset precedence, selected-candidate-only asset loading, and cache identity changes for any binding/pack/version/asset/digest/state/motion change.

```python
def test_runtime_miss_falls_back_to_portrait_without_exposing_a_path(...) -> None:
    result = resolve_persona_visual(..., requested_state="custom:missing")
    assert result.source == "persona_portrait"
    assert result.reason == "persona_visual_idle_unavailable"
    assert result.cache_identity == (...stable identifiers...)
    assert not hasattr(result, "storage_path")
```

- [ ] **Step 2: Run resolver tests and verify RED**

```bash
TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Persona_Visual/test_persona_visual_runtime.py
```

- [ ] **Step 3: Implement the runtime resolver**

Load one active graph, validate its stored manifest, and build the finite ordered candidate chain: requested mapping, its validated manifest fallbacks, then `idle` and its validated fallbacks without duplicates. Read one candidate animation at a time. If any referenced frame fails read/decode/digest validation, reject that whole animation and continue to the next candidate; never return a partial animation. Use portrait/unavailable with a fixed reason only after every candidate is exhausted. For reduced motion choose `preview_frame`, then a frame naming the manifest preview asset, then frame zero, and set `animate=False`.

- [ ] **Step 4: Run contract/assets/repository/runtime tests and verify GREEN**

```bash
TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Persona_Visual/test_persona_visual_contract.py Tests/Persona_Visual/test_persona_visual_assets.py Tests/Persona_Visual/test_persona_visual_repository.py Tests/Persona_Visual/test_persona_visual_runtime.py
```

- [ ] **Step 5: Commit the resolver slice**

```bash
git add tldw_chatbook/Persona_Visual/runtime.py Tests/Persona_Visual/test_persona_visual_runtime.py
git commit -m "feat: resolve Persona Visual runtime states"
```

### Task 5: Publish immutable versions and clean owned orphans

**Files:**
- Create: `tldw_chatbook/Persona_Visual/publication.py`
- Create: `Tests/Persona_Visual/test_persona_visual_publication.py`

- [ ] **Step 1: Write publication RED tests**

Cover first activation and later version publication; exact old/new full identities; source Persona revision/binding/pack/version ABA; final authority guard; bounded materialization; fsync/atomic replace/parent fsync; source identity swap; package/profile root overlap; rollback after filesystem publication; returned opaque profile-relative cleanup token; pinned cleanup success/refusal/reference race; cancellation-neutral failure (the synchronous boundary never leaves a half-updated DB graph); and path-free errors/logs.

- [ ] **Step 2: Run publication tests and verify RED**

```bash
TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Persona_Visual/test_persona_visual_publication.py
```

- [ ] **Step 3: Implement publication with the smallest proven workflow**

Use one publication input snapshot and no UI candidate model. Materialize validated bytes under a profile-private `.staging-<uuid>` directory, pin source/staging/final identities, rename atomically, revalidate the final inode inside the reserved repository transaction, then return:

```python
PersonaVisualPublicationResult(
    old_identity: PersonaVisualIdentity | None,
    new_identity: PersonaVisualIdentity,
    cleanup_candidate: str | None,
)
```

Cleanup accepts only opaque `.staging-<uuid>` or final-version tokens returned by this module, reserves SQLite against reference insertion, checks both manifest and asset path references before/after pinned deletion, and fails closed on identity substitution.

- [ ] **Step 4: Run publication + repository + assets tests and verify GREEN**

```bash
TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Persona_Visual/test_persona_visual_publication.py Tests/Persona_Visual/test_persona_visual_repository.py Tests/Persona_Visual/test_persona_visual_assets.py
```

- [ ] **Step 5: Commit the publication slice**

```bash
git add tldw_chatbook/Persona_Visual/publication.py Tests/Persona_Visual/test_persona_visual_publication.py
git commit -m "feat: publish Persona Visual pack versions"
```

### Task 6: Verify the touched foundation and close TASK-19053

**Files:**
- Modify: `backlog/tasks/task-19053 - Add-local-Persona-Visual-pack-foundation.md`
- Modify only if generated diagnostics require it: `Docs/security/production-diagnostic-inventory.json`

- [ ] **Step 1: Add and verify the architecture boundary test**

Create `Tests/Architecture/test_persona_visual_runtime_boundary.py` with AST/import and public-dataclass assertions proving: Persona Visual production modules do not import Shared Visual Identity or UI/Buddy/provider/server modules; the four `persona_visual_*` tables are distinct from `visual_identity_*`; no public result/error dataclass exposes a `path`, `relpath`, or exception-detail field; and the package introduces no dependency. Run it first against a deliberate temporary forbidden import to prove RED, restore the boundary, and rerun GREEN.

```bash
TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Architecture/test_persona_visual_runtime_boundary.py
```

- [ ] **Step 2: Run the fresh focused Persona Visual + touched migration gate**

```bash
TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Persona_Visual Tests/ChaChaNotesDB/test_persona_visual_migration.py
```

- [ ] **Step 3: Run only touched adjacent Shared Visual Identity and ChaChaNotes coverage**

```bash
TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/ChaChaNotesDB \
  Tests/Character_Chat/test_visual_identity_migration.py \
  Tests/Character_Chat/test_visual_identity_repository.py \
  Tests/Character_Chat/test_visual_identity_contract.py \
  Tests/Character_Chat/test_visual_identity_assets.py \
  Tests/Character_Chat/test_visual_identity_resolution.py \
  Tests/Character_Chat/test_visual_identity_lifecycle.py \
  Tests/Character_Chat/test_visual_identity_publication.py
```

Do not run the full repository suite.

- [ ] **Step 4: Run mutation checks**

Temporarily and individually remove the authority CAS, required-state validation, fallback-to-idle, source/final inode guard, cleanup-reference guard, transaction rollback trigger, one full-identity cache field, and the path/exception redaction guard. Each owning regression must fail. Restore the production code and rerun the exact cases.

- [ ] **Step 5: Run isolated-profile provenance and privacy gates**

Create a `mktemp -d` root with private `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME`, explicit `TLDW_CONFIG_PATH`, and parsed `[paths].data_dir` before importing Chatbook. Before any product import, set and record the assigned worktree root; after import, assert `Path(tldw_chatbook.__file__).resolve().is_relative_to(assigned_worktree_root)` and make the same assertion for `Persona_Visual.contracts`, `repository`, `assets`, `runtime`, and `publication`. Exercise a real migrated SQLite DB, publish and resolve one pack, and assert no writes outside the isolated roots and no private-root token in errors/log captures.

- [ ] **Step 6: Run scoped static checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  tldw_chatbook/Persona_Visual Tests/Persona_Visual Tests/ChaChaNotesDB/test_persona_visual_migration.py Tests/Architecture/test_persona_visual_runtime_boundary.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/Persona_Visual Tests/Persona_Visual Tests/ChaChaNotesDB/test_persona_visual_migration.py Tests/Architecture/test_persona_visual_runtime_boundary.py
PYTHONPYCACHEPREFIX=/tmp/tldw-task-19053-pyc \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q \
  tldw_chatbook/Persona_Visual tldw_chatbook/DB/ChaChaNotes_DB.py
git diff --check
```

Run the repository's diagnostic inventory, privacy, architecture, and ADR-067 governance checks that include the touched modules. If and only if the generated diagnostic inventory changes, review the exact semantic delta and commit that file separately.

```bash
TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Architecture/test_persona_visual_runtime_boundary.py \
  Tests/Architecture/test_persistent_diagnostic_inventory.py \
  Tests/DB/test_core_sqlite_owner_privacy.py \
  Tests/test_database_path_privacy.py
```

- [ ] **Step 7: Commit the verified architecture test**

```bash
git add Tests/Architecture/test_persona_visual_runtime_boundary.py
git commit -m "test: verify Persona Visual runtime boundary"
```

- [ ] **Step 8: Self-review scope and exclusions**

Confirm the final diff contains no Workbench UI, Buddy, provider generation, server writes, Shared Visual Identity schema reuse, Persona JSON mutation, export/import archive flow, or third-party dependency.

- [ ] **Step 9: Complete Backlog evidence**

Use Backlog CLI to check all eight ACs, set status Done, and add concise Implementation Notes listing approach, tests, mutation evidence, isolated roots, ADR-074, touched files, and the explicit full-suite deviation requested by the user.

- [ ] **Step 10: Commit closeout metadata**

```bash
git add "backlog/tasks/task-19053 - Add-local-Persona-Visual-pack-foundation.md"
git commit -m "docs: complete Persona Visual foundation"
```

## Execution constraints

- Use `superpowers:test-driven-development` for every behavior change.
- Use `superpowers:systematic-debugging` before fixing any unexpected failure.
- Use `ponytail` full: no generic visual-runtime framework, no refactor of existing Shared Visual Identity, no draft/job/provider abstractions, and no new dependency.
- Use `superpowers:verification-before-completion` before every completion claim.
- Preserve unrelated worktree changes and stage exact files only.
- Do not run the repository-wide full pytest suite; the user explicitly requested touched-component tests only.
