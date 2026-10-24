# Portrait Component Compositing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Retain the user's inline execution preference; self-review each atomic task before starting the next. This update authorizes planning, not implementation.

**Goal:** Generate three expression edits and assemble six exact eye/mouth pairs on one prepared portrait, with reusable manually selected regions.

**Architecture:** Keep the existing provider validation, disposable draft, six-frame storage and Console/TTS playback. Add a bounded pure compositor, explicit source-bound region settings and optional applied provenance. The editor saves settings independently; component requests stage dependent frame groups atomically and Apply publishes the complete set.

**Tech Stack:** Python >=3.12, Textual >=8.0.0,<9, SQLite, existing Pillow, image validators and shared decode reservations. No new dependency or provider.

**Spec:** [Updated Console radio portrait design](../specs/2026-10-23-console-radio-portraits-design.md), especially “One-time source regions”, “Three generated components” and “Region settings and composition provenance”.

**Review:** [Planning review and revised-plan second pass](../reviews/2026-10-24-portrait-compositing-plan-review.md#revised-plan-review-after-three-findings).

ADR required: yes
ADR path: backlog/decisions/243-source-bound-portrait-compositing.md
Reason: Changes the generation/retry contract, adds durable independently saved region settings and optional performance provenance, and refines authoring UX. The three review fixes clarify existing ADR-243 decisions (fresh candidates, prepared-base identity and pair review); no new architecture decision or dependency is needed. [ADR-243](../../../backlog/decisions/243-source-bound-portrait-compositing.md) amends [ADR-242](../../../backlog/decisions/242-console-radio-portrait-performance.md); playback and speech contracts remain unchanged.

## Global Constraints

- Roles are exactly `rest`, `small`, `open`, `blink_rest`, `blink_small`, `blink_open`.
- Each normalized still PNG has at most 1,048,576 pixels and 4 MiB encoded; a set has at most 16 MiB encoded. Stricter applicable existing limits win.
- Retain the existing 64 MiB shared active/in-flight preparation budget and independent decode caps. Account for source, edits, masks, finished planes, output and concurrent editor/Console work; explicitly close images and release reservations on all exits.
- Three integer, non-overlapping rectangles, each at least 3×3 pixels, use prepared-image coordinates. Canonical recipe JSON is at most 2048 UTF-8 bytes. Algorithm version 1 and its feather equation are fixed by the spec.
- The user confirms the base has open eyes and closed mouth. `rest` equals the prepared base; do not infer semantic correctness, alter the original, add a fourth request or silently scale masks.
- Generate discloses exactly three sequential reference-image edits using one prepared reference; a selected component retry discloses exactly one edit and its affected group. No provider fallback, automatic paid retry or native-mask API is introduced.
- Prepared-base identity uses canonical RGBA pixels/dimensions, never encoded PNG bytes; original source and artifact hashes keep their existing byte-identity purpose.
- Fresh batch admission replaces all disposable composition state with base rest/zero components; composed Apply/sample requires all three current components plus six roles. Failed retries preserve the prior valid snapshot.
- Preserve base alpha and exact RGBA pixels outside mask support. Reuse finished planes without feathering twice. A failed group replacement leaves the previous candidate intact; incomplete candidates cannot Apply.
- Save regions is an independent explicit CAS write and never alters an active performance. Only Apply/Remove changes performance revision and publishes Console invalidation.
- Full-profile backups carry regions/provenance. Existing portable cards/Actor Packs/Chatbooks/sync omit them. Existing/manual six-frame imports remain valid without composition metadata.
- Source/profile/service/DB/editor/config/region revision are revalidated across async boundaries and before each request/publication. Keep original source bytes, canonical source references, exact sample ownership and scoped cancellation.
- UI uses central `$ds-*` tokens, component patterns, keyboard-accessible fields and production CSS. Do not bind reserved global/terminal keys. Edit CSS sources and rebuild with `python tldw_chatbook/css/build_css.py`.
- TASK-36554 implements the in-app 2×3 pair viewer using original frame PNGs, with a scaled overview and unresampled shared detail crops. Review raster <=1,048,576 pixels; sequential source decode/cleanup, shared budgets and stale-worker fencing are mandatory.
- No automatic face detection, freehand mask tool, runtime layering, new graphics dependency, speech change or portable recipe format.
- Run targeted checks only; ask before a full sweep. Use actual file-backed SQLite for worker/UI tests and the existing owned private-profile harness for app/backup runs. Never bypass recovery/ownership guards.
- Read relevant Architecture docs and `backlog/docs/{design-language,component-patterns,lessons-bounded-artwork-imports,lessons-actor-pack-portability,lessons-testing-evidence,lessons-live-verification,lessons-backlog-hygiene}.md` before owning work.
- Before implementation, read the task, move it In Progress, then add its task-specific Implementation Plan with ADR fields. To Do tasks intentionally have no Implementation Plan or Implementation Notes. Add notes, evidence and checked AC only after actual implementation and review.

## Atomic Backlog tasks

| Unit | Task | Dependencies | Independently reviewable outcome |
| --- | --- | --- | --- |
| 1 | [TASK-36551](../../../backlog/tasks/task-36551%20-%20Compose-exact-paired-portrait-frames-from-bounded-expression-regions.md) | 36550 | Pure compositor, pixel invariants and atomic draft-group replacement |
| 2 | [TASK-36552](../../../backlog/tasks/task-36552%20-%20Persist-source-bound-portrait-regions-and-applied-composition-provenance.md) | 36551 | Durable recipe CAS, provenance, migration and recovery/export semantics |
| 3 | [TASK-36553](../../../backlog/tasks/task-36553%20-%20Select-and-reuse-portrait-eye-and-mouth-regions-in-Personas.md) | 36552 | Usable one-time region selection and independent Save |
| 4 | [TASK-36554](../../../backlog/tasks/task-36554%20-%20Generate-three-portrait-components-and-recompose-dependent-pose-groups.md) | 36551, 36552, 36553 | Three-call generation and one-component dependent retries |
| 5 | [TASK-36555](../../../backlog/tasks/task-36555%20-%20Qualify-composited-portrait-generation-through-restore-and-Console-playback.md) | 36554 | Installed/provider/reopen/restore qualification and accurate final guidance |

All are To Do. No umbrella or scaffolding-only task is needed. Creation used explicit IDs after refreshed all-ref/object and 35-worktree sweeps (observed task max 36550, ADR max 242); recheck before integration because the sweep is not a reservation.

## File and interface map

Planned files below are not claims of implemented APIs.

| File | Responsibility |
| --- | --- |
| Create `tldw_chatbook/Character_Chat/portrait_compositing.py` | Recipe/rectangle/component types, canonical RGBA base identity, masks, bounded composition, paired assembly, invariant-checked recovery |
| Modify `Character_Chat/portrait_performance_authoring.py` | Immutable composition candidate, authoritative snapshot replacement and mode-aware readiness alongside manual-frame import |
| Modify `Character_Chat/portrait_performance.py` | Optional applied-recipe field on `PortraitPerformanceRecord`, default absent |
| Create `tldw_chatbook/DB/portrait_region_settings.py` | Region-settings read/save/CAS and recipe serialization behind existing DB ownership/transaction helpers |
| Modify `tldw_chatbook/DB/ChaChaNotes_DB.py` | Thin repository entry points, next migration version and optional provenance in Apply/get/Remove |
| Create one executed SQL migration under `tldw_chatbook/DB/migrations/` | Derive the exact versioned filename from the live schema immediately before implementation; do not reserve a number during planning |
| Modify `tldw_chatbook/DB/sql_validation.py` | Chachanotes new-table allowlist; verify the actual inventory/runtime guards |
| Create `tldw_chatbook/Widgets/Persona_Widgets/portrait_region_editor.py` | Focused region modal, coordinate transform, numeric fields and detailed mask preview |
| Modify `Widgets/Persona_Widgets/portrait_performance_editor.py` | Region entry/status, mode-aware Apply/sample readiness, component selector and Review pairs action |
| Create `tldw_chatbook/Widgets/Persona_Widgets/portrait_composition_review.py` | Static in-app labeled matrix, original-frame detail crops, keyboard pan and bounded stale-safe rendering (Unit 4) |
| Modify `UI/Persona_Modules/portrait_performance.py` | Prepare/load/save region settings, source/recipe epochs, manual/composed state and Apply provenance |
| Create `tldw_chatbook/UI/Persona_Modules/portrait_composition_generation.py` | Replacement for the existing `_generate` body: component request loop and fenced group publication; avoid enlarging the current controller with a second parallel generation path |
| Modify `Character_Chat/expression_generation.py` | Closed component prompt vocabulary; preserve ordinary expression-generation callers |
| Modify `css/core/_variables.tcss` only if needed | Missing central visual tokens, then rebuild generated bundle |
| Test files in each unit below | Pure pixels, repository, UI/request boundaries and installed/restore behavior |

## Shared contracts

Unit 1 defines these in `Character_Chat/portrait_compositing.py`; `PerformanceFrame` remains the existing normalized type:

```python
ComponentRole = Literal["eyelids", "small", "open"]
COMPONENT_ROLES: tuple[ComponentRole, ...] = ("eyelids", "small", "open")

@dataclass(frozen=True, slots=True)
class PixelRect:
    x: int
    y: int
    width: int
    height: int

@dataclass(frozen=True, slots=True)
class PortraitCompositionRecipe:
    algorithm_version: int
    width: int
    height: int
    base_pixel_sha256: str  # canonical decoded RGBA identity defined in the spec
    left_eye: PixelRect  # image-left, not anatomical left
    right_eye: PixelRect
    mouth: PixelRect
    neutral_base_confirmed: bool

def prepared_base_pixel_sha256(base_png: bytes) -> str: ...
def validate_recipe(base_png: bytes, recipe: PortraitCompositionRecipe) -> None: ...
def compose_component(
    base_png: bytes, recipe: PortraitCompositionRecipe,
    role: ComponentRole, edit_png: bytes,
) -> bytes: ...  # normalized finished full-canvas plane, feathered once

def assemble_components(
    base_png: bytes, recipe: PortraitCompositionRecipe,
    components: Mapping[ComponentRole, bytes],
) -> tuple[PerformanceFrame, ...]: ...  # available roles in FRAME_ROLES order

def recover_components(
    base_png: bytes, recipe: PortraitCompositionRecipe,
    frames: tuple[PerformanceFrame, ...],
) -> dict[ComponentRole, bytes]: ...  # complete-set pixel invariants required
```

Algorithm implementation bodies are owned by Unit 1; signatures above fix the contract rather than prescribing extra abstractions. Unknown roles/versions and booleans passed as integers are invalid. An empty component map yields only `rest`; a mouth yields its open-eye frame; eyelids yields `blink_rest` plus blink counterparts for available mouth planes. Composed Apply still requires all three ready components and exactly six roles from the same snapshot; manual Apply retains the six-role rule. Components are already-composited planes from this recipe/base, never arbitrary imported frames. The controller treats these bytes as trusted only after composition or successful invariant-checked recovery.

Unit 1 also defines this immutable disposable state in `portrait_performance_authoring.py` (import `dataclass, field` from `dataclasses`):

```python
@dataclass(frozen=True, slots=True)
class PortraitCompositionCandidate:
    recipe: PortraitCompositionRecipe
    regions_revision: int
    components: tuple[tuple[ComponentRole, bytes], ...] = field(repr=False)
    # Finished planes in role order; image bytes never enter routine repr/logs.
    frames: tuple[PerformanceFrame, ...] = field(repr=False)
    # All currently available derived roles; immutable normalized frame records.

# PortraitPerformanceDraft methods/properties:
def replace_composition(self, candidate: PortraitCompositionCandidate) -> None: ...
@property
def composition(self) -> PortraitCompositionCandidate | None: ...
@property
def can_apply(self) -> bool: ...
def stage_group(self, frames: tuple[PerformanceFrame, ...]) -> None: ...
```

`replace_composition` validates duplicate/known component keys, positive region revision,
normalized metadata/dimensions/aggregate limits and the exact available-role set implied
by the component keys, then replaces the frame map and candidate together without a
merge. The initial candidate is `PortraitCompositionCandidate(recipe, regions_revision,
(), (rest,))`. Build/validate its pixel content on the existing drain worker before
admission; install it after consent and before any provider call. Cancelled consent
or initial validation failure preserves the previous draft. A fresh admitted batch
cannot inherit even one old slot. Each successful edit creates a new complete snapshot
of its currently available state and calls the same replacement API. An exception
preserves the prior snapshot object and all its state. Check captured previous snapshot
identity after awaits, in addition to source/owner/region/config fences.

`can_apply` is true for manual mode only with six valid roles, or for composed mode
only with `COMPONENT_ROLES` all present and all six current roles. `frames()` enforces
this property; Apply and Play speech sample use it instead of slot-count checks.
`stage_group` is only the atomic manual-import merge path; `stage(frame)` delegates to
it. Successful manual stage/clear drops composition lineage together with the frame
edit; a failed import changes neither. Saving a changed recipe installs its empty
composition candidate, never turns the old composed frames into a complete manual set.
No image decode runs on the event loop; snapshot content is composed/normalized by
the existing drain worker. Unchanged snapshots share immutable bytes rather than
copying encoded frame/component buffers. `cancel()`/discard clears the stored candidate
as well as frames/source bytes; bounded worker/viewer snapshots release their refs
when drained. Do not log or serialize component/image bytes in routine diagnostics.

Canonical pixel identity uses the spec's exact domain tag, big-endian dimensions and
unpremultiplied row-major RGBA bytes. The original encoded source hash and encoded
`PerformanceFrame.sha256` remain unchanged. Account hashing scratch in decode budgets.
Changing PNG compression/metadata with equal decoded pixels must not invalidate a
recipe, increment saved-region revision or disable reopen/retry. Actual source, any
RGBA byte, dimensions or unsupported algorithm changes still invalidate as specified.

Unit 2 defines `PortraitRegionSettings(source: PerformanceSource, recipe: PortraitCompositionRecipe, revision: int)` and these `CharactersRAGDB` seams:

```python
def get_character_portrait_regions(
    self, *, character_id: int, expression_key: str,
) -> PortraitRegionSettings | None: ...

def save_character_portrait_regions(
    self, source: PerformanceSource, recipe: PortraitCompositionRecipe, *,
    expected_revision: int,
) -> int: ...

# Existing method gains optional keyword-only arguments; old callers remain valid.
def apply_character_portrait_performance(
    self, source: PerformanceSource, frames: tuple[PerformanceFrame, ...], *,
    expected_revision: int,
    composition_recipe: PortraitCompositionRecipe | None = None,
    expected_regions_revision: int | None = None,
) -> int: ...
```

`PortraitPerformanceRecord.composition_recipe` defaults to `None`. Imported Apply clears it. Composited Apply requires a valid recipe plus matching saved-region revision/content; validate the base/frames outside the write transaction and fence the source/region/performance stamps inside it. Store a canonical recipe snapshot with the applied set; do not depend on future settings edits. The saved row stores one current full source identity and recipe per character/expression with a monotonically increasing revision, even when explicitly replacing coordinates for a different source. A mismatching saved source is shown as “selection required”, not reused. An unchanged Save with the current expected revision returns that revision without invalidating components; a stale revision still conflicts. No new helper may copy, clear or bypass profile ownership registries.

## Unit 1 — TASK-36551: bounded compositor and atomic staging

**Files:** Create `tldw_chatbook/Character_Chat/portrait_compositing.py`, `Tests/Character_Chat/test_portrait_compositing.py`. Modify `portrait_performance_authoring.py` and `Tests/Character_Chat/test_portrait_performance_authoring.py`.

**Consumes:** Existing `PerformanceFrame`, `normalize_performance_frame(s)`, `FRAME_ROLES`, bounded decode reservation/cleanup seams and `PortraitPerformanceDraft`. **Produces:** All Unit 1 contracts above; no provider or DB write.

- [x] **1. Write a red pixel-oracle test.** Add local fixtures that encode a 64×64 RGBA base and visibly different whole-canvas edits, with rectangles left `(8,8,10,10)`, right `(42,8,10,10)`, mouth `(22,36,20,16)`. Construct the recipe with `prepared_base_pixel_sha256(base)` and the exact decoded dimensions. Decode returned PNGs into pixel lists; independently select coordinate sets for eye/mouth/outside regions. Assertions include:

```python
frames = assemble_components(base, recipe, {
    role: compose_component(base, recipe, role, edit)
    for role in ("eyelids", "small", "open")
})
assert tuple(f.role for f in frames) == FRAME_ROLES
assert rgba(frames[0].png) == rgba(base)
for x, y in outside_rectangles:
    assert all(rgba(f.png)[x, y] == rgba(base)[x, y] for f in frames)
for mouth_role in ("rest", "small", "open"):
    assert pixels(mouth_role, mouth_region) == pixels("blink_" + mouth_role, mouth_region)
assert all(alpha(f.png) == alpha(base) for f in frames)
```

`rgba`, `pixels`, `alpha` and coordinate sets are local test helpers using Pillow `getpixel`; define them in this test file, not a new production utility. Also assert eye pixels match across mouth states, every rectangle outer-border pixel (zero mask alpha) equals the base, and whole outputs differ. Use a deliberately globally darkened edit as the control proving no exposure drift outside the masks.
- [x] **2. Run the focused red case.** `.venv/bin/python -m pytest Tests/Character_Chat/test_portrait_compositing.py -q -o addopts=''`; expected failure is the missing compositor contract, not bootstrap/recovery setup.
- [x] **3. Implement the fixed v1 algorithm and bounded cleanup.** Validate types/geometry/digest/confirmation, decode admitted stills, create inward masks from the spec equation and preserve base alpha. Compose raw edits once; assemble using binary eye-support copying. Recover finished planes from validated `small/open/blink_rest` and compare all six decoded outputs against an independent reassembly before reuse. Implement manual `stage_group` and authoritative `replace_composition` as separate atomic paths; wire `can_apply` and `frames()` to candidate completeness rather than occupied slots. The core assembly copies finished eye pixels, preserving alpha:

```python
# These names denote the bounded images/masks admitted earlier in this step.
finished = Image.composite(edit_rgb, base_rgb, inward_mask)
finished.putalpha(base_alpha)
blink = Image.composite(eyelid_plane, mouth_plane, binary_eye_support)
blink.putalpha(base_alpha)
```

Create masks using the spec's integer equation; mask support, not feather alpha, controls the second copy. Reuse prior normalized frames for unaffected roles during retries so their encoded bytes remain identical.
- [x] **4. Exercise failure and reuse controls.** Reject float/bool/negative/overlapping/edge-exceeding/too-small coordinates, unknown versions, wrong-sized/animated/oversized/malformed edits. Cover alpha-zero pixels, non-square canvases, masks touching image borders, incomplete maps, repeat retries, normalization/set-byte failure and held concurrent decode reservations. Inject encode failure; assert every image core closes and the reservation counter returns to its entry value. A failed replacement leaves its prior snapshot/frames unchanged. Add encoding-equivalence and candidate-reset controls:

```python
# png0/png9 encode the same source pixels using compression levels 0 and 9.
assert png0 != png9
assert prepared_base_pixel_sha256(png0) == prepared_base_pixel_sha256(png9)
validate_recipe(png9, recipe_for_png0)
# draft previously contains a complete applied/manual set.
draft.replace_composition(PortraitCompositionCandidate(recipe, 1, (), (rest,)))
assert draft.staged_roles == ("rest",)
assert not draft.can_apply
with pytest.raises(ValueError):
    draft.frames()
```

Also change one RGB byte, one alpha byte, RGB under zero alpha, and dimensions to require different identities. Inject bad replacement metadata after a valid complete candidate and assert candidate object, encoded frames and readiness are unchanged. A failed retry of a complete candidate retains `can_apply`; a fresh incomplete candidate cannot inherit old slots.
- [x] **5. Verify, document and review.** Run only both affected test files, Ruff check/format on edited files and `git diff --check`; compare any pre-existing diagnostics by file. Update task notes with exact evidence and ADR-243, check AC, self-review memory lifetime/pixel invariants, then commit `feat: compose exact paired portrait frames from source regions`.

**Execution checkpoint:** implemented and self-reviewed. New core/authoring gate:
27 passed; affected core/legacy generation/motion/Personas callers: 58 passed before
three additional cleanup/max-size controls and the successful retry control. Final
core checks include those controls. Ruff lint/format and diff checks pass. No new
provider/UI integration is claimed. See [atomic review](../reviews/2026-10-24-task-36551-review.md).

## Unit 2 — TASK-36552: durable settings and optional provenance

**Files:** Create `tldw_chatbook/DB/portrait_region_settings.py`, `Tests/ChaChaNotesDB/test_portrait_region_settings.py`, `Tests/ChaChaNotesDB/test_portrait_regions_migration.py`, and the actual next migration SQL. Modify `DB/ChaChaNotes_DB.py`, `Character_Chat/portrait_performance.py`, `DB/sql_validation.py`, `Tests/DB/test_sql_validation.py`, `Tests/ChaChaNotesDB/test_portrait_performance_repository.py`, `Tests/Backup_Recovery/test_portrait_performance_restore.py`, and existing targeted character/Actor Pack/Chatbook export tests.

**Consumes:** Unit 1 recipe validation/recovery and existing performance source/write fences. **Produces:** Unit 2 record/repository signatures above and backward-compatible optional provenance.

- [x] **1. Write red repository and upgrade controls.** Use real SQLite; an old-version fixture has an existing applied six-frame set. After upgrade, assert its bytes/revision unchanged and provenance absent. Save regions independently and reopen. Add a stale-CAS control:

```python
before = db.get_character_portrait_performance(character_id=cid, expression_key="neutral")
rev = db.save_character_portrait_regions(source, recipe, expected_revision=0)
assert db.get_character_portrait_performance(character_id=cid, expression_key="neutral") == before
with pytest.raises(PerformanceConflict):
    db.save_character_portrait_regions(source, recipe, expected_revision=0)
assert db.get_character_portrait_regions(character_id=cid, expression_key="neutral").revision == rev
```

Use the existing DB fixture/source builder; add cases for different exact source bindings with the same bytes and profile replacement at the controller boundary in Unit 3.
- [x] **2. Run the new repository/migration cases red.** `.venv/bin/python -m pytest Tests/ChaChaNotesDB/test_portrait_region_settings.py Tests/ChaChaNotesDB/test_portrait_regions_migration.py -q -o addopts=''`; attribute setup failures before changing unrelated code.
- [x] **3. Implement one migration and the contract.** Discover `_CURRENT_SCHEMA_VERSION` immediately before allocating the SQL filename. Add one region-settings table with character FK/unique character-expression/source fields/revision/canonical bounded recipe JSON; add nullable provenance JSON to the existing performance parent. Register/package migration and update the chachanotes allowlist. Delegate settings persistence from thin DB methods. Validate recipe/base before entering SQL; use parameterized transactions and existing source stamps plus both CAS revisions inside publication. Default optional arguments to absent and clear provenance on manual Apply/Remove. Separate content validation from re-encoding for unchanged stored roles: when the submitted role exactly matches the currently stored bytes/hash/dimensions at the expected performance revision, validate format/limits/content but preserve those bytes. Recheck that stored identity under the publication CAS; newly changed/unrecognized input still uses the existing normalization choke point. Do not require an unchanged trusted stored frame to reproduce the current encoder's byte output as an extra validity condition.
- [x] **4. Verify lifetime/rollback/export boundaries.** Test interrupted Apply rollback of both frames and provenance, stale region changes during publication, changed source/prepare geometry or RGBA identity, equivalent prepared PNG encodings with stable settings revision/reopen/retry, an already-stored valid frame encoded differently from the current encoder retaining its exact bytes through a component retry/Apply, malformed/oversized/unknown recipe JSON, legacy/manual/invalid-provenance read fallback, Remove retaining regions, soft-delete/restore and permanent FK cascade. Extend native backup/restore to check settings/provenance after fresh DB open. Assert ordinary actual archive/JSON/Chatbook outputs and sync logs are unchanged. No new sync registration or sidecar paths. Persist only `base_pixel_sha256` for prepared-base identity in both saved settings and applied provenance, with the same canonical algorithm; never substitute an encoded PNG hash. Keep the new schema payload bounded in SQL as well as Python:

```sql
-- Fields in character_portrait_regions, alongside the exact source columns/FK:
recipe_json TEXT NOT NULL CHECK(
    json_valid(recipe_json) AND length(CAST(recipe_json AS BLOB)) BETWEEN 1 AND 2048
),
revision INTEGER NOT NULL CHECK(revision > 0),
PRIMARY KEY(character_id, expression_key)
```

The optional parent provenance column uses the same JSON/UTF-8 byte bound or NULL. Verify migration SQL is executed and shipped with `Tests/Packaging/test_installed_distribution.py`; do not create a decorative migration twin.
- [x] **5. Verify and commit the durable unit.** Run the new files plus affected repository/restore/export/SQL inventory and migration packaging cases, Ruff and diff checks. Update `Docs/Architecture/character-chat.md` and task notes, review source authority/transaction ownership/old caller compatibility, then commit `feat: persist reusable portrait regions and composition provenance`.

**Execution checkpoint (2026-10-24):** Unit 2 implemented and self-reviewed; schema 83,
independent region CAS and atomic optional provenance passed repository/migration,
actual native F9 restore/fresh-open/edit, portable export and built-migration checks.
See [TASK-36552 review](../reviews/2026-10-24-task-36552-review.md). No UI/provider
workflow change is claimed by this unit.

## Unit 3 — TASK-36553: one-time region selection

**Files:** Create `Widgets/Persona_Widgets/portrait_region_editor.py` and `Tests/UI/test_portrait_region_editor.py`. Modify `Widgets/Persona_Widgets/portrait_performance_editor.py`, `UI/Persona_Modules/portrait_performance.py`, `Tests/UI/test_personas_portrait_performance.py`, `Docs/User_Guide/roleplay-chat-dictionaries/characters-and-personas.md`, and central CSS tokens/bundle only if the language lacks a required value.

**Consumes:** Unit 1 recipe/geometry validation and Unit 2 settings CAS. **Produces:** Explicit Edit/Save regions workflow and reusable region state for generation.

- [ ] **1. Add red coordinate and mounted interaction tests.** Expose a pure transform on the region widget, `image_pixel_at(cell_x: int, cell_y: int) -> tuple[int, int] | None`, derived from the currently rendered contain/crop transform. Exercise wide/tall images, terminal pixel aspect, letterbox clicks and resize; compare the same image point after resize. Use mounted labeled fields to save three rectangles and confirm the neutral base, without calling repository saves directly from the test:

```python
assert editor.image_pixel_at(letterbox_x, letterbox_y) is None
await pilot.click("#portrait-regions-save")
settings = db.get_character_portrait_regions(character_id=cid, expression_key="neutral")
assert settings.recipe == selected_recipe
assert db.get_character_portrait_performance(character_id=cid, expression_key="neutral") == active_before
```

Test fixtures must await the actual selected-character load worker and visible editor, rather than just the scheduling wrapper.
- [ ] **2. Run `Tests/UI/test_portrait_region_editor.py` red** using the existing bootstrap-profile/file-backed DB fixture and `-q -o addopts=''`.
- [ ] **3. Implement the smallest complete selection UI.** Three labeled region choices, two-corner pointer selection, x/y/width/height fields, source/detail-mask preview, neutral-base checkbox and Save/Cancel. Numeric fields provide complete keyboard operation even without graphics. Store only source-pixel rectangles; reset unfinished pointer anchors on resize. All preview image work is off-thread, bounded and fenced. Capture source/owner/region revision when opening; Save revalidates after its await and commits only settings. Reopening matching settings preloads them even if the prepared PNG encoding differs but its canonical RGBA identity is equal. Cancel leaves durable settings and active artwork alone. Save uses the captured settings revision, never a newly reread value that would hide a concurrent edit:

```python
def save_regions():
    return authority.db.save_character_portrait_regions(
        source, recipe, expected_revision=captured_revision
    )

outcome = await controller._native(authority, save_regions)
```

This uses the existing `_native(authority, function, *args)` worker/transaction ownership seam rather than a new write executor. After awaiting, check the editor/source/region epoch before publishing the saved state.
- [ ] **4. Add stale and accessibility controls.** No Save/Generate for missing or invalid regions/neutral confirmation. Source/profile/editor changes, stale concurrent Save, visual binding replacement and late preview results cannot reach a new owner. Saving changed regions first stops/awaits only this draft sample through the existing scoped cancellation, then installs a rest-only/zero-component snapshot for the new recipe, requiring a fresh batch; manual frames remain usable and applied frames remain untouched. Verify Tab/focus, named fields, keyboard entry, production CSS, narrow layouts and Static/Reduce motion. No reserved terminal/global binding is introduced.
- [ ] **5. Verify and commit.** Run the new widget file plus affected mounted Personas tests, token governance, component-pattern and CSS sync guards if CSS changes; Ruff/diff checks. Document one-time-save semantics and neutral-base requirement. Review actual source/mask captures at full detail and a narrow layout, then commit `feat: select and reuse portrait expression regions`.

## Unit 4 — TASK-36554: component generation and dependent retries

**Files:** Create `UI/Persona_Modules/portrait_composition_generation.py`, `Widgets/Persona_Widgets/portrait_composition_review.py` and `Tests/UI/test_portrait_composition_review.py`. Modify the existing controller, `Character_Chat/expression_generation.py`, `Widgets/Persona_Widgets/portrait_performance_editor.py`, `Tests/Character_Chat/test_portrait_performance_generation.py`, `Tests/UI/test_personas_portrait_performance_generate.py`, `Tests/UI/test_personas_portrait_performance.py`, `Docs/Architecture/media-generation.md`, `Docs/Architecture/character-chat.md` and the character user guide.

**Consumes:** Saved recipe/current revision, pure compositor/group staging, existing configured registry/build_request/run_generation and original source authority. **Produces:** Three-component full batch, exact one-component retry, manual/composed transitions, Apply provenance and the actual in-app full-resolution pair review.

- [ ] **1. Replace six-call expectations with red component controls.** Keep actual request building/validation; fake only the external adapter. Return whole-canvas edits with deliberately wrong surrounding lighting but correct dimensions. Drive the real mounted controls and capture confirmation text:

```python
assert await controller.handle(CharacterPerformanceRequested("generate"))
assert len(calls) == 3
assert "3 image-edit requests" in disclosures[0]
assert all(c.reference_image is calls[0].reference_image for c in calls)
assert controller.draft.staged_roles == FRAME_ROLES
```

Extend `CharacterPerformanceRequested` with keyword `component: ComponentRole | None = None` while retaining `role` for imports/preview. Retry `component="small"`, assert one added call and only `small/blink_small` change; retry `"eyelids"`, assert only three blink frames change. Rest and unaffected PNG bytes stay exact. Define `compose_performance_component_prompt(*, component: ComponentRole, expression_key: str) -> str` in `expression_generation.py` without altering ordinary expression prompts/callers.
- [ ] **2. Run affected generation tests red** with `-q -o addopts=''`; existing six-request expectations should fail for the deliberately changed behavior, not disappear without replacements.
- [ ] **3. Replace the generation body using existing admission.** `_generate` delegates to `async generate_components(controller, authority, draft, component: ComponentRole | None, epoch: int) -> bool` in the focused new module. A full batch freezes source/prepared reference/recipe/revision/config, obtains consent, then builds the validated rest-only snapshot off-thread and installs it with `draft.replace_composition` before the first provider request. It clears all old draft frames/components without changing the applied DB set. Derive all button/publication readiness from `draft.can_apply`. After each validated same-size result, call compositor/assembly on the existing worker and stage the resulting dependent group once after owner checks. Retain successful same-candidate components on failure and permit explicit retries. Reopen recovers planes only after full invariant validation against applied provenance and matching saved settings. Hold candidate planes only for this disposable editor and release on retirement. Here `component` is the current requested component role. The pure work happens inside the existing drain worker, with staging after all source/settings checks:

```python
previous = draft.composition  # capture one immutable snapshot before the worker

def recompose():
    updated = dict(previous.components)
    updated[component] = compose_component(base_png, recipe, component, result.content)
    affected = {
        "eyelids": {"blink_rest", "blink_small", "blink_open"},
        "small": {"small", "blink_small"},
        "open": {"open", "blink_open"},
    }[component]
    previous_by_role = {frame.role: frame for frame in previous.frames}
    frames = tuple(
        previous_by_role.get(frame.role, frame) if frame.role not in affected else frame
        for frame in assemble_components(base_png, recipe, updated)
    )
    return PortraitCompositionCandidate(
        recipe, previous.regions_revision,
        tuple((r, updated[r]) for r in COMPONENT_ROLES if r in updated), frames,
    )
```

Call `await controller._drain(recompose, task_name="portrait-performance-compose")`.
Reject cancellation/error/stale source/owner/region/config and require
`draft.composition is previous` before `draft.replace_composition(outcome.value)`.
Never stage frames and then separately publish a mutable component dictionary.
A failed retry preserves the old snapshot; partial progress belongs only to the
current admitted fresh batch. Region-setting changes use the same rest-only reset.
- [ ] **4. Preserve controls and write boundaries.** Generate and Regenerate labels/counts come from actual component requests, not six output roles. Keep six-role import/preview selector separate from the three-component selector. Disable/reject a component retry before any provider call when `draft.composition is None`; a missing component in an existing partial candidate remains retryable. Never infer components from manual frames. A successful Import or explicit Clear drops composition lineage/cache together with the frame edit; cancelled/invalid imports retain both frames and lineage; normal Apply remains possible without masks. Composited Apply passes the captured recipe and region revision to the existing DB seam. Region save changes clear generated candidate state; source/DB/editor/config/recipe changes cancel remaining requests/reject late completions. Do not call providers while selecting/saving/reviewing. No change to `_sample` owners, preview clocks or Console playback.
- [ ] **5. Exercise races and repeated edits.** Cover uncertain middle-component response, invalid PNG/wrong dimensions, normalization/set-limit failure after an edit, late uncancellable job, duplicate actions, region CAS change during await, source becoming dirty, cancel/profile/config replacement, manual import after generation, old/manual provenance, Apply rollback and repeated retry after fresh reopen. Start from a complete existing performance and fail each first/middle/all component combination in a new full batch; assert no old slot survives, Apply/sample/publication stay blocked until all three new components are ready and the active DB frames/revision stay unchanged. Cancel consent and fail initial snapshot validation as controls retaining the old draft and sending zero calls. Ensure no automatic extra call and no stale group publication. Compare eye/mouth pixels and unaffected encoded frames; do not assert only role names/counts.
- [ ] **6. Implement and test the in-app pair review.** Unit 4 owns `PortraitCompositionReview` in the new review widget, opened by **Review pairs…** beside Apply. It receives immutable original normalized frame PNGs, the recipe and an `is_current` closure bound to candidate/source/owner identity. Admit it through `CharacterPerformanceRequested("review")` only for a complete composed candidate. Its constructor is `PortraitCompositionReview(frames: tuple[PerformanceFrame, ...], recipe: PortraitCompositionRecipe, is_current: Callable[[], bool])`; it uses the existing safe-modal cancellation pattern. The controller owns its handle and explicitly retires it on discard, source/profile/recipe change or candidate replacement; passive worker predicates alone do not retire an already displayed stale review. Layout is three mouth columns × two eye rows. Provide a labeled scaled overview plus common source-coordinate detail crops for left eye/right eye/mouth, with keyboard/numeric pan. Never feed it avatar `current_image` or a fitted raster. Detail generation copies original RGBA crop pixels without pre-resampling; derive native crop bounds from viewport/backend geometry, clamp the assembled review raster to 1,048,576 pixels and pan through larger regions. Decode/crop/close one original source at a time under shared reservations; retain no six decoded full canvases. Reject late crop workers after candidate/source/profile/recipe change, resize or close. Static/Reduce motion and keyboard-only mode work; no provider/audio/save/Apply action or review-acknowledgement gate is added.

Add mounted `Tests/UI/test_portrait_composition_review.py` controls with a one-pixel marker absent from a deliberately lossy fitted thumbnail:

```python
# raw_frame.png is the actual normalized draft frame, not avatar.current_image.
assert detail_tile.tobytes() == raw_rgba.crop(common_crop_box).tobytes()
assert review_raster.width * review_raster.height <= MAX_FRAME_PIXELS
assert observed_cell_roles == (("rest", "small", "open"),
                               ("blink_rest", "blink_small", "blink_open"))
```

Construct the marker in a local fixture and assert the raw crop still contains it;
make use of the fitted thumbnail fail this assertion. Test pan/resize and all six
identical crop coordinates, sequential decode/cleanup including encode failure,
shared-budget refusal, stale completion, close and unchanged draft/DB state. The
crop raster is exact source data; graphics/mosaic presentation still has terminal
capability limits and must be labeled honestly.
- [ ] **7. Verify, review and commit.** Run the two affected generation files plus the new pair-review widget tests, authoring/UI and guarded TTS sample controls, Ruff/diff checks and UI token/CSS guards if changed. Rewrite Architecture/user guidance around three requests and dependent retries; mark old six-request descriptions historical. Self-review actual request count, no double feathering, candidate atomicity and async authority, then commit `feat: generate portrait components with exact dependent pose retries`.

## Unit 5 — TASK-36555: installed and real workflow qualification

**Files:** Extend `Tests/ProductionApp/test_console_radio_portraits.py`, `Tests/Backup_Recovery/test_portrait_performance_restore.py` and affected export cases. Create evidence under `Docs/superpowers/qa/portrait-compositing/2026-10-24/`, including an opt-in live adapter probe and `qualification.md`; create final task review. Update the linked user/Architecture docs to actual behavior and this plan's checkboxes.

**Consumes:** Units 1–4 as installed application behavior. **Produces:** Reviewed end-to-end evidence and an honest completion/coverage record.

- [ ] **1. Add the installed/private-profile journey red.** Reuse the actual TldwCli production-entry bootstrap, network guard and native restore harness from TASK-36550. Through visible editor actions mark/save regions, generate, retry small mouth and eyelids, open the actual Review pairs modal, inspect overview and unresampled eye/mouth crops for all six, Apply, navigate Console and reopen. A separately generated QA image cannot replace this viewer path. The deterministic normal-CI adapter remains fake only at the external call boundary. Assertions bind real saved state and pixels:

```python
assert reopened_regions.recipe == selected_recipe
assert reopened_record.composition_recipe == selected_recipe
assert tuple(f.sha256 for f in reopened_record.frames) == applied_hashes
assert source_bytes_after == source_bytes_before
assert set(observed_preview_roles) == set(FRAME_ROLES)
assert observed_pair_review_opened
assert review_crop_pixels == original_frame_crop_pixels
```

Use canonical character-context navigation and hydrated visible-avatar barriers. Do not call avatar ticks, fabricate speech snapshots or bypass ownership/recovery to make the journey pass.
- [ ] **2. Verify native backup/restore through another edit.** After full-profile backup/restore, fresh app open must preload the same regions, recover exact components and permit a one-request mouth retry. Confirm old source/session ownership cannot publish, old frame bytes persist until explicit Apply, and ordinary exports/sync remain unchanged. Use a real migrated DB and preserve original artwork. Reencode the prepared reference with different PNG compression while keeping source bytes/pixels fixed; restored settings revision and component retry eligibility must remain unchanged. Then alter one actual RGBA byte as the rejection control.
- [ ] **3. Qualify one real configured reference-image adapter separately.** In an isolated profile, record destination/model, three actual requests, returned hashes, selected retry requests, saved regions/base and corrected full-resolution 2×3 matrix. Open the actual pair viewer, inspect original-frame detail crops, alignment/edges/lighting and actual Console-size output. Also run a fresh-batch failure from a prior complete set in the deterministic installed journey to prove no stale draft role can satisfy Apply; prior active DB frames remain intact. Use existing offline cycle/idle and guarded speech sample for the final set; no new audio subsystem requires recalibrating unchanged DAC timing. Historical external ChatGPT patches can qualify composition/import but cannot count as this in-app adapter evidence. If no reachable configured backend exists, retain this AC as open and report the exact missing evidence; do not add a provider or silently substitute a service.
- [ ] **4. Run the affected gate.** New compositor/region/repository/migration/request/UI/installed/restore/export cases, SQL allowlist and migration-package guards, token/component/CSS checks for touched UI, Ruff/format and `git diff --check`. Use `-q -o addopts='' -p no:cacheprovider --tb=short --show-capture=no` and an owned private `--basetemp`. Attribute baseline failures to the same pre-change source; no full sweep unless explicitly requested. Record commands, exit results, source/runtime identities and hash-checked assets.
- [ ] **5. Self-review and commit only proved completion.** Match every AC to evidence; keep unavailable live backend/terminal/device coverage explicit. Update user/Architecture docs and the final review, task notes and plan checkboxes. Add an incident-backed lesson only if new general knowledge emerged. Mark Done only when all AC and DoD are met, then commit `test: qualify composited portrait generation and reusable regions`.

## Review checklist and handoff

- [ ] Core pixel oracle covers alpha/outside/pair invariants, globally retouched edits and PNG-encoding-independent base identity.
- [ ] Recipe save and performance Apply have separate revisions; both are fenced for composited publication.
- [ ] Applied recipe snapshot is independent of future settings; recovery never blends finished planes again.
- [ ] Fresh batches authoritatively replace frames/components; composed readiness requires all three current components, and failed retries preserve their prior snapshot.
- [ ] Partial candidates, manual imports and dependent-group retries have explicit transitions.
- [ ] TASK-36554 delivers the actual raw-frame pair viewer; TASK-36555 opens it and verifies native crop pixels, memory bounds and stale rejection.
- [ ] Exact source coordinates survive view transforms; region selection is possible with keyboard and graphics fallback.
- [ ] Neutral-base restriction, three-call consent and real-provider qualification limits are visible.
- [ ] Migration/restore/edit/export boundaries and original source preservation have executable tests.
- [ ] Every To Do task has outcome AC, only existing lower-numbered dependencies, linked ADR/spec/plan and no premature notes/checkmarks.

Planning self-review is documented separately; these boxes track implementation evidence. Product implementation has not started. The previously selected execution style remains inline, sequential and reviewed per task; wait for the user to start execution rather than reopen an execution-mode choice or create chats/subagents.
