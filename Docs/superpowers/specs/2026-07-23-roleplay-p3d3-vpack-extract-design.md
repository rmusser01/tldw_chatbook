# Roleplay P3d-3 — `.tldw-persona-vpack` Extraction — Design

**Date:** 2026-07-23
**Program:** Roleplay (Personas) redesign — P3d cycle 3 (after P3d-1 reactive expression avatar #792 and P3d-2 zip-set import/export #797).
**Status:** design approved (brainstorming); ready for implementation plan.

## Goal

Let a user import a `.tldw-persona-vpack` archive (tldw_server2's Persona Visual pack format) through the character editor's existing **"Import set…"** button and extract each of our four states' **static** image (idle / thinking / speaking / error), feeding the same `_apply_expression_set` orchestrator P3d-2 built. This is the terminal-portable slice of the server's animated pack format: one static frame per state; everything animation-shaped is inert.

## Context

- **P3d-1** (#792): reactive Console avatar; `character_expression_images` table; states idle/thinking/speaking/error.
- **P3d-2** (#797): pure `Character_Chat/expression_set_io.py` — `resolve_local_expression_set(list[Path]) -> ExpressionSetResolution{images, skipped, notes}` (zip/dir/files by filename stem, security-capped, never-raises) → screen orchestrator `_apply_expression_set` (idle staged in the editor, three written to the table, single render-token bump) → summary notification. Import UI: single-file `EnhancedFileOpen`, screen-boundary `validate_path_simple`, `_io_dialog_active` gate.
- **P3d-3** adds a second archive format to the SAME entry point. No new UI verb, no new orchestrator, **no migration** (dev schema is v24 via an unrelated Console-branching migration; P3d-3 touches no schema).

## Reference-format ground truths (scouted in tldw_server2 — verified, use verbatim)

Files: `tldw_Server_API/app/core/Persona/visuals.py`, `visual_portability/{constants,exporter,importer,preview}.py`.

1. A `.tldw-persona-vpack` is a **zip** (so `zipfile.is_zipfile()` is True). Required members: `manifest.json`, `metadata/pack.json`, `metadata/assets.json`, `checksums/sha256.json` (constants.py `REQUIRED_MEMBERS`). Assets live under `assets/persona_visuals/{source_asset_id}{ext}`.
2. **Manifest V1** (`manifest_version: 1`, `renderer_type: "sprite_frames"`): `states` maps state→`animation_id`; `animations[id]` has `frame_rate` + `frames[]`, each frame `{asset_id, region?, duration_ms?}`. Animations may instead declare an **`asset_ids` shorthand** (a list), normalized server-side to `frames = [{"asset_id": a} for a in asset_ids]` (visuals.py:210-220) — the extractor must handle both shapes.
3. **`region`** is optional; when present it is `{x, y, width, height}` — all ints, `x,y >= 0`, `width,height > 0` (visuals.py `_validate_frame_region`). Absent region = the whole asset is the frame.
4. **`preview_frame`** is an optional int index validated server-side as `0 <= preview_frame < len(frames)` (visuals.py:280-283). Frame choice for extraction: **use `preview_frame` when it is a valid in-range int, else `frames[0]`**.
5. **`metadata/assets.json`** is `{"assets": [...]}` where each entry carries `source_asset_id` (matches the manifest's `asset_id`), **`asset_path`** (the archive member path), `asset_bytes_status` (`"present"`/`"missing"`), plus sha256/size/width/height. Asset resolution: manifest `asset_id` → the entry with `source_asset_id == asset_id` and `asset_bytes_status == "present"` → read `asset_path`.
6. Server state vocabulary: `VISUAL_STATE_IDS = {idle, wake_armed, listening, thinking, speaking, tool_running, approval_needed, error, offline}` + custom states. `REQUIRED_VISUAL_STATES = {idle, listening, thinking, speaking, error}` — **all four of our states are required for server-side activation**, so any pack that ever validated server-side defines them directly. This justifies keeping `fallbacks` inert (honoring them would only serve packs the server itself rejects — YAGNI).
7. Server-side manifest validation hard-raises (`PersonaVisualManifestError`) to gate *activation*. We do NOT port the validator — extraction is lenient/best-effort; we port only what extraction touches (region-bounds sanity before a PIL crop).

## Architecture — one new pure extractor, dispatched inside the existing zip branch

### Unit 1 — vpack detection + dispatch (in `resolve_local_expression_set`)

The dispatch lives in `resolve_local_expression_set` (not `_candidate_pairs`, whose `(name, bytes)` pair contract doesn't fit an extractor that returns a full resolution): for each input path that `zipfile.is_zipfile()`, sniff the **namelist** (cheap; zero member reads). If the archive contains BOTH `manifest.json` and `metadata/assets.json` → hand the open `ZipFile` to the vpack extractor and merge its `ExpressionSetResolution` into the call's result; else the path flows into `_candidate_pairs`'s generic stem-mapping iteration unchanged. (The UI passes exactly one path, so in practice a vpack call returns the extractor's resolution directly; the merge rule for mixed inputs is first-writer-wins per state, same as the existing tie handling.)

**Budget sharing mechanism (pinned):** the running size total moves to the resolver level — `_candidate_pairs` gains a `start_total: int = 0` parameter (it has exactly one caller, verified, so the signature change is safe), and both it and the vpack extractor report the total they consumed, so vpack reads and generic reads draw on one `MAX_TOTAL_BYTES` budget per call.

**Nested-root leniency (the re-zip papercut):** the server always emits members at the archive **root** (`REQUIRED_MEMBERS` are root-level paths), but a user who extracts a pack and re-compresses the folder gets `MyPack/manifest.json` — a classic macOS papercut that would otherwise fall through to stem mapping and yield a confusing "Nothing imported". Detection therefore also accepts a **single shared top-level directory**: if all members share one root prefix and `<root>/manifest.json` + `<root>/metadata/assets.json` exist, the extractor strips that prefix for all member lookups. Deeper nesting or multiple roots are NOT supported (fall through to stem mapping).

**Why targeted dispatch is load-bearing (not just tidy):** the generic branch slices to the first `MAX_ZIP_MEMBERS=64` members and reads them all. Zip member order is arbitrary — in a pack with >64 members (big packs are legitimate), `manifest.json` could be member #70 and get sliced away; meanwhile the generic reads would waste the size budget on assets we don't need. The vpack path therefore **never enumerate-reads**: it reads ONLY the members it needs (`manifest.json`, `metadata/assets.json`, ≤4 resolved asset members), each size-capped before read, drawing on the same shared `total` budget as the rest of the call. The 64-member cap is irrelevant to vpacks — bounded work regardless of archive size. Detection is by content, so a vpack renamed `.zip` routes correctly and a plain zip named `.vpack` falls back to stem mapping.

### Unit 2 — `_resolve_vpack_expression_set` (pure extractor)

```
_resolve_vpack_expression_set(zf: zipfile.ZipFile, total_budget_state) -> ExpressionSetResolution
```

Per state in `EXPRESSION_STATES` (idle, thinking, speaking, error):
1. `manifest["states"].get(state)` → `animation_id` (missing → skip: "state not in pack").
2. `manifest["animations"].get(animation_id)` → animation (missing → skip: "unknown animation"). Normalize the `asset_ids` shorthand to frames.
3. Pick the frame: `preview_frame` if a valid in-range int, else `frames[0]` (no frames → skip).
4. Resolve `frame["asset_id"]` via the assets index (`source_asset_id` match) → `asset_path`. Skip only when `asset_bytes_status` is **explicitly `"missing"`** — an absent status key is tolerated and the member read is attempted anyway (a failed lookup then skips with its own reason). This mirrors the server's own defensive `.get("asset_bytes_status")` reads and stays robust to format drift.
5. Read the member **through the caps** (size-checked vs `ZipInfo.file_size` before read, shared running total) — **with a per-call cache of distinct member bytes AND the opened PIL image**: the common pack shape is ONE `sprite_sheet` asset + N regions, and a naïve per-state read would count the same sheet 4× against the total budget (a legit 16 MiB sheet × 4 = 64 MiB → false total-cap trip) and decode it 4×. Each distinct asset is read once, decoded once, cropped up to 4×.
6. If the frame has a `region`: bounds-check it against the actual decoded image (ints, non-negative origin, positive size, within the image — bad bounds → skip: "invalid region"), PIL-crop, and **re-encode the crop as PNG bytes** (a crop must produce standalone image bytes). No region → pass the whole asset bytes through **verbatim**.
7. PIL-validate the final bytes (`_valid_image`) → `images[state]`. (Satisfies the orchestrator's callers-pass-validated-bytes contract by construction.)

Non-matching server states (`listening`, `wake_armed`, `tool_running`, `approval_needed`, `offline`, custom) are ignored. `frame_rate` / `duration_ms` / `fallbacks` / `alignment` are inert. **Checksums are NOT verified** — PIL validation of the actual bytes is the safety gate; a checksums mismatch on bytes that decode fine is not a reason to reject a state.

**Leniency:** every step is per-state best-effort with a `skipped`/`notes` reason. A structurally broken manifest or assets.json (unparseable JSON, wrong types) degrades to an empty resolution with a note. The extractor NEVER raises (same contract as `resolve_local_expression_set`).

### Unit 3 — UI (one-line change)

Widen the existing "Import set…" picker filter to accept both extensions: `("Archives", lambda p: p.suffix.lower() in (".zip", ".tldw-persona-vpack"))`. Everything else — `validate_path_simple` at the screen boundary, `_io_dialog_active` gate, `_import_expression_set_from_path`, `_apply_expression_set`, the summary — is unchanged (the resolver returns the same `ExpressionSetResolution` either way).

## Security posture (unchanged + two explicit safe-by-construction notes)

- Identical to P3d-2: in-memory reads only, `MAX_MEMBER_BYTES`/`MAX_TOTAL_BYTES` checked against `ZipInfo.file_size` BEFORE each read (manifest + assets.json members included — both parsed with plain `json.loads` under the member-size cap), PIL-validate everything, `PurePosixPath` normalization where member names are compared.
- **`asset_path` is only ever a zip-member key, never a filesystem path**: a traversal-shaped value (`../../etc/passwd`) simply fails the member lookup → skip. Nothing is extracted to disk.
- **Region cropping decodes an untrusted image**: PIL's default `MAX_IMAGE_PIXELS` decompression-bomb guard (raises past ~2× the threshold) + the per-state try/except→skip covers it — the same posture as the existing thumbnail decodes in P3d-1/P3d-2.

## Data flow

Pick a `.vpack` (or `.zip`) → screen validates the path → `resolve_local_expression_set([path])` → zip branch sniffs namelist → vpack? → `_resolve_vpack_expression_set` (targeted capped cached reads, per-state extraction) → `ExpressionSetResolution` → `_apply_expression_set` (idle staged, three to the table, one render) → summary: `Imported: idle (staged), thinking, speaking, error.` / `Nothing imported (pack has no matching states).`

## Testing

- **A crafted in-test vpack builder** (manifest + pack.json + assets.json + assets; variants with a shared sprite-sheet + regions, per-state standalone assets, and the `asset_ids` shorthand):
  - full four-state extraction (sprite-sheet + regions) with **pixel-checked crops** (crop a known-color region and assert the extracted image's color);
  - `preview_frame` honored over `frames[0]`; invalid `preview_frame` (out of range / non-int) falls back to `frames[0]`;
  - `asset_ids` shorthand works;
  - per-state skips each carry a reason: missing state, unknown animation, no frames, unknown asset, `asset_bytes_status: "missing"`, traversal-shaped `asset_path` (fails lookup), invalid region (out of image bounds), oversize asset;
  - **shared-sheet budget**: a sheet sized so 4 naive reads would trip `MAX_TOTAL_BYTES` but one cached read does not → all four states extract (RED against a non-caching implementation);
  - **manifest beyond member #64**: a pack with >64 members where `manifest.json` is written LAST → still extracts (RED against generic-slicing dispatch);
  - broken manifest JSON → empty resolution + note, no raise; plain P3d-2 zip still stem-maps (no regression); a vpack renamed `.zip` auto-detects; a plain zip named `.vpack` falls back;
  - **nested-root pack** (`MyPack/manifest.json`, single shared prefix) extracts; a two-root or doubly-nested archive falls through to stem mapping;
  - an assets.json entry with **no `asset_bytes_status` key** still resolves (only explicit `"missing"` skips);
  - **mixed-input budget**: a vpack path plus a standalone image in one call share the `MAX_TOTAL_BYTES` budget.
- **UI/end-to-end**: the picker filter accepts `.tldw-persona-vpack`; `_import_expression_set_from_path` with a crafted vpack lands the three states in the table + stages idle (real editor + file-backed DB, the P3d-2 harness).

## Global constraints (for the plan)

- **NO migration** (dev schema is v24 via unrelated work; P3d-3 touches no schema). All new logic in the PURE `Character_Chat/expression_set_io.py` (zipfile + PIL + json only; NO Textual/DB imports).
- The vpack path **never enumerate-reads** — targeted members only, size-capped before read, shared `total` budget, per-call byte+PIL cache for distinct assets.
- Frame choice: `preview_frame` if valid in-range int, else `frames[0]`. Region crops re-encode as PNG; whole assets pass through verbatim. Everything PIL-validated before it enters `images`.
- Lenient/best-effort per state; checksums not verified; `fallbacks` inert (all four of our states are in `REQUIRED_VISUAL_STATES`); the extractor never raises.
- UI change is ONLY the picker filter; the P3d-2 flow (path validation, gates, orchestrator, summary) is untouched.
- Established process constraints: file-backed `CharactersRAGDB(tmp_path/...)` for DB tests; `Tests/UI` asyncio_mode rules; concurrent-session hazard on `personas_screen.py` (localize; expect rebase); implementers stage only task files; no background/broad sweeps; never pkill.

## Out of scope — later / eventual

- Animation / multi-frame playback (terminals do static frame-swap only — P3d-1's model).
- Checksum verification; manifest V2 renderer envelopes (`renderer_assets`, `fallback_preview_asset_id` — design-only server-side; V1 is what ships).
- Server/MCP pack *fetch* (this imports a local file); writing/exporting vpacks.
- Custom / mood / tool / voice states.
