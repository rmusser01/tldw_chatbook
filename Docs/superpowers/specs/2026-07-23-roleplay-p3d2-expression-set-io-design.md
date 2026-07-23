# Roleplay P3d-2 — Bulk Import + Export of a Character's Expression Set — Design

**Date:** 2026-07-23
**Program:** Roleplay (Personas) redesign — character-presence theme, P3d cycle 2 (after P3d-1 reactive expression avatar, PR #792).
**Status:** design approved (brainstorming); ready for implementation plan.

## Goal

Make getting a character's expression images **in and out** fast and shareable: assign the whole set (idle + thinking + speaking + error) at once by importing a `.zip`, and export a character's set as a `.zip` for backup or sharing — instead of P3d-1's three one-at-a-time file-dialog slots. (The `.zip` is the portable, shareable unit and is symmetric: an exported set re-imports directly. The importer's core is written to also accept a folder / multiple files, so a future picker upgrade can add loose-file import with no core change.)

## Context

P3d-1 shipped a reactive Console avatar that swaps among **idle / thinking / speaking / error** as a reply generates. Storage: the three reactive states are BLOBs in `character_expression_images` (`set/get/list/delete_character_expression_image`, state-validated in Python to `{thinking, speaking, error}`); **idle reuses `character_cards.image`** (the main avatar). Authoring today = three upload/preview/clear slots in the character editor (`personas_screen.py` / `personas_character_editor_widget.py`), reusing the `_character_editor_generation` render-token discipline.

P3d-2 adds **bulk local import** and **export** on top of that, with **no schema change** (the table and the avatar column already exist). The heavier `.tldw-persona-vpack` server-archive extractor is **deferred to P3d-3**, which will plug into the same `apply_expression_set` orchestrator this spec defines.

## Verified ground truths (from code scout — use verbatim)

1. **idle is staged, the three are immediate.** The main avatar is staged in the editor via `editor.set_avatar_image(bytes)` (`personas_screen.py:~3949`) and read back via `editor.current_avatar_bytes()` (`:~3979`); it persists only on **card save** (a version-bumping `update_character_card`). The three reactive states write immediately to the table via `db.set_character_expression_image(...)` (P3d-1, card-version-independent). So a bulk import must handle idle differently from the three.
2. **File picker** (`Widgets/enhanced_file_picker.py`): `EnhancedFileOpen` returns a **single file** — every real usage (`ccp_character_handler`, `chat_attachment_handler`, `ccp_dictionary_handler`, …) gets one `Path` from `push_screen_wait`; `_should_return(candidate: Path)` gates one file. There is **no reliable multi-file return** and no confirmed directory-select. `EnhancedFileSave` is the save dialog. So the import UI selects a **single `.zip`**; multi-select/folder are not available through the standard picker.
3. **P3d-1 authoring surface** (to reuse): `_apply_expression_upload(character_id, state, image, mime)`, `_clear_expression_slot(...)`, the `char-expression-slot-{state}` slots, `_character_editor_generation` render token, the off-thread thumbnail render, `_avatar_upload_dialog_worker` (the file-dialog pattern), and `EXPRESSION_IMAGE_STATES = ("thinking","speaking","error")` from `Chat/console_expression_state.py`.
4. **No migration** — `character_expression_images` and `character_cards.image` already exist (schema v23).

## Architecture — one orchestrator, two flows (import, export)

### Unit 1 — `apply_expression_set` orchestrator (screen-level)

A screen method that applies a resolved set to the currently-edited, **saved** character:

```
_apply_expression_set(character_id: int, images: dict[str, bytes]) -> ExpressionSetApplyResult
```

- `images` maps a state name (`idle`/`thinking`/`speaking`/`error`) → raw bytes; any subset may be present.
- **idle** → `editor.set_avatar_image(bytes)` — **staged** in the editor exactly like a manual avatar upload (persists on card save).
- **thinking/speaking/error** → `db.set_character_expression_image(character_id, state, bytes, mime)` — immediate (off-thread, as P3d-1 does).
- Each image is **PIL-validated** before applying; an invalid image for one state is skipped, not fatal (best-effort partial).
- After applying, bump `_character_editor_generation` once and re-render the affected slot thumbnails + the avatar thumbnail (P3d-1 render-token discipline).
- Returns `ExpressionSetApplyResult{applied: list[str], skipped: list[(state, reason)]}` so the caller can show a summary.
- This orchestrator is the single sink; P3d-3's `.vpack` extractor will feed it the same `images` dict.

A pure helper does the DB-only part so it is unit-testable without the editor:
`apply_expression_images_to_db(db, character_id, images_without_idle) -> (applied, skipped)`.

### Unit 2 — Local importer (`.zip` → `dict[state, bytes]`; resolver stays general)

A pure function that resolves selected inputs to a validated set:

```
resolve_local_expression_set(paths: list[Path]) -> ExpressionSetResolution
```

- **UI input is a single `.zip`** (ground truth #2 — the standard picker returns one file). The resolver is nonetheless written to accept a general `list[Path]` — a single `.zip`, a directory, or multiple image paths — so it is unit-testable without a zip and ready for a future folder/multi-select picker or P3d-3's needs. The UI passes `[selected_zip_path]`.
- If a path is a `.zip`: read its members **into memory** (never extract attacker paths to disk); enforce the security caps (Unit 5); treat each member as a candidate file by its base name.
- If a path is a directory: enumerate its image files (non-recursive). If a path is an image file: use it directly.
- **Filename → state:** the file's stem, **case-insensitive**, matched exactly against `{idle, thinking, speaking, error}`. Non-matching or non-image files are skipped. If two files match one state, prefer `.png`, else the first alphabetically (and record the tie in the resolution's notes).
- Every candidate is **PIL-validated**; invalid → skipped with a reason.
- Returns `ExpressionSetResolution{images: dict[str, bytes], skipped: list[(name, reason)], notes: list[str]}`.

### Unit 3 — Exporter (character's set → `.zip`)

```
build_expression_set_zip(character_name: str, images: dict[str, bytes]) -> bytes
```

- Input: the character's current set — idle from `editor.current_avatar_bytes()` (the live on-screen avatar), the three from `db.get_character_expression_image(...)`.
- Output: a `.zip` (bytes) containing one file per present state named `{state}.{ext}`, where `ext` is derived by **PIL-detecting the format from the bytes** (`Image.open(BytesIO(b)).format` → `png`/`jpg`/`webp`; default `png`) — the stored `mime` is NOT readable (`get_character_expression_image` returns bytes only), so detection is from the bytes, not the column. So a JPEG-stored state exports as `speaking.jpg`, not a mislabeled `.png`. Plus a minimal self-describing `expression_set.json` (`{"format": "tldw-expression-set/1", "character": <name>, "states": [...]}`).
- **Symmetric with Unit 2:** an exported zip re-imports directly — the importer maps by filename **stem** (extension-agnostic), so `{state}.{ext}` still resolves to `state`; `expression_set.json` is ignored on import (provenance only).
- The screen writes these bytes to the path chosen via `EnhancedFileSave`.

### Unit 4 — UI (character editor)

- An **"Import set…"** and **"Export set…"** button pair next to the P3d-1 expression slots. **Saved-character-only** (the three table writes need a `character_id`) and **characters-only** (personas have no images) — same gate as the P3d-1 slots.
- **Import** opens `EnhancedFileOpen` filtered to `.zip`, gets one path, resolves via Unit 2 (`[zip_path]`), applies via Unit 1, and shows a **summary** notification: `Applied: idle (staged), speaking. Skipped: thinking, error (not found).` idle always reads as "staged — save the character to keep it." (A `.zip` is the bulk unit and is symmetric with Export; loose-folder / multi-select import is a future enhancement gated on picker capability, not this cycle.)
- **Export** reads the current set, builds the zip (Unit 3), and saves via `EnhancedFileSave` (default filename `<character-name>-expressions.zip`).
- Both run in the editor's IO worker group with the same guard pattern as `_avatar_upload_dialog_worker` (never let an exception escape into an `exit_on_error=True` worker).

### Unit 5 — Security (untrusted `.zip`)

Both the import `.zip` and (in P3d-3) the `.vpack` are untrusted archives. On read:
- **Read entries into memory** via `zipfile`; never write attacker-controlled paths to disk.
- Cap **member count** (e.g. ≤ 64), **per-file uncompressed bytes** (e.g. ≤ 16 MB), and **total uncompressed bytes** (zip-bomb guard, e.g. ≤ 64 MB) — checked against `ZipInfo.file_size` *before* reading each member.
- **PIL-validate** every image member; reject non-images.
- On any single bad/oversize member: skip that member (best-effort), do not abort the whole import. On a structurally broken zip (not a zip, encrypted): fail the import cleanly with a notification, never raise into the worker.

## Data flow

**Import:** pick one `.zip` (`EnhancedFileOpen`, `.zip` filter) → `resolve_local_expression_set([zip_path])` (in-memory, validated, security-capped) → `_apply_expression_set(character_id, images)` (idle staged in editor, 3 immediate to DB) → bump generation, re-render thumbnails → summary notification.

**Export:** read idle (`current_avatar_bytes()`) + 3 (`get_character_expression_image`) → `build_expression_set_zip(name, images)` (extensions PIL-detected from bytes) → `EnhancedFileSave` → write bytes.

## Error handling / fail-soft

- Import/export are best-effort and never crash the editor: the resolver and orchestrator return structured skip reasons; the workers wrap everything and degrade to a notification. A missing image, a non-image file, an oversize member, a broken zip, or a DB write failure for one state all produce a summary line, not an exception.
- idle staging never touches the card version until the user saves; the three writes are independent of the card version (P3d-1 invariant preserved).

## Testing

- **`resolve_local_expression_set`** (pure): multi-select images → correct state map; `.zip` → same; directory → same; case-insensitive stems; non-matching/non-image skipped; two-files-one-state tie resolution; corrupt image skipped with reason; **security** — member-count / per-file / total-size caps reject (real crafted zips), not-a-zip fails cleanly.
- **`build_expression_set_zip`** + round-trip: export then `resolve_local_expression_set` yields the same states; `expression_set.json` present and ignored on import.
- **`apply_expression_images_to_db`** (pure DB, file-backed `CharactersRAGDB(tmp_path/...)` — NOT `:memory:`, per the P3d-1 off-thread/threading.local rule): writes the three states, skips idle, best-effort partial.
- **`_apply_expression_set`** (mounted editor): idle staged (`current_avatar_bytes()` reflects it, card unsaved), three in the table, thumbnails refresh, result summary correct.
- **UI**: buttons present saved-character-only / characters-only; import summary; export save flow; never raises.

## Global constraints (for the plan)

- **NO migration** (schema stays v23; reuses `character_expression_images` + `character_cards.image`).
- **idle is staged via `editor.set_avatar_image()`** (persists on card save); the three reactive states write immediately via `db.set_character_expression_image` (card-version-independent — P3d-1 invariant). Export reads idle from `editor.current_avatar_bytes()`.
- Reuse the P3d-1 authoring surface (`_character_editor_generation` render token — bump once per import + re-render affected slots; the `_avatar_upload_dialog_worker` file-dialog + IO-worker-guard pattern; `EXPRESSION_IMAGE_STATES`).
- **Saved-character-only** (needs `character_id`) and **characters-only** (personas mode shows no import/export).
- **Untrusted-zip handling** (Unit 5): in-memory reads, member/per-file/total-size caps, PIL-validate, best-effort skip, never raise into the worker.
- **Best-effort partial**: every state applies independently; one bad state never fails the whole operation; the summary reports skips.
- File-backed `CharactersRAGDB(tmp_path/...)` for DB tests (never `:memory:`). Tests/UI `asyncio_mode=auto` — don't mix dirs OR add explicit `@pytest.mark.asyncio`.
- CONCURRENT-SESSION HAZARD: `personas_screen.py` / `personas_character_editor_widget.py` are heavily edited by other sessions — keep P3d-2 localized to the new orchestrator + importer/exporter + the two buttons; expect a rebase.
- Implementers stage ONLY their task's files (never `git add -A`, never `.superpowers/`). NO background/broad test sweeps; NEVER broad-pkill pytest — scope to the worktree.

## Out of scope — later cycles

- **`.tldw-persona-vpack` extraction → P3d-3** (its own spec→plan→PR), built on this cycle's `_apply_expression_set` orchestrator: parse `manifest.json`, resolve each of our four states → `frames[0]` → asset → crop region → static image → feed the orchestrator; ignore non-matching server states and animation.
- Server/MCP pack **fetch** (this cycle imports a local `.zip`, not a server-pulled pack).
- Animation / sprite playback (terminals do static frame-swap only).
- Checksum *verification* as a hard gate (lenient extract).
- Custom / mood / tool / voice states.
