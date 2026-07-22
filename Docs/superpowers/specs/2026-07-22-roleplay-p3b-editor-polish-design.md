# Roleplay P3b — Editor polish (character + persona editors)

**Program:** Personas → "Roleplay" workbench redesign. Sub-project **P3b** — second of the **P3 (Characters/Personas flow polish)** phase, and the first cycle of the **"character presence"** theme (P3b editor polish → P3c persistent chat avatar → P3d reaction/expression images). P3a (library pagination/sort/tag-filter) is merged. See the northstar `Docs/superpowers/specs/2026-07-13-roleplay-p0-reframe-northstar-design.md`.

**Date:** 2026-07-22
**Status:** Design (awaiting user review before writing-plans)
**Schema:** ChaChaNotes v22 — **P3b adds NO migration** (reads/writes existing `character_cards` columns; personas are file-backed JSON).

---

## Goal

Polish the character and persona editors so authoring is smoother and less error-prone: show the character's avatar as an actual thumbnail (not a text label), edit alternate greetings as a real ordered list (not a fragile newline blob), validate inline and per-field as you type, keep you in the editor after saving (with dirty-tracking correctly re-armed), and surface the persona fields the schema already stores. Also lays the avatar-render groundwork that P3c (persistent chat avatar) reuses.

## Non-goals

- **No persona avatars** (user decision: characters-only for now). Persona profiles gain no image field.
- **No chat/Console changes** — the persistent chat avatar is P3c; reaction/expression images are P3d.
- **No schema migration** (v22 stays); no new DB tables/columns.
- **No new image-rendering engine** — P3b reuses the existing `ConsoleImageRenderCache` + `[chat.images]` mode resolution verbatim.

## Success criteria

- The character editor shows a small rendered thumbnail of the current avatar (honoring `[chat.images]` render mode + terminal capability), updates on upload, and offers a Remove action; a text fallback appears when there's no image or rendering is unavailable.
- Alternate greetings are edited as an add/update/delete/reorderable list; multi-line greetings survive round-trips (the current newline-corruption hazard is gone).
- Both editors validate live and per-field (the offending field is marked, plus the existing footer summary), before Save — not only at Save.
- After a successful Save the editor stays open, the unsaved badge/title clear, and a subsequent edit re-flags dirty (re-arm works).
- The persona editor exposes `personality_traits` and `enabled` (already stored by the schema/service), grouped sensibly.

---

## Background — current state (verified via source scout; file:line approximate, re-verify at plan time)

**Character editor** — `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py` (`PersonasCharacterEditorWidget(Container)`, id `ccp-character-editor-view`):
- compose (~:149–216): `VerticalScroll#personas-char-editor-body` with primary fields name/first-message/description/personality/system-prompt, an **Advanced** collapsible `#personas-char-editor-advanced` (toggle `_set_advanced_open` ~:408) holding scenario/post-history/creator-notes/creator/version/tags/alt-greetings. Then the **avatar row** `Horizontal#personas-char-editor-avatar-row` (~:200, CSS `height:1`), a validation `Static#personas-char-editor-validation` (~:207), and the Save/Cancel toolbar.
- **Avatar today:** `_set_avatar_status_from_record` (~:415) renders literal text `"Avatar: embedded"`/`"Avatar: none"`; `set_avatar_image(bytes)` (~:279) stages bytes at `self._character_data["image"]` then `_mark_dirty()`. Upload button posts `CharacterImageUploadRequested()`.
- **Alt-greetings today:** a single `TextArea#personas-char-editor-alt-greetings` (~:199), newline-joined. Load (`_populate_form` ~:266) builds `_loaded_greetings: list[str]` + `_loaded_greetings_text = "\n".join(...)`. Save (`get_character_data` ~:338) applies a **fidelity rule**: if the TextArea text is byte-identical to `_loaded_greetings_text`, return `list(self._loaded_greetings)` verbatim (protects multi-line greetings); only if edited, re-split on `splitlines()`. Stored on the record as `alternate_greetings: list[str]`.
- **Dirty machinery:** `_loading`, `_loaded_snapshot: tuple | None`, `_dirty_posted: bool`. `_form_snapshot()` (~:391) = raw values of all fields. `load_character` (~:228) populates then **re-arms** (`_loaded_snapshot = _form_snapshot()`, `_dirty_posted = False` at ~:240). `_field_changed` (`@on(Input.Changed)`/`@on(TextArea.Changed)`, ~:433) → `_mark_dirty` (~:425) posts `EditorContentChanged()` **once** (guarded by `_dirty_posted`).
- **Validation today:** the footer `Static` surface **already exists**. `_save_pressed` (~:464) does an inline name-required check → writes to the Static and returns without posting when blank. `show_validation(errors)` (~:294) is the shared entry the screen calls. **No per-field affordance.** Posts `CharacterSaveRequested(get_character_data())` / `CharacterEditorCancelled()`.

**Persona editor** — `tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py` (`PersonaProfileEditorWidget(Container)`, id `ccp-persona-editor-view`):
- compose (~:61–78): a **flat 3-field form** — name/description/system-prompt — plus the footer `Static#personas-editor-validation` and Save/Cancel toolbar. **No** `personality_traits`/`enabled` fields, no Advanced split. `_persona_id`/`_version` tracked internally (~:51) for optimistic locking.
- Same dirty machinery (own copy, ~:57–153); `validate()` (~:155) returns `("name: required",)` when blank; `_save_pressed` (~:171) validates then posts `PersonaProfileSaveRequested(self.collect())`; `collect()` (~:111) returns name/description/system_prompt (+ id/version). `PersonaProfileEditCancelled()` on cancel.
- **Schema gap:** `PersonaProfileCreate` (`tldw_chatbook/tldw_api/character_persona_schemas.py:53`) supports `personality_traits` + `enabled` — the editor doesn't expose them.

**Both editors** post the same `EditorContentChanged` (`personas_pane_messages.py:75`); the dirty/validation state machines are **duplicated per-widget, not shared**.

**Screen seams** — `tldw_chatbook/UI/Screens/personas_screen.py`:
- `_handle_editor_content_changed` (~:3750): sets `state.has_unsaved_changes = True`, `inspector.set_unsaved(True)`, badges the row, updates title.
- **Save finishers (the re-arm site):** `_after_character_save` (~:4804) resets `has_unsaved_changes=False` (~:4830), reloads the card, and **flips to the read-only card** via `_show_center("#ccp-character-card-view")` (~:4838). `_after_profile_save` (~:4913) mirrors it (~:4934, flip at ~:4948). **Neither touches the editor widget's `_dirty_posted`** — the re-arm gap. It's masked today *only because save closes the editor and reopen reloads*.
- `_validate_character` (~:4748) checks name + (if present) `character_book`; `_handle_save_requested` (~:4760) calls `editor.show_validation(errors)` then `_save_character_worker` (~:4780). Persona `_handle_profile_save_requested` (~:4843) (editor pre-validated) → scope-service create/update (optimistic `expected_version`) → `_after_profile_save`.
- Unsaved guard: `_run_guarded`/`_confirm_then_run`/`_confirm_discard_unsaved` (~:5063–5116) push `UnsavedChangesDialog`. Cancel routes through it.
- Avatar upload: `_handle_character_image_upload_requested` (~:3889) → `_avatar_upload_dialog_worker` (`EnhancedFileOpen`, suffixes `PERSONAS_AVATAR_IMAGE_SUFFIXES` ~:198) → `_stage_character_avatar_from_path` (~:3841, `_read_avatar_image_bytes` enforces suffix + `PERSONAS_AVATAR_MAX_BYTES` = 5 MB ~:200) → `editor.set_avatar_image(bytes)`.
- `_show_center` (~:5006) exclusive-toggles `_CENTER_VIEW_IDS` (~:213). Both editors + both cards are always mounted.

**Render primitive (reuse verbatim)** — `tldw_chatbook/Chat/console_image_view.py` (pure, Textual-free): `ConsoleImageRenderCache.prepare(id, bytes) -> bool` (PIL decode + LANCZOS downscale ≤1024px, LRU-16, **must run off the event loop**), `get_pil(id)`, `get_pixels(id)` (rich-pixels thumbnail). `resolve_default_mode(app_config)` reads `[chat.images].default_render_mode` (+ `terminal_overrides`) → `"pixels"`/`"graphics"`. Mount pattern from `Widgets/Console/console_transcript.py` `_image_row_widget` (~:970): graphics → `textual_image.widget.Image(pil)`; pixels/fallback → `Static(Pixels)`; box capped `max_width:80, max_height:40`.

**List-editor pattern to mirror (alt-greetings)** — `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py`: a `DataTable` (`cursor_type="row"`) + edit form + Add/Update/Delete/Move-up/Move-down button row; `selected_entry_id` from the cursor, tracked on both `DataTable.RowSelected` and `DataTable.RowHighlighted`; reorder posts the **full reordered id list**. NOTE: lore entries persist via a DB manager; **alt-greetings persist widget-local** (a `list[str]` on the record), so the greetings editor is entirely in-widget — no screen round-trip, no messages to the screen.

---

## Design decisions (resolved with the user)

1. **Sequence:** P3b editor → P3c chat avatar → P3d reactions (deferred to its own program).
2. **Avatars are characters-only** (personas gain no image).
3. **Save-in-place:** after Save the editor **stays open** (clears unsaved state, re-arms dirty) rather than flipping to the card.
4. **Surface persona `personality_traits` + `enabled`** (already schema/service-supported).
5. All five items: avatar thumbnail preview, alt-greetings list editor, live per-field validation, dirty re-arm + save-in-place, field grouping.

---

## Architecture

### A. Avatar thumbnail preview (character editor)

Reuse the pure render primitive; the editor owns a small render step driven by the screen (which has `app.config` + can run off-thread).

- Replace the avatar row's text-only status with a **thumbnail widget** mounted into `#personas-char-editor-avatar-row` (row height grows from 1 to a small fixed cell box, e.g. `height: 12`, `max_width/max_height` small — a compact editor thumbnail, smaller than the 80×40 chat box). Keep the Upload button; add a **Remove** button that clears `_character_data["image"]`, re-renders the empty state, and marks dirty.
- **Rendering** goes through a screen worker (the widget can't run `asyncio.to_thread` cleanly and needs `app.config`): on avatar change (upload, remove, or editor open with an image), the screen calls `await asyncio.to_thread(cache.prepare, key, bytes)` on a screen-owned `ConsoleImageRenderCache` (keyed by a stable editor key, e.g. `"char-editor-avatar"`), resolves the mode once via `resolve_default_mode(app.config)`, then hands the editor a ready renderable (a `pil` for graphics or a `Pixels` for pixels) to mount — mirroring `_build_console_image_specs`/`_image_row_widget`. A small editor method `set_avatar_thumbnail(renderable | None, mode)` mounts/replaces the widget (graphics `Image` vs `Static(Pixels)`), or shows the text fallback ("No avatar" / "Preview unavailable") when `None`.
- **Never blocks / never crashes:** decode is off-thread; a decode failure (`prepare` returns False) or missing `textual_image` falls back to pixels then to text. The editor session-token guard already used for `set_avatar_image` (`_stage_character_avatar_from_path`) extends to the render step so a late render doesn't paint onto a different character's editor.

### B. Alternate-greetings list editor (character editor)

Replace `TextArea#personas-char-editor-alt-greetings` with a compact list editor **inside the Advanced section**, mirroring the lore skeleton but **widget-local** (no screen messages, no DB):
- A `DataTable#personas-char-editor-greetings-table` (`cursor_type="row"`) showing one row per greeting (a single-line **preview** — first line, truncated), plus an edit `TextArea#personas-char-editor-greeting-edit` for the selected greeting's full (possibly multi-line) text, and a button row **Add / Update / Delete / Move up / Move down** (`#personas-char-editor-greeting-{add,update,delete,move-up,move-down}`).
- Internal state: `_greetings: list[str]` (the source of truth). Add appends a new (empty or editor-text) greeting; Update writes the edit area back to the selected row; Delete removes it; Move up/down swap with the neighbor (clamped). Every mutation re-renders the table (preview text via a `_greeting_preview(text)` helper) and marks dirty.
- **Fidelity by construction:** because each greeting is a discrete `str` (edited in its own TextArea, never blob-joined), the multi-line-corruption hazard disappears. `get_character_data` returns `list(self._greetings)` directly (dropping the old `_loaded_greetings_text` diff rule). `load_character` seeds `_greetings` from `record["alternate_greetings"]` (coercing non-str/None safely). Selection tracking mirrors lore (`RowSelected` + `RowHighlighted`).

### C. Live per-field validation (both editors)

Keep the footer `Static` summary; add **live, per-field** validation:
- A `validate()` method on each editor returns a list of `(field_id, message)` pairs (character extends the current name check; persona extends its `validate()`). Checks: **name required** (both); **oversized avatar** (>`PERSONAS_AVATAR_MAX_BYTES`) and **empty/whitespace-only greeting** (character); (optionally) empty system prompt as a soft warning. The rules live in the widget (pure, testable) — the screen's `_validate_character` `character_book` check stays screen-side and merges into the same surface.
- **When it runs:** on field change (debounced, reusing the editor's existing `@on(Input.Changed)`/`@on(TextArea.Changed)` path — the same handler that marks dirty also schedules a debounced `_run_validation()`), and always at Save. Results: mark each offending field row with a `.is-invalid` class (CSS border/color) and render the aggregated messages into the footer `Static` (existing `show_validation`, extended to also toggle the per-field classes). Clearing a field's error removes its class live.
- Save is blocked (no `*SaveRequested` posted) while any **error**-level finding stands; warnings don't block.

### D. Dirty re-arm + save-in-place (both editors + screen finishers)

- Add `mark_saved(saved_record)` to **both** editor widgets: update the internal version/id from `saved_record` (character: version on `_character_data`; persona: `_version`/`_persona_id`), re-baseline dirty (`_loaded_snapshot = _form_snapshot()`, `_dirty_posted = False`, and character also refreshes the greetings baseline), and clear the footer validation. This re-arms **without** re-rendering the form (the form already shows the saved state) and carries the new optimistic-lock version for the next save.
- **Save-in-place** in the finishers: `_after_character_save` / `_after_profile_save` stop flipping to the card (`_show_center(...card...)` removed from the save path). Instead they keep `_edit_mode` in its edit state, keep the editor center visible, clear `has_unsaved_changes`/`inspector.set_unsaved(False)`/`_set_active_row_unsaved(False)`, obtain the freshly-persisted record (by id) and call `editor.mark_saved(record)`, and notify "Saved". (The read-only card is still reached by selecting the entity in the list, which exits edit mode via the existing guard — unchanged.)
- **Create → edit transition:** after saving a *new* entity (create), the persisted record gets an id (character) / id+version (persona). `mark_saved(record)` sets that id and the finisher transitions `_edit_mode` from `create` to `edit`, so the editor is now editing the just-created entity in place (subsequent saves are updates with the correct `expected_version`). The library list is refreshed and the new row is marked active/selected, without leaving the editor.
- The `UnsavedChangesDialog` guard is unaffected: after `mark_saved`, `has_unsaved_changes` is False so navigating away is clean; a fresh edit re-posts `EditorContentChanged` (because `_dirty_posted` was reset) and re-arms the guard.

### E. Field grouping / disclosure (both editors)

- **Character:** re-tune the primary-vs-Advanced split (keep name/first-message/description/personality/system-prompt primary; the greetings list editor lives in Advanced). Minor reordering only — no field removals.
- **Persona:** add `personality_traits` (a `TextArea` or the same greetings-style list if traits are a list — confirm the schema shape at plan time; `PersonaProfileCreate.personality_traits`) and `enabled` (a `Switch`/checkbox) to `persona_profile_editor_widget.py`, wired through `collect()`/`load_persona`/`_form_snapshot`/`validate`. Group them under the existing form (optionally an Advanced split if the field count warrants). `PersonaProfileCreate`/`PersonaProfileUpdate` and the scope-service round-trip already carry these fields — this is UI surfacing, not schema work.

### Error handling / edge cases

- **No avatar / decode failure / missing `textual_image`:** thumbnail falls back pixels→text; never raises; off-thread decode.
- **Late avatar render onto a switched editor:** session-token guard drops a stale render (mirrors `_stage_character_avatar_from_path`).
- **Greetings:** empty list → empty table + empty-state hint; a greeting that is only whitespace → validation warning (or dropped on save per the current strip behavior — keep the strip); multi-line greeting preview shows the first line + "…"; deleting the last row clears the edit area.
- **Save-in-place version drift:** `mark_saved` must set the new version or the *next* save raises a `ConflictError`; the finisher fetches the persisted record to get it. If the fetch fails, fall back to today's flip-to-card (so we never leave the editor with a stale version) and notify.
- **Validation debounce vs save:** a Save while a debounced validation is pending still runs the synchronous `validate()` at Save (authoritative), so the debounce can't let an invalid save through.
- **Persona `enabled`/`personality_traits` absent on older profiles:** default `enabled=True`, `personality_traits=""`/`[]` on load (coerce safely).

### Testing strategy

- **Character editor widget tests:** avatar thumbnail mounts for image bytes (graphics + pixels modes) and shows text fallback for none/decode-fail; Remove clears image + marks dirty; greetings Add/Update/Delete/Move-up/Move-down mutate `_greetings` and re-render; **multi-line greeting round-trips byte-identical** (the fidelity guarantee); `get_character_data` returns the list; `mark_saved` re-arms (`_dirty_posted` False, snapshot rebaselined, version updated) and a subsequent change re-posts `EditorContentChanged`; per-field validation marks the name row invalid on blank + oversized avatar + whitespace greeting, and clears live.
- **Persona editor widget tests:** `personality_traits` + `enabled` load/collect/round-trip; dirty + `mark_saved` re-arm; validation marks blank name; `enabled` toggle marks dirty.
- **Screen integration tests (real screen + real DB):** save-in-place keeps the editor open + clears the row/title unsaved state + re-arms so a second edit re-flags dirty (drives the re-arm fix end-to-end); a validation error blocks save and surfaces inline; avatar upload → thumbnail renders; persona save-in-place with the new fields persists + re-arms.

---

## Task decomposition (for writing-plans → subagent-driven-development)

1. **Dirty re-arm + save-in-place** — `mark_saved(record)` on both widgets; `_after_character_save`/`_after_profile_save` stay-in-place + call it; fetch-persisted-record-for-version (fallback to flip-to-card on fetch error). Widget + screen integration tests. *(Foundational — the other items build on the editor staying open.)*
2. **Avatar thumbnail preview** — screen render worker (reuse `ConsoleImageRenderCache`/`resolve_default_mode`), editor `set_avatar_thumbnail`, grow the row, Remove button, fallbacks + session-token guard. Widget + screen tests.
3. **Alt-greetings list editor** — replace the TextArea with the DataTable+edit+buttons list editor, `_greetings` state, fidelity-preserving load/save. Widget tests (incl. multi-line round-trip).
4. **Live per-field validation** — `validate()` rules + debounced runner + `.is-invalid` per-field marking + footer surface, both editors; Save-blocks-on-error. Widget tests.
5. **Persona fields + field grouping** — surface `personality_traits` + `enabled` in the persona editor (collect/load/snapshot/validate); re-tune character Advanced grouping. Widget + persona save-in-place integration test.

Five focused tasks → one PR.

---

## Global constraints

- **NO schema migration** (ChaChaNotes stays v22); no new DB tables/columns. Persona fields already exist in the schema/service.
- **Reuse the render primitive verbatim** — `ConsoleImageRenderCache` + `resolve_default_mode` from `Chat/console_image_view.py`; do not build a new decoder. Avatar decode runs **off-thread** (`asyncio.to_thread`), driven from the screen (which has `app.config`).
- **Preserve the alt-greetings no-corruption guarantee** — multi-line greetings must round-trip byte-identical; the list model replaces the newline-blob + fidelity-diff, it must not regress it.
- **Save-in-place must carry the new optimistic-lock version** — the next save after a save-in-place must not raise `ConflictError`; fall back to flip-to-card + notify if the persisted record can't be re-read.
- **Apply dirty-re-arm + validation to BOTH editors** — their state machines are duplicated; keep them consistent (extract a shared mixin only if it's clearly simpler, otherwise mirror the change in both).
- **Characters-only avatars** — no persona image field.
- **All screen DB/decode I/O off-thread**, wrapped so errors notify (never crash the worker).
- **Branch off the LATEST `origin/dev`** (`bf21cfb4b` at spec time — re-verify at plan/merge time; dev is moving fast with concurrent RP-UX work). **Concurrent-session hazard:** other sessions are actively editing `personas_screen.py` (RP-UX-review tasks 425–445, e.g. Start-Chat #754) — expect a non-trivial rebase before merge; keep P3b changes localized to the editor widgets + the save/validation/avatar seams.
- **Implementers stage ONLY their task's files** — never `git add -A`, never stage `.superpowers/`.
- **No background/broad test sweeps; never broad-`pkill` pytest** — scope to this worktree.
- **Test env:** `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest … -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread` (venv in the MAIN checkout; imports resolve to worktree source; UI tests are slow — generous timeouts).

## Deferred / follow-ups

- **P3c — persistent character avatar in the Console chat** (the next cycle): a docked "Character" panel in the Console left rail rendering the active character's image, resolving `character_id` via `conversations.character_id` (note the Start-Chat handoff is being changed by task-427/#754 — reconcile at P3c time). Reuses P3b's avatar-render step.
- **P3d — reaction/expression image system** (own future program): net-new image-set data model + authoring + trigger + swap, ported from the tldw_server webui/browser extension (needs the external `../tldw_server` repo).
- Possible shared editor base/mixin to de-duplicate the dirty/validation machinery (if not done inline in P3b).
