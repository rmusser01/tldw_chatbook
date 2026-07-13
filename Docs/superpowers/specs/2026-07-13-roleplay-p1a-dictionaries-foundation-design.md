# Roleplay P1a — Dictionaries mode foundation (design)

**Status:** Design approved (brainstorm), pending spec review.
**Program:** Personas→Roleplay redesign. P0 (reframe + north-star) MERGED (#619). **P1 = Dictionaries mode**, decomposed into **P1a → P1b → P1c → P1d**, each its own spec → plan → PR. This is **P1a**, the foundation the rest build on.
**Research input:** `Docs/superpowers/research/2026-07-13-server-dictionaries-port.md` (note: several digest claims were corrected by a ground-truth backend audit — see Backend reality below).
**North-star contract:** the P0 spec's binding **List → Detail → Try-it** pattern (`Docs/superpowers/specs/2026-07-13-roleplay-p0-reframe-northstar-design.md`, Part A).
**Worktree/branch:** `.claude/worktrees/personas-redesign`, `claude/roleplay-p1-dictionaries` off dev `dc5fbfd6` (includes P0).

## Problem

The Roleplay strip's **Dictionaries** chip is a dead "· soon" placeholder over a fully-built but unused backend (`chat_dictionary_scope_service` + local/server services). Chat Dictionaries (find/replace substitution rules injected into chats) have **no management UI** — the only existing dictionary widgets (`ccp_dictionary_editor_widget.py`, `ccp_dictionary_handler.py`, `chat_events_dictionaries.py`) are **broken/dead code** that call functions which do not exist. P1a makes Dictionaries a real, working mode: author dictionaries and their entries, and verify substitution behavior — instantiating the north-star three-pane pattern for the first non-Characters/Personas mode.

## Backend reality (ground-truth audit — overrides the research digest)

The design is built on what the code **actually** does today, not the digest's assumptions:

- **Data layer:** `app.chat_dictionary_scope_service` (`ChatDictionaryScopeService`), all methods `async`, all take `mode="local"` (P1a is local-only). Relevant methods: `list_dictionaries`, `get_dictionary(id)`, `create_dictionary(req)`, `update_dictionary(id, req, expected_version=…)`, `delete_dictionary(id)` (soft-delete only), `add_entry(dict_id, req)`, `list_entries(dict_id)`, `update_entry(entry_id, req)`, `delete_entry(entry_id)`, `reorder_entries(dict_id, req)`, `process_text(req)`. Instance already wired at `app.py` (`self.chat_dictionary_scope_service`).
- **Dictionary record** (table `chat_dictionaries`, schema v17, soft-delete + `version` optimistic-lock): `id`, `name` (**UNIQUE**), `description`, `content`, `entries_json`, `strategy` (default `sorted_evenly`), `max_tokens` (default 1000), `enabled`. `list_dictionaries`/`get_dictionary` normalize to dicts exposing `id`, `name`, `enabled`/`is_active`, and an `entries` list.
- **Entry model** (per-entry, stored only inside `entries_json` — no per-entry table locally). Fields that **actually persist**: `key` (raw pattern; `/pat/flags` marks regex), `content` (replacement), `is_regex` (bool), `probability` (int 0–100), `group` (str|None), `max_replacements` (int), `timed_effects` (dict). Fields the digest lists that **do NOT exist locally**: per-entry `enabled`, `case_sensitive`, `priority`, `type` literal, `usage_count` → **deferred to P1c** (they need schema/engine work).
- **API naming ≠ storage naming.** The service request/response layer speaks `pattern`, `replacement`, `probability` (**0–1 float**), `type` (`"literal"|"regex"`), `group`, `max_replacements` — the local adapter maps these to the stored `key`/`content`/int-0–100 shape. **The UI speaks the API names only.** The form displays probability as 0–100% (convert at the seam). Regex `pattern`s come back slash-delimited (`/pat/flags`) and round-trip safely (`_entry_from_payload` wraps only when slashes are missing).
- **Entry identity is positional.** `entry_id = "local:chat_dictionary_entry:<dict_id>:<index>"`. `add`/`update`/`delete`/`reorder` each **rewrite the whole `entries_json`** and self-fetch the record `version` for the optimistic guard. Indices shift on insert/delete. **`reorder_entries`** takes `{entry_ids: [...]}` and computes `selected + remainder` (moves the listed entries to the front) — it is **not** "here is the full new order."
- **`process_text` (Try-it backend).** Local returns exactly `{"text", "processed_text", "dictionary_id", "source"}` — **no** per-entry fire data, replacement counts, or budget stats (those are **P1b**). Request fields used: `text`, `dictionary_id` (loads that dict by id from the DB — so it previews **saved** state), `token_budget` (default 5000). Raises `ValueError` on a missing id.

## Goal / Acceptance

- **AC1 — mode is live.** Selecting the Dictionaries chip shows a working three-pane workbench (List · Detail · Try-it), not the placeholder. The chip is no longer marked "· soon".
- **AC2 — List.** The rail lists local dictionaries as **height-2 rows** (name line + dim meta line `"{n} entries · on|off"`), with **+ New**, **Duplicate**, **Delete**, enable/disable toggling (the meta line *displays* state; **`space` on the highlighted row toggles it** — rows are non-interactive `ListItem(Static)`, so no per-row Switch — plus the Settings `enabled` field), client-side **name search**, and a **"Create your first dictionary"** empty state.
- **AC3 — Detail.** Selecting a dictionary shows a `TabbedContent` with **Entries** (a `DataTable` of `pattern · replacement · type · prob% · group` + an inline edit form: add / update / delete / move-up / move-down) and **Settings** (name, description, strategy, max_tokens, enabled). Editing uses today's real entry fields only, in the API naming.
- **AC4 — Try-it.** The right pane is a substitution preview: a sample-text `TextArea` + **Run** (Button, and Ctrl+Enter while focused) → a **highlighted word-diff** (original with removed spans struck/dim, processed with added spans highlighted) backed by `process_text`, with a "No differences" empty state. Run is disabled until a saved dictionary is selected.
- **AC5 — correctness under the backend model.** Every entry mutation is a discrete, immediately-persisted service call that **reloads the entry list** afterward; row→entry-id mapping is never cached across a mutation. Move-up/down is expressed by passing the full current id list in the desired order to `reorder_entries`.
- **AC6 — no collision.** Only Personas-owned files change (`personas_screen.py`, `personas_library_pane.py`, new `Persona_Widgets/*`, tests). Route id `personas`, `screen_registry.py`, `shell_destinations.py`, `route_inventory.py` are **untouched**. No backend/schema changes.

## Architecture

**Approach: focused widgets + thin screen orchestration** — two new self-contained widgets the screen loads data into and toggles by mode, mirroring the existing `PersonasCharacterCardWidget` / `PersonasPreviewPane` pattern. *Rejected:* a monolithic `DictionaryWorkbench` (fights the shared rail structure); reviving the dead CCP widgets (broken imports, key→value-only entry model — mined only for layout reference).

**Files**

- **New** `Widgets/Persona_Widgets/personas_dictionary_detail.py` — `PersonasDictionaryDetailWidget(Vertical)`. `TabbedContent` with `TabPane` **Entries** and **Settings**. Entries: a `DataTable` (columns key · replacement · type · prob · group) + an inline form (key `Input`, replacement `TextArea`, regex `Switch`, probability `Input`, group `Input`, max-replacements `Input`) with Add/Update/Delete/Move-up/Move-down buttons. Settings: name/description/strategy/max_tokens/enabled + Save. Emits intent messages (`DictionaryEntryAddRequested`, `DictionaryEntryUpdateRequested`, `DictionaryEntryDeleteRequested`, `DictionaryEntriesReorderRequested`, `DictionarySettingsSaveRequested`); the **screen** performs all persistence and re-feeds the widget. The widget holds no DB handle.
- **New** `Widgets/Persona_Widgets/personas_dictionary_tryit.py` — `PersonasDictionaryTryItWidget(Vertical)`. Sample `TextArea`, Run `Button` (+ `Ctrl+Enter` binding scoped to the widget), a results area rendering the word-diff, and status/empty states. Emits `DictionaryTryItRunRequested(text)`; the screen calls `process_text` and calls back `render_diff(original, processed)`.
- **Modify** `personas_screen.py` — `_apply_mode("dictionaries")` branch (populate rail, show detail, show Try-it, leave `self.preview` idle — no gateway start); register the detail widget id in `_CENTER_VIEW_IDS`; compose both new widgets into `personas-work-area` (**pane swap is explicit:** in dictionaries mode `PersonasPreviewPane.display = False` and Try-it shows; every other mode the reverse); `_select_dictionary` off the existing `PersonaEntitySelected(entity_kind="dictionary")`; handlers for New/Duplicate/Delete/space-toggle and the widgets' intent messages; dictionaries search render; drop `"dictionaries"` from `_COMING_SOON_MODES` and `_MODE_PLACEHOLDER_BODY`; change the compose placeholder default (`:439`) from `"dictionaries"` to `"lore"`.
- **Toolbar/actions:** **+ New** reuses the existing `PersonaActionRequested(action="create")` routed by mode. **Duplicate** and **Delete** are mode-gated affordances (visible/active only in dictionaries mode), with the `PersonaAction` Literal extended (`"duplicate"`, reusing `"delete"`-adjacent flows where they exist). **Delete** reuses the screen's existing `ConfirmationDialog` delete-confirm worker idiom (including its refuse-reentry guard).
- **Inspector:** selecting a dictionary calls the existing `PersonasInspectorPane.show_selection(name=…, kind="Dictionary", authority="Local")`; Console actions stay blocked with an honest reason via the existing `_console_action_block_reason` seam ("Attach arrives with Attachments" — P1d).
- **Dirty guard:** edits in the Settings tab mark the screen dirty (`- unsaved` title suffix) and reuse the existing `_confirm_then_run` / `_confirm_discard_unsaved` guard, so switching rows/modes with unsaved Settings prompts instead of silently discarding. Entry-form edits do not need the guard (each entry op persists immediately).
- **Modify** `Widgets/Persona_Widgets/personas_library_pane.py` — add optional `meta: str | None = None` to `LibraryRow`; render a **height-2** row (name + dim meta) when `meta` is set, keeping height-1 for name-only modes. Requires the ListItem to allow `height: auto`/2 for meta rows (the current CSS pins `height: 1`).

## Verified integration facts (implementer: don't re-derive these)

- The rail's **Import** button is already mode-gated: `PersonasLibraryPane.set_mode` shows it only in characters mode — no P1a work needed there.
- **`space` is unbound** across the personas widgets and screen (and not a ListView default binding) — free for the enable-toggle.
- `_handle_entity_selected` currently ignores `entity_kind="dictionary"` with an explicit "wired in follow-up tasks" comment — the new branch is a clean `elif`.
- `update_dictionary` accepts `name/description/content/strategy/max_tokens/enabled/entries` (plus `is_active`/`default_token_budget` aliases); `expected_version` is a keyword passed through the scope service's `**kwargs`.
- **`list_dictionaries` defaults to `include_inactive=False`** — a disabled dictionary silently vanishes from the default listing (a toggled-off row would disappear). The UI must always pass `include_inactive=True`. The response key is `"dictionaries"`, not `"items"`.
- The screen already binds `ctrl+enter` → `personas_attach`; Try-it's widget-scoped `ctrl+enter` binding shadows it only while focus is inside Try-it (intended precedence).

## Data flow

Enter mode → `list_dictionaries(mode="local")` → build `LibraryRow(item_id=id, kind="dictionary", name=…, meta="{len(entries)} entries · on|off")` → `set_mode("dictionaries")` + `update_rows`. Select → `PersonaEntitySelected` → `_select_dictionary(id)` → `get_dictionary(id)` → feed Detail (settings + entries). Entry action → widget message → screen calls the matching scope method → **reload** via `list_entries`/`get_dictionary` → re-feed Detail + refresh the row's meta. Enable toggle → `update_dictionary(id, {"enabled": …})` → refresh row meta. Try-it → sample text → `process_text({"text", "dictionary_id": id, "token_budget": dict.max_tokens})` → diff `text` vs `processed_text` → render.

All backend calls run through the screen's existing guarded async pattern (`_run_guarded` / workers), never blocking the UI thread.

## Editor model (dictated by the backend)

- **Immediate persistence, reload after mutation.** Each entry op is one service call; on success the screen reloads the entry list so indices/ids stay valid. No dirty-entry buffer, no multi-edit-then-save-all. Settings is the one exception: its fields save together via one `update_dictionary` on the Save button.
- **Reorder.** Move-up/down recomputes the desired full order of current entry ids and passes them as `{entry_ids: [full order]}` (selected = all → remainder empty). A test pins this against the "move-to-front" semantics.
- **Naming.** New creates a disambiguated `"Untitled dictionary"` / `"Untitled dictionary 2"`, immediately selected with Settings focused for rename. Duplicate copies via `get_dictionary` + **one** `create_dictionary` call (the payload accepts `entries` inline; response entries round-trip through `_entry_from_payload` unchanged) with `"{name} (copy)"` / `"… (copy 2)"`. Both probe existing names (from the loaded list) to avoid the UNIQUE `ConflictError`. **Strategy caveat:** `create_dictionary` ignores `strategy` (column default `sorted_evenly`), so Duplicate follows with `update_dictionary(new_id, {"strategy": source.strategy})` when the source's differs. `create_dictionary` returns the full new record (id included) — New/Duplicate select it directly.

## Try-it (substitution preview)

Right pane. Input: multiline sample `TextArea`. Run (Button + `Ctrl+Enter`) is enabled only when a **saved** dictionary is selected. On run, the screen calls `process_text` with the selected id and the dict's `max_tokens` as `token_budget`, then renders a **word-level diff** of `text` → `processed_text`: unchanged spans plain, removed spans struck/dim, added spans highlighted (colors from the app stylesheet, structure-only CSS in the widget). Empty states: "No differences" when the texts match; a hint when no dictionary is selected. Try-it previews the dictionary regardless of its enabled flag (id is passed explicitly) — correct for a test surface. A bad regex in an entry already no-ops at the engine, so it simply shows no change, never a hard error.

## Error handling

- `ValueError` (bad id) and `ConflictError` (UNIQUE name clash on New/Duplicate; concurrent-writer version mismatch) surface as a **non-blocking status line** in the Detail/status area, not a crash.
- `process_text` `ValueError` (missing/deleted dict) → Try-it status "couldn't run preview", Run stays available.
- Empty-key guard on entry add/update (a keyless entry can't fire) → inline form validation message, no service call.
- Delete requires the `ConfirmationDialog` confirm (soft-delete underneath); deleting the selected dictionary clears the Detail + Try-it and returns focus to the list.
- Ctrl+Enter-to-run inside the sample `TextArea` is attempted as a widget-scoped binding; if TextArea key handling conflicts, the Run button alone is the accepted fallback (plan verifies).

## Collision constraints (parallel Library-Prompts branch)

- **Do not touch** `screen_registry.py` / `shell_destinations.py` / `route_inventory.py` / route id `personas`.
- `MODE_CHIP_ORDER` still contains `"prompts"` (the parallel branch removes it — not P1a's job). P1a only removes `"dictionaries"` from `_COMING_SOON_MODES`/`_MODE_PLACEHOLDER_BODY`; it does not reorder or drop chips.
- New widgets live under `Widgets/Persona_Widgets/` (Personas-owned); no shared-shell files change.

## Testing

Established `app.run_test()` personas harness (`PersonasTestApp` + `_mounted(pilot)` in `Tests/UI/test_personas_workbench.py`, or a sibling `test_personas_dictionaries.py`), with a **fake `chat_dictionary_scope_service`** on the mock app returning the real shapes (list of dicts; `{text, processed_text, …}` for `process_text`; positional entry ids; **response-named entry fields** — `pattern`/`replacement`/`probability` 0–1 float/`type`). Assertions:

- Switching to dictionaries populates rows whose meta line reads `"{n} entries · on|off"`; the chip is **not** "· soon" and the placeholder is not shown.
- Selecting a dictionary loads Detail (settings + entry table).
- Add / update / delete / move-up / move-down each round-trip through the fake service and the entry table re-renders from the reloaded list (indices correct after delete/reorder).
- `reorder_entries` receives the full ordered id list (pins the move-to-front quirk).
- New and Duplicate produce disambiguated, non-colliding names; Duplicate preserves entries **and a non-default strategy** (pins the create-then-update-strategy flow).
- `space` on a highlighted dictionary row toggles enabled; the change persists and the row meta flips `on`↔`off`.
- Selecting a dictionary drives the inspector (`show_selection` with kind "Dictionary"); Console actions report the honest block reason.
- Unsaved Settings edits trigger the discard-confirm guard on row/mode switch; the title carries the `- unsaved` suffix.
- Delete shows the confirm dialog; cancel keeps the dictionary, confirm removes the row and clears Detail/Try-it.
- Try-it renders a highlighted diff for a substituting dictionary and "No differences" otherwise; Run is disabled with no selection.
- **Geometry pilot:** a dictionary row's dim meta line actually renders (height-2), not clipped by the rail's `height:1` ListItem CSS (QA at the project's standard capture size).
- **P0 test debt:** update `test_mode_chips_are_self_explaining_and_mark_coming_soon` (dictionaries no longer "· soon"; assert on `lore` instead) and `test_coming_soon_mode_shows_inviting_copy` (drive it with `lore`, not `dictionaries`); adjust the `:439` placeholder-default expectation.

Follow the project's CSS-presence pin discipline for any new class names.

## Scope / non-goals

- **P1a does NOT** surface which-entries-fired or substitution stats in Try-it (**P1b** — additive engine diagnostics); add per-entry `enabled`/`case_sensitive`/`priority` or the structured validation panel (**P1c** — schema/engine); build Attachments/Stats/Versions tabs, import/export, starter templates, bulk ops, or a local/server toggle (**P1d**).
- **Forward map:** **P1b** instruments a diagnostics-returning substitution path (chat-time `str` contract untouched) → Try-it shows fired entries + counts + budget. **P1c** adds the richer entry fields + validation taxonomy. **P1d** adds the remaining Detail tabs and portability (import/export JSON+Markdown, attach-to-conversation + reverse used-by index, statistics, version history, local/server toggle).
