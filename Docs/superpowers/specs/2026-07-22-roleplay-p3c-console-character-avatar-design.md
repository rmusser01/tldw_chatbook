# Roleplay P3c — Persistent character avatar in the Console chat

**Program:** Personas → "Roleplay" workbench redesign. Sub-project **P3c** — the second cycle of the **"character-presence"** theme (P3b editor polish → **P3c persistent chat avatar** → P3d reaction/expression images). P3a, P3b, and the enabling task-427/#754 (character-on-Console-session) are all merged to dev.

**Date:** 2026-07-22
**Status:** Design (awaiting user review before writing-plans)
**Schema:** ChaChaNotes v22 — **P3c adds NO migration** (reads existing `character_cards.image` + `conversations.character_id`, both already present/written by #754).

---

## Goal

Show the active character's profile image persistently in the native Console chat, so you can see who you're talking to. A new collapsible **"Character"** section in the Console left rail renders the active character's avatar + name, following the session as it changes (Start-Chat, resume, tab switch) and reusing the image renderer P3b/P3a built.

## Non-goals

- **No reaction/expression images** (that's P3d — a net-new image-set model + trigger + swap, ported from the tldw_server webui).
- **No persona avatars** (personas have no image — characters-only, consistent with P3b).
- **No new image decoder / renderer** — reuse `ConsoleImageRenderCache` + `resolve_default_mode` + the shared `fit_image_cell_size` (from #775).
- **No schema migration** — `conversations.character_id` is already written by #754; `character_cards.image` already exists.
- **No change to how the character is set** — #754 owns that; P3c only reads it.

## Success criteria

- Starting a chat with a character (Start-Chat) shows that character's avatar + name in the Console left rail's "Character" section.
- Resuming a persisted character conversation repopulates the panel (the session's `character_id` is restored from `conversations.character_id` by #754).
- A generic (no-character) Console session shows a clean empty state, not a broken/blank panel.
- Switching the active session/tab updates the panel to the new session's character.
- The avatar honors `[chat.images]` render mode (graphics/pixels) with a text fallback, and never blocks the UI (off-thread decode) or crashes the sync tick.
- `[chat.images].show_character_avatar` (default **on**) gates the whole feature.

---

## Background — current state (verified via source scout at dev `dc21e3f04`; file:line approximate, re-verify at plan time)

**Active-character resolution (post-#754).** The live session object is `ConsoleChatSession` (`Chat/console_chat_store.py`), which now carries `character_id: int | None` (~:181) + `character_name: str | None` (~:182). Set at three sites: Start-Chat native handoff `_start_character_console_session` (`chat_screen.py` ~:9724-9730, id/name from `_character_session_identity_from_handoff` ~:472-505); DB-resume `_resume_console_workspace_conversation` (~:4212-4237, reads `conversation["character_id"]` → `session.character_id`, resolves name via `_resolve_resumed_character_name` → `get_character_card_by_id`); screen-state serialize/restore (~:9153/9212). **`None` for a never-character generic session.**
- Active session accessor: `_active_native_console_session()` (`chat_screen.py` ~:3853) → the live `ConsoleChatSession | None`.
- Conversation-id accessor: `_current_console_rail_conversation_id()` (~:3866). **There is NO character-id accessor yet** — P3c adds one.
- **Leftover hole:** `_active_console_dictionary_scope_ids()` (~:6756-6772) still `return self._current_console_rail_conversation_id(), None` — the `character_id` is hardcoded `None` with a "once it lands, source a real character id here" note. #754 did not update it.
- **Split-brain guard:** the native Console must NOT read legacy `app.current_chat_*` reactives — resolve only off the live session.

**Durable persistence (already written by #754).** `persist_session_if_needed` (`console_chat_store.py` ~:1035-1055) writes `character_id`/`character_name`/`assistant_kind="character"` into `create_conversation` (`chat_persistence_service.py` ~:70-72) → `conversations.character_id` (`ChaChaNotes_DB.py` :268, index `idx_conv_char` :280). The resume path already reads it back, so a resumed session's `session.character_id` is repopulated — **P3c reads the session field and needs no separate fallback query.**

**Character image.** `get_character_card_by_id(character_id: int)` (`ChaChaNotes_DB.py` ~:4336, `SELECT *`) returns the row dict with `"image"` as raw `bytes` (schema `image BLOB` :194). No lighter image-only fetch exists — `SELECT *` is the path. `character_id` is INT.

**Renderer (reuse verbatim).** `chat_screen.py` already owns `self._console_image_cache: ConsoleImageRenderCache | None` (~:1873, lazily created in `_ensure_console_image_view` ~:2945, which also sets `_console_image_default_mode = resolve_default_mode(app_config)` ~:2956). `ConsoleImageRenderCache.prepare(id: str, bytes) -> bool` (`console_image_view.py` ~:257) is keyed by an arbitrary string and does PIL decode off the event loop. `fit_image_cell_size(pixel_w, pixel_h, box_cols, box_lines)` (`console_image_view.py`, added in #775) returns explicit aspect-preserving cell dims. The graphics-vs-pixels mount is `console_transcript._image_row_widget` (a copyable pattern) — graphics → `textual_image.widget.Image(pil)` with explicit `fit_image_cell_size` dims, pixels/fallback → `Static(Pixels)`.

**Left rail (mount point).** `compose_content` (~:8285); `#console-left-rail` (~:8399) → `#console-left-rail-body` (VerticalScroll ~:8437) holds four collapsible sections, each a `ConsoleRailSectionHeader` (imported ~:292) + `Vertical(id="console-rail-section-body-<id>")`: Session (~:8441/8447), Model (~:8472/8478), Agent (~:8603/8610), Details (~:8650/8657). A new **Character** section mounts as a 5th pair here. Section open/closed state is threaded through `ConsoleRailState` (toggle list built ~:6222; header/body requery pattern ~:6346-6349) — P3c adds a `character_open` flag.

**Refresh hook + the pattern to mirror.** Central re-sync `_sync_native_console_chat_ui` (~:10102) runs on tab/session switch, Start-Chat, resume, setup changes, and a 0.2s poll. The proven "what's in play" pattern: a cache field (`_active_dictionaries_summary` ~:1911) + a scope-guard field (`_last_console_dictionary_scope_ids` ~:1918) + a change-gated refresh (`_refresh_active_dictionaries_summary_if_scope_changed` ~:6808-6824, early-returns when the scope tuple is unchanged, else runs the DB work `asyncio.to_thread`) called from the sync tick; compose/recompose reads ONLY the cache (zero-DB-on-recompose).

**Config.** `[chat.images]` (`Config.py` ~:2740-2754: `enabled`, `show_attach_button`, `default_render_mode`, `max_size_mb`, `terminal_overrides`). Read via `_chat_images_config(app_config)` (`console_image_view.py`). A `show_character_avatar` toggle joins here.

---

## Design decisions (resolved with the user)

1. **Panel form:** a **collapsible "Character" section** (a 5th rail section, matching Session/Model/Agent/Details).
2. **Config default:** `show_character_avatar` **on** by default.
3. **Characters-only** (no persona image).
4. **Do NOT touch `_active_console_dictionary_scope_ids`** (reversed after review). Verification showed the `None` there is **not an inert hole**: that tuple flows into `refresh_active_dictionaries_summary` → `_run_dictionary_summary_off_thread(service, conversation_id, character_id)` → `Chat_Dictionary_Lib._resolve_active_dictionaries(db, conversation_id, char_data)`, which **uses the character id to include character-embedded dictionaries** in the "what's in play" summary (same for world-books). Sourcing the real id would change what those inspectors show and could **break the "shown=applied" invariant** if the native send does not apply character dicts (character-dicts-on-native-send was deferred pre-#754; whether #754 changed that needs its own analysis). So the `None` is arguably the *correct* current behavior, and this is a separate feature question — **deferred to its own follow-up**, out of P3c scope. P3c adds only its OWN independent `_current_console_rail_character_id()` accessor for the avatar; it does not read or change `_active_console_dictionary_scope_ids`.

---

## Architecture

### A. Resolution accessor + adjacent scope fix (`chat_screen.py`)

- Add `_current_console_rail_character_id() -> int | None` and `_current_console_rail_character_name() -> str | None`, mirroring `_current_console_rail_conversation_id`: read the live session via `_active_native_console_session()` and return `session.character_id` / `session.character_name` (or `None`). This accessor is **used only by the avatar** (§C–E); it is independent of `_active_console_dictionary_scope_ids`.
- **Do NOT modify `_active_console_dictionary_scope_ids`** — see design decision 4: sourcing the real id there changes the dictionary/world-book "what's in play" content (character-embedded dicts) and touches the shown=applied invariant; it's a separate feature question, deferred.

### B. Config helper (`console_image_view.py`)

- Add a pure `resolve_show_character_avatar(app_config) -> bool` mirroring `resolve_default_mode`: `_chat_images_config(app_config).get("show_character_avatar", True)` coerced to bool (default **True**). Document the new key in the `[chat.images]` config section (`Config.py`) alongside the others.

### C. The "Character" rail section (`chat_screen.py` compose + `ConsoleRailState`)

- Add a `character_open: bool` field to `ConsoleRailState` (default open) and thread it through the toggle list (~:6222) + header/body requery (~:6346-6349), exactly like the other four sections.
- In `compose_content`, when `resolve_show_character_avatar(app_config)` is True, compose a 5th `ConsoleRailSectionHeader("Character", section_id="character", …)` + `Vertical(id="console-rail-section-body-character")` inside `#console-left-rail-body` (after Details, or positioned per the rail's ordering). When the config is off, the section is not composed.
- The section body holds: a thumbnail container (`#console-character-avatar`) + a name `Static` (`#console-character-name`). The body reads only the cached rendered avatar + name — **no DB/decode in compose**.
- **Empty state:** when there's no active character (`character_id is None` — generic session), show a compact placeholder ("No character in this chat") and no image.

### D. Avatar cache + scope-guarded off-thread refresh (`chat_screen.py`)

Mirror the dictionary "what's in play" machinery:
- Cache fields: `self._active_character_avatar` = a small **spec (data, NOT a live widget)** — e.g. `{character_id, name, mode, pil | pixels}` — enough to build a fresh mount widget without re-decoding. (A Textual widget can't be re-mounted on recompose, so the cache holds the decoded PIL/Pixels + mode, and each mount builds a new `Image`/`Static`, exactly like the transcript's `ConsoleImageRowSpec` → `_image_row_widget`.) Plus `self._active_character_avatar_name`.
- Scope-guard: `self._last_console_avatar_scope: tuple | None` = `(character_id,)`.
- `_refresh_active_character_avatar_if_scope_changed()`:
  - Compute `scope = (self._current_console_rail_character_id(),)`. If `scope == self._last_console_avatar_scope`, return (zero DB).
  - Set `self._last_console_avatar_scope = scope`. If `character_id is None`: clear the cache (empty state) + update the section, return.
  - Otherwise run off-thread (`asyncio.to_thread`): `card = get_character_card_by_id(character_id)`; `image = card.get("image")`; if bytes, `self._console_image_cache.prepare(f"character:{character_id}", image)`; build the renderable via `resolve_default_mode`/`fit_image_cell_size` sized to a compact rail box (new `CHARACTER_AVATAR_COLS`/`CHARACTER_AVATAR_LINES` constants). Wrap so any failure → text fallback / empty (never raises in the sync tick).
  - After the await, re-check the scope is still current (a session switch mid-decode supersedes), then update the cache + mount into `#console-character-avatar` + set `#console-character-name`.
- Reuse `self._console_image_cache` (do NOT create a second cache).

### E. Wire the refresh into the sync tick (`chat_screen.py`)

- Call `_refresh_active_character_avatar_if_scope_changed()` from `_sync_native_console_chat_ui` (~:10102), alongside `_refresh_active_dictionaries_summary_if_scope_changed()` (~:10129). The scope-guard makes it a cheap no-op when the character is unchanged; it only fetches/decodes when the active character changes.

### Error handling / edge cases

- **No active session / generic session:** `character_id is None` → empty-state placeholder, cache cleared, no DB.
- **Character has no image** (`image` is None/empty): show the name + a text "no avatar" placeholder (not a blank box).
- **Decode failure / missing `textual_image`:** graphics → pixels → text fallback (same as P3b); off-thread; the refresh never raises into the sync tick (wrapped).
- **Session switch mid-decode:** the post-await scope re-check drops a stale render (mirrors the P3b avatar session-token guard and the dictionary summary's scope re-check).
- **Config off:** section not composed; refresh early-returns (guard on `resolve_show_character_avatar`).
- **Deleted character:** `get_character_card_by_id` returns None (`deleted = 0` filter) → empty state.
- **Zero-DB-on-recompose:** rail recompose reads only `_active_character_avatar`/`_name`; the DB fetch happens only in the scope-changed refresh.

### Testing strategy

- **Config helper:** `resolve_show_character_avatar` default True; explicit False/True; live-config-shape (COMPREHENSIVE_CONFIG_RAW).
- **Accessor:** `_current_console_rail_character_id`/`_name` return the active session's fields; `None` when no session/character. (A pin that P3c leaves `_active_console_dictionary_scope_ids` unchanged — still `(conversation_id, None)` — so the dictionary/world-book summaries are unaffected.)
- **Refresh scope-guard (real screen + real DB):** seed a character with an image; set the active session's `character_id`; run the sync tick → the panel mounts the avatar + name; unchanged scope → the DB fetch does NOT run again (spy the fetch); change `character_id` → it re-fetches and updates; set `character_id=None` → empty state; character with no image → name + placeholder. Assert off-thread (the fetch runs off the event loop) and that a stale post-await render is dropped when the scope changed during decode.
- **Rail section:** the "Character" section composes when config on, absent when off; the `character_open` toggle collapses/expands it like the others; empty state renders for a generic session.
- **Integration:** Start-Chat with a character → avatar shows; resume a persisted character conversation → avatar shows; generic session → empty state.

---

## Task decomposition (for writing-plans → subagent-driven-development)

1. **Config helper + resolution accessors** — `resolve_show_character_avatar` (console_image_view.py) + `_current_console_rail_character_id`/`_name` (chat_screen.py). Unit tests. *(Foundational, isolated; does NOT touch `_active_console_dictionary_scope_ids`.)*
2. **The "Character" rail section** — `ConsoleRailState.character_open` + compose the 5th `ConsoleRailSectionHeader` + body (config-gated) + empty state + toggle wiring. Rail widget tests.
3. **Avatar cache + scope-guarded off-thread refresh + render** — the cache fields, `_refresh_active_character_avatar_if_scope_changed` (reuse `_console_image_cache` + `resolve_default_mode` + `fit_image_cell_size`, compact rail box), mount into the section, never-raise. Real-screen + real-DB refresh tests.
4. **Wire into `_sync_native_console_chat_ui` + integration** — call the refresh from the sync tick; Start-Chat / resume / generic / character-change integration tests.

Four focused tasks → one PR.

---

## Global constraints

- **NO schema migration** (ChaChaNotes stays v22); reads existing `character_cards.image` + `conversations.character_id` (already written by #754).
- **Resolve the active character ONLY off the live `ConsoleChatSession`** (`_active_native_console_session().character_id`/`.character_name`); never read legacy `app.current_chat_*` reactives (split-brain).
- **Reuse the renderer verbatim** — `self._console_image_cache` (do not create a second cache), `resolve_default_mode`, and `fit_image_cell_size` (#775); avatar decode runs **off-thread** (`asyncio.to_thread`).
- **Zero-DB-on-recompose** — the DB fetch/decode happens only in the scope-changed refresh (mirror `_refresh_active_dictionaries_summary_if_scope_changed`); compose reads only the cache. Re-check the `(character_id,)` scope after the await to drop a stale render.
- **The refresh must never raise into `_sync_native_console_chat_ui`** (wrap; failure → text/empty fallback).
- **Config-gated** on `[chat.images].show_character_avatar` (default True); section not composed when off.
- **Characters-only** (no persona image).
- **Branch off the LATEST `origin/dev`** (`dc21e3f04` at spec time — has #754 + #775; re-verify at plan/merge time). **Concurrent-session hazard:** `chat_screen.py` and `personas_screen.py` are heavily edited by other sessions — keep P3c localized to the new accessor/section/refresh + the one `_active_console_dictionary_scope_ids` line; expect a rebase before merge.
- **Implementers stage ONLY their task's files** — never `git add -A`, never `.superpowers/`.
- **No background/broad test sweeps; never broad-`pkill` pytest** — scope to this worktree.
- **Test env:** `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest … -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread` (venv in MAIN checkout; imports resolve to worktree; UI tests slow — generous timeouts). Note `Tests/UI/pytest.ini` sets `asyncio_mode = auto`; a run that mixes `Tests/UI/` with another dir drops that (rootdir shift) — keep async tests in `Tests/UI/` and run them without cross-dir mixing, or add explicit `@pytest.mark.asyncio`.

## Deferred / follow-ups

- **P3d — reaction/expression image system** (own future program): a net-new image-set data model + authoring + trigger + swap, ported from the tldw_server webui/browser extension (needs the external `../tldw_server` repo). Renders into P3c's panel.
- **Character-aware "what's in play" (separate concern, pulled out of P3c):** decide whether the native Console dictionary/world-book summaries should include character-embedded dicts/books now that #754 gives sessions a real `character_id` — i.e. whether `_active_console_dictionary_scope_ids` (and its world-book twin) should source `session.character_id`. This requires verifying the native **send** path applies character dicts/books post-#754 first (to preserve "shown=applied"); if it doesn't, either wire that too or leave the summary conversation-only. File as its own task.
- Migrate the personas editor's `_fit_avatar_cell_size`/`_build_avatar_pixels` to the shared `fit_image_cell_size` (partly done by #775 for the transcript; the personas editor still has its own copy).
- A live config-toggle recompose (if `show_character_avatar` is toggled mid-session) — P3c composes on the config at compose time; a mid-session toggle picks up on the next natural recompose.
