# Roleplay P3d-1 — Reactive Character Expression Avatar (Console) — Design

**Date:** 2026-07-22
**Program:** Roleplay (Personas) redesign — character-presence theme, cycle 3 (P3b editor → P3c chat avatar → **P3d reactions**)
**Status:** design approved (brainstorming); ready for implementation plan.

## Goal

Make the Console Character-rail avatar (shipped in P3c, PR #782) **react as the character thinks and speaks**: swap among a small set of per-state images — *idle / thinking / speaking / error* — driven by the Console's own send/stream lifecycle, using images stored locally on the character. Fully additive: a character with no expression images just shows its normal avatar, exactly as today.

## Context

P3d is the character-presence theme's final, most ambitious cycle. The user's north-star framing was "copying the webui/browser-extension for tldw_server … multiple reaction images / changes in the profile pic as the message renders (like animating a character reacting as they speak)."

That reference is real: **tldw_server2 has a "Persona Visual Packs" / "Persona Buddy" system** — a manifest-backed pack (`states` → animations → ordered `frames`), a fixed core state set (`idle, listening, thinking, speaking, tool_running, approval_needed, error, offline`) plus freeform custom states, portable `.tldw-persona-vpack` archives, a renderer capability registry, and an external MCP pack-provider contract. Crucially, in that system **"reacting as it speaks" is NOT an LLM emotion classifier** — it is a **deterministic client-side state machine** driven by the app's runtime state (voice/tool/session), priority-ordered.

The full system is far too large for one spec (it mirrors ~30 server-side design tasks) and much of it assumes a graphical renderer (smooth sprite animation, atlas cropping, Live2D) that a terminal cannot do. So **P3d decomposes into cycles**, and this spec is the first: the **minimal terminal-portable core** — a local `terminal_static` renderer that frame-swaps one static image per state, driven by the Console lifecycle. Later cycles (out of scope here) add archive import, the server/MCP pack provider, the full capability registry, and custom/mood/tool/voice states.

### Foundation this builds on (P3c, PR #782, merged to dev)

- Config-gated **"Character"** collapsible section in `#console-left-rail` (`[chat.images].show_character_avatar`, default True).
- Active character resolved **only** off the live `ConsoleChatSession` (`_current_console_rail_character_id()` / `_name()`), never legacy `app.current_chat_*`.
- Avatar render pipeline: cache-as-**spec** (data dict, not a live widget) + off-thread decode via the single `self._console_image_cache` (`ConsoleImageRenderCache`) + `fit_image_cell_size` + graphics/pixels/text fallback in `_build_character_avatar_widget`, mounted by the async `_render_character_avatar_into_section`.
- `_refresh_active_character_avatar_if_scope_changed`: scope-guarded on `(character_id,)`, guard set **before** the `await`, off-thread fetch, post-await scope re-check drops a stale render, **never raises** into the sync tick, config-off early-return that also resets the scope guard to `None`.
- Called from `_sync_native_console_chat_ui`.

## Verified ground truths (from code scout — use verbatim, do not re-derive)

1. **Message status lifecycle** (`Chat/console_chat_store.py`): `_initial_status(role, content)` returns `"pending"` for an empty ASSISTANT message; `append_stream_chunk` sets status `"streaming"`; terminal statuses are `"complete"`, `"stopped"` (user stop), `"failed"` (error). **There is no `"error"` status — the error state maps to `"failed"`.** `stopped` is a user action, not an error → maps to idle.
2. **The trigger is already wired.** `_sync_native_console_chat_ui` runs every 0.2s via `_poll_transcript` (`chat_screen.py:~10434`, `set_interval(0.2, _poll_transcript)`) during a send, **and once on completion** — `_poll_transcript` calls `await self._sync_native_console_chat_ui()` *before* it checks `run_state.status` and stops the timer. So the avatar refresh already fires on every streaming tick and on the terminal tick. P3d needs **no new hook** — only a wider scope tuple.
3. **Character image today** is a single BLOB: `character_cards.image` (schema `image BLOB`), read via `get_character_card_by_id(character_id: int)`. `character_id` is INT.
4. **Character editor** (authoring host) is in `personas_screen.py` (P3b) with an existing avatar upload + thumbnail path (`_fit_avatar_cell_size`, `ConsoleImageRenderCache`, per-generation render token `_character_editor_generation`).
5. **Schema** is ChaChaNotes v22 (P3a/P3b/P3c added no migration). The next migration is `v22→v23`. `CharactersRAGDB` uses `threading.local()` connections.
6. **DO NOT touch** `_active_console_dictionary_scope_ids` (P3c pin — it feeds the "what's in play" summaries; not inert).

## Architecture

Five units, each independently testable:

### 1. Data model — `character_expression_images` table (migration v22→v23)

New table storing one raster image per (character, **non-idle** state):

```
character_expression_images(
  id            INTEGER PRIMARY KEY,
  character_id  INTEGER NOT NULL REFERENCES character_cards(id),
  state_id      TEXT NOT NULL,          -- 'thinking' | 'speaking' | 'error'
  image         BLOB NOT NULL,
  mime          TEXT,
  created_at    TEXT,
  updated_at    TEXT,
  deleted       INTEGER NOT NULL DEFAULT 0,   -- soft-delete, mirrors character_cards
  UNIQUE(character_id, state_id)
)
```

- **idle is never stored here** — it reuses `character_cards.image`.
- Migration is idempotent (PRAGMA/`CREATE TABLE IF NOT EXISTS`-guarded), bumps schema to v23. Pre-flight at plan time: re-verify dev is still v22 (concurrent sessions may have taken v23) and that no other in-flight migration collides.
- On character hard-delete, expression rows cascade; on soft-delete, they follow the card (excluded from reads).
- CRUD seam on `CharactersRAGDB`: `set_character_expression_image(character_id, state_id, image, mime)`, `get_character_expression_image(character_id, state_id) -> bytes|None`, `list_character_expression_states(character_id) -> list[str]`, `delete_character_expression_image(character_id, state_id)`. All parameterized; `state_id` validated against the known set.
- **Independent save:** expression images write straight to this table and do **not** touch the card's optimistic-lock version — authoring an expression never conflicts the card.

### 2. State derivation — `resolve_console_expression_state`

A pure, **DB-free** function (reads only in-memory store state), safe to call on the 0.2s tick:

```
resolve_console_expression_state(store, session) -> "idle" | "thinking" | "speaking" | "error"
```

Derived from the active assistant message's status for the session's conversation:
- an assistant message with status `"pending"` → **thinking**
- status `"streaming"` → **speaking**
- no in-flight message, last assistant `"complete"` or `"stopped"` → **idle**
- last assistant `"failed"` → **error** (persists until the next send resets to thinking/idle)

No emotion classification of reply text. Non-streaming replies may show a brief/absent "speaking" phase — acceptable and documented, not engineered around.

### 3. Trigger — extend the P3c scope

Change the avatar scope-guard from `(character_id,)` to **`(character_id, expression_state)`**. `_refresh_active_character_avatar_if_scope_changed` computes `state = resolve_console_expression_state(...)` (config-gated on the new sub-gate), builds the scope, and returns early when unchanged — so nothing rebuilds except on an actual state transition. The existing `_sync_native_console_chat_ui` calls (0.2s poll + completion + send/stop + session change) drive it. The post-await re-check now compares the full `(character_id, state)` tuple (state can flip mid-decode during fast streaming).

### 4. Rendering — reuse P3c, add a per-state decode cache

- **Keep** P3c's `_active_character_avatar` as "the spec the render currently mounts" (the render path `_render_character_avatar_into_section` reads it unchanged). **Add** a separate `(character_id, state)`-keyed **decode cache** dict → cached spec (`{character_id, state, name, mode, pil, pixels}`). On a state change, the refresh looks up (or decodes-and-stores) the cache entry for `(character_id, state)`, sets `_active_character_avatar` to it, and renders — so re-visiting a state (idle↔speaking↔idle) is instant, no re-decode. idle's entry is the P3c avatar spec (from `character_cards.image`).
- Fetch for a state: **fallback chain** = expression-table image for `state` → `character_cards.image` (idle) → text placeholder. If the idle image is null but a state image exists, that state renders its image and idle renders text — consistent with "missing → fall back."
- Everything else is P3c unchanged: off-thread decode via `self._console_image_cache`, `fit_image_cell_size` to `CHARACTER_AVATAR_COLS/LINES`, `_build_character_avatar_widget` (graphics → pixels → text), async `_render_character_avatar_into_section`, never-raise.

### 5. Authoring — expression slots in the character editor

In the P3b character editor (`personas_screen.py`), a small **"Expressions"** area with three upload slots (thinking / speaking / error), reusing the existing avatar upload + thumbnail machinery:
- Each slot: upload / preview (thumbnail) / clear. Upload writes to `character_expression_images`; clear soft-deletes the row.
- Each slot's async decode uses the `_character_editor_generation` render-token discipline (P3b lesson: any action invalidating an in-flight decode must bump the token; the swapped/removed slot must drop stale renders).
- An empty slot ⇒ that state falls back to idle in chat.

## Configuration

- Reuse `[chat.images].show_character_avatar` (default True): off ⇒ no Character section at all, expressions moot.
- New sub-gate `[chat.images].react_character_expressions` (default **True**): when False, the Character section shows only the static idle avatar (P3c behavior) and never swaps. Pure config helper `resolve_react_character_expressions(app_config) -> bool` in `console_image_view.py`, mirroring `resolve_show_character_avatar`. Documented in `config.py`'s `[chat.images]` block.
- **UX caveat (documented):** terminals without a graphics protocol render via pixels, which can flicker on rapid swaps; `react_character_expressions=false` is the escape hatch. Not auto-disabled — user-controllable.

## Data flow (one reply)

1. User sends → controller creates an empty assistant message (`pending`) and starts the 0.2s `_poll_transcript`.
2. Poll tick → `_sync_native_console_chat_ui` → refresh → state `thinking` (scope changed) → swap to the thinking image (or idle fallback).
3. First token → `append_stream_chunk` flips status to `streaming` → next poll tick → state `speaking` → swap to the speaking image.
4. Completion → message `complete`; the terminal poll tick calls sync **before** stopping the timer → state `idle` → swap back to the idle avatar. (Failure → `failed` → `error` image, persisting until the next send.)

## Error handling / fail-soft

Inherited from P3c and made total: state-derivation never raises (pure, guarded), the refresh never raises into the sync tick (P3c invariant preserved, now guarding the wider scope), the render degrades state-image → idle-image → text. A missing/corrupt expression image, a decode failure, or a torn table read all degrade to idle-or-text, never crash the 0.2s tick.

## Testing strategy

- **DB seam:** real in-memory-unsafe note — use a file-backed `CharactersRAGDB(tmp_path/...)` (the fetch runs off-thread; `:memory:` is connection-private). Table CRUD, UNIQUE(character_id, state_id) upsert, soft-delete exclusion, migration applies idempotently and reports v23.
- **State derivation:** a table of constructed store states → expected state (pending→thinking, streaming→speaking, complete/stopped→idle, failed→error, none→idle). Pure, no live LLM.
- **Trigger + render:** drive status transitions on a real mounted Console screen (P3c harness) and assert the mounted avatar spec changes across idle→thinking→speaking→idle; assert scope-guard prevents rebuild when state is unchanged; assert per-state cache reuse (second visit to a state does not re-decode).
- **Fallback:** state with no image ⇒ idle image mounted; no idle image either ⇒ text; corrupt image ⇒ never raises (mirrors P3c `test_refresh_never_raises_on_bad_image`).
- **Authoring:** upload → row present → thumbnail renders; clear → row soft-deleted → chat falls back to idle; save is independent of the card version.
- **Config:** `react_character_expressions=false` ⇒ static idle only, no swap; `show_character_avatar=false` ⇒ no section (reuse P3c test).

## Global constraints (for the plan)

- **Migration v22→v23**, idempotent, pre-flight re-verify v22 at plan/merge time. Only DB change; character_cards untouched (idle still `character_cards.image`).
- Resolve active character **only** off the live `ConsoleChatSession`; resolve expression state **only** off in-memory store status — never DB on the derivation path (safe on the 0.2s tick).
- **Reuse** the single `self._console_image_cache`, `resolve_default_mode`, `fit_image_cell_size`, and the P3c render/refresh methods. Do not create a second render cache. Decode stays off-thread.
- **Never raise** into `_sync_native_console_chat_ui`; preserve every P3c invariant (config-off early-return + guard reset, post-await re-check — now on `(character_id, state)`, cache-as-spec-not-widget).
- **Do not touch** `_active_console_dictionary_scope_ids`.
- Scope-guard means **no rebuild/decode except on a real state change** (honors the perf-audit "no per-tick DB/rebuild" rule).
- Authoring slots apply the `_character_editor_generation` render-token discipline per slot; expression saves are independent of the card's optimistic-lock version.
- Config-gated: `show_character_avatar` (section) and `react_character_expressions` (swapping), both default True.
- Characters-only (personas have no image). CONCURRENT-SESSION HAZARD: `chat_screen.py` / `personas_screen.py` / `ChaChaNotes_DB.py` edited by other sessions — keep P3d localized to the new table + seam + derivation + scope-tuple + authoring slots; expect rebase.
- Tests: `Tests/UI/pytest.ini` sets `asyncio_mode=auto` — don't mix Tests/UI with another dir in one invocation, or add explicit `@pytest.mark.asyncio`. Never broad-pkill pytest; scope to the worktree.

## Out of scope — later P3d cycles

- Smooth animation / sprite atlas / frame-rate playback / Live2D (browser-only).
- `.tldw-persona-vpack` archive import; the full manifest V2 contract.
- Server/MCP pack provider; the renderer capability **registry** (a `terminal_static` capability marker with no consumer is YAGNI now — the essential fail-soft is already inherited).
- Custom / mood / tool_running / voice (listening) states; approval_needed / offline.
- Any emotion classification of reply text (state stays runtime-driven, per the server model).
