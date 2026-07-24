# Console Inline Image Rendering (TASK-215) — Design

- **Date:** 2026-07-13
- **Status:** Approved pending user spec review
- **Scope anchor:** TASK-215 — inline image rendering in the Console transcript with pixel and terminal-graphics modes, per-message Toggle View, ported semantics from ChatMessageEnhanced, safe under the transcript's row-signature reconcile loop. Fast-follow to PR #621.

## Decisions (user-approved)

| Decision | Choice |
|---|---|
| Default behavior | Inline by default, honoring existing `[chat.images].default_render_mode` (`auto`/`pixels`/`regular`) + `[chat.images.terminal_overrides]`. No new config keys. **Honesty note (verified):** legacy chat *defines* these keys (and the Settings UI edits them) but never reads them — `ChatMessageEnhanced.pixel_mode` hardcodes graphics-first — and `terminal_utils.detect_terminal_capabilities()` currently has zero consumers. Console's `resolve_default_mode` becomes the first real consumer of all three, giving the existing Settings controls their first actual effect. |
| Chip fate | Chip line stays in the message row (metadata + fallback); the rendered image is an additive row below it. Plain-text exports unchanged. |
| Toggle cycle | Per-message three-state: `pixels` → `graphics` → `hidden` → … starting from the config-resolved mode. |
| View persistence | Screen-state only: override map rides the existing serialization allowlist (JSON-safe strings), survives screen switches, resets on relaunch. |
| Architecture | New `image` row kind in the keyed reconciler + off-loop render cache (Approach A). |

## Naming note

The terminal-graphics mode is called **`graphics`**, not "TGP": `textual_image.widget.Image` is the auto-negotiating variant (TGP/Sixel/halfcell/unicode per terminal) — verified against the installed library. `regular` in legacy config maps to `graphics`.

## Components

### New pure module — `tldw_chatbook/Chat/console_image_view.py`

- `ConsoleImageViewMode = Literal["pixels", "graphics", "hidden"]`
- `resolve_default_mode(app_config) -> Literal["pixels", "graphics"]` — resolution semantics (defined here, since no prior consumer exists to mirror): explicit `default_render_mode = "pixels"`/`"regular"` wins outright (`regular` → `graphics`); `"auto"` consults `detect_terminal_capabilities()` for the terminal name/flags, then `terminal_overrides[<terminal>]` if present, else `terminal_overrides["default"]` (shipped defaults: kitty/wezterm/iterm2 → `regular`, default → `pixels`); missing/empty behaves as `auto`; unrecognized values fall back to `pixels` (safest). |
- `next_view_mode(current) -> ConsoleImageViewMode` — the three-state cycle.
- `ConsoleImageViewState` — per-message overrides (only non-default entries); `serialize() -> dict[str, str]` / `restore(payload)`; `prune(live_message_ids)` drops stale entries (called at serialize time).
- `ConsoleImageRenderCache` — bounded cache of prepared images:
  - Entry: decoded PIL image, downscaled at decode time to ≤1024 px longest side (one decoded image serves both modes; verified ~16 MB for an uncapped 2048px RGBA — the cap plus LRU bound is mandatory, not optional).
  - LRU bounded at 16 entries; `evict_session(message_ids)` wired where stream buffers are cleaned on session close.
  - `prepare(message_id, image_data)` is synchronous CPU work — always invoked off-loop (see data flow).
  - A per-entry lazily built `Pixels` renderable (legacy sizing: ≤80 cols × 40 lines) so pixels-mode row builds are cheap.

### Transcript — `tldw_chatbook/Widgets/Console/console_transcript.py`

- `_TranscriptRow` gains kind `"image"`, key `image:{message_id}`, emitted directly after the message row when: message has an image, resolved mode ≠ `hidden`, and a prepared cache entry exists. Row order for a selected image message is message → image → actions → action-help (legacy reading order: text, then its image, then controls). Consequence to watch in QA: the action row (including Toggle View) sits below a potentially tall image.
- Signature: `("image", message_id, mode, prepared_flag)` — streaming reconcile ticks never rebuild image rows; toggling mode changes the signature and swaps the widget.
- Row widget: `pixels` → `Static(pixels_renderable)`; `graphics` → `textual_image.widget.Image(prepared_pil)` behind a guarded import (import failure → pixels). Build errors → log + no image row (chip remains).
- The transcript stays dumb: the screen supplies a prebuilt `{message_id: ConsoleImageRowSpec}` map (mode + renderable/PIL) via a setter alongside `set_messages`. Unmounted unit-test builds without descriptors produce no image rows. (Verified: `Static(Pixels)` and `Image(pil)` both construct safely unmounted, so this is defense-in-depth, not a correctness requirement.)
- Image rows are excluded from plain-text export paths — structurally free: `to_plain_text()` (console_transcript.py:440) iterates messages and action labels, never row widgets; the chip line already carries the attachment into plain text.

### Toggle View action

Established 3-point plumbing: `("toggle-image-view", "View")` in `ConsoleMessageActionService`, gated on the same `_has_image` used by Save Image; dispatch returns `completed` + copy naming the next mode; screen handler prefix `console-message-action-toggle-image-view-` cycles `ConsoleImageViewState` and triggers one sync.

### Screen wiring — `chat_screen.py`

- Owns `ConsoleImageViewState` + `ConsoleImageRenderCache`.
- During message sync: image messages with bytes but no cache entry → batched prep in a worker with `group="console-image-prep"`, `exclusive=True` (dedicated group per the PR #621 lesson: sync workers must not cancel it), running `prepare` via `asyncio.to_thread`; completion triggers one `_sync_native_console_chat_ui`.
- Builds the image-row spec map each sync from (messages × view state × cache) and hands it to the transcript.
- Serialization: override map joins `_serialize_native_console_state` / restore allowlists (strings only — the no-raw-bytes constraint is untouched; the cache is never serialized).

## Data flow

Message with image bytes arrives (send, resume, or restore-rehydration) → sync notices missing cache entry → off-loop prep (decode, cap at 1024px, LRU insert) → next sync emits the image row in the resolved mode → Toggle cycles pixels/graphics/hidden per message; `hidden` removes the row key and the reconciler unmounts it. Metadata-only messages (failed DB rehydration) have no bytes → no prep, no row — chip-only, exactly today's behavior. Cache eviction (LRU or session close) simply re-preps on next sight.

## Edge cases

- Prep failure (corrupt bytes): log with message id, negative-cache the id (avoid re-prep loops), chip remains.
- `graphics` mode under non-graphics terminals (incl. textual-serve): `textual_image` auto-negotiates down to halfcell/unicode — acceptable, matches legacy "regular".
- Image sizing (live-verified hazard): `textual_image.widget.Image` scales up to fill its container's width — unconstrained in the transcript it would render even tiny images ~180 cols wide. Graphics rows therefore get explicit style caps: `max-width: 80` cells (matching the pixels-mode cap) and `max-height: 40` rows. Pixels renderable is already capped at 80 cols × 40 lines (legacy parity).
- Session close: `evict_session` + view-state prune prevents unbounded growth.

## Testing

1. Pure module: mode resolution (auto × terminal overrides matrix), cycle order, serialize/restore/prune round-trip, cache LRU bounds + downscale cap + negative-cache.
2. Transcript: image row present/absent per mode and cache readiness; signature stability across simulated streaming ticks; unmounted builds with no descriptors.
3. Action service: gating (image messages only, absent on failed rows per existing action-row semantics), dispatch result.
4. Mounted flow (`app.run_test()`): full toggle cycle; hidden collapse; screen-switch survival of overrides; prep worker populating a row after arrival.
5. Standing visual gate: textual-serve captures of pixels mode + graphics mode (which will show halfcell under serve — recorded as expected; true TGP evidence optional via local kitty/iTerm2) before merge, user approval required.

## Out of scope

Clipboard paste/drag-drop (TASK-216); multi-attachment (TASK-217); config-driven filter/caps unification (TASK-222); any legacy chat rendering changes; persisting view modes beyond screen state.

## Key file touch list

| File | Change |
|---|---|
| `Chat/console_image_view.py` | **New** — modes, state, render cache |
| `Widgets/Console/console_transcript.py` | Image row kind + spec map setter |
| `Chat/console_message_actions.py` | Toggle View action |
| `UI/Screens/chat_screen.py` | State/cache ownership, prep worker, spec-map build, serialization, action handler |
