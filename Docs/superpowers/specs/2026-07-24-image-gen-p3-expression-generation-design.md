# Image Generation P3 — Generate Character Avatar & Expression Images

**Status:** Design approved (user: "lets target the existing expression system"), verified against `origin/dev@238ac3041`
**Date:** 2026-07-24
**Program:** In-chat image generation (P1 #800 engine, P2a #832 card+variants, P2b #850 speak/styles/context — all merged). P3 fills the **existing** character expression system with generated images; it invents no new canvas, gallery, or mood states.

## 1. User-visible behavior

- In the character editor (Personas/Roleplay screen), each image slot gains a **✨ Generate** button beside its existing Upload: the **avatar row** and the three **expression slots** (thinking / speaking / error — `EXPRESSION_IMAGE_STATES`). Plus one **Generate all** button (avatar + 3 states, sequential).
- The prompt is auto-composed from the editor's **live** form text: name + description (+ personality when non-empty), plus a per-state modifier fragment (avatar/idle → neutral portrait framing; thinking → pensive/looking-away; speaking → mid-speech/animated; error → confused/sheepish). Style presets compose on top exactly like P2b (`compose_styled_request`; user-text-always-included invariant).
- **Style**: an optional style readout + "Pick style…" control in the expression section opens the existing `ConsoleStylePickerModal` (verified screen-agnostic); the chosen style applies to subsequent generations this editor session (not persisted). Unset = no template ("Custom").
- **Persistence matches each slot's existing rule** (verified):
  - Generated **avatar** → staged via `set_avatar_image` (dirty-mark; persists on Save; Save-blocking validation unchanged). Works for unsaved characters too.
  - Generated **expression images** → immediate write via the existing `_apply_expression_upload` seam (DB upsert + editor-generation bump + notify + slot thumbnail re-render, all reused). Requires a saved character — same gating as Upload (`_sync_expression_slots_enabled` extended to the new buttons).
- **Refusals** (status/notify, no generation): empty description ("Add a description first"); backend not configured/enabled (same copy as Console); a generation already running for that (character, slot).
- Generation runs off the UI loop; the slot shows a "Generating…" hint; failures surface via the screen's `_notify(..., "error")`; **regenerate = click again** (immediate-overwrite for expressions, restage for avatar — cheap redo replaces any review-queue UI).
- Switching characters/screens mid-generation never writes into the wrong character (session-token guard, §2).

## 2. Implementation seams (all verified on dev — reuse, don't rebuild)

- **Buttons/messages:** mirror the upload flow exactly. New `CharacterExpressionGenerateRequested(state)` + `CharacterAvatarGenerateRequested()` + `CharacterExpressionGenerateAllRequested()` messages in `Widgets/Persona_Widgets/personas_pane_messages.py` (sibling of `CharacterExpressionUploadRequested`, `:59`). Editor buttons `personas-char-editor-expr-{state}-generate` (+ avatar/all) posted from handlers mirroring `_expression_upload_pressed` (`personas_character_editor_widget.py:1064`); the new buttons JOIN `_sync_expression_slots_enabled`'s loop (`:577` — else they never enable). Avatar-generate is enabled whenever the editor is active (staged path needs no saved id).
- **Screen handlers** (`UI/Screens/personas_screen.py`): mirror the 5-seam upload chain — `message.stop()`, `_character_editor_is_active()`, saved-id gate (expressions only), then `run_worker(self._generate_expression_worker(...), group="personas-io", exit_on_error=False)` (the 5-of-20 explicit-`False` convention; the repo's documented `exit_on_error=True` app-panic trap) with the whole body in try/except and the in-flight guard cleared in `finally`.
- **Worker body:** capture `_character_editor_session_token()` BEFORE the await; `await asyncio.to_thread(run_generation, request)` (blocking engine contract); **re-check the token after** (the screen's established stale-guard idiom, `personas_screen.py:4789+`); then:
  - expressions → `await self._apply_expression_upload(character_id, state, result.content, result.content_type)` (the single call that does write+bump+notify+re-render, `:4663`);
  - avatar → explicit `len(result.content) > PERSONAS_AVATAR_MAX_BYTES` pre-check (the 5 MB rule; the path-based validator is unreachable for in-memory bytes — verified) → `editor.set_avatar_image(result.content)` + avatar thumb refresh (`_render_character_editor_avatar` worker).
- **Prompt composition:** new pure module `tldw_chatbook/Character_Chat/expression_generation.py`: per-state modifier constants (NOT in `BUILTIN_TEMPLATES` — internal fragments, not user styles) + `compose_expression_prompt(name, description, personality, state, style_template) -> (prompt, negative, params)` reusing P2b's `compose_styled_request` semantics (user text always included). Reads: `editor._area("description").text` etc. — the LIVE form values, never `_character_data` (verified stale).
- **Engine:** `Image_Generation.worker.build_request`/`run_generation` unchanged; backend = `cfg.default_backend`; seed −1; one image per slot. Engine output stored **as-is** (768×1024-ish per template params; every existing write path stores raw bytes verbatim; both display surfaces thumbnail render-copies; engine caps — `max_pixels` 1 Mpx, `inline_max_bytes` 4 MB — already sit under the avatar's 5 MB validator).
- **In-flight guard:** screen-level `set[(int|None, str)]` keyed (character_id-or-None-for-unsaved-avatar, slot); "Generate all" iterates the four slots through the same guard sequentially in one worker.
- **Style picker:** `push_screen_wait(ConsoleStylePickerModal())` from the worker (the screen's dialog convention) or `push_screen(callback=...)` from sync context; restore focus to the editor after dismiss (the modal's documented caller responsibility).
- **No DB/schema/config changes.** The v23 table + CRUD is the storage; `[image_generation]` config is reused as-is.

## 3. Testing
- Pure: `compose_expression_prompt` table (each state's modifier present; name+description included; personality only when non-empty; style template composes; **empty/whitespace description → the SCREEN refuses before composing (the user-facing path), and `compose_expression_prompt` additionally raises `ValueError` as a defensive guard — both pinned**).
- Screen-level (personas harness style): generate-expression handler writes via `_apply_expression_upload` with generated bytes (mocked `run_generation`); saved-id gate refuses; session-token change between dispatch and completion → NO write (stale-guard pin); avatar path stages via `set_avatar_image` + oversized-bytes refusal (fake >5 MB result); in-flight guard blocks a second click; generate-all runs 4 sequential requests with per-state prompts; failure → `_notify` error + no write; `exit_on_error=False` on the new workers (grep-pin like sibling tests if precedent exists).
- Editor-level: new buttons enable/disable with `_sync_expression_slots_enabled`; button→message posting.
- Regression: personas screen suites green vs baseline; live smoke at the end (editor → Generate → SwarmUI conn-refused surfaces as notify, no crash; with a backend, a real image lands in the slot).

## 4. Out of scope
tldw_server-style visual packs / candidate galleries / review queues; new mood states beyond the 4-state machine; changing Console avatar display; persisting a per-character default style; batch-count >1 per slot; editing generated images. Follow-ups continue in tasks 497/498/558/559.
