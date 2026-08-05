# Image Generation P2b — Speak (TTS), Style Presets, Generate-from-Conversation

**Status:** Design approved (user: "all three" → "yes"), ready for implementation
**Date:** 2026-07-24
**Program:** In-chat image generation (P1 #800, P2a #832 merged). This slice completes the reference card's remaining affordances (🔊, real Style field) + context generation.
**Base:** `dev@05abb57d3` (P2a merged; Textual 8.2.7).

## 1. Feature contracts (user-visible)

### 1a. 🔊 Speak (Console messages)
- A new `speak` action (`🔊`) appears in the Console message action row for any **completed message with non-empty text content** (user or assistant; generation-card messages included — it reads their `[image] …` marker text, harmless).
- Clicking it posts the app's existing `TTSRequestEvent(text=<message content>, message_id=<id>)` (`Event_Handlers/TTS_Events/tts_events.py:25`). Everything downstream is the existing pipeline: validation (5000-char cap, 2s per-message cooldown), synthesis via `[app_tts]` config (default provider `openai`), system-player playback, and the existing graceful failure toast ("TTS failed: …") when TTS isn't configured. **No new TTS machinery.**

### 1b. Style presets for `/generate-image`
- New leading token `@<style>`: `/generate-image @anime a red dragon`, composable with the backend token in either order (`/generate-image :swarmui @anime a dragon`). Token resolution against `Media_Creation/generation_templates.py` `BUILTIN_TEMPLATES` (13 presets), case-insensitive: exact id → exact name (spaces/underscores interchangeable) → unique prefix of id or name; ambiguous/unknown → refusal status listing the valid ids (no generation).
- A resolved style wraps the request: the user's prompt substitutes into the template's `{{placeholders}}` (build the context dict by mapping every `context_mappings` target key to the user prompt), the template's `negative_prompt` is used when the user supplied none, and `default_params` (width/height/steps/cfg_scale) flow into `build_request`. `GenerationVariantMeta.style` = the template **name** (e.g. "Anime Style") → the card's Style row shows it (P2a already renders `meta.style or "Custom"`, `console_generation_card.py:137`). Regenerate inherits the stored composed prompt + style label (P2a behavior — it rebuilds from position-0 meta; no change needed, but pinned by a test).
- **Style picker:** a `ConsoleStylePickerModal` mirroring `ConsoleSkillPickerModal` (`Widgets/Console/console_skill_picker_modal.py:93` — filter input, keyboard-first, category shown per row), launched from a command-palette action ("Insert image style"), which **inserts `@<style-id> ` into the composer draft** (mirror the prompt-insert launch pattern, `chat_screen.py:1444`). It does not generate by itself.

### 1c. Generate from conversation
- `/generate-image` with **no prompt** (with or without `@style`/`:backend` tokens): if the session has ≥1 completed message with text, build the prompt from conversation context — `ImageGenerationService.extract_context_from_messages(messages)` (`Media_Creation/image_generation_service.py:99`; pure helper, instantiate nothing else from that legacy service) over the session's last messages (store side; content only, most recent last), then apply the **`chat_scene_visual`** template by default (or the explicit `@style` template if given) via `apply_template_to_prompt(template_id, context)`.
- The composed prompt is what generates AND what the card/marker show (`[image] <composed…>`) — the user sees exactly what was asked. `style` records the template used. Empty conversation (no usable text) → the existing usage status line.
- Known limitation, accepted: the extractor is shallow (keyword mood + visual-hint fragments; `mentioned_characters`/`mentioned_settings` are never populated). This slice wires it as-is; a richer context builder is future work.

## 2. Implementation seams (verified on base)

- **Speak:** `ConsoleMessageActionService` action tuples + `available_actions` (`Chat/console_message_actions.py:59,102`); button-id prefix table `chat_screen.py:13438`; dispatch side-effect chain `chat_screen.py:12790+` — a `speak` branch posts `TTSRequestEvent` with `store.get_message(message_id).content`. Availability rule: completed + non-empty content (mirror how copy/edit gate).
- **Parser:** extend `GenerateImageArgs` (frozen) with `style: str | None`; generalize `parse_generate_image_args` (`console_generate_image.py:62`) to consume leading prefixed tokens in any order — `:x` → backend, `@x` → style token (raw; resolution happens later) — stopping at the first unprefixed token; bare `:`/`@` are prompt text (existing bare-`:` semantics preserved).
- **Style resolution + composition:** new pure helpers in `console_generate_image.py`: `resolve_style_token(token) -> GenerationTemplate | None` (+ an "ambiguous" signal) and `compose_styled_request(prompt, template) -> (prompt, negative, params)` wrapping `apply_template_to_prompt`. **Invariant: the user's prompt text always appears in the composed prompt** — if the template has no `context_mappings` (or substitution leaves the user prompt unconsumed), fall back to `f"{composed_base}, {user_prompt}"` rather than silently dropping it. `run_generation_batch` gains optional `style_name`, `negative_prompt`, and `default_params` threading (params → `build_request(width=…, height=…, steps=…, cfg_scale=…)`; explicit-seed rule unchanged; `GenerationVariantMeta.style=style_name`, currently hardcoded `None` at `console_generate_image.py:198`).
- **Context:** new pure helper `build_context_prompt(messages: Sequence[tuple[str, str]], template) -> (prompt, negative, params) | None` (role/content pairs in, None when no usable content) — internally shapes the dicts `extract_context_from_messages` expects. The handler branch replaces the empty-prompt usage-return when the conversation has content.
- **Picker:** `Widgets/Console/console_style_picker_modal.py` mirroring the skill picker (rows: name, id, category; filter matches all three); palette action inserts the token into the draft via the same composer-insert idiom as prompt-insert.
- **No DB/schema changes.** `GenerationVariantMeta.style` already persists via the P2a sidecar. No config changes (TTS uses existing `[app_tts]`; styles need none).

## 3. Testing
- Parser table: token orders, both tokens, bare `:`/`@`, unknown/ambiguous style token surfaces as parse output (resolution refusal tested at resolver level).
- Resolver: exact id / name / unique prefix / ambiguous / unknown.
- Composition: prompt substitution into placeholders, negative fallback (user-supplied wins), params flow into `build_request` (assert on captured request), style name lands in meta + card details text.
- Context: builds from role/content pairs; None on empty/whitespace-only conversations; handler generates with composed prompt (mocked batch) and shows composed marker; usage line preserved for truly-empty sessions.
- Speak: action appears per gating; dispatch posts `TTSRequestEvent` with the right text/id (assert on posted message, no TTS execution); absent for empty-content/failed messages.
- Picker: modal filter/selection unit-level; insert-into-draft callback.
- Regression: P2a suites stay green; baseline = 3 ChaChaNotesDB legacy-parity + 1 anthropic + **4** chat-flow (dev's own, incl. 2 rail-label renames).

## 4. Out of scope
Richer context extraction (characters/settings/multi-turn narrative); per-style user-defined templates; TTS playback controls (pause/save — legacy enhanced-chat has them; Console gets fire-and-forget speak); reaction images / character canvas (P3); tasks 497/498/522 backlogs.
