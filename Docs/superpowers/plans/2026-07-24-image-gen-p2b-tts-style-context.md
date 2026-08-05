# Image Gen P2b — Speak (TTS), Style Presets, Context Generation: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 🔊 speak action on Console messages (reusing the existing `TTSRequestEvent` pipeline), `@style` presets for `/generate-image` (token + picker modal + real Style on the card), and `/generate-image` with no prompt generating from conversation context.

**Architecture:** All three features are thin wiring over verified existing seams — the TTS event pipeline, the P2a parser/batch/meta/card chain (`style` field already persists), the 13 `generation_templates` presets, and the Console modal-picker family. **No DB schema changes, no config changes.**

**Design spec (read first):** `Docs/superpowers/specs/2026-07-24-image-gen-p2b-tts-style-context-design.md`

## Global Constraints

- **Worktree:** ALL work in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/image-gen-p2b`, branch `claude/image-gen-p2b`. Subagents start in the MAIN checkout — `cd` in first; never touch the main checkout's `tldw_chatbook/`.
- **Test command** (worktree has NO `.venv`; MAIN venv + cwd=worktree): `source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate && cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/image-gen-p2b && python -m pytest <paths> -q`
- **Git hygiene:** repo tracks a stale `.superpowers/sdd/progress.md` — stage ONLY your own files by explicit path; never `git add -A`/`.`/`-am`.
- **Test hygiene:** `ruff check` changed files clean before each commit; no unused imports/fixtures.
- Reuse, don't rebuild: `TTSRequestEvent` (`Event_Handlers/TTS_Events/tts_events.py:25`) for speech; `apply_template_to_prompt`/`BUILTIN_TEMPLATES` (`Media_Creation/generation_templates.py`) for styles; `extract_context_from_messages` (`Media_Creation/image_generation_service.py:99` — call the METHOD via a lightweight instantiation-free path: it only uses `self` incidentally, so call it on a cheaply-constructed service OR extract via `ImageGenerationService.extract_context_from_messages(None, messages)` if it never touches self — READ IT and pick the honest option) for context.
- Composition invariant (spec §2): the user's prompt text ALWAYS appears in the composed prompt.
- Baseline failures (NOT yours): 3 ChaChaNotesDB legacy-parity, 1 anthropic-native-tools, 4 chat-flow (incl. 2 rail-label renames). Only NEW failures block.
- `loguru`; type hints + Google docstrings on public helpers.

## File structure

```
tldw_chatbook/Chat/console_generate_image.py        # MODIFY: parser @style, resolver, composition, batch threading, context builder
tldw_chatbook/Chat/console_message_actions.py       # MODIFY: speak action
tldw_chatbook/UI/Screens/chat_screen.py             # MODIFY: speak dispatch + button-id prefix; handler style/context branches; picker palette action
tldw_chatbook/Widgets/Console/console_style_picker_modal.py  # NEW: style picker
Tests/Chat/test_console_generate_image.py           # extend (parser/resolver/composition/context/batch)
Tests/Chat/test_console_message_actions.py          # extend (speak gating)
Tests/Chat/test_console_generation_actions.py       # extend (speak dispatch posts TTSRequestEvent)
Tests/Chat/test_console_style_picker.py             # NEW
```

---

### Task 1: Speak action (🔊)

**Files:** Modify `Chat/console_message_actions.py`, `UI/Screens/chat_screen.py`; extend `Tests/Chat/test_console_message_actions.py` + `Tests/Chat/test_console_generation_actions.py`.

**Interfaces:**
- Action id `"speak"`, glyph `"🔊"`, tooltip "Speak message". Available for completed messages with non-empty `content.strip()` (any role; mirror how copy gates on completed — READ `available_actions` at `console_message_actions.py:102` and the `_COMPLETED_ACTIONS` tuple at `:59`; place speak after copy).
- Screen: entry in `_parse_console_message_action_button_id`'s prefix table (`chat_screen.py:13438`); dispatch branch in `handle_console_message_action` (`:12790+`) posting `TTSRequestEvent(text=message.content, message_id=message.id)` via `self.app_instance.post_message(...)` (import from `tldw_chatbook.Event_Handlers.TTS_Events.tts_events`). No worker needed — the event handler does everything.

- [ ] **Step 1 (RED):** action-service tests — speak present for completed text message (user AND assistant), absent for empty-content and failed messages; existing action sets otherwise unchanged (pin one legacy case). Screen test (existing harness in `test_console_generation_actions.py`): dispatching speak posts a `TTSRequestEvent` with the message's text + id (capture via `app_instance.post_message` mock/messages list — read how the harness observes posted messages; if it can't, assert via monkeypatched `post_message`).
- [ ] **Step 2 (GREEN):** implement per Interfaces.
- [ ] **Step 3:** `python -m pytest Tests/Chat/test_console_message_actions.py Tests/Chat/test_console_generation_actions.py -q` green; ruff; commit `feat(console): speak (TTS) action on message rows`.

---

### Task 2: `@style` parsing, resolution, composition, batch threading

**Files:** Modify `Chat/console_generate_image.py`; extend `Tests/Chat/test_console_generate_image.py`.

**Interfaces:**
- `GenerateImageArgs` gains `style: str | None = None` (raw token, unresolved).
- `parse_generate_image_args`: consume leading prefixed tokens in ANY order/combination — `:x`→backend, `@x`→style — stop at first unprefixed token; bare `:` or `@` alone = prompt text (preserve existing bare-`:` test).
- `StyleResolution` (frozen): `template | None`, `ambiguous: tuple[str, ...] = ()`. `resolve_style_token(token: str) -> StyleResolution` — case-insensitive: exact id → exact name (spaces≡underscores) → unique prefix over ids+names; multiple prefix hits → ambiguous listing matched ids; none → template None.
- `compose_styled_request(user_prompt: str, template) -> tuple[str, str, dict]` — context = {target: user_prompt for target in template.context_mappings.values()} → `apply_template_to_prompt(template.id, context)`; **invariant:** if `user_prompt` not a substring of the composed prompt, return `f"{composed}, {user_prompt}"` (comma-join, strip artifacts). Returns (prompt, template_negative, default_params).
- `run_generation_batch(..., style_name: str | None = None, negative_prompt=..., width=None, height=None, steps=None, cfg_scale=None)` — thread into `build_request` and set `GenerationVariantMeta(style=style_name, ...)` (currently `style=None` at `:198`; negative already a param — user-supplied negative wins over template's, decided by the CALLER).

- [ ] **Step 1 (RED):** parser table additions (`@anime dragon`, `:swarmui @anime dragon`, `@anime :swarmui dragon`, `@` alone, `@x` unknown passes through as style token); resolver table (exact id `style_anime`, name `anime style`, unique prefix `waterc`, ambiguous `style_`→lists 3, unknown); composition (placeholder substitution for `quick_simple`; no-mapping template invariant appends prompt; template negative returned); batch threading (captured `build_request` kwargs carry width/height/steps/cfg from params; meta.style == style_name).
- [ ] **Step 2 (GREEN):** implement.
- [ ] **Step 3:** file's suite green; ruff; commit `feat(console): @style token — resolution + prompt composition + batch threading`.

---

### Task 3: Handler wiring — style + context generation

**Files:** Modify `UI/Screens/chat_screen.py` (`_console_command_generate_image` ~`:11740+`), `Chat/console_generate_image.py` (context builder); extend `Tests/Chat/test_console_generate_image.py`.

**Interfaces:**
- `build_context_prompt(messages: Sequence[tuple[str, str]], template) -> tuple[str, str, dict] | None` in `console_generate_image.py` — role/content pairs (chronological); returns None when no non-whitespace content; else shapes `[{"role": r, "content": c}...]`, calls `extract_context_from_messages` (READ it first — it's effectively static; call it the honest cheap way per Global Constraints), then `apply_template_to_prompt(template.id, context)`; same always-include guard using `context["last_message"]` as the anchor text.
- Handler flow (extend, don't rewrite): parse → resolve style token (unknown/ambiguous → status line listing valid ids, return, draft intact) → if prompt empty: gather session messages from the store (completed, non-empty content) → `build_context_prompt(..., style_template or get_template("chat_scene_visual"))`; None → existing usage line. → else `compose_styled_request` when styled. Composed prompt feeds the batch AND `generation_content_marker`. Batch call passes template params + style NAME (template.name).

- [ ] **Step 1 (RED):** context-builder tests (composes from pairs incl. mood/hints path; None on empty/whitespace; explicit style template used over chat_scene_visual); handler-level tests at whatever level the P2a flow tests sit — minimum: batch-level assertion that the composed prompt/params/style reach `run_generation_batch` (extract a pure `prepare_generation_request(args, conversation_pairs, cfg) -> PreparedGeneration | UsageError | StyleError` helper in `console_generate_image.py` so the decision logic is fully unit-testable; the handler then just executes it — REQUIRED, don't bury logic in the screen).
- [ ] **Step 2 (GREEN):** implement helper + slim handler branches.
- [ ] **Step 3:** suites green (`test_console_generate_image.py` + `python -c "import tldw_chatbook.app"`); ruff; commit `feat(console): style + generate-from-conversation in /generate-image`.

---

### Task 4: Style picker modal + palette insert action

**Files:** NEW `Widgets/Console/console_style_picker_modal.py`; modify `UI/Screens/chat_screen.py`; NEW `Tests/Chat/test_console_style_picker.py`.

**Interfaces:**
- `ConsoleStylePickerModal(ModalScreen[Optional[Mapping[str, object]]])` — mirror `ConsoleSkillPickerModal` (`console_skill_picker_modal.py:93`) near-verbatim: filter `Input`, Up/Down/Enter keyboard flow, Escape cancels. Rows from `BUILTIN_TEMPLATES`: display "name — category" + id; filter matches id/name/category (casefold substring). Selection returns `{"id": ..., "name": ...}`.
- Palette action "Insert image style" (register alongside the existing Console palette actions — find where `action_open_console_prompt_insert` (`chat_screen.py:1444`) is exposed and mirror it) → `push_screen(ConsoleStylePickerModal(), callback=...)` → callback inserts `f"@{id} "` into the composer draft via the same insert idiom the prompt picker uses (READ `_open_console_prompt_picker_for_insert` and its apply path).

- [ ] **Step 1 (RED):** modal unit tests in the skill-picker test file's style (READ it first — find via `grep -rl "SkillPickerModal" Tests/`): rows built from templates; filter narrows; selection returns id/name; escape returns None. Callback test: insert produces `@style_anime ` in the draft (screen-level if harness supports; else test the pure insert helper).
- [ ] **Step 2 (GREEN):** implement.
- [ ] **Step 3:** suites + ruff + app import; commit `feat(console): style picker modal + palette insert`.

---

### Task 5: Round-trip + sweep + live sanity

- [ ] Card shows the real style: extend `Tests/Chat/test_console_generation_card.py` — a spec whose meta has `style="Anime Style"` renders "Anime Style" in `generation_card_details_text`; regenerate-path test (Task-8 P2a harness) pins that an appended variant inherits the style label.
- [ ] Sweep: `python -m pytest Tests/Chat/ Tests/Image_Generation/ Tests/UI/test_console_native_transcript.py Tests/UI/test_console_native_chat_flow.py -q` — only baseline failures (per Global Constraints).
- [ ] `ruff check` all changed files; `python -c "import tldw_chatbook.app"`.
- [ ] Live tmux sanity (scratch `TLDW_CONFIG_PATH`): `/generate-image @nope x` → refusal lists ids; `/generate-image @anime` in a conversation with text → composed-prompt marker appears (backend refusal acceptable without a server); speak action visible on a text message (clicking without TTS config → red toast, no crash). Report honestly what was covered.
- [ ] Commit fixes/tests: `test(console): P2b round-trip + sweep`.
