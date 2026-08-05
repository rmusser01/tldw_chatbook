# Image Gen P3 — Generate Avatar & Expression Images: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ✨ Generate buttons on the character editor's avatar row + three expression slots (and a Generate-all), auto-prompting from live form fields (+optional style preset), generating via the existing engine, and persisting through each slot's existing seam.

**Architecture:** Pure prompt composition in a new `Character_Chat/expression_generation.py`; new sibling messages + editor buttons mirroring the upload flow; screen workers following the personas screen's own conventions (`run_worker(..., group="personas-io", exit_on_error=False)`, `asyncio.to_thread`, session-token stale guard) landing on `_apply_expression_upload` (expressions, immediate) / `set_avatar_image` (avatar, staged + explicit 5 MB pre-check). **No schema/config changes.**

**Tech Stack:** Existing `Image_Generation` engine (`worker.build_request`/`run_generation` — blocking, thread-only), `ConsoleStylePickerModal` (verified screen-agnostic), Textual.

**Design spec (read first):** `Docs/superpowers/specs/2026-07-24-image-gen-p3-expression-generation-design.md`

## Global Constraints

- **Worktree:** ALL work in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/image-gen-p3`, branch `claude/image-gen-p3-expressions`. Subagents start in the MAIN checkout — `cd` in first; never touch the main checkout's `tldw_chatbook/`.
- **Test command** (worktree has NO `.venv`; MAIN venv + cwd=worktree): `source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate && cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/image-gen-p3 && python -m pytest <paths> -q`
- **NO BACKGROUND PROCESSES in subagent verification** — the harness kills them; run sweeps foreground, chunked if long.
- **Git hygiene:** repo tracks a stale `.superpowers/sdd/progress.md` — stage ONLY your own files by explicit path; never `git add -A`.
- **Worker safety (repo-documented app-panic trap):** every new worker = `run_worker(<async coroutine>, group="personas-io" or "personas-avatar-render", exit_on_error=False)` AND full-body try/except; blocking `run_generation` only ever inside `await asyncio.to_thread(...)`.
- **Stale-write guard:** capture `self._character_editor_session_token()` before any await; re-check after EVERY await before touching the editor/DB (the screen's established idiom, `personas_screen.py:~4789`).
- **Live-form reads:** prompts read `editor._area("description").text` / `_input("name").value` / `_area("personality").text` — NEVER `_character_data` (stale copy).
- **Persistence seams (verbatim reuse):** expressions → `await self._apply_expression_upload(character_id, state, content, content_type)` (`personas_screen.py:~4663`); avatar → `len(content) > PERSONAS_AVATAR_MAX_BYTES` check (`:~221`, 5 MB) then `editor.set_avatar_image(content)` + `self.run_worker(self._render_character_editor_avatar(), group="personas-avatar-render", exit_on_error=False)`.
- Expression states = `EXPRESSION_IMAGE_STATES` (thinking/speaking/error) from `Chat/console_expression_state.py`. Avatar = the fourth "slot" (state key `"avatar"` in OUR in-flight/message layer only — never passed to the DB API).
- `ruff check` changed files clean per commit; type hints + Google docstrings; loguru.
- Baseline discipline: attribute any unexpected failing test by running it at the branch base (`git stash` / base worktree) before treating it as yours.
- Anchors below are from the verification recon at `origin/dev@238ac3041` — they may drift a few lines; search by name.

## File structure

```
tldw_chatbook/Character_Chat/expression_generation.py    # NEW: pure composition
tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py  # MODIFY: 3 new messages
tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py  # MODIFY: buttons + gating + handlers
tldw_chatbook/UI/Screens/personas_screen.py              # MODIFY: handlers + workers + style state
Tests/Character_Chat/test_expression_generation.py       # NEW
Tests/UI/test_personas_expression_generate.py            # NEW (screen+editor level)
```

---

### Task 1: Pure composition — `expression_generation.py`

**Files:**
- Create: `tldw_chatbook/Character_Chat/expression_generation.py`
- Test: `Tests/Character_Chat/test_expression_generation.py` (check `Tests/Character_Chat/__init__.py` exists; create if missing, mirroring siblings)

**Interfaces:**
- `EXPRESSION_PROMPT_STATES: tuple[str, ...] = ("avatar", "thinking", "speaking", "error")`.
- `STATE_MODIFIERS: dict[str, str]` — avatar: `"neutral friendly expression, head and shoulders portrait, looking at viewer"`; thinking: `"pensive thoughtful expression, hand near chin, looking away"`; speaking: `"mid-speech, animated engaged expression, mouth open"`; error: `"confused sheepish expression, embarrassed, sweatdrop"`.
- `compose_expression_prompt(*, name: str, description: str, personality: str = "", state: str, style_template=None) -> tuple[str, str, dict]` — returns `(prompt, negative_prompt, params)`. Raises `ValueError` on unknown `state` or empty/whitespace `description` (defensive; the screen refuses first). Base text = `f"{name}, {description}"` (name omitted when blank) + `", {personality}"` when non-empty + `", {STATE_MODIFIERS[state]}"`. With `style_template`, wrap via `Chat.console_generate_image.compose_styled_request(base_text_with_modifier, template)` (import inside the function to keep module import-light); without, return `(base, "", {})`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Character_Chat/test_expression_generation.py
import pytest

from tldw_chatbook.Character_Chat.expression_generation import (
    EXPRESSION_PROMPT_STATES,
    STATE_MODIFIERS,
    compose_expression_prompt,
)


@pytest.mark.parametrize("state", EXPRESSION_PROMPT_STATES)
def test_each_state_modifier_lands_in_prompt(state):
    prompt, negative, params = compose_expression_prompt(
        name="Sayori", description="short coral pink hair, red bow", state=state,
    )
    assert STATE_MODIFIERS[state] in prompt
    assert "Sayori" in prompt and "coral pink hair" in prompt
    assert negative == "" and params == {}


def test_personality_included_only_when_nonempty():
    with_p, _, _ = compose_expression_prompt(
        name="A", description="desc", personality="cheerful", state="avatar",
    )
    without_p, _, _ = compose_expression_prompt(
        name="A", description="desc", personality="   ", state="avatar",
    )
    assert "cheerful" in with_p and "cheerful" not in without_p


def test_blank_name_omitted():
    prompt, _, _ = compose_expression_prompt(name="  ", description="desc", state="avatar")
    assert not prompt.startswith(",") and "desc" in prompt


def test_empty_description_raises():
    with pytest.raises(ValueError):
        compose_expression_prompt(name="A", description="   ", state="avatar")


def test_unknown_state_raises():
    with pytest.raises(ValueError):
        compose_expression_prompt(name="A", description="desc", state="angry")


def test_style_template_composes_and_keeps_user_text():
    from tldw_chatbook.Media_Creation.generation_templates import get_template

    template = get_template("style_anime")
    prompt, negative, params = compose_expression_prompt(
        name="Sayori", description="coral pink hair", state="thinking",
        style_template=template,
    )
    assert "coral pink hair" in prompt                     # user text survives (P2b invariant)
    assert STATE_MODIFIERS["thinking"] in prompt
    assert negative == template.negative_prompt
    assert params == template.default_params
```

- [ ] **Step 2: Run → FAIL** (`ModuleNotFoundError`). `python -m pytest Tests/Character_Chat/test_expression_generation.py -q`
- [ ] **Step 3: Implement** per Interfaces (read `Chat/console_generate_image.py`'s `compose_styled_request` first — reuse it, don't reimplement the invariant).
- [ ] **Step 4: Run → PASS** (7 passed). `ruff check` both files.
- [ ] **Step 5: Commit** `feat(character): expression-generation prompt composition` (stage the 2-3 files by path).

---

### Task 2: Messages + editor buttons + gating

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py` (sibling of `CharacterExpressionUploadRequested`, `:59`)
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py` (avatar row `:~101-130`; expression slots `:~372-391`; `_sync_expression_slots_enabled` `:~577`; upload-press handlers `:~1041-1069`)
- Test: `Tests/UI/test_personas_expression_generate.py` (NEW — find the existing editor/personas test files first: `grep -rl "personas_character_editor\|PersonasCharacterEditorWidget" Tests/ | head`; mirror their harness style)

**Interfaces:**
- New messages (each a dataclass-style Textual `Message` mirroring `CharacterExpressionUploadRequested`): `CharacterAvatarGenerateRequested()` (no fields), `CharacterExpressionGenerateRequested(state: str)`, `CharacterExpressionGenerateAllRequested()` (no fields).
- Editor buttons: `#personas-char-editor-avatar-generate` ("✨ Generate") in the avatar row beside Upload; `#personas-char-editor-expr-{state}-generate` (class `personas-char-editor-expr-generate`) in each expression slot beside Upload; `#personas-char-editor-expr-generate-all` ("✨ Generate all") in the expression section header area.
- Gating: the three per-state generate buttons + generate-all JOIN `_sync_expression_slots_enabled`'s enable/disable loop (disabled until the character is saved, same as upload); the avatar-generate button is always enabled while the editor is active (staged path).
- Press handlers mirror `_expression_upload_pressed` (`:1064`): `event.stop()` + post the matching message (state recovered via `_expression_state_from_button_id(..., suffix="-generate")`).

- [ ] **Step 1: Write the failing tests** — in the existing harness style; MANDATORY assertions: (a) pressing each generate button posts the right message type (+state); (b) per-state generate + generate-all buttons are disabled before the character is saved and enabled after (drive `_sync_expression_slots_enabled` the way existing slot tests do); (c) avatar-generate stays enabled pre-save. Write real arrange/act/assert after reading the sibling tests — the assertions as named are required.
- [ ] **Step 2: RED.** **Step 3: Implement.** **Step 4: GREEN** + existing editor suites stay green (`python -m pytest Tests/ -q -k "personas_character_editor or personas_expression"` — attribute any pre-existing failures at base). `ruff check`.
- [ ] **Step 5: Commit** `feat(personas): generate buttons + messages on avatar and expression slots`.

---

### Task 3: Screen — expression generate handler + worker (the core)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (handler cluster near `_handle_character_expression_upload_requested` `:~4710`; worker near `_expression_upload_dialog_worker` `:~4755`)
- Test: `Tests/UI/test_personas_expression_generate.py` (extend)

**Interfaces:**
- `@on(CharacterExpressionGenerateRequested)` handler: `message.stop()`; `_character_editor_is_active()` gate; `editor.expression_character_id()` None → notify "Save the character to add expressions." (same copy as upload); empty/whitespace live description → notify "Add a description first." severity warning, return; in-flight `(character_id, state)` in `self._expression_generate_inflight: set` → notify "already generating", return; else add to set + `run_worker(self._generate_expression_image_worker(character_id, state), group="personas-io", exit_on_error=False)`.
- `async def _generate_expression_image_worker(self, character_id: int, state: str) -> None`: whole body try/except/finally (finally discards the in-flight key). Read live fields from the editor (name/description/personality) + `self._expression_generate_style` (Task 4; `getattr(self, ..., None)` until then); `compose_expression_prompt(...)`; `build_request(backend=cfg.default_backend, prompt=..., negative_prompt=... or None, seed=-1, image_format="png", width/height/steps/cfg from params when present)`; token = `self._character_editor_session_token()`; `result = await asyncio.to_thread(run_generation, request)`; **re-check token** (changed → log + return, NO write); `await self._apply_expression_upload(character_id, state, result.content, result.content_type)`. Failures → `self._notify(f"Expression generation failed: {exc}", "error")`.
- Backend refusal: before dispatch, resolve config (`get_image_generation_config()`) — if `default_backend` unresolvable/not enabled per `listing.list_image_models_for_catalog()`, notify the Console's copy (`Image backend ... is not enabled/configured. Check [image_generation] settings.`), return (no worker).

- [ ] **Step 1: Write the failing tests** — screen-level in the existing personas-screen harness style (find how sibling tests drive handlers/workers; the upload flow's tests are the template). MANDATORY: (a) happy path — mocked `run_generation` returns a fake PNG result; assert `_apply_expression_upload` was awaited with (cid, state, content, content_type) [patch it and record]; (b) stale-token pin — token changes between dispatch and completion (mutate `self._character_editor_generation` inside the mocked `run_generation`) → NO `_apply_expression_upload` call + no crash; (c) empty-description refusal → notify called, no worker; (d) in-flight second click → refusal notify, single generation; (e) failure path → error notify, no write, in-flight cleared; (f) unsaved character → save-first notify.
- [ ] **Step 2: RED.** **Step 3: Implement.** **Step 4: GREEN** + `python -c "import tldw_chatbook.app"` clean + `ruff check`.
- [ ] **Step 5: Commit** `feat(personas): generate expression images via the image-gen engine`.

---

### Task 4: Screen — avatar generate, generate-all, style picker

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py` (ONLY if a style-readout label is added to the expression section — a small Static `#personas-char-editor-style-readout` "Style: Custom" + a `#personas-char-editor-style-pick` button posting a new `CharacterExpressionStylePickRequested()` message; add the message in `personas_pane_messages.py`)
- Test: `Tests/UI/test_personas_expression_generate.py` (extend)

**Interfaces:**
- `@on(CharacterAvatarGenerateRequested)`: gates = editor active + non-empty live description + backend configured + in-flight `(character_id_or_None, "avatar")`; worker mirrors Task 3 but: state `"avatar"` composition; token re-check; then `if len(result.content) > PERSONAS_AVATAR_MAX_BYTES: notify error, return`; else (still on the worker/UI seam — `set_avatar_image` touches widgets, so marshal like the upload flow does; READ how `_stage_character_avatar_from_path`/equivalent calls it and mirror) `editor.set_avatar_image(result.content)` + `self.run_worker(self._render_character_editor_avatar(), group="personas-avatar-render", exit_on_error=False)` + notify "Avatar image generated — Save to keep it."
- `@on(CharacterExpressionGenerateAllRequested)`: gates once (saved character required — avatar included in the sweep only when the description gate passes; same single in-flight key `(character_id, "all")` PLUS per-slot keys), one worker iterating `("avatar", "thinking", "speaking", "error")` sequentially, reusing the SAME per-slot logic (factor the per-slot body into a helper both workers call: `async def _generate_one_slot(self, character_id, state, style_template) -> bool`), notify a summary "k/4 generated" at the end.
- Style picker: `@on(CharacterExpressionStylePickRequested)` → worker (io-dialog convention: `_io_dialog_active` flag like `_expression_upload_dialog_worker` `:~4755`) → `await self.app.push_screen_wait(ConsoleStylePickerModal())` → store `self._expression_generate_style: GenerationTemplate | None` (resolve via `get_template(choice["id"])`), update the readout Static ("Style: {name}" / "Style: Custom"), restore focus to the editor (the modal's documented caller responsibility — mirror how the upload dialog restores).

- [ ] **Step 1: Write the failing tests** — MANDATORY: (a) avatar happy path stages via `set_avatar_image` (patched/recorded) + render worker kicked + NOT `_apply_expression_upload`; (b) oversized fake result (> `PERSONAS_AVATAR_MAX_BYTES`) → error notify + no staging; (c) generate-all with one failing state → 3 writes + "3/4" summary; (d) style pick stores the template and subsequent generation's request carries its params (record `build_request` kwargs via the mocked path); (e) style readout text updates.
- [ ] **Step 2: RED.** **Step 3: Implement** (factor `_generate_one_slot` FIRST, refactor Task 3's worker onto it — a pure refactor with Task 3's tests staying green is the proof). **Step 4: GREEN** + app import + ruff.
- [ ] **Step 5: Commit** `feat(personas): avatar generate, generate-all, expression style picker`.

---

### Task 5: Sweep + live smoke

- [ ] **Step 1: Full sweeps (foreground, chunked):** `python -m pytest Tests/Character_Chat/ Tests/UI/test_personas_expression_generate.py -q`; then the personas/editor suites (`python -m pytest Tests/ -q -k "personas"`); then `python -m pytest Tests/Chat/ -q` and `python -m pytest Tests/Image_Generation/ -q`. Baseline attribution for anything unexpected (run at branch base).
- [ ] **Step 2:** `git diff --name-only <plan-commit>..HEAD -- '*.py' | xargs ruff check` clean; `python -c "import tldw_chatbook.app"` clean; `python -m tldw_chatbook.css.check_bundle_sync` (only relevant if CSS changed — buttons may need slot CSS; if any `.tcss` source changed, regenerate the bundle via `./build_css.sh`, NEVER hand-edit).
- [ ] **Step 3: Live tmux smoke** (per `.claude/skills/verify` + the P2b lesson: pre-seed the scratch config with `[chat_defaults] provider="llama_cpp"` + `[api_settings.llama_cpp] api_url="http://127.0.0.1:9099"` to clear the first-run gate): RP&CD tab → open/create a character with a description → expression slots show ✨ Generate; click one → without a backend server: "Expression generation failed: ... Connection refused" notify, no crash; avatar ✨ pre-save works and stages. Report honestly what was covered.
- [ ] **Step 4: Commit** any fixes + final ledger note.

---

## Self-review notes (already applied)

- Spec §1 behaviors → T2 (buttons/gating), T3 (expression path + refusals), T4 (avatar/all/style); §2 seams pinned in Global Constraints; §3 tests distributed with the stale-token and oversized-avatar pins mandatory.
- Type consistency: `compose_expression_prompt` (T1) is the single composition entry used by T3/T4's workers; `_generate_one_slot` introduced in T4 refactors T3's body (T3 tests must stay green — the refactor's gate).
- Out of scope guarded: no galleries/review queues/new states/persisted style/schema changes.
