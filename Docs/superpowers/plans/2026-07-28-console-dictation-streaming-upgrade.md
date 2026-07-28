# Console Dictation Streaming Upgrade — Revised Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Supersedes** Tasks 4–10 of `2026-07-27-console-voice-dictation.md`. Tasks 1, 2, 2b, 3, 3b of that plan are complete, reviewed, and retained.

## Why this plan replaces the original

The original plan was written against a stale baseline. It assumed the Console had no voice affordance; in fact `ca4825b15 — feat(console): add local microphone dictation` was already on `origin/dev` before this branch was created. Executing the original Task 4 produced a **second** Mic button beside the existing one, which is how the error surfaced.

**Goal:** keep everything that ships, and replace only its backend — so dictation gains live partial feedback and works with any installed STT provider instead of requiring a local Parakeet ONNX model directory.

## What ships today (verified, do not break)

`tldw_chatbook/Audio/console_dictation.py` — `ConsoleDictationSession`, 228 lines:
- One-shot: `start(on_buffer_limit=...)`, `stop_and_transcribe() -> str`, `discard()`
- Local Parakeet ONNX only, resolved from a model directory via `_resolve_model_dir()` / `_required_files_present()`
- 16 kHz mono 16-bit, `CONSOLE_DICTATION_MAX_SECONDS = 60.0`, bounded PCM buffer with an `on_buffer_limit` callback
- English only, no partial output — nothing is visible until you stop

`tldw_chatbook/Widgets/Console/console_composer_bar.py`:
- `#console-dictation` "Mic" button in the action row
- `_set_dictation_state(state)` driving four states — `idle` / `starting` / `recording` / `transcribing` — each with label, tooltip, `disabled`, `variant`, and a `console-dictation-recording` class

`tldw_chatbook/UI/Screens/chat_screen.py`:
- `_request_console_dictation_start` / `_stop`, workers grouped `console-dictation-start` / `console-dictation-stop`
- A hard wall timer on `CONSOLE_DICTATION_MAX_SECONDS`
- `_console_dictation_origin_session_id` — attributes a transcript to the session it was spoken into, and declines to insert if the user has since switched away
- `_insert_console_dictation` → `_dictation_insertion(draft, cursor_index, transcript)` → `composer.insert_text(...)` → `store.set_session_draft(...)`

`Tests/UI/test_console_dictation.py` — 4 tests: state exposure, caret insertion without sending, wall timer and limit transition, failure visibility with draft preservation and idle recovery. **All four must still pass at every step.**

## What the retained controller adds

`tldw_chatbook/Chat/console_voice_input.py` (Tasks 1/2/3/3b, 51 tests, four review rounds):
- `probe()` — capture vs. provider availability, distinguished, using `find_spec` only so it never drags faster-whisper or NeMo into startup
- `resolve()` — honors `[transcription] default_provider`, verifies the provider is installed, and prevents the service silently substituting `parakeet-mlx`
- `ConsoleVoiceInputController` — injected `emit` / `spawn` / `service_factory`; no wedge paths; abandon race closed; per-attempt error latch; guaranteed microphone release
- Streaming partials and per-segment finals via `VoicePartial` / `VoiceFinal`

**Net gain of the upgrade:** live partial text while speaking, and dictation that works wherever faster-whisper is installed rather than only where a Parakeet ONNX model directory exists.

## Global Constraints

Everything from the original plan's Global Constraints still binds. Restated, plus what this plan adds:

- **Never import `tldw_chatbook.Audio` (the package)** — it chains to `Local_Ingestion/transcription_service`, which imports faster-whisper and NeMo at module scope. Import submodules directly, inside function bodies. `Tests/Chat/test_console_voice_input.py` asserts this; keep it passing.
- **Never use `call_from_thread`** — it blocks the caller, and the caller is the audio path. Use `post_message`, which is thread-safe.
- **Escape every transcript string** with `rich.markup.escape` before it reaches a `Static`, Button label, or tooltip.
- **Workers:** `run_worker(..., thread=True, group=..., exit_on_error=False)`. Never `exclusive=True` without `group=`.
- **CSS in `tldw_chatbook/css/components/_agentic_terminal.tcss`** (source). Never hand-edit the generated `tldw_cli_modular.tcss`.
- **Test markers:** `Tests/UI/` runs wholesale; anything under `Tests/Chat/` needs `pytestmark = pytest.mark.unit`.
- **Never run the whole `Tests/UI` directory** — it takes ~2 hours. Use the targeted commands each task names.
- **The four existing dictation tests are the contract.** A change that requires editing `Tests/UI/test_console_dictation.py` to keep it green is a design error unless the task explicitly says otherwise.
- **Preserve the ordering invariant:** `VoiceFailed` before `VoiceStateChanged(idle)`, state mutated before either emit. Pinned by two index-comparison tests.
- Run pytest in the FOREGROUND via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest ...`.

---

### Task 11: Remove the duplicate affordance

Task 4 of the superseded plan added `#console-voice-toggle` beside the existing `#console-dictation`, leaving two Mic buttons in one row. Remove mine; keep the status chip, which the existing feature has no equivalent of and which Task 13 needs for partials.

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `Tests/UI/test_console_voice_chip.py`

**Interfaces:**
- Consumes: `set_voice_status(state, *, partial, elapsed_seconds, message)` from Task 4 (retained).
- Produces: `#console-voice-status` chip only. `#console-voice-toggle` no longer exists.

- [ ] **Step 1: Write the failing test**

Replace `test_mic_button_exists_in_the_actions_row` in `Tests/UI/test_console_voice_chip.py` with:

```python
@pytest.mark.asyncio
async def test_there_is_exactly_one_microphone_button():
    """The composer ships #console-dictation; this feature must not add a second."""
    from textual.widgets import Button

    app = ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        mic_like = [
            button
            for button in composer.query(Button)
            if "mic" in str(button.label).lower() or "dictat" in (button.id or "")
        ]
        assert len(mic_like) == 1
        assert mic_like[0].id == "console-dictation"
        assert not composer.query("#console-voice-toggle")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_voice_chip.py -v`
Expected: FAIL — two mic-like buttons found.

- [ ] **Step 3: Write minimal implementation**

In `console_composer_bar.py`, delete the `#console-voice-toggle` `_bounded_button(...)` yield added by the superseded Task 4. Leave `#console-dictation` and the `#console-voice-status` chip untouched.

In `_agentic_terminal.tcss`, delete the `#console-voice-toggle` rule. Leave `.console-voice-status` and `.console-voice-status-error` in place.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
.venv/bin/python -m pytest Tests/UI/test_console_voice_chip.py Tests/UI/test_console_dictation.py -v
```
Expected: all pass, including the four pre-existing dictation tests.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_composer_bar.py \
        tldw_chatbook/css/components/_agentic_terminal.tcss \
        Tests/UI/test_console_voice_chip.py
git commit -m "fix(console): drop the duplicate mic button, keep the status chip"
```

---

### Task 12: Drive the chip from the existing dictation state

Make the chip reflect the shipping four-state lifecycle, so it is wired to real state before Task 13 puts partials in it.

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Test: `Tests/UI/test_console_voice_chip.py`

**Interfaces:**
- Consumes: `_set_dictation_state(state)` (existing), `set_voice_status(...)` (Task 4).
- Produces: the chip mirrors `idle` / `starting` / `recording` / `transcribing`.

State mapping — the chip vocabulary differs from the button's, so map explicitly rather than passing the string through:

| Button state | Chip |
|---|---|
| `idle` | hidden (`width: 0`) |
| `starting` | `◌ Preparing microphone…` |
| `recording` | `● 0:07` plus partial once Task 13 lands |
| `transcribing` | `◌ Transcribing…` |

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_chip_mirrors_the_shipping_dictation_states():
    app = ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        chip = composer.query_one("#console-voice-status", Static)

        composer._set_dictation_state("starting")
        assert _visible(chip)
        assert "Preparing" in str(chip.renderable)

        composer._set_dictation_state("recording")
        assert "●" in str(chip.renderable)

        composer._set_dictation_state("transcribing")
        assert "Transcribing" in str(chip.renderable)

        composer._set_dictation_state("idle")
        assert chip.styles.width.value == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_voice_chip.py -v -k mirrors`
Expected: FAIL — the chip does not react to `_set_dictation_state`.

- [ ] **Step 3: Write minimal implementation**

At the end of `_set_dictation_state`, after the existing button updates, translate the state and call `set_voice_status`. Do not alter any existing button label, tooltip, `disabled`, `variant`, or class assignment — the four shipping tests assert on those.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
.venv/bin/python -m pytest Tests/UI/test_console_voice_chip.py Tests/UI/test_console_dictation.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_composer_bar.py Tests/UI/test_console_voice_chip.py
git commit -m "feat(console): mirror dictation state in the voice status chip"
```

---

### Task 13: Swap the backend to the streaming controller

The substance of the upgrade. Replace `ConsoleDictationSession` with `ConsoleVoiceInputController` behind the *existing* button, preserving every externally-observable behavior the four shipping tests assert.

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_dictation_streaming.py` (new — do NOT edit `test_console_dictation.py`)

**Interfaces:**
- Consumes: `ConsoleVoiceInputController`, `VoicePartial`, `VoiceFinal`, `VoiceStateChanged`, `VoiceFailed`, `STATE_*` from `Chat/console_voice_input.py`.
- Produces: the same `_set_console_dictation_state` transitions and the same `_insert_console_dictation` call the shipping code makes.

**Behavior that must be preserved exactly** — each is asserted by a shipping test:
1. The four button states appear in the same order for a normal capture.
2. The transcript inserts at the caret and does **not** send.
3. The wall timer still fires at `CONSOLE_DICTATION_MAX_SECONDS` and transitions visibly.
4. A failure is visible, preserves the draft, and recovers to `idle`.
5. `_console_dictation_origin_session_id` still gates insertion — a transcript must never land in a session the user switched to mid-capture.

**Mapping:** controller `preparing` → button `starting`; controller `listening` → button `recording`; controller `finishing` → button `transcribing`; controller `idle` → button `idle`. `VoiceFailed` drives the existing failure path.

**Segment accumulation:** the controller emits per-segment `VoiceFinal` events, but the shipping insertion contract is one transcript at the end. Accumulate finals and insert once on reaching `idle`, so behavior 2 is unchanged. Live partials go to the chip only (Task 14) and never to the draft.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_console_dictation_streaming.py` asserting: a `VoicePartial` never mutates the draft; accumulated `VoiceFinal` segments insert once, space-joined, at the caret; and a `VoiceFailed` leaves the draft untouched and the button `idle`. Use the existing `_configure_native_ready_console` helper that `test_console_dictation.py` imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_dictation_streaming.py -v`
Expected: FAIL — the screen still drives `ConsoleDictationSession`.

- [ ] **Step 3: Write minimal implementation**

Replace `_create_console_dictation_session` with controller construction, `spawn` backed by `run_worker(thread=True, group="console-dictation-start", exit_on_error=False)`. Route the controller's events through `post_message`, never `call_from_thread`. Keep `_insert_console_dictation`, the origin-session gate, and the wall timer exactly as they are.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
.venv/bin/python -m pytest Tests/UI/test_console_dictation_streaming.py Tests/UI/test_console_dictation.py Tests/Chat/test_console_voice_input.py -v
```
Expected: new tests pass; **all four shipping tests still pass unmodified**.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_dictation_streaming.py
git commit -m "feat(console): stream dictation through the hardened voice controller"
```

---

### Task 14: Live partials in the chip

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Test: `Tests/UI/test_console_dictation_streaming.py`

- [ ] **Step 1: Write the failing test** — a `VoicePartial` shows escaped text in the chip and leaves the draft empty; `[silence]` renders literally; the chip drops the partial below the width floor.
- [ ] **Step 2: Run it and watch it fail.**
- [ ] **Step 3: Route `VoicePartial` to `set_voice_status(..., partial=...)`**, escaping the text. Start the 1 s elapsed timer on `recording`, stop it on every exit path.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest Tests/UI/test_console_dictation_streaming.py Tests/UI/test_console_dictation.py -v`
- [ ] **Step 5: Commit** `feat(console): show live partial transcripts while dictating`

---

### Task 15: Provider availability on the existing button

Today the Mic button offers no guidance when the local model directory is missing. `probe()`/`resolve()` already distinguish the two failure causes.

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Test: `Tests/UI/test_console_dictation_streaming.py`

- [ ] **Step 1: Write the failing test** — with capture missing, the button is visible-but-disabled and its tooltip names the `speech_recording` extra; with capture present but no provider, the tooltip names `transcription_faster_whisper`. Two different remedies, never hidden, never only in a hover.
- [ ] **Step 2: Run it and watch it fail.**
- [ ] **Step 3: Probe at mount for the initial label and re-probe on every activation**, so installing an extra mid-run is picked up. Surface `was_overridden` once per app run when the configured provider is unavailable.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest Tests/UI/test_console_dictation_streaming.py Tests/UI/test_console_dictation.py -v`
- [ ] **Step 5: Commit** `feat(console): explain why dictation is unavailable`

---

### Task 16: Verification, retirement, and follow-ups

- [ ] **Step 1: Guard the import cost.** Confirm `tldw_chatbook.Audio`, `transcription_service`, `faster_whisper` and `nemo` are all absent from `sys.modules` after a Console mount. The existing subprocess test in `Tests/Chat/test_console_voice_input.py` covers the module; extend it to the screen path.
- [ ] **Step 2: Baseline diff by NAME.** Compare against `.superpowers/sdd/2026-07-27-console-voice-dictation/baseline-full.txt` (189 failed / 8174 passed / 14 errors, none of them this branch's). Use `comm` on sorted `FAILED`/`ERROR` name lists — never counts. Note `Tests/UI/test_chat_shell_bar.py` fails to COLLECT on the base (`ImportError: TabState`); `--continue-on-collection-errors` is required.
- [ ] **Step 3: Decide `ConsoleDictationSession`'s fate.** If Task 13 leaves it with no callers, propose retirement in the report — do not delete it in this task. `Audio/console_dictation.py` is 228 lines with its own model-directory resolution that nothing else uses.
- [ ] **Step 4: Live verification with a real microphone.** No test in this plan exercises real audio. Confirm by hand: the Mic button starts capture; partials appear while speaking; the transcript lands at the caret without sending; Send becomes enabled; navigating away releases the microphone (OS indicator clears); and with `speech_recording` uninstalled the button is disabled with an actionable tooltip. Record each outcome.
- [ ] **Step 5: File follow-ups** in `backlog/tasks/`, assigning IDs via a Python `os.listdir` + regex scan across **all** worktrees against `origin/dev` (`git ls-tree | uniq` misses em-dash filenames), re-verified immediately before writing:
  1. Delete `Widgets/voice_input_button.py` — zero callers, and it touches widgets from the transcription worker thread.
  2. Composer undo/redo (`ctrl+z`/`ctrl+shift+z`) — keys belong in `ChatScreen.on_key`'s whitelist next to `ctrl+u`, **not** in `BINDINGS`.
  3. `"lightning-whisper"` in the two legacy Dictation Window dropdowns (`Dictation_Window_Improved.py:351,359`, `Dictation_Window.py:227`) — same id bug Task 2b fixed in the service.
  4. `_release()` frees the microphone but never sets `stop_processing`, so the `DictationProcessor` daemon thread outlives every abandon and mid-session error.
  5. `config.py:920` computes the non-macOS `STT_settings.default_stt_provider` fallback as `"faster_whisper"` (underscore) while every provider id elsewhere is hyphenated.
- [ ] **Step 6: Commit.**

---

## Deferred minors carried forward

From the superseded plan's ledger, for the final review to triage: `_module_installed`'s missing Args docstring; `Availability.kind` typed `str` not `Literal`; tests catch a narrowed but not a widened `except`; mid-file `import threading` in the Chat test module; no `thread.is_alive()` assertion after join; `STATE_ERROR`/`STATE_UNAVAILABLE` defined but never assigned; the parked selectively-raising-`emit` edge; `_release()`'s spurious "Not currently recording" warning on the happy path; `_run_begin()`'s `if not started:` discarding a service without `_release()`.
