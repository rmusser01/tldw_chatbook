# Console voice dictation (voice2voice V1) — design

**Date:** 2026-07-27
**Status:** Approved for planning
**Scope:** V1 of a four-phase voice programme. This spec covers V1 only.

## Why

The Console (`UI/Screens/chat_screen.py`) has no voice affordance of any kind. The
mic button that once existed belonged to the retired `Chat_Window_Enhanced`. Speaking
is faster than typing for composing a prompt, and every later voice capability —
voice control, a hands-free conversation loop, a native realtime backend — needs the
same capture seam, provider resolution, and permission handling that V1 builds.

## Programme context

"Voice2voice" decomposes into four sub-projects with a real dependency chain. Each
gets its own brainstorm → spec → plan cycle. Only V1 is specified here.

| Phase | Goal | Depends on |
|---|---|---|
| **V1** | Push-to-talk dictation into the Console composer | — |
| V2 | Voice control + spoken status/errors (accessibility) | V1 capture seam |
| V3 | Hands-free loop: VAD turn-taking, spoken replies, barge-in | V1; **a streaming PCM sink that does not exist yet** |
| V4 | Native realtime speech API (audio in/out over WebSocket) | V3's audio sink |

Two programme-level constraints, recorded here so V1 does not have to rediscover them:

- `TTS/audio_player.py` shells out to `afplay`/`mpv`/`ffplay` on **complete files**. It
  cannot begin speaking before generation finishes and cannot be interrupted
  mid-utterance. V3 must build a streaming sink; V4 cannot consume a realtime audio
  stream without it.
- `LLM_Calls/` has no WebSocket transport at all. V4 is a from-scratch integration.

`webrtcvad` is already declared under the `speech_recording` optional feature, so V3's
VAD dependency is in place.

## Hard constraints (verified, not assumed)

1. **Press-and-hold is impossible.** Textual 8.2.7 enables kitty flags
   `DISAMBIGUATE | REPORT_ALL_KEYS | REPORT_ASSOCIATED_TEXT` — *not*
   `REPORT_EVENT_TYPES` — and has no `KeyRelease` event
   (`textual/drivers/linux_driver.py`, `textual/events.py`). Activation is a toggle.
2. **`post_message` is thread-safe; `call_from_thread` is not the right tool.**
   `message_pump.py:882` detects a foreign thread and routes via
   `loop.call_soon_threadsafe`. `call_from_thread` *blocks the caller* — here the
   audio/transcription path — so it is never used.
3. **`#console-composer-status` is invisible.** It carries `console-hidden-control`
   permanently (`console_composer_bar.py:1886`); that class is `display: none`
   (`tldw_cli_modular.tcss:3489`). It is a hidden compat element for tests, a sibling
   of the hidden `#console-command-input`. Voice status must not live there.
4. **`import tldw_chatbook.Audio` pulls in the heavy transcription stack.**
   `Audio/__init__.py` → `dictation_service` → `Local_Ingestion/transcription_service`,
   which at module scope runs `from faster_whisper import WhisperModel` and
   `import nemo.collections.asr` (torch). Multi-second startup cost for anyone with
   those extras installed.
5. **Both service entry points block.** `start_dictation()` triggers the cold model
   load; `stop_dictation()` does `processing_thread.join(timeout=2.0)`
   (`dictation_service_lazy.py:604`).
6. **The service silently rewrites the configured provider.**
   `dictation.privacy.local_only` defaults to `True`, and
   `_initialize_streaming_transcriber` (`dictation_service_lazy.py:339`) then forces
   the provider to `parakeet-mlx` whenever it is not in
   `["parakeet-mlx", "faster-whisper", "lightning-whisper"]`. `parakeet-mlx` is
   Apple-Silicon-only, so on other platforms this rewrites to something that cannot
   load. The allowlist entry `"lightning-whisper"` also never matches the real
   provider id `"lightning-whisper-mlx"` — a live bug, filed separately.
7. **Whisper emits bracketed tokens.** `[BLANK_AUDIO]`, `[Music]`, `[silence]` are
   routine output. Rich parses `[...]` as markup, so an unescaped transcript in a
   `Static`, Button label, or tooltip raises `MarkupError`.

## Architecture

### New module: `tldw_chatbook/Chat/console_voice_input.py`

Headless. No Textual imports.

```
ConsoleVoiceInputController
  probe()      -> Availability(ok | MissingCapture(remedy) | MissingProvider(remedy))
  resolve()    -> EffectiveConfig(provider, model, language, was_overridden)
  start()      -> None   # non-blocking
  stop()       -> None   # non-blocking
  is_active
  _emit: Callable[[VoiceEvent], None]    # injected
  _service_factory: Callable[..., Any]   # injected; fake in tests
```

Emits frozen dataclasses — `VoicePartial`, `VoiceFinal`, `VoiceStateChanged`,
`VoiceFailed` — through `_emit`. The composer owns the corresponding `Message`
subclasses and wraps them, so thread-safety lands on `post_message` while the
controller stays importable and unit-testable with zero Textual.

**Ownership:** `ConsoleComposerBar` owns the controller instance and supplies an
`_emit` that wraps each dataclass in its matching `Message` and calls
`self.post_message`. Everything the controller drives directly — mic
button, chip, draft insertion — is composer-scoped. The emitted messages bubble to
`ChatScreen`, which owns only the cross-cutting concerns: the `alt+r` action, the
deferred send, and the shutdown triggers. The screen stops dictation by calling
`composer.stop_dictation()`, never by reaching for the controller itself.

**Service:** `LazyLiveDictationService`, constructed with `enable_commands=False`
(V2 flips it) and `save_audio=False`. It is chosen for two reasons: it defers the
model load to `start_dictation()`, and it is the only one of the two dictation
services that does not drag the transcription stack in at module scope (it imports
config and numpy only).

**Import discipline** (constraint 4):

- the controller imports `..Audio.dictation_service_lazy` **directly — never `..Audio`** —
  and does so **inside `start()`**
- `chat_screen.py` imports the controller under `TYPE_CHECKING` / lazily
- **`probe()` must never import `transcription_service`.** It uses
  `importlib.util.find_spec` and `optional_deps` flags. `get_available_providers()`
  is cheap once imported, but importing is the entire cost.

**Provider resolution belongs to the controller, not the service** (constraint 6).
`resolve()` reads `[transcription]` / `[STTSettings]`, verifies the provider is
installed, and passes a validated provider down, so the service never gets the chance
to swap it. When the resolved provider differs from the configured one,
`was_overridden` is surfaced in the UI rather than swallowed.

**Worker discipline** (constraint 5): both `start_dictation()` and `stop_dictation()`
run in `run_worker(thread=True, group="console-dictation", exit_on_error=False)`.

### Touched files

| File | Change |
|---|---|
| `Chat/console_voice_input.py` | new; headless; unit-testable |
| `Widgets/Console/console_composer_bar.py` | mic `Button` in `#console-composer-actions`; new `#console-voice-status` chip; `insert_dictated_text()` |
| `UI/Screens/chat_screen.py` | `alt+r` binding + action; message handlers; shutdown triggers |
| `css/components/_agentic_terminal.tcss` | chip + mic styling (source file; the bundle regenerates at boot) |

The chip follows the `console-composer-recovery` pattern: `width: 0` collapsed,
expanded when active. `#console-composer-expanded` is a `Horizontal`, so this is an
inline chip and **`COMPOSER_CHROME_ROWS` stays 4**. It has a bounded `max-width` with
left-truncation (keep the newest words) and a width floor below which it shows only
`●`, so the `1fr` draft never collapses. Every transcript string is `escape()`d before
reaching any renderable (constraint 7).

`alt+r` is free and reachable: `ChatScreen.on_key` is a whitelist that stops only
named keys plus `is_printable` characters (`chat_screen.py:11296+`), so non-printable
unmatched keys fall through to the screen's `BINDINGS`.

## Interaction model

**States:** `unavailable → idle → preparing → listening → finishing → idle`, plus
`error → idle`. The chip renders one line per state and is the single source of truth;
the mic button's label and variant derive from it.

**Start** (via `alt+r` or the mic button — identical paths):

- setup modal blocking (`_console_setup_modal_blocking()`) → refuse with a notify, stay `idle`
- composer collapsed → expand via `_set_console_composer_collapsed(False)`, then start
- otherwise → `preparing` immediately (`◌ Preparing microphone…`, then `◌ Loading model…`),
  worker spawned

Toggling during `preparing` or `finishing` is rejected by the controller's own state.
Delegating this to the service's `state_lock` yields a `False` return and a log line
while the UI sits on "Preparing…" forever.

**Listening:** the chip shows `● 0:07  …and compare them to`, driven by a 1s
`set_interval` created on start and stopped on every exit path. Each `VoiceFinal`
inserts via `insert_dictated_text()`, which delegates to `insert_text()` (caret
insertion, no paste-collapse — dictation must never trip `PASTE_COLLAPSE_THRESHOLD`
and become a collapsed token) and then calls
`_sync_console_workbench_actions_from_draft()`. Omitting that last call leaves
dictated text in the draft while Send stays disabled.

Finals are joined with a single space, suppressed when the draft is empty or already
ends in whitespace. Dictation inherits caret behavior: move the caret mid-dictation
and the next final lands there, exactly as typing would. `ctrl+a` during dictation
means the next final clears the draft, because `insert_text` honors
`_draft_selection_all` (`console_composer_bar.py:1034`) — identical to what typing
does, and accepted as such.

**Stop:** `finishing` (`◌ Finishing…`), worker joins, the in-flight partial is
finalized and inserted, chip collapses, timers stop. Text stays; there is no discard,
because `ctrl+z` covers it (see Dependencies).

**Enter while listening defers the send.** Enter already presses
`#console-send-message` (`chat_screen.py:11368`). Because stop is non-blocking, firing
both in the same tick would put the last words spoken into the *next* message. Enter
therefore sets a `pending_send` flag and requests stop; the send fires from the
`VoiceStateChanged(idle)` handler after the final insert. If stop errors, the flag is
cleared and nothing is sent. Repeated Enter while pending is a no-op.

`escape` is untouched. Its two existing bindings, one `priority=True`, stay as they are.

**No input-level meter in V1.** `get_audio_level()` exists on both the lazy service
(`:678`) and the recording service (`:513`), and "is it hearing me?" is exactly what a
meter answers — but it is a poll, requiring a 5–10 Hz tick, which is the sub-second
widget-write pattern that has caused problems in this codebase before. V1 keeps the 1s
elapsed counter. This is a decision, not an oversight.

## Failure and safety

**Availability is two independent questions,** probed at mount for the initial label
and **re-probed on every activation attempt** — mic button or `alt+r` alike, and cheap
since probe is `find_spec` only — so installing an extra or plugging in a microphone
mid-run is picked up rather than leaving the button permanently dead:

- no pyaudio/sounddevice → *"Microphone support isn't installed. Install with `pip install 'tldw_chatbook[speech_recording]'`."*
- capture present, no transcription provider → *"No speech-to-text provider installed. Install with `pip install 'tldw_chatbook[transcription_faster_whisper]'`."*

The mic button stays **visible and disabled** with the reason in its tooltip *and* in
the chip on press — never hidden, and never remedy copy that exists only on hover.

**macOS permission denial** is its own case, detected from the recording service's
error, with the existing three-step remedy (System Settings → Privacy & Security →
Microphone → **restart the app**). That copy is already written in the orphaned
widgets and should be mined before they are deleted.

**Provider override surfaces.** If the `local_only` rule would have swapped the
provider, the chip reports the effective provider once per app run, and the log
records both the configured and the effective value.

**Hot-mic safety.** Capture stops on all of: toggle-off, pending-send completion,
**session/tab switch**, `on_screen_suspend`, `on_unmount`, and a dictation-length cap
that auto-stops and notifies. The cap reads `dictation.max_session_seconds` (default
300), matching the existing `dictation.buffer_duration_ms` / `dictation.privacy.*`
keys, so a long dictation is never truncated by a hardcoded limit. Session switch matters because
the draft is per-session and swapped via `load_draft` (`chat_screen.py:3083, 7594,
11645, 11664`) — without it the mic stays hot and finals land in a different session's
draft. A microphone left live because a screen changed is a privacy bug, not a UX bug.

`on_screen_suspend` is a real Textual event (`events.ScreenSuspend`). It must
early-return when not recording so it does not re-earn the task-247 performance
removal, and must **not** call `super()` — `BaseAppScreen` has no such method
(`mcp_screen.py:240`).

**Shutdown does not go through the join.** `on_unmount` during app quit releases the
mic via the recording service's stop directly; the full `stop_dictation()` join is for
the interactive stop only.

**Liveness is not `is_mounted`.** In this Textual version it means "has been mounted at
least once" and is never reset (`library_screen.py:1328`). Use `post_message`'s own
closed-check return (`message_pump.py:875`).

**Worker failures** never reach `exit_on_error`. Any exception moves the machine to
`error`, shows the reason in the chip, notifies, and returns to `idle`. No path leaves
the chip stuck or the state machine wedged.

## Testing

CI runs `pytest -m unit`, `pytest -m integration`, and `pytest Tests/UI` plus
`pytest Tests -m ui --ignore=Tests/UI`. **Anything outside `Tests/UI/` without an
explicit marker is never executed.** `Tests/UI/` tests use a minimal `App` harness
mounting just the widget under test with `app.run_test()`.

**`Tests/Chat/test_console_voice_input.py`** — `@pytest.mark.unit`, fake service via
the injected factory, no Textual import in the file:

1. `probe()` distinguishes missing-capture, missing-provider, and ok — three remedies asserted separately.
2. `probe()` does **not** import `transcription_service` — assert absence from `sys.modules`. Regression guard for the startup-cost rule.
3. `resolve()` returns the configured provider when installed, sets `was_overridden` when `local_only` would swap it, never returns an uninstalled provider.
4. State machine: toggle during `preparing`/`finishing` is a no-op; `stop` from `idle` is a no-op; every error path returns to `idle`.
5. The fake drives callbacks from a real non-main thread; the controller touches nothing thread-affine.
6. The factory receives `enable_commands=False` and `save_audio=False`.

**`Tests/UI/test_console_voice_chip.py`:**

7. The chip is **actually displayed** when active — assert the display chain, not that its renderable holds text. This test exists because the first draft of this design put the indicator in a `display: none` element, which no text-only assertion would have caught.
8. A `[BLANK_AUDIO]` / `[Music]` partial renders literally and raises nothing.
9. Below the width floor the chip collapses to `●`; the draft keeps its `1fr` share.
10. `insert_dictated_text` joins with a single space **and leaves Send enabled** — catches an omitted `_sync_console_workbench_actions_from_draft()`.
11. A long dictation produces literal text, not a collapsed paste token.

**`Tests/UI/test_console_dictation.py`:**

12. `alt+r` reaches the action while the composer has focus.
13. Enter while listening defers the send until idle, and the sent payload contains the final utterance.
14. The mic stops on each of session switch, screen suspend, unmount, and cap expiry — one test per trigger.
15. Setup-modal blocking refuses start; a collapsed composer expands on start.

**Manual, and required.** A real-mic run against the live TUI: cold model-load latency,
the macOS permission-denial copy, and reading the chip while actually speaking. Every
finding in this design came from reading code that passed its own tests.

## Definition of done

CLAUDE.md §8, plus:

- All ACs checked; `## Implementation Notes` written
- The three new test files green; full suite compared to baseline **name-by-name**, not by count
- `transcription_service` absent from `sys.modules` after a Console mount — measured, not asserted
- Live run on at least one platform, including the deps-missing path
- No edits to the legacy voice widgets; CSS changed in the source `.tcss`, never the generated bundle
- The three follow-up tasks below filed in `backlog/tasks/`

## Out of scope for V1

TTS, VAD, auto-send, barge-in, realtime transport, any settings UI, an input-level
meter, and any change to the legacy voice widgets
(`voice_input_button.py`, `voice_input_widget.py`, `chat_voice_handler.py`,
`Dictation_Window*.py`).

## Dependencies and follow-ups

**Sequenced dependency — composer undo/redo.** V1's discard story *is* `ctrl+z`, so
undo should land first or alongside. It does **not** block V1 from merging: without
it, a misheard dictation is cleared with `ctrl+u` (`clear_draft`, already bound in
`ChatScreen.on_key`) or edited by hand, which is exactly the status quo for typed
text. Undo is not voice work and is not specified here. One implementation note
carries over: the composer's edit keys live in
`ChatScreen.on_key`'s whitelist next to `ctrl+u` (`chat_screen.py:11382`), gated by
`_should_capture_console_input` — **not** in `BINDINGS`. `ctrl+z` / `ctrl+shift+z`
must join that whitelist.

**Follow-up tasks to file:**

- `"lightning-whisper"` vs `"lightning-whisper-mlx"` allowlist mismatch in
  `dictation_service_lazy.py:341` — lightning users are silently rewritten to
  `parakeet-mlx`.
- Delete `Widgets/voice_input_button.py` (zero callers) after mining its macOS
  permission copy.
- Consider whether `local_only` defaulting to `True` should force a provider swap at
  all, rather than refusing with an explanation.
