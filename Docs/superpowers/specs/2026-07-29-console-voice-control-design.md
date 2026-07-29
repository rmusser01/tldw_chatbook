# Console voice control and spoken feedback (voice2voice V2) — design

**Date:** 2026-07-29
**Status:** Approved for planning
**Scope:** V2 of the four-phase voice programme. V1 (streaming dictation) merged as PR #1085.

## Why

V1 made the Console's Mic button stream dictation through a provider-agnostic
controller. The capture is still entirely hand-operated: starting, stopping,
sending, and recovering from errors all require the keyboard or mouse, and all
feedback is visual. V2 is the accessibility phase: spoken commands while the
microphone is open, and an opt-in spoken-feedback mode so the Console can be
operated without reading the screen.

Chosen scope (user decisions, 2026-07-29): dictation-session commands,
read-back plus spoken status, and the Console command surface. App-wide
navigation was explicitly excluded. Activation is a **prefix word spoken
mid-dictation**; spoken feedback is **opt-in, default off**.

## What already ships (verified on dev at `a2947be90`, not assumed)

- **V1 dictation:** `Chat/console_voice_input.py` — headless
  `ConsoleVoiceInputController` (probe/resolve/state machine, ~70 unit tests),
  driven by `ChatScreen` through `ConsoleStreamingDictationSession` behind the
  shipping `#console-dictation` button. The controller constructs the service
  with `enable_commands=False`; the service's own command machinery
  (`_detect_command`, `Audio/voice_commands.py`) is dormant and stays dormant.
- **Per-message TTS:** task-559 added Speak/Stop actions on Console messages,
  posting `TTSRequestEvent(text, message_id=None, voice=None)` to the app's
  existing synthesis/playback pipeline. The player is a single-slot global
  singleton. `TTSRequestEvent` accepts arbitrary text with no message id, so
  status speech can ride the same pipeline.
- **Command routing:** `ChatScreen._CONSOLE_COMMAND_NAME_TO_HANDLER_ID` maps
  registry command names to handler ids for typed `/`-commands.
- **The four contract tests** in `Tests/UI/test_console_dictation.py` pin the
  shipping button lifecycle, caret insertion without sending, the wall timer,
  and failure recovery. They remain the contract: needing to edit them is a
  design error.

## Hard constraints (verified during design review)

1. **Mid-capture segment finals are dead code today.** `VoiceFinal` fires
   exactly once, at stop: `LazyLiveDictationService._audio_callback` resets
   `last_speech_time` on every delivered chunk, and the recording loop
   delivers every chunk — `use_vad=True` is stored by
   `AudioRecordingService.__init__` and never applied (verified:
   `recording_service.py` recording loop has no VAD gate). Without fixing
   this, a spoken command could not execute until the user manually stopped —
   defeating V2's purpose. Hence Task 0 below.
2. **Speaker-to-microphone echo.** TTS played while the microphone is open is
   captured and transcribed into the draft. V2 has no echo cancellation
   (that is V3 barge-in territory), so **audio output and an open microphone
   are mutually exclusive** by rule.
3. **The prefix word collides with the app's own vocabulary.** Users dictating
   about this application will legitimately start sentences with "Console…".
   The grammar below is designed so that can never fire a command or lose
   prose.
4. **There is no composer undo** (task-1281 open). A false-positive `send`
   ships a half-written draft irreversibly. The grammar must make that
   effectively impossible, not merely unlikely.
5. **Single-slot player:** any new speech stops whatever is playing. A status
   line will cut off an in-flight read-back. Accepted and documented.
6. All V1 invariants carry forward: `probe()` stays `find_spec`-only; the
   subprocess import guard must keep passing; controller events leave via
   `post_message`, never `call_from_thread`; `VoiceFailed` precedes
   `VoiceStateChanged(idle)`; chip text renders via `textual.Content` and
   tests assert on `render_line(0).text`, never `renderable`.

## Task 0 (prerequisite): VAD-gated segment finalization

Make per-segment finals live in `LazyLiveDictationService`:

- Apply the already-requested VAD (webrtcvad, already a declared dependency of
  the `speech_recording` extra) to delivered chunks, and refresh
  `last_speech_time` **only for chunks containing speech**. The existing
  2.0-second silence branch in `_process_audio_buffer` then becomes reachable
  and fires `on_final_transcript` per pause, exactly as its code already
  intends.
- Where VAD is unavailable (import missing), behavior must degrade to today's
  finals-at-stop — never a crash, never a changed default for the three other
  service consumers. In that degraded mode spoken commands still parse but
  cannot execute until the capture ends; in practice this is a non-case, since
  `webrtcvad` ships in the same `speech_recording` extra that gates capture
  itself — but the degraded consequence is stated here so nobody mistakes it
  for a bug later.
- This is compatible with the shipped insertion contract: the Console adapter
  already accumulates `VoiceFinal` segments and inserts once at stop. The four
  contract tests must pass unmodified.
- Independent user benefit: pause-delimited segments improve transcript
  assembly for plain dictation regardless of commands.

The user-facing consequence, stated plainly in docs and help: **pause briefly
before and after a command**. Pauses are what delimit segments; a command is a
segment.

## Grammar (in the controller, headless)

`Chat/console_voice_input.py` gains a command matcher applied to each
finalized segment **before** it is emitted as `VoiceFinal`:

- **Normalization:** lowercase; remove **all** punctuation (leading, trailing,
  and internal); collapse whitespace. This is a deliberate tradeoff, decided
  here so the plan does not relitigate it: recognizers near-universally emit a
  comma after a vocative prefix ("Console, send.") — preserving internal
  punctuation would make the command never match, i.e. the feature would ship
  broken. The cost is that staccato dictated prose ("Console. Send.") that
  finalizes as one segment can false-fire; that is rare, visible (chip ack),
  and a named check in the live verification below.
- **Whole-segment match only.** A segment is a command **iff** the normalized
  segment is exactly `<prefix> <command phrase>` — nothing before, nothing
  after. "Console send button is broken" has trailing words, fails the match,
  and flows to the draft as text. This rule is what makes constraint 3 and 4
  survivable.
- **Fail open to text.** Any segment that is not a whole-segment command match
  — including prefixed-but-unrecognized ones like "console sned" — is emitted
  as ordinary `VoiceFinal` text. Misrecognitions become visible, editable
  words in the draft, never invisible actions and never vanished prose. There
  is no `VoiceCommandUnrecognized` event; YAGNI.
- **Prefix:** config `dictation.command_prefix`, default `"console"`,
  normalized the same way. Multi-word prefixes are permitted by construction.
  A blank or whitespace-only configured prefix falls back to the default —
  an empty prefix would make every segment "prefixed" (the nan-timeout
  lesson: validate config at the seam).
- **Latency, stated honestly:** a command executes ~`silence_threshold`
  (default 2.0 s) after you finish saying it, because the pause is what
  finalizes the segment. Task 0 makes the threshold config-backed
  (`dictation.silence_threshold_seconds`, default 2.0, finite-and-positive
  validated) so users who lean on voice control can shorten it. The full
  choreography — pause, command, pause — costs roughly two thresholds; the
  docs and the F1 help must say so rather than letting it read as lag.
- **Emission:** a matched command emits `VoiceCommand(name: str)` (frozen
  dataclass, same family and threading rules as the existing events) instead
  of `VoiceFinal`. The segment text is consumed — it never reaches the draft.

### Command table (all parameterless, whole-segment)

| Spoken (after prefix) | Kind | Effect |
|---|---|---|
| `new paragraph` | inline | append `\n\n` to the accumulating transcript; capture continues; chip-only ack |
| `new line` | inline | append `\n`; capture continues; chip-only ack |

Inline breaks interact with the adapter's `" ".join(self._segments)`
(`chat_screen.py:824`): a naive break-as-segment yields `"para. \n\n para"`.
The join must be break-aware — no space padding around inserted breaks.
| `stop` | capture-ending | end capture, insert accumulated text at caret (identical to pressing the button) |
| `send` | capture-ending | end capture, insert, then press `#console-send-message` once insertion has completed |
| `discard` | capture-ending | end capture, insert nothing (existing cancel semantics; **no confirmation** — explicit intent, same as pressing cancel) |
| `read that back` | capture-ending | end capture (insert accumulated text), then speak the latest assistant message via the task-559 pipeline, tracking `_console_speaking_message_id` |
| `new session` | capture-ending | end capture (insert), then invoke the existing new-tab action |

`rewind` is cut from V2: it needs arguments to mean anything, it is
destructive, and a spoken confirmation round-trip is impossible under
constraint 2 (the capture that would hear "confirm" has just been closed).

### Capture-outcome correction

A capture whose segments were all consumed as commands has an empty
transcript. Today that raises "No audio was captured from the microphone." —
a false error after a successful command. The capture-outcome logic must
count command-consumed segments as heard-and-handled: no error, no insertion,
no stray whitespace in the draft (the V1 silent-capture rules still apply to
genuinely empty captures).

## Screen routing (thin, in `ChatScreen`)

A small dispatch table maps `VoiceCommand.name` to existing paths:

- `stop`/`discard` → the existing stop/cancel flows.
- `send` → request stop, set a pending-send flag, and press
  `#console-send-message` only after insertion completes (the stop path is
  asynchronous; pressing in the same tick would send a draft missing its last
  segment — the same race V1's review history documents). **The flag is
  cleared without sending if the stop path fails** — a failed dictation must
  never ship the user's message. This ordering is why `VoiceFailed` precedes
  `VoiceStateChanged(idle)` in the first place.
- `new session` → the existing new-tab action (`ctrl+t` path).
- `read that back` → latest assistant message of the active session; post
  `TTSRequestEvent(text=..., message_id=...)`; update
  `_console_speaking_message_id` and resync, mirroring task-559's handler.
  Works regardless of the spoken-feedback toggle. If there is no assistant
  message yet, ack "nothing to read" (spoken if the toggle is on, toast
  otherwise).
- Inline commands (`new paragraph`, `new line`) are handled inside the
  controller/adapter accumulator and never reach the screen table.

Registry-backed commands beyond this table are **not** wired in V2 (`prompt`,
`system`, `skills`, `prefill`, `generate-image` all take arguments;
parameterized spoken commands are out of scope). The dispatch table and
`_CONSOLE_COMMAND_NAME_TO_HANDLER_ID` give V3 the seam.

## Spoken feedback (opt-in)

- Config `dictation.spoken_feedback`, default `false`, read with the existing
  `dictation.*` conventions.
- When on, a thin `_speak_status(text)` posts `TTSRequestEvent(text)` for:
  capture ended, command acknowledgements ("Sent.", "Discarded.",
  "New session."), and dictation errors (the same reason strings the toasts
  carry). **"Capture started" is deliberately NOT spoken** — the microphone is
  already open at that moment, so speaking it would violate the mutual
  exclusion below and transcribe itself into the draft. This is a real gap
  for a screen-free user (they must trust the keypress); a pre-capture earcon
  is the V3-era answer, not more TTS.
- **Mutual exclusion (constraint 2):** `_speak_status` never fires while the
  microphone is open. Inline commands ack via the chip only. Capture-ending
  commands speak *after* the capture has fully closed (state `idle`, recorder
  released). **Starting a capture explicitly stops any in-flight TTS playback
  first** by posting `TTSPlaybackEvent(action="stop")` — the single-slot
  player does NOT do this on its own (it only stops the previous clip when a
  *new clip starts*; opening the microphone plays nothing), so without this
  rule a status ack or read-back still playing at capture start would be
  transcribed straight into the new draft.
- "Read that back" is independent of the toggle (it is an explicit request,
  not ambient feedback).

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `dictation.command_prefix` | `"console"` | prefix word(s) for spoken commands; blank falls back to default |
| `dictation.spoken_feedback` | `false` | speak status/acks/errors via TTS |
| `dictation.silence_threshold_seconds` | `2.0` | pause length that finalizes a segment (and thus fires a command); finite-and-positive validated |

Both live in the established `dictation.*` namespace beside
`buffer_duration_ms`, `max_session_seconds`, `stop_join_timeout_seconds`,
`warm_model_before_capture`.

## Testing

- **Task 0:** hermetic service tests with a fake VAD: speech-gated
  `last_speech_time`, per-pause finals firing, VAD-unavailable degrading to
  finals-at-stop. No real audio, no model downloads. The two real-hardware
  test files stay untouched and unrun.
- **Grammar:** unit tests in `Tests/Chat/test_console_voice_input.py`
  (`pytestmark = pytest.mark.unit`): normalization (case, punctuation),
  whole-segment acceptance, trailing-words rejection, prefixed-typo fail-open,
  prefix configurability, every table entry emitting `VoiceCommand` and
  consuming its text, `VoiceFinal` untouched for plain segments.
- **Routing:** screen tests per command; the send-after-insertion ordering
  pinned the way V1's interleaving tests do; command-only capture produces no
  error and no draft mutation; read-back with no assistant message acks
  instead of failing.
- **Spoken feedback:** toggle off = no `TTSRequestEvent` for status; toggle on
  = events for the enumerated moments and **never** while the mic is open
  (mutual-exclusion test).
- The four contract tests pass unmodified throughout. Mutation-check every
  behavioral change. Any chip-text assertion uses the painted line.
- **Live verification before merge** (the V1 lesson, non-negotiable): a real
  microphone run covering — a command executing mid-capture ("console, stop")
  and its latency feeling acceptable at the default threshold;
  staccato prose ("Console. Send.") checked for the documented false-fire;
  starting a capture while a read-back plays, confirming the playback stops
  and nothing self-transcribes;
  prose beginning with "console" landing in the draft; "console, send"
  shipping the full utterance including the last segment; a command-only
  capture producing no error; spoken feedback audible with the toggle on and
  absent with it off; and no TTS self-transcription.

## Out of scope

App-wide navigation; parameterized commands; wake word / always-listening;
barge-in and echo cancellation; confirmation dialogs; any new TTS machinery;
`Audio/voice_commands.py` (stays dormant — filing its retirement is a
follow-up task, not V2 work).

## Follow-ups to file during V2

- Retire `Audio/voice_commands.py` (stale APP_NAVIGATION actions, no callers
  once V2 ships its own grammar) — or rebind it if app-wide nav (V3+) wants it.
- `AudioRecordingService.stop_recording()` joins its thread for up to 2 s, so
  `abandon()` is not strictly non-blocking today (pre-existing, noted in the
  1283 review).
