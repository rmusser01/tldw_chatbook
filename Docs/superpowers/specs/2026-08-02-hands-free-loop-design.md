# Hands-Free Conversation Loop — design

**Date:** 2026-08-02
**Programme:** Console voice2voice, V3 phase 2 of 2. V1 = PR #1085, V2 = PR #1171,
V3 phase 1 (streaming PCM sink) = PR #1203 (`d9cb4fcfe`).
**Decided with the user:** keyboard-first barge-in with acoustic opt-in; two-stage silence
auto-send with a visible countdown; sentence-chunked reply speech; mode entered by spoken
command/shortcut, exited by "console stop"/Esc/mic press.

## Why

Dictation (V1), spoken commands (V2), and interruptible playback (phase 1) exist; nothing
yet closes the loop: speak → it sends → the reply is spoken → speak again. This phase adds
that loop as composition of the existing pieces, with the minimum new machinery: a state
machine, a sentence sequencer, one new dictation event, one new speech-path entry, and one
playback-completion signal.

## Decisions that bound everything below

- **No AEC exists.** Default barge-in is keyboard/mic-press. Acoustic barge-in is an
  opt-in config (`dictation.acoustic_barge_in`, default `false`) documented
  "headphones recommended" — on speakers the recognizer transcribes the assistant's own
  voice.
- **Send = V2's machinery, not a new path.** Under segment-at-silence the draft is EMPTY
  during a live capture (segments live in the session until `stop_and_transcribe`). The
  countdown's expiry intent is `RequestStopAndSend`, which drives exactly the existing
  `_console_pending_voice_action = "send"` flow (stop → transcribe → insert → dispatch),
  live-verified in V2. No second send path.
- **Countdown cancellation needs a signal that does not exist yet.** Mid-segment partials
  were deliberately removed; raw VAD frames never leave the recorder. New event:
  **`VoiceSpeechResumed`**, emitted once per silence→speech transition — the transition
  the recorder ALREADY detects (the VAD pre-roll flush keys on it,
  `recording_service._process_audio_chunk`). Flows through the existing dictation event
  plumbing with the capture-generation token. It is also the acoustic barge-in trigger.
- **Sequential reply speech must not ride the ad-hoc cooldown.**
  `handle_tts_request` runs `_enforce_cooldown_limit()`; sentence N+1 seconds after
  sentence N is exactly what it throttles. The sequencer uses a dedicated internal entry
  that shares the generation/playback path but not the ad-hoc cooldown (same class of
  seam the message-speech path already has). Exact seam pinned at plan time.
- **Honest timing:** pause-to-send ≈ silence gate (2.0 s) + segment transcription
  (~0.3–1 s parakeet) + countdown (default 1.5 s,
  `dictation.handsfree_send_delay_seconds`). Cancellable until expiry.

## Architecture

```
dictation events ──► HandsFreeController ──intents──► ChatScreen wiring
(VoiceFinal/Command/     (headless FSM)                (send via V2 flow, capture
 SpeechResumed/state)         │                         open/close, chip states)
reply lifecycle ─────────────►│
(delta, complete, failed)     └──► SentenceSequencer ──► internal speech entry ──► sink
user interrupts ─────────────►         (headless)         (no ad-hoc cooldown)     or legacy
(keypress, mic, commands)                                       ▲                  player
                                        utterance-completion ───┘  (both paths signal)
```

### `Chat/console_hands_free.py` — `HandsFreeController` (headless FSM)

States: `IDLE → LISTENING → COUNTDOWN → AWAITING_REPLY → SPEAKING → LISTENING…`;
`exit` reachable from every state.

Inputs (all existing events plus the two new ones): `VoiceFinal`, `VoiceCommand`,
`VoiceSpeechResumed`, dictation state changes/failures; reply lifecycle (text delta,
completed, failed); interrupts (composer keypress, mic press, Esc, "console stop").

Outputs, via injected emit (the `ConsoleVoiceInputController` pattern):
`RequestStopAndSend`, `SilenceSpeech`, `OpenCapture`, `CloseCapture`,
`CountdownTick(remaining)`, `ModeChanged(state)`, `ExitLoop`.

Transition rules:
- `LISTENING` + `VoiceFinal` → `COUNTDOWN` (armed with the finalized-segment fact only;
  the text stays in the session).
- `COUNTDOWN` + `VoiceSpeechResumed` or any composer keypress → `LISTENING` (cancelled).
- `COUNTDOWN` expiry → emit `RequestStopAndSend` → `AWAITING_REPLY`; capture closes
  (mic/speaker exclusion) unless acoustic opt-in.
- `AWAITING_REPLY` + first speakable sentence queued → `SPEAKING`.
- Reply completed AND sequencer drained → `OpenCapture` → `LISTENING`. A reply with zero
  speakable sentences (empty, whitespace, pure code) short-circuits the same way.
- `SPEAKING` + interrupt (keypress/mic) or, with acoustic opt-in, `VoiceSpeechResumed`
  → `SilenceSpeech` + sequencer flush → `LISTENING`. Generation is NOT cancelled — the
  reply finishes silently into the transcript; only audio stops. With acoustic opt-in the
  interrupting segment lands in the next draft as normal capture.
- V2 commands keep working mid-loop; `stop` (spoken), Esc, or mic press → `ExitLoop`
  from any state, tearing down to today's idle behavior.
- Precedence: Esc / mic press / spoken `stop` = exit; every other composer key =
  barge-in-and-listen (in `SPEAKING`/`COUNTDOWN`) or ordinary typing (the controller
  never swallows keys outside those states).

Entry: new grammar phrase `hands free` (→ command name `hands-free`) in
`COMMAND_PHRASES`, plus a key binding on the screen. Entering from a live capture keeps
that capture as the first turn; entering from idle opens one.

### `Chat/reply_sentence_sequencer.py` — `SentenceSequencer` (headless)

- Consumes reply text deltas; maintains a boundary buffer; emits utterances at sentence
  boundaries (`.` `!` `?` and newline), with a minimum-length guard (abbreviations,
  decimals — "Dr.", "3.14" do not chop) and a maximum-length force-split.
- **Speakable-text normalization:** strips markdown syntax (emphasis, links → link text,
  headings), skips fenced code blocks entirely, collapses whitespace. A sentence that
  normalizes to nothing is dropped.
- Hands exactly ONE utterance at a time to the speech entry; advances on
  utterance-completion; never displaces its own previous sentence (sequential waits, not
  one-voice displacement — displacement remains the cross-utterance safety net only).
- `flush()` on barge-in: clears the queue, abandons the in-flight utterance via the
  speech entry's stop, emits drained.
- On reply-completed: emits the final partial sentence (if any) as the last utterance.

### Reply-delta tap

The transcript renders streamed replies live, so a delta seam exists; the spec requires
the plan to NAME the verified seam (expected: the console agent bridge / store append
path) and wire the sequencer as a read-only subscriber with the assistant message id as
the correlation key. Fallback ONLY if live deltas are genuinely inaccessible: per-message
speak-on-complete (explicitly a degradation, not the design).

### Utterance-completion signal (the one playback edit)

- Sink path: already observable (`pump` result / terminal reason).
- Legacy file path: `play_audio_file`'s worker gains a completion callback (and failure
  signal on the error path), surfaced as one event. Additive; existing callers unchanged.

### ChatScreen wiring (thin)

- Grammar entry + key binding; mode state on the existing voice chip
  (`hands-free · listening`, `sending in 1.5s…` countdown from `CountdownTick`,
  `thinking…` for `AWAITING_REPLY`, `speaking`).
- Keypress hook scoped to the states that need it; mic button exits the loop.
- Capture open/close intents drive the existing dictation start/stop machinery with the
  existing capture-generation guards; `AWAITING_REPLY`/`SPEAKING` keep the mic closed by
  default (the V2 exclusion rule), open under acoustic opt-in.
- No new config keys beyond `dictation.handsfree_send_delay_seconds` and
  `dictation.acoustic_barge_in`.

## Error handling

- Reply generation fails → existing error surface; loop → `LISTENING` (never traps).
- One utterance's synthesis/playback fails → skip it, log once, continue the queue.
- Capture fails to reopen → `ExitLoop` through the standard dictation error surface.
- Unmount/shutdown → V2-style abandon teardown; capture-generation tokens drop stale
  events; the sequencer's flush guarantees no orphaned utterance (and the sink's
  terminal-call rule holds — every opened sink reaches a terminal call).
- Countdown expiry racing a simultaneous `VoiceSpeechResumed`: the controller resolves
  by arrival order at the FSM; a resume that loses the race rides into the NEXT capture
  (opened for the following turn) rather than being lost.

## Testing

- **Controller:** pure FSM tests, scripted sequences: every transition; every interrupt
  in every state; countdown cancel races (resume-vs-expiry both orders); empty-draft
  expiry; zero-sentence replies; exit from each state. Mutation-checked.
- **Sequencer:** boundary cases (abbreviations, decimals, ellipses, newline-heavy text,
  fenced code skipped, markdown stripped, max-length force-split); one-at-a-time
  discipline (completion-gated advance); flush mid-utterance.
- **`VoiceSpeechResumed`:** recorder-level test on the silence→speech transition
  (extends the pre-roll suite); generation-gated delivery like every dictation event.
- **Speech entry:** cooldown-free sequential utterances (pin: N sentences, zero
  throttling); shares generation path (no duplicate synthesis logic).
- **Completion signal:** both paths (fake sink, fake player worker), including failure.
- **Wiring:** real-app harness — grammar entry, chip states painted (CSS-true harness),
  keypress barge-in routing, two-stage send driving the real V2 send flow with a stub
  gateway.
- **Live gate (human):** full loop on real hardware — speak, pause, watch countdown,
  auto-send, hear the reply begin sentence-by-sentence, keyboard barge-in mid-reply,
  speak the next turn; acoustic mode checked with headphones.

## Out of scope

AEC; wake-word; TASK-1880 (provider streaming unlock — the loop works today via
per-sentence legacy synthesis; audio.cpp streams); V4 realtime; Settings UI; speaking
replies for non-hands-free sends (the existing per-message speak affordance is
untouched).
