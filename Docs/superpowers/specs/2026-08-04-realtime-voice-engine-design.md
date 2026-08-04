# Realtime Voice Engine (voice2voice V4) — Design

**Status:** approved by owner 2026-08-04 (four shaping decisions + two review-round
amendments recorded inline).
**Programme:** phase V4 of the voice2voice programme
(`2026-07-27-console-voice-dictation-design.md`). V1 dictation, V2 voice control,
V3 (streaming PCM sink + hands-free loop) are merged; this phase replaces the
hands-free loop's *engine* with a native realtime speech API while the V3
pipeline remains the fallback.

## What V4 buys

The V3 loop is a half-duplex pipeline: silence gate → whole-segment STT → text
LLM → sentence-chunked TTS, ≈4 s from pause to reply. A realtime API holds one
WebSocket to a speech-native model: audio streams both directions continuously,
the server detects turns, replies begin in well under a second, the model hears
tone rather than a transcript, and interruption is a first-class protocol
operation. Same loop, radically lower latency.

## Owner decisions (2026-08-04)

1. **Same loop, new engine.** The existing hands-free entry/exit/chip UX stays;
   a realtime engine replaces the middle when configured. V3 remains the
   fallback engine.
2. **OpenAI Realtime first**, behind a provider-neutral session protocol.
   Gemini Live is a named follow-up behind the same seam.
3. **Full continuity.** Voice turns land in the Console transcript as ordinary
   messages; entering the loop seeds the realtime session from the session's
   existing history.
4. **No tools in V4.** Pure spoken conversation. Tool/agent integration is its
   own future phase (approval-flow design required).
5. **V3 barge-in semantics.** Keyboard-first, speaker-safe by default;
   `dictation.acoustic_barge_in = true` opts into always-hot mic + server-VAD
   interruption. Same key, same meaning, same headphones warning.
6. **Settings surface** (owner addition): the realtime options are editable in
   the Settings screen's speech area, not config-file-only.

## Architecture: a second controller behind the V3 intent vocabulary

`HandsFreeController` (V3) talks to `ChatScreen` exclusively through typed
intents (`OpenCapture`, `CloseCapture`, `SilenceSpeech`, `ModeChanged`,
`CountdownTick`, `ExitLoop`, …) routed by one dispatcher
(`_handle_console_hands_free_intent`). V4 adds a sibling
**`RealtimeLoopController`** emitting the same vocabulary from a different
internal machine — minus the pipeline-only intents, with chip nuance riding
on new `ModeChanged` values, and with `ExitLoop`/`ModeChanged` gaining an
optional `reason` payload (idle ceiling, reconnect, double-failure) so the
wiring can toast honestly; V3's controller never sets it and its handling is
unchanged:

```
IDLE → CONNECTING → LIVE ─ overlay: reply_active (thinking → speaking)
                     │
                     └→ RECONNECTING (once) → LIVE | exit
```

Screen-side wiring — chip painting, Esc/mic/any-key routing, teardown — stays
shared. Rejected alternatives, recorded: making realtime a provider gateway
under the V3 FSM (forfeits full-duplex — the FSM's cadence *is* the 4 s), and
growing realtime states into `HandsFreeController` (entangles a merged,
live-gated, 51-test machine with semantics that don't apply to it).

### New components

1. **`LLM_Calls/realtime/`** — everything provider-shaped.
   - `transport.py`: WebSocket connect/auth over the `websockets` library
     (new `realtime` pip extra; not a core dependency), close, and the
     single-reconnect policy hook. No business logic.
   - `protocol.py`: the provider-neutral session surface the controller
     consumes. Outbound: `append_audio(frames)`, `send_text_item(text)`,
     `cancel_response(played_ms)` (server-side truncation so the model's
     context matches what was heard), `close()`. Inbound callbacks:
     `on_ready`, `on_audio_delta(bytes)`, `on_reply_started`, `on_reply_done`,
     `on_turn_committed`, `on_input_transcript(text)`,
     `on_output_transcript_delta(text)`, `on_speech_started` (server VAD),
     `on_usage(dict)`, `on_error(exc)`, `on_closed(reason)`.
   - `openai_session.py`: the OpenAI Realtime implementation. Session config
     at connect: pcm16 in/out, input transcription enabled, voice/model from
     config, `turn_detection` server-VAD (always on for turn-taking; the
     *client* gates outbound audio in default barge-in mode).
   - Key resolution: the app's standard chain (`api_settings.openai` →
     `OPENAI_API_KEY` env). No new key surface.
2. **Raw mic-frame tap** on the existing capture stack: continuous pcm16
   mono frames **at the provider's required input rate** — OpenAI Realtime is
   24 kHz both directions, and since realtime mode runs no client-side VAD,
   webrtcvad's 8/16/32/48 kHz constraint does not apply; the recorder opens
   at 24 kHz directly (client-side resampling is the fallback only if a
   device refuses the rate). **No STT is loaded at all** in this mode (the
   model does the hearing), and the tap must neither import the transcription
   stack nor trigger the lazy model load — the programme's oldest trap
   (`Audio/__init__` → transcription imports torch at module scope) applies
   to this new entry point. The device stays open for the whole loop; the
   *stream* is gated client-side per barge-in mode. Frames buffer locally
   (bounded, ~10 s) from entry until `on_ready`, then flush — first words
   during the connect handshake are not lost.
3. **Audio out**: the API's 24 kHz pcm16 deltas feed an async iterator into
   the existing `pump()`/`StreamingPcmSink` (`open(24000, 1)`). Barge-in
   aborts in ≤2 blocks exactly as V3 does. Played-bytes accounting supplies
   `played_ms` for truncation.

### Engine selection, entry, exit

```toml
[realtime]
enabled = true                # the switch (default false; absence = false)
provider = "openai"           # only value in V4
# model = "gpt-realtime"      # optional override
# voice = "marin"             # optional
idle_timeout_minutes = 5      # cost protection; see Lifecycle

[dictation]
handsfree_engine = "auto"     # "auto" | "pipeline" | "realtime"
```

- Explicit `enabled = true` is the opt-in — never inferred from key presence
  (the TASK-2110 lesson: no silent engine substitution). A Settings-screen
  toggle maps 1:1 onto it.
- `handsfree_engine`: `auto` (realtime iff enabled) / `pipeline` (force V3
  without deleting config) / `realtime` (force; honest toast if not enabled).
- Entry and exit are V3's: `Ctrl+Shift+H`, spoken "Console, hands free."
  (classified by the *pipeline* capture that is open at that moment), Esc, mic
  button. Entry from a live capture: the capture stops and transcribes
  normally; that text is sent as the realtime session's first turn
  (`send_text_item` + response request) — nothing dictated is lost. If the
  user has already begun speaking to the live session when the adopted text
  arrives, items append in server order; the minor interleaving is accepted.
- **Connect failure falls back loudly and only to a viable pipeline**: if the
  WS/handshake fails, and the V3 speech stack is viable (STT available, VAD
  not degraded), enter the pipeline loop with a toast naming the realtime
  failure. If the pipeline is not viable either, toast both reasons and do
  not enter. Never a dead entry; never a silent substitution.
- **Honest UX difference, documented in guide + chip copy:** spoken commands
  do not exist inside realtime mode — no client STT is running, so
  "Console, stop." cannot be classified. Exits are key/mouse only.
- **Privacy honesty, documented in the guide's privacy section:** in realtime
  mode, microphone audio streams to the provider continuously for the whole
  live session (subject to barge-in gating) — not just after a silence gate
  as in the pipeline engine.

### Continuity

- **Seeding:** at connect (and re-connect), prior session turns are sent as
  text conversation items, newest-first retained under a fixed budget
  (constant in V4: last 20 turns / ≈8 k chars; configurability is a named
  follow-up), plus the session's system prompt via session instructions.
- **Turns out:**
  - User speech: the user-message row is created at `on_turn_committed` with
    a pending placeholder and its text fills on `on_input_transcript` —
    ordering in the pane stays correct even though the server's transcription
    arrives after the reply has begun (review amendment #2).
  - Model speech: `on_output_transcript_delta` streams into an assistant
    message through the store's existing streaming-append path. Turns carry
    engine metadata (`engine="realtime"`, provider, model).
- **Interruption honesty (review amendment #1):** precise text-at-cut
  trimming is not implementable — transcript streams ahead of audio and no
  ms→chars mapping exists; the API truncates its own context but returns no
  corrected transcript. The stored assistant message therefore keeps the full
  streamed text **marked interrupted** (metadata + visible marker). Documented
  divergence: the model remembers less than the transcript shows, and
  post-reconnect reseeding sends slightly more than the model previously had.
- **Turn metadata deferred (task 5, fix round 1 / F4):** the per-turn
  `engine`/`provider`/`model` and `interrupted` metadata above has no field to
  live in — `ConsoleChatStore`'s message row exposes no metadata column, and
  adding one is a schema change well outside a wiring task. Deferred pending
  that field; today the provenance a reader actually needs is carried by the
  visible `⏹ interrupted` marker on the cut reply and by the usage attached to
  the row (recorded against the realtime provider/model).
- Accepted oddity: reply text finishes rendering before the voice finishes
  speaking; the chip's `speaking` state carries the truth.

### Barge-in

- **Default (speaker-safe):** outbound mic stream gated while a reply is
  active (device open, frames dropped client-side; the server sees silence).
  Any keypress → sink abort (≤2 blocks) → `cancel_response(played_ms)` → mic
  ungates. Sub-second interruption with V3's any-key discipline.
- **Acoustic (`dictation.acoustic_barge_in = true`):** mic never gates;
  `on_speech_started` during a reply mirrors the same local cut. Same config
  key and headphones warning as V3.
- **Wiring rule (review amendment #3):** the V3 store tap
  (`_install_console_hands_free_store_tap`) is pipeline-engine-only. In
  realtime mode the engine writes transcript deltas through the same store
  method the tap watches; installing the tap would drive the V3 sentence
  sequencer into TTS-ing a reply the API is already speaking. Engine
  selection gates the install; a test pins it.

### Chip states

`connecting…` → ordinary recording indicator (live/listening) →
`hands-free · thinking…` (turn committed, no audio yet) →
`hands-free · speaking` → recording again. All via `ModeChanged` values; no
new UI surfaces.

### Lifecycle, errors, cost

- Ends by: user exit; transport drop; provider session limit (~30 min).
- Unexpected drop mid-loop: **one** automatic reconnect (fresh WS, re-seed
  from store, toast "reconnected"). A second failure in the same loop entry
  exits loudly with the reason. No infinite retry; no silent death.
- **Idle ceiling:** no activity — user turn commit or reply-audio end,
  whichever is later — for `idle_timeout_minutes` (default 5) → exit the
  loop with a toast naming the reason. Never fires while a reply is active
  (a long reply must not be cut mid-sentence by the cost guard). Unattended
  sessions must not bill indefinitely.
- `on_usage` per-response token counts are recorded onto turn metadata.
  Feeding the Console cost chip is a named follow-up task, not V4 scope.

### Settings screen

A Realtime block in the existing Settings speech area: enable toggle,
provider dropdown (OpenAI only), optional model/voice fields, engine
override, idle timeout. Writes through the existing settings mutation path —
no second config writer (TASK-2111 adjacency).

## Testing

- **Session layer:** local fake WebSocket server replaying recorded protocol
  scripts — connect/handshake, audio+transcript interleavings, truncation,
  usage, drop-and-reconnect. No network, no key, deterministic.
- **`RealtimeLoopController`:** headless FSM tests in the V3 style — typed
  intents pinned, injected clock, no wall-time.
- **Wiring:** fake realtime session behind the same ChatScreen harness
  patterns as V3's suites (entry/fallback/chips/barge-in/tap-gating/
  continuity ordering).
- **Contract:** the V3 pipeline suites stay untouched and byte-identical —
  the engine seam's proof.
- **Live gate:** first by the agent with the repo-root keys, then the owner
  on real hardware: connect, first-words buffering, sub-second turn, both
  barge-in modes, continuity both directions, reconnect, idle ceiling.

## Out of scope

Gemini Live (follow-up behind the same protocol seam), tool calling, WebRTC
transport, wake word, per-session voice picker (config-only voice), cost-chip
integration (follow-up task), any change to V3 pipeline behavior.
