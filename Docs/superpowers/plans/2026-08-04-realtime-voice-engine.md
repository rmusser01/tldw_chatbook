# Realtime Voice Engine (V4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hands-free loop's engine with an OpenAI Realtime WebSocket session (audio in/out, sub-second turns) behind a provider-neutral seam, with the V3 pipeline as the loudly-selected fallback.

**Architecture:** A sibling `RealtimeLoopController` emits the V3 intent vocabulary (`ModeChanged`/`ExitLoop` gain an optional `reason`); under it sit three new units — `LLM_Calls/realtime/` (transport + provider session), a raw 24 kHz mic tap with pre-ready buffering, and the existing `StreamingPcmSink` fed by WS audio deltas. Continuity flows both ways through `ConsoleChatStore`.

**Tech Stack:** Python 3.11+, Textual, `websockets` (new optional extra `realtime`), existing `AudioRecordingService` + `StreamingPcmSink`/`pump`.

**Spec:** `Docs/superpowers/specs/2026-08-04-realtime-voice-engine-design.md` — binding; read it first.

## Global Constraints

- Branch `feat/realtime-voice-v4` in worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/voice-control-v2`. Never `git stash`; never `git checkout --` a file with uncommitted work.
- Foreground pytest only: `./.venv/bin/python -m pytest <files> -p no:randomly -q` from the worktree root. No background runs.
- **Contract files (byte-identical across the whole branch):** `Tests/UI/test_console_dictation.py`, `Tests/Chat/test_console_hands_free.py`, `Tests/UI/test_console_hands_free_wiring.py`. The V3 engine must be provably untouched.
- `[realtime] enabled = true` is the only opt-in — never infer from key presence (TASK-2110).
- New config readers follow `console_voice_input.py:881`'s sibling-validation shape (invalid → log + default, never raise). Remember the `get_cli_setting` dotted-form trap: pass section and key separately (`get_cli_setting("realtime", "enabled", False)`).
- OpenAI Realtime audio is **24 kHz pcm16 mono both directions**. No client VAD runs in realtime mode.
- The mic tap must not import the transcription stack nor trigger lazy model load (`Audio/__init__` → torch trap).
- Commits: conventional style, ending `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- OpenAI Realtime event names drift between beta and GA. Task 2 pins them with a **live probe first** (repo-root `openai-api-key.txt` is for agent use); the fake server encodes what the probe observed.

---

### Task 1: Config readers, engine selection logic, and the realtime protocol module

**Files:**
- Create: `tldw_chatbook/LLM_Calls/realtime/__init__.py`
- Create: `tldw_chatbook/LLM_Calls/realtime/protocol.py`
- Modify: `tldw_chatbook/Chat/console_voice_input.py` (append readers after `acoustic_barge_in_enabled`, ~line 930)
- Modify: `pyproject.toml` (add `realtime = ["websockets>=12.0"]` to `[project.optional-dependencies]`)
- Test: `Tests/Chat/test_realtime_config_readers.py`, `Tests/LLM_Calls/test_realtime_protocol.py`

**Interfaces:**
- Produces (readers in `console_voice_input.py`):
  - `realtime_enabled() -> bool` (default False)
  - `realtime_provider() -> str` (default `"openai"`)
  - `realtime_model() -> str` (default `"gpt-realtime"`)
  - `realtime_voice() -> str | None` (default None = server default)
  - `realtime_idle_timeout_seconds() -> float` (from `realtime.idle_timeout_minutes`, default 5 minutes → returns 300.0; non-numeric/non-positive → default)
  - `handsfree_engine() -> str` (from `dictation.handsfree_engine`; one of `"auto" | "pipeline" | "realtime"`, anything else → `"auto"` with a log line)
  - `resolve_handsfree_engine() -> str` — pure combination: `"pipeline"` unless (`handsfree_engine()=="realtime"`) or (`handsfree_engine()=="auto"` and `realtime_enabled()`); returns `"realtime"` in those cases. Forcing `"realtime"` while not `realtime_enabled()` still returns `"realtime"` — the wiring toasts and refuses there (it needs the distinction to be honest).
- Produces (`protocol.py`):

```python
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

@dataclass(frozen=True)
class RealtimeSessionConfig:
    api_key: str
    model: str
    voice: str | None = None
    input_sample_rate: int = 24000
    output_sample_rate: int = 24000
    instructions: str | None = None

@dataclass
class RealtimeCallbacks:
    """Mutable bundle; all optional. Fired from the session's receive loop
    thread/task — consumers must be thread-safe or trampoline themselves."""
    on_ready: Callable[[], None] | None = None
    on_audio_delta: Callable[[bytes], None] | None = None
    on_reply_started: Callable[[str], None] | None = None      # assistant item_id
    on_first_audio: Callable[[], None] | None = None
    on_reply_done: Callable[[], None] | None = None
    on_turn_committed: Callable[[], None] | None = None
    on_input_transcript: Callable[[str], None] | None = None
    on_output_transcript_delta: Callable[[str], None] | None = None
    on_speech_started: Callable[[], None] | None = None
    on_usage: Callable[[dict], None] | None = None
    on_error: Callable[[Exception], None] | None = None
    on_closed: Callable[[str], None] | None = None             # reason string

class RealtimeSession(Protocol):
    async def connect(self) -> None: ...
    def append_audio(self, frames: bytes) -> None: ...
    def send_seed(self, items: list[tuple[str, str]], instructions: str | None) -> None: ...
    def send_text_item(self, text: str, *, request_response: bool) -> None: ...
    def cancel_response(self, played_ms: int) -> None: ...
    async def close(self) -> None: ...
```

- [ ] **Step 1: Write the failing reader tests** — `Tests/Chat/test_realtime_config_readers.py`. Follow the exact monkeypatch style of `Tests/UI/test_console_hands_free_wiring.py:83` (`_spy_get_cli_setting`): patch `tldw_chatbook.Chat.console_voice_input.get_cli_setting` and assert each reader passes the exact `(section, key, default)` triple and applies the validation:

```python
import tldw_chatbook.Chat.console_voice_input as cvi

def _patch_setting(monkeypatch, mapping):
    calls = []
    def fake(section, key, default=None):
        calls.append((section, key, default))
        return mapping.get((section, key), default)
    monkeypatch.setattr(cvi, "get_cli_setting", fake)
    return calls

def test_realtime_enabled_reads_exact_key_and_defaults_false(monkeypatch):
    calls = _patch_setting(monkeypatch, {})
    assert cvi.realtime_enabled() is False
    assert ("realtime", "enabled", False) in calls

def test_realtime_enabled_accepts_truthy_string(monkeypatch):
    _patch_setting(monkeypatch, {("realtime", "enabled"): "true"})
    assert cvi.realtime_enabled() is True

def test_idle_timeout_converts_minutes_and_rejects_nonpositive(monkeypatch):
    _patch_setting(monkeypatch, {("realtime", "idle_timeout_minutes"): 2})
    assert cvi.realtime_idle_timeout_seconds() == 120.0
    _patch_setting(monkeypatch, {("realtime", "idle_timeout_minutes"): -3})
    assert cvi.realtime_idle_timeout_seconds() == 300.0
    _patch_setting(monkeypatch, {("realtime", "idle_timeout_minutes"): "soon"})
    assert cvi.realtime_idle_timeout_seconds() == 300.0

def test_handsfree_engine_rejects_unknown_values(monkeypatch):
    _patch_setting(monkeypatch, {("dictation", "handsfree_engine"): "hyperspace"})
    assert cvi.handsfree_engine() == "auto"

def test_resolve_engine_matrix(monkeypatch):
    for engine, enabled, expect in [
        ("auto", False, "pipeline"), ("auto", True, "realtime"),
        ("pipeline", True, "pipeline"), ("realtime", False, "realtime"),
    ]:
        _patch_setting(monkeypatch, {
            ("dictation", "handsfree_engine"): engine,
            ("realtime", "enabled"): enabled,
        })
        assert cvi.resolve_handsfree_engine() == expect, (engine, enabled)
```

Also `Tests/LLM_Calls/test_realtime_protocol.py`: construct `RealtimeSessionConfig` (frozen, defaults 24000/24000), `RealtimeCallbacks()` (all None), and assert `RealtimeSession` is a `typing.Protocol` (`from typing import get_protocol_members` is 3.13+; instead assert `issubclass` behavior via a minimal conforming stub).

- [ ] **Step 2: Run to verify failure** — `./.venv/bin/python -m pytest Tests/Chat/test_realtime_config_readers.py Tests/LLM_Calls/test_realtime_protocol.py -p no:randomly -q` → FAIL (attributes missing).
- [ ] **Step 3: Implement** — readers in `console_voice_input.py` mirroring `handsfree_send_delay_seconds`'s docstring/validation shape; `protocol.py` exactly as the Interfaces block; `realtime/__init__.py` re-exports names lazily (module must import in <50 ms — no websockets import at package import time; `transport`/session import websockets inside their own modules only). Add the pyproject extra.
- [ ] **Step 4: Run to green**, plus `./.venv/bin/python -c "import time; t=time.monotonic(); import tldw_chatbook.LLM_Calls.realtime; print(time.monotonic()-t)"` — confirm no heavy import.
- [ ] **Step 5: Commit** — `feat(realtime): config readers, engine resolution, provider-neutral session protocol`.

---

### Task 2: WebSocket transport + OpenAI Realtime session (fake server + live probe)

**Files:**
- Create: `tldw_chatbook/LLM_Calls/realtime/transport.py`
- Create: `tldw_chatbook/LLM_Calls/realtime/openai_session.py`
- Test: `Tests/LLM_Calls/test_openai_realtime_session.py`

**Interfaces:**
- Consumes: Task 1's `RealtimeSessionConfig`, `RealtimeCallbacks`, `RealtimeSession` protocol.
- Produces: `OpenAIRealtimeSession(config: RealtimeSessionConfig, callbacks: RealtimeCallbacks, *, url: str | None = None)` implementing `RealtimeSession`. `url` override exists solely so tests point at the fake server; production default is the OpenAI realtime endpoint with `?model=<config.model>`. `transport.py` exposes `class WsTransport` with `async connect(url, headers) -> None`, `async send_json(obj) -> None`, `async recv_loop(on_event: Callable[[dict], None]) -> str` (returns close reason), `async close() -> None` — no OpenAI knowledge in it.

- [ ] **Step 1: LIVE PROBE FIRST (event-name ground truth).** Write `Tests/LLM_Calls/openai_realtime_probe.py` (a script, not a test): read the key from `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/openai-api-key.txt`, connect with `websockets` to the realtime endpoint for `gpt-realtime`, send `session.update` (audio+text modalities, pcm16, input transcription on, server VAD on), then a text `conversation.item.create` + `response.create`, and print every received event's `type` field until `response.done`. Run it once: `./.venv/bin/python Tests/LLM_Calls/openai_realtime_probe.py`. Record the observed event-type names in a comment block at the top of `openai_session.py`. **The names below are the expected GA set — if the probe disagrees, the probe wins everywhere:** `session.created`, `session.updated`, `input_audio_buffer.speech_started`, `input_audio_buffer.speech_stopped`, `input_audio_buffer.committed`, `conversation.item.input_audio_transcription.completed`, `response.created`, `response.output_item.added`, `response.output_audio.delta`, `response.output_audio_transcript.delta`, `response.done`, `error`.
- [ ] **Step 2: Write the failing fake-server tests.** Fixture: `websockets.serve` on `127.0.0.1:0`; the handler runs a per-test script list of `("expect", predicate)` / `("send", event_dict)` steps. Core tests (all use `asyncio` marker):

```python
async def test_connect_sends_session_update_and_fires_ready(fake_server):
    # script: expect type=="session.update" with audio pcm16 + transcription
    # + server VAD; send session.updated
    ...
    assert fired["ready"] == 1

async def test_append_audio_base64_roundtrip(fake_server):
    session.append_audio(b"\x01\x02" * 480)
    # server asserts input_audio_buffer.append with base64 payload decoding
    # back to the exact bytes

async def test_audio_delta_decodes_to_bytes_and_first_audio_fires_once(fake_server):
    # send two response.output_audio.delta events; on_audio_delta gets raw
    # bytes twice, on_first_audio exactly once, on_reply_started carried the
    # assistant item_id from response.output_item.added

async def test_transcripts_route_to_both_callbacks(fake_server): ...
async def test_speech_started_fires_during_active_response(fake_server): ...

async def test_cancel_response_sends_cancel_then_truncate_with_played_ms(fake_server):
    session.cancel_response(1234)
    # server asserts response.cancel, then conversation.item.truncate with
    # audio_end_ms == 1234 and the CURRENT assistant item_id

async def test_send_seed_creates_items_in_order_without_response(fake_server): ...
async def test_send_text_item_with_request_response_true_sends_response_create(fake_server): ...
async def test_response_done_fires_reply_done_and_usage(fake_server): ...
async def test_server_close_fires_on_closed_with_reason(fake_server): ...
async def test_error_event_routes_to_on_error_not_crash(fake_server): ...
```

- [ ] **Step 3: RED run.**
- [ ] **Step 4: Implement `transport.py` then `openai_session.py`.** Session owns: an asyncio task running `recv_loop`; outbound helpers building the exact event dicts; `append_audio` is sync + thread-safe (`loop.call_soon_threadsafe` posting onto an outbound queue — the mic tap calls it from the recorder thread); item-id tracking from `response.output_item.added` for truncation; base64 both directions. `close()` idempotent; every callback invocation isolated (`try/except` → `on_error`), and per the Global Constraints, any new error log carries operation + context.
- [ ] **Step 5: GREEN run** (fake-server suite), plus rerun the live probe once more to confirm the implementation's `session.update` is accepted live (probe prints `session.updated`).
- [ ] **Step 6: Commit** — `feat(realtime): WebSocket transport and OpenAI Realtime session with scripted fake-server suite`.

---

### Task 3: Raw 24 kHz mic tap with pre-ready buffering and gating

**Files:**
- Create: `tldw_chatbook/Audio/realtime_mic_tap.py`
- Test: `Tests/Audio/test_realtime_mic_tap.py`

**Interfaces:**
- Consumes: `Audio/recording_service.py`'s `AudioRecordingService(backend=None, sample_rate=..., channels=1, use_vad=False)` + `start_recording(callback=cb)` / `stop_recording()` (verified anchors :144/:356).
- Produces:

```python
class RealtimeMicTap:
    def __init__(self, on_frames: Callable[[bytes], None], *,
                 sample_rate: int = 24000,
                 recorder_factory: Callable[..., Any] | None = None,
                 max_buffer_seconds: float = 10.0): ...
    def start(self) -> bool: ...      # False + logged reason on device failure
    def mark_ready(self) -> None: ... # flush buffer, then stream live
    def set_gated(self, gated: bool) -> None: ...
    def stop(self) -> None: ...
```

Semantics: before `mark_ready()`, frames accumulate in a bounded deque (`max_buffer_seconds * sample_rate * 2` bytes; oldest dropped); `mark_ready()` flushes buffered frames to `on_frames` in order, then subsequent frames stream directly. `set_gated(True)` drops frames (device stays open). After `stop()`, no callbacks fire. `recorder_factory` defaults to `AudioRecordingService` and exists so tests inject a fake.

- [ ] **Step 1: Failing tests** — with a fake recorder (records constructor kwargs; exposes the captured callback so tests push frames): buffering order + flush-on-ready; bound eviction (push > max, assert oldest dropped, newest kept); gating drops; no-callbacks-after-stop (push after stop → nothing); constructor kwargs pinned (`sample_rate=24000, channels=1, use_vad=False`); `start()` returning False when the fake's `start_recording` returns False. Plus the **import-lightness test**: run `./.venv/bin/python -c "import sys, tldw_chatbook.Audio.realtime_mic_tap; assert 'faster_whisper' not in sys.modules; assert 'torch' not in sys.modules; assert 'nemo' not in str(sys.modules.keys())"` via `subprocess` inside a test.
- [ ] **Step 2: RED.** Step 3: implement (lock-guarded state; the recorder callback runs on the recording thread — `on_frames` is documented as called from that thread). Step 4: GREEN. Step 5: Commit — `feat(realtime): raw 24kHz mic tap with pre-ready buffering and gating`.

---

### Task 4: RealtimeLoopController FSM

**Files:**
- Create: `tldw_chatbook/Chat/console_realtime_loop.py`
- Modify: `tldw_chatbook/Chat/console_hands_free.py` — `ModeChanged` gains `reason: str | None = None`; `ExitLoop` gains `reason: str | None = None` (frozen dataclasses, additive defaults; V3 controller never sets them). `ModeChanged.state`'s annotation widens to `HandsFreeState | RealtimeLoopState`.
- Test: `Tests/Chat/test_console_realtime_loop.py`

**Interfaces:**
- Consumes: intent dataclasses from `console_hands_free.py` (`ModeChanged`, `ExitLoop`, `SilenceSpeech`).
- Produces:

```python
RealtimeLoopState = Literal["idle", "connecting", "live", "thinking",
                            "speaking", "reconnecting"]

class RealtimeLoopController:
    def __init__(self, emit: Callable[[object], None], *,
                 acoustic_barge_in: bool,
                 idle_timeout_seconds: float): ...
    state: RealtimeLoopState
    mic_gated: bool          # wiring syncs the tap on every ModeChanged
    # inputs (all called by wiring):
    def enter(self) -> None                      # idle -> connecting
    def on_session_ready(self) -> None           # connecting/reconnecting -> live
    def on_connect_failed(self) -> None          # -> ExitLoop(reason="connect-failed")
    def on_turn_committed(self, now: float) -> None   # live -> thinking
    def on_reply_started(self) -> None           # (no state change; disarms nothing in V4)
    def on_first_audio(self) -> None             # thinking -> speaking
    def on_reply_done(self, now: float) -> None  # thinking|speaking -> live
    def on_speech_started(self) -> None          # acoustic barge-in mid-reply
    def on_keypress(self) -> None                # keyboard barge-in mid-reply
    def on_transport_closed(self, *, error: bool) -> None
    def on_exit_request(self) -> None            # any state -> ExitLoop
    def tick(self, now: float) -> None           # idle-ceiling clock
```

Rules (each is a test): barge-in (either kind, only while `thinking|speaking`) emits `SilenceSpeech` then `ModeChanged("live")` — the wiring's realtime silence handler both aborts the sink and calls `cancel_response(played_ms)`; in default mode `mic_gated` is True exactly during `thinking|speaking`, in acoustic mode always False; `on_transport_closed(error=True)` from `live|thinking|speaking` transitions to `reconnecting` and emits `ModeChanged("reconnecting", reason="reconnecting")` the FIRST time, `ExitLoop(reason="connection-lost")` the second time within one loop entry; `on_transport_closed(error=False)` (our own close) is a no-op after exit; idle ceiling: `tick(now)` emits `ExitLoop(reason="idle-timeout")` only when `state == "live"` and `now - last_activity >= idle_timeout_seconds`, where `last_activity` updates on `enter`, `on_session_ready`, `on_turn_committed`, and `on_reply_done` — it can never fire in `thinking|speaking` (a long reply is not activity-starved); every state reaches `ExitLoop` via `on_exit_request`; all emissions go through one `_transition()` chokepoint mirroring V3's.

- [ ] **Step 1: Failing FSM suite** (~22 tests; verbatim style of `Tests/Chat/test_console_hands_free.py` — construct with a recording `emit`, drive inputs, assert intent sequences and state):

```python
def _make(acoustic=False, idle=300.0):
    intents = []
    c = RealtimeLoopController(intents.append, acoustic_barge_in=acoustic,
                               idle_timeout_seconds=idle)
    return c, intents

def test_enter_connect_ready_reaches_live_with_mode_intents(): ...
def test_connect_failed_exits_with_reason(): ...
def test_turn_committed_thinking_first_audio_speaking_done_live(): ...
def test_keypress_mid_speaking_emits_silence_then_live(): ...
def test_keypress_while_live_is_a_noop(): ...
def test_speech_started_barges_only_in_acoustic_mode(): ...
def test_mic_gated_true_during_reply_default_mode(): ...
def test_mic_gated_always_false_acoustic_mode(): ...
def test_transport_error_reconnects_once_then_exits_with_reason(): ...
def test_idle_ceiling_fires_only_in_live_and_resets_on_activity(): ...
def test_idle_ceiling_never_fires_mid_reply_even_past_deadline(): ...
def test_exit_reachable_from_every_state(): ...
def test_intents_after_exit_are_dropped(): ...
```

- [ ] **Step 2: RED.** Step 3: implement. Step 4: GREEN, **plus the contract check**: `git diff --stat Tests/Chat/test_console_hands_free.py` must be empty and the full V3 FSM suite green (`./.venv/bin/python -m pytest Tests/Chat/test_console_hands_free.py -p no:randomly -q`). Step 5: Commit — `feat(realtime): RealtimeLoopController FSM with reasoned exits, reconnect-once, idle ceiling`.

---

### Task 5: ChatScreen wiring — engine selection, continuity, barge-in, fallback

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` — engine fork in `_enter_console_hands_free_loop` (:7386); new `_enter_console_realtime_loop`, `_handle_console_realtime_intent` (delegating shared intents to the V3 dispatcher's handlers), session/tap/sink lifecycle, continuity writers, fallback logic.
- Test: `Tests/UI/test_console_realtime_wiring.py`

**Interfaces:**
- Consumes: Tasks 1–4 surfaces exactly as specified; `ConsoleChatStore.append_message(session_id, *, role, content, ...) -> ConsoleChatMessage` (:1070) and `append_stream_chunk(message_id, chunk)` (:1971); `StreamingPcmSink.open(24000, 1)` + `pump(sink, chunks_aiter)`; the app-level injection seam **`app.console_realtime_session_factory`** (new; mirrors `console_provider_gateway_factory`, chat_screen.py:5079's getattr idiom) so tests inject a fake session.
- Produces: the feature. Key wiring rules, each pinned by a named test:
  1. **Engine fork**: `resolve_handsfree_engine()` at entry; `"realtime"` + not `realtime_enabled()` → toast + refuse (forced-but-unconfigured); `"realtime"` otherwise → realtime entry; else V3 path **unchanged**.
  2. **Tap gating**: `_install_console_hands_free_store_tap()` is NOT called on the realtime path (pin: tap-installed flag stays False).
  3. **Connect flow**: chip `connecting…`; mic tap `start()` immediately (buffering); session `connect()` in a worker with an 8 s timeout; `on_ready` → seed (last 20 turns / 8000 chars from `store.messages_for_session`, plus system prompt as instructions) → `mark_ready()` → live.
  4. **Loud viable fallback**: connect failure/timeout → if V3 stack viable (not `_console_hands_free_vad_degraded` and dictation available) → toast naming the realtime failure + enter the V3 pipeline loop; else toast both reasons, no loop.
  5. **Continuity out**: `on_turn_committed` → `append_message(role=USER, content="")` (row created now for ordering) and remember its id; `on_input_transcript(text)` → `append_stream_chunk(user_row_id, text)`. `on_reply_started(item_id)` → `append_message(role=ASSISTANT, content="")`; `on_output_transcript_delta` → `append_stream_chunk`; barge-in → `append_stream_chunk(assistant_row_id, " ⏹ interrupted")`; usage → the existing usage-attach seam on the assistant row.
  6. **Audio out**: deltas → `asyncio.Queue` → aiter → `pump(sink, aiter)`; per-reply pump task; `on_reply_done` closes the aiter; fed-bytes accounting → `played_ms = fed_bytes / (24000*2) * 1000` (over-counts by ≤ buffered depth — truncating late is the safe direction; comment this).
  7. **Barge-in**: `SilenceSpeech` in realtime mode → sink stop + `session.cancel_response(played_ms)`; any-key routing reuses V3's existing hands-free key hook, now consulting whichever controller is active; `ModeChanged` syncs `tap.set_gated(controller.mic_gated)`.
  8. **Reasoned toasts**: `ExitLoop(reason=...)`/`ModeChanged(reason="reconnecting")` → toast copy: `"Hands-free ended: connection lost"`, `"Hands-free ended: idle for N minutes"`, `"Realtime reconnecting…"`.
  9. **Reconnect**: on `ModeChanged("reconnecting")` the wiring builds a fresh session (same factory), re-seeds from the store, `on_session_ready` resumes live.
  10. **Adopted capture**: entry with a live pipeline capture → stop+transcribe via the existing V2 stop path; when the transcript lands, `send_text_item(text, request_response=True)`.
  11. **Exit teardown**: tap stop → session close → sink close → chip repaint; unmount abandon-teardown mirrors V3's (:14700 region) for the realtime session.
- Fake session for tests: implements the protocol, records calls, exposes `fire_*` helpers to drive callbacks on the event loop.

- [ ] **Step 1: Failing wiring suite** (~16 tests, harness identical to `test_console_hands_free_wiring.py` — `_build_test_app`, `ConsoleHarness`, `_mounted_console`): engine fork honesty (×3: auto+disabled→pipeline untouched; forced-unconfigured→refuse+toast; enabled→realtime chip), tap-gating pin, connect→ready→live chip sequence, seed contents+budget pin, first-words buffering (frames pushed pre-ready arrive after `mark_ready` in order), turn-commit row ordering + transcript fill, assistant streaming + interrupted marker, keypress barge-in (sink stop + `cancel_response` with computed ms + regate), acoustic mode never gates, loud fallback (×2: viable→pipeline entered+toast; unviable→no loop+both reasons), reconnect-once re-seeds, idle-timeout toast, exit teardown order, V3 contract (all three contract files byte-identical + their suites green).
- [ ] **Step 2: RED.** Step 3: implement. Step 4: GREEN + full named sweep (the 11 V3 files + the 5 new test files). Step 5: Commit — `feat(realtime): wire the realtime engine into the Console hands-free loop`.

---

### Task 6: Settings panel, docs, sweeps

**Files:**
- Modify: `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py` (Realtime block), `Docs/Features/Speech-Services-Guide.md`, `README.md` (one line in the voice section)
- Test: extend the panel's existing test file (locate via `grep -rn "SpeechTTSSettingsPanel" Tests/`), plus a docs-presence check in the wiring suite is NOT needed — guide edits are reviewed by eye.

**Interfaces:**
- Consumes: Task 1 readers; the panel's existing save path (it must write `realtime.enabled`, `realtime.provider`, `realtime.model`, `realtime.voice`, `realtime.idle_timeout_minutes`, `dictation.handsfree_engine` through the same mutation helper its sibling fields use — **no second config writer** (TASK-2111)).
- Produces: Settings UI + docs.

- [ ] **Step 1**: Failing panel tests — the Realtime block renders (enable switch default off, provider select with only `openai`, model/voice inputs, idle timeout, engine select), toggling writes the exact keys through the existing helper (spy it), invalid idle input refuses per the panel's sibling validation.
- [ ] **Step 2: RED → implement → GREEN.**
- [ ] **Step 3: Docs.** Guide: a "Realtime engine" subsection under the Hands-Free section — what it is, the `[realtime]` block, the spoken-commands-don't-exist difference, the privacy line (continuous mic streaming while live), reconnect/idle behavior, cost note; README: one line under Voice Conversation mentioning the optional realtime engine. Copy the spec's wording — do not soften the privacy or cost language.
- [ ] **Step 4: Sweeps** — full named sweep (V3's 11 files + the 5 new files + panel tests) and `./.venv/bin/python -m pytest Tests/ --collect-only -q 2>&1 | tail -2` (collection health). Contract-file byte-identity check one final time.
- [ ] **Step 5: Commit** — `feat(realtime): Settings realtime block + guide and README documentation`.

---

## Self-review notes (already applied)

- Spec coverage: engine opt-in/fork (T1/T5), transport+session+truncation+usage (T2), 24 kHz tap+buffering+lazy-import (T3), FSM+idle+reconnect+reasons (T4), continuity both ways+interrupted marker+ordering (T5), barge-in both modes (T4/T5), loud viable fallback (T5), Settings (T6), privacy/UX docs (T6), live-gate prep (T2 probe + the gate itself is post-plan).
- Type consistency: `RealtimeCallbacks.on_reply_started(item_id: str)` is consumed in T5 rule 5; `cancel_response(played_ms: int)` consistent T2/T4-rules/T5 rule 6-7; `RealtimeLoopState` values match the chip states in T5 rule 8 and the spec's chip section.
- The V3 contract files list is the seam's proof; every task's GREEN step reruns what it touches.
