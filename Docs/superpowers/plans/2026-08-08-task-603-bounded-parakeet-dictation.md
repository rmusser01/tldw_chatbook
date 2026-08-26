# TASK-603 Bounded Parakeet ONNX Dictation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route public and Console Parakeet ONNX PCM transcription through the one app-owned `LocalSTTExecutor`, with a bounded one-item dictation mailbox, dictation-next admission, explicit limit recovery, and faster-whisper retry while preserving current Console insertion and voice-command behavior.

**Architecture:** Keep `LocalSTTExecutor` queue-less and add one small app-owned `LocalSTTDispatchCoordinator` in front of it. File jobs and bounded PCM requests share the same exact Parakeet model/artifact resolver and resident worker. Live Console capture seals silence-delimited segments into an asynchronous capture handle so its processing thread never blocks behind batch work; the handle coalesces at most one pending request and preserves segment boundaries. Existing retained providers remain behind `LegacyTranscriptionBridge`; only an explicit user-approved faster-whisper retry replays failed bounded PCM there.

**Tech Stack:** Python 3.11+, Textual 8, existing `BufferAudioSource`/normalized STT contracts, existing `LocalSTTExecutor`, `onnx-asr[cpu]==0.12.0`, existing Console dictation controllers, pytest.

## Global Constraints

- ADR required: no new ADR.
- ADR path: `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`.
- Reason: ADR-025 already decides the shared executor boundary, bounded in-memory dictation, dictation-next/non-preemption policy, retained-provider fallback, and release gates. This task implements that decision without changing it.
- Read `backlog/docs/lessons-testing-evidence.md`, `backlog/docs/lessons-live-verification.md`, and `backlog/docs/lessons-backlog-hygiene.md` before implementation or verification.
- Do not create another executor, worker process, model cache, downloader, general priority queue, persisted audio queue, or temporary WAV for microphone PCM.
- Keep `parakeet_defaults_enabled=False`; do not promote semantic defaults, delete legacy NeMo/MLX/faster-whisper code, or perform TASK-605 cleanup.
- TASK-603.1 already shipped the Mic control, 60-second capture, off-loop stop,
  caret insertion, and direct legacy Parakeet buffer vertical slice. Upgrade its
  execution/admission seam; do not add a second Mic flow or reimplement the
  established Console lifecycle.
- English remains the Console default and resolves to Parakeet v2; precision
  defaults to INT8. Do not alter `auto` or non-English routing owned by the
  existing batch policy.
- INT8 is the fallback only when `transcription.default_precision` is absent.
  Honor the first-run wizard's persisted explicit `f32` selection for public
  and Console Parakeet buffers; reject any other precision clearly. Do not read
  the legacy MLX-only `transcription.parakeet_precision` key.
- `LocalSTTExecutor` remains single-slot and queue-less. The coordinator may hold one dictation reservation and one pending same-capture PCM request only; Library ordering remains in `LibraryIngestJobRegistry`.
- One canonical Console ceiling is 60 seconds. Derive bytes as `sample_rate * channels * sample_width * seconds`; do not preserve the current 1.5 headroom as a second behavioral limit.
- An active native inference is never preempted. A Mic press gates only future local-STT heavy Library jobs; light parsing remains dispatchable.
- The retained silence-transcription warm-up must be skipped for deferred shared-executor Parakeet capture, or an active batch will block the Mic before recording opens.
- At the limit, stop and transcribe retained PCM, show `Limit reached — press Mic to continue.`, return to idle, and do not auto-reopen even from hands-free mode.
- Retry copy is exactly `Parakeet failed. Retry this audio with faster-whisper?`; never expose native exception text and never retry automatically.
- Run only tests covering modified functionality. Do not run the full repository suite or wait on unrelated CI.
- macOS native evidence may be collected locally. Keep Windows/Linux release gates open and do not claim them without native evidence.

---

## Task 1: Extend the executor protocol from file-only to file-or-buffer

**Files:**
- Modify: `tldw_chatbook/STT/executor.py`
- Modify: `tldw_chatbook/STT/executor_worker.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/STT/executor_test_support.py`
- Modify: `Tests/STT/test_local_stt_executor.py`
- Modify: `Tests/STT/test_transcribe_cpp.py`
- Modify: `Tests/Library/test_library_ingest_runner.py`

**Interfaces:**
- Consumes: existing `FileAudioSource`, `BufferAudioSource`, `ModelIdentity`, executor callbacks, and Library file requests.
- Produces: a picklable `ExecutorRequest.source`, optional `job_id`, validated logical segment boundaries, and typed rejection for provider/source mismatches.
- Invariant: exactly one source object per request; buffer boundaries are frame offsets, not byte offsets.

Implement the request shape directly in the existing executor module:

```python
@dataclass(frozen=True, slots=True)
class ExecutorRequest:
    generation: int
    attempt_id: str
    job_id: str | None
    source: FileAudioSource | BufferAudioSource = field(repr=False)
    identity: ModelIdentity
    options: dict[str, Any] = field(repr=False)
    segment_end_frames: tuple[int, ...] = ()
    local_source: LocalSourceSnapshot | None = field(default=None, repr=False)
    managed_store_root: Path | None = field(default=None, repr=False)
    managed_artifact_ref: tuple[str, str, str] | None = None
```

Validation rules in `ExecutorRequest.__post_init__`:

```python
if self.job_id is not None:
    _require_nonempty_text("job_id", self.job_id)
if type(self.source) not in (FileAudioSource, BufferAudioSource):
    raise TypeError("source must be a FileAudioSource or BufferAudioSource")
if self.segment_end_frames:
    if type(self.source) is not BufferAudioSource:
        raise ValueError("segment_end_frames require a buffer source")
    frame_bytes = self.source.channels * self.source.sample_width
    total_frames = len(self.source.audio) // frame_bytes
    if (
        any(type(end) is not int or end <= 0 for end in self.segment_end_frames)
        or any(a >= b for a, b in zip(self.segment_end_frames, self.segment_end_frames[1:]))
        or self.segment_end_frames[-1] != total_frames
    ):
        raise ValueError("segment_end_frames must increase to the final PCM frame")
```

The executor submission seam becomes:

```python
def submit(
    self,
    *,
    attempt_id: str,
    job_id: str | None,
    source: FileAudioSource | BufferAudioSource,
    identity: ModelIdentity,
    options: dict[str, Any],
    segment_end_frames: tuple[int, ...] = (),
    local_source: LocalSourceSnapshot | None = None,
    managed_store_root: Path | None = None,
    managed_artifact_ref: tuple[str, str, str] | None = None,
    on_event: Callable[[ExecutorEvent], None] = _ignore_event,
    on_result: Callable[[ExecutorResult], None] = _ignore_result,
    on_failure: Callable[[ExecutorFailure], None] = _ignore_failure,
    explicit_retry: bool = False,
) -> int:
```

- [ ] Add failing protocol tests for file and buffer sources, `job_id=None`, invalid source types, boundaries on a file, non-increasing/zero/past-end boundaries, and a final boundary that does not equal the buffer frame count.
- [ ] Update the pickling/redaction test to prove PCM bytes, paths, and snapshot tokens never appear in `repr` while both source variants round-trip through `pickle`.
- [ ] Add a worker test proving `FileAudioSource` still calls the existing parser with the exact path/options/runner shape and unchanged payload.
- [ ] Add a worker test proving a `BufferAudioSource` never calls `parse_job`; it calls a provider buffer runner and returns its bounded payload.
- [ ] Add a worker test proving `BufferAudioSource` with `transcribe-cpp` or a provider without a buffer runner returns `UNSUPPORTED_CAPABILITY` plus the existing recovery actions.
- [ ] Run `python -m pytest Tests/STT/test_local_stt_executor.py Tests/STT/test_transcribe_cpp.py Tests/Library/test_library_ingest_runner.py -q` and confirm the new tests fail for the intended missing contract/branch, not import or fixture errors.
- [ ] Replace `source_path` with `source`, make `job_id` optional, and add boundary validation without changing executor admission, cancellation, recycling, containment, or callback delivery.
- [ ] Extend `ProviderRuntime` minimally with `buffer_runner: Callable[..., dict[str, Any]] | None = None`; keep the existing `runner` field for file parsing and existing fake providers.
- [ ] Branch once in `_run_executor_worker`: file -> the unchanged `parse_job` call using `source.path`; Parakeet buffer -> `buffer_runner(source, segment_end_frames=request.segment_end_frames)`; otherwise raise `_ProviderLoadFailure(UNSUPPORTED_CAPABILITY)`.
- [ ] Update the app's existing Library call and direct request construction in the listed test/support files mechanically from `source_path=path` to `source=FileAudioSource(path)`. Update the Library fake executor assertion to the same source object so this commit leaves production file dispatch working; do not change admission policy yet.
- [ ] Re-run `python -m pytest Tests/STT/test_local_stt_executor.py Tests/STT/test_transcribe_cpp.py Tests/Library/test_library_ingest_runner.py -q` and commit: `feat(stt): admit bounded executor buffers`.

## Task 2: Share exact Parakeet dispatch identity and add in-memory recognition

**Files:**
- Create: `tldw_chatbook/STT/parakeet_dispatch.py`
- Modify: `tldw_chatbook/STT/parakeet_onnx.py`
- Modify: `tldw_chatbook/STT/executor_worker.py`
- Create: `Tests/STT/test_parakeet_dispatch.py`
- Modify: `Tests/STT/test_parakeet_onnx.py`
- Modify: `Tests/STT/test_local_stt_executor.py`

**Interfaces:**
- Consumes: model ID, precision, optional configured model directory, existing managed/legacy artifact helpers, `BufferAudioSource`, and optional logical frame boundaries.
- Produces: one `ParakeetDispatch` used by both Library and Console plus one normalized in-memory result with ordered logical segment text.
- Invariant: resolving identity never downloads; recognizing PCM never writes it to disk.

Create one immutable resolution result:

```python
@dataclass(frozen=True, slots=True)
class ParakeetDispatch:
    identity: ModelIdentity
    local_source: LocalSourceSnapshot | None
    managed_store_root: Path | None
    managed_artifact_ref: tuple[str, str, str] | None
    option_updates: Mapping[str, Any]


def resolve_parakeet_dispatch(
    *,
    model_id: str,
    precision: str,
    model_dir: str | Path | None,
) -> ParakeetDispatch:
    """Resolve an installed configured, managed, or verified legacy artifact."""
```

`option_updates` contains only the already-supported private worker options (`transcription_model_dir`, `_verify_legacy_parakeet_v2`) and is copied into each caller's own options dictionary. The function must not accept a Library job or invent an attempt/job ID.

Add an internal normalized carrier in `parakeet_onnx.py`:

```python
@dataclass(frozen=True, slots=True)
class ParakeetBufferResult:
    normalized: TranscriptionResult
    logical_segments: tuple[str, ...]


def transcribe_buffer(
    self,
    *,
    source: BufferAudioSource,
    segment_end_frames: tuple[int, ...],
    attempt_id: str,
    language: str,
    job_id: str | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> ParakeetBufferResult:
```

PCM conversion is in memory and explicit:

```python
samples = np.frombuffer(source.audio, dtype="<i2").reshape(-1, source.channels)
mono = samples.astype(np.float32).mean(axis=1) / 32768.0
ends = segment_end_frames or (len(mono),)
starts = (0, *ends[:-1])
logical_waveforms = tuple(mono[start:end] for start, end in zip(starts, ends))
```

Only 16-bit PCM is admitted by Parakeet in this task; the provider boundary returns `UNSUPPORTED_CAPABILITY` for other widths rather than silently mis-decoding them. For total duration over 30 seconds, use the resident managed VAD when present and report `produced_capabilities.vad=True`. A configured or verified legacy v2 runtime without VAD recognizes each logical waveform directly up to the 60-second public/Console bound and reports `vad=False`.

- [ ] Add failing resolver tests showing Library-style and Console-style calls
  produce identical v2 INT8 and explicit v2 F32 `ModelIdentity`, local
  snapshot/managed closure identity, and option updates from the same
  configured, managed, and verified-legacy fixtures.
- [ ] Add a failing resolver test proving it never imports or calls provision/download functions and fails clearly when no installed artifact is resolvable.
- [ ] Add failing runtime tests for mono and stereo int16 PCM conversion, exact duration, `job_id=None` provenance, v2 English semantics, v3 warning semantics, and no filesystem/tempfile/wave staging.
- [ ] Add failing boundary tests where two logical segments (`ordinary text`, `console stop`) return two ordered texts even when sent as one executor request.
- [ ] Add a resident-reuse test with two buffer requests sharing one model identity but different attempt IDs; prove the second result uses the second request's attempt/language/job metadata (including `job_id=None`) rather than state captured when the runtime first loaded.
- [ ] Add failing 30–60-second tests: managed closure invokes fake VAD in memory and reports `vad=True`; verified legacy v2 without VAD recognizes directly and reports `vad=False`.
- [ ] Add cancellation coverage proving the token is checked before every logical/VAD inference and a cancellation before segment two prevents its native call.
- [ ] Add worker coverage proving `buffer_runner` serializes `text`, `logical_segments`, `duration`, `transcription_model`, and validated normalized `transcription_provenance` without a synthetic job ID.
- [ ] Run `python -m pytest Tests/STT/test_parakeet_dispatch.py Tests/STT/test_parakeet_onnx.py Tests/STT/test_local_stt_executor.py -q` and confirm intended failures.
- [ ] Extract the Parakeet half of `LibraryIngestQueueMixin._build_local_stt_dispatch` into `resolve_parakeet_dispatch`; leave transcribe.cpp resolution exactly where it is.
- [ ] Factor the runtime's result/provenance assembly so file and buffer methods cannot drift, while keeping `_prepared_wav` file-only.
- [ ] Add the Parakeet provider's `buffer_runner`, mapping unexpected native errors to the existing sanitized `INFERENCE_FAILED` failure and preserving `retry_faster_whisper`.
- [ ] Pass the current request's attempt/job/context/language into
  `buffer_runner` on every worker loop iteration. The resident provider closure
  may capture artifact/runtime identity only; it must not reuse the first
  request's provenance metadata.
- [ ] Re-run the three focused tests and commit: `feat(stt): transcribe Parakeet PCM in shared worker`.

## Task 3: Add the bounded dictation-next dispatch coordinator

**Files:**
- Create: `tldw_chatbook/STT/dispatch_coordinator.py`
- Create: `Tests/STT/test_dispatch_coordinator.py`

**Interfaces:**
- Consumes: the existing queue-less `LocalSTTExecutor`, a resolved `ParakeetDispatch`, bounded segment PCM, and existing executor callbacks.
- Produces: `submit_library`, blocking one-shot `transcribe_buffer`, and asynchronous `begin_dictation`/`DictationCaptureHandle`.
- Invariant: one native inference active; zero or one same-capture dictation request pending; no persisted Library queue duplicated here.

Keep the public shape small. These are the complete public operations; private
helpers may only support their locking and callback delivery:

```python
DICTATION_MAX_SECONDS = 60.0


def pcm_byte_limit(*, sample_rate: int, channels: int, sample_width: int) -> int:
    return int(sample_rate * channels * sample_width * DICTATION_MAX_SECONDS)
```

Coordinator operations are exactly:

- `submit_library(**executor_kwargs) -> int`, where `executor_kwargs` is the
  existing executor submission contract from Task 1;
- `transcribe_buffer(*, source, dispatch, language) -> dict[str, Any]`;
- `begin_dictation(*, capture_generation, dispatch, sample_rate, channels,
  sample_width, language, on_logical_segment) -> DictationCaptureHandle`;
- read-only `dictation_reserved -> bool`; and
- idempotent `close() -> None`.

Capture-handle operations are exactly read-only `waiting_for_executor -> bool`,
`append_segment(audio) -> DictationAppendStatus`, `finish() -> None`, `wait()
-> None`, `cancel(*, force=False) -> bool`, and `take_retry_buffer() ->
RetryableDictationBuffer | None`.

`DictationAppendStatus` has only `ACCEPTED` and `LIMIT_REACHED`. `RetryableDictationBuffer` contains one `BufferAudioSource` plus logical frame offsets; it is held only after a retryable Parakeet failure and released by `take_retry_buffer`, cancel, or close.

The handle's blocking caller receives one sanitized typed failure:

```python
class RetryableDictationFailure(RuntimeError):
    def __init__(self, retry_buffer: RetryableDictationBuffer) -> None:
        self.retry_buffer = retry_buffer
        super().__init__("Parakeet transcription failed.")
```

It contains no native exception text or private path. `wait()` raises it only
after a retryable Parakeet terminal; non-retryable failure and cancellation use
their existing normalized categories without a retry buffer.

Coordinator state under one `threading.RLock`:

```python
self._active_kind: Literal["library", "dictation"] | None
self._active_attempt_id: str | None
self._reservation: _DictationCapture | None
self._pending: _PendingDictation | None
self._closed: bool
```

Admission order on every executor terminal callback is fixed:

```python
with self._lock:
    clear_active_once()
    if self._pending is not None:
        next_request = snapshot_pending_locked()
    elif self._reservation is finished:
        clear_reservation_once()
outside_lock_submit(next_request)
outside_lock_deliver_original_callback()
outside_lock_notify_library_top_up_if_gate_cleared()
```

Never invoke executor or user callbacks while holding the coordinator lock.
The executor terminal callback itself runs on the executor reader thread, so it
must not retire/re-submit a different resident identity inline: doing so can
make worker retirement join its own reader thread. The terminal callback starts
one bounded daemon dispatch thread whose body performs the three `outside_lock`
operations above in order. There is never more than one such transition thread
because the executor itself has only one active terminal guard.

- [ ] Write a dependency-free fake executor that records submissions and exposes explicit `succeed`, `fail`, `cancel`, and `force_stop` terminal controls.
- [ ] Add failing tests for idle dictation immediate dispatch; batch active -> one pending dictation; batch terminal -> dictation before a later Library submit; and dictation terminal -> gate clear/top-up callback.
- [ ] Add a test proving a Mic reservation with no PCM blocks only `submit_library` heavy work, then cancellation clears the gate and wakes the waiter.
- [ ] Add coalescing tests: several same-capture segments become one pending `BufferAudioSource`, frame boundaries stay ordered, and an active dictation permits exactly one next pending request.
- [ ] Add mismatch tests for capture generation, identity, sample rate, channels, and width; each returns a visible busy failure and never combines audio.
- [ ] Add exact-bound/one-frame-over tests proving frame-aligned accepted PCM is retained, `LIMIT_REACHED` is returned once, and no accepted sample disappears.
- [ ] Add stale terminal, duplicate terminal, worker failure, pending cancel, active cooperative cancel, force-stop, and close tests; every path clears reservation/pending state exactly once.
- [ ] Add retry-buffer tests proving a retryable active failure merges only
  that failed request, its not-yet-run pending segments, and later same-capture
  segments accepted before `finish`; earlier successful segments are excluded,
  native details are absent, and `take_retry_buffer` transfers ownership once.
- [ ] Add an identity-change regression where a v3 batch terminal callback runs on a named fake reader thread and pending v2 dictation is submitted by the bounded transition thread, never by that reader thread; assert callback/top-up ordering remains unchanged.
- [ ] Add a real event-controlled delay test: hold a fake batch longer than a 50 ms processing-thread join bound, append dictation, finish capture, and prove the processing thread exits promptly while `handle.wait()` completes only after batch then dictation. Do not sleep 30 seconds.
- [ ] Run `python -m pytest Tests/STT/test_dispatch_coordinator.py -q` and confirm intended failures.
- [ ] Implement only the state above plus the single terminal-transition thread; do not add a standing scheduler thread, priorities, multiple pending entries, polling, timers, persistence, worker ownership, or model ownership.
- [ ] Deliver logical-segment callbacks in monotonically increasing sequence order and ignore stale executor generations/attempt IDs.
- [ ] Retain only failed+not-yet-run PCM for explicit retry; release successful request bytes immediately and release all bytes on cancel/close.
- [ ] Re-run the focused coordinator test and commit: `feat(stt): coordinate dictation before batch top-up`.

## Task 4: Put Library and Console on the same app-owned coordinator

**Files:**
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/Library/test_library_ingest_runner.py`
- Modify: `Tests/App/test_submit_library_ingest_job.py`

**Interfaces:**
- Consumes: app-owned `LocalSTTExecutor`, `LibraryIngestQueueMixin` heavy dispatch, and the new `LocalSTTDispatchCoordinator`.
- Produces: one lazily constructed coordinator, one Console dictation service factory, and deterministic heavy-lane gating/top-up.
- Invariant: shutdown detaches coordinator and executor once, off the Textual event loop.

Add app ownership beside the existing executor fields:

```python
self._local_stt_executor_lock = threading.RLock()
self._local_stt_executor: LocalSTTExecutor | None = None
self._local_stt_dispatch_coordinator: LocalSTTDispatchCoordinator | None = None
```

The lazy accessor constructs both under the existing lock:

```python
def _ensure_local_stt_dispatch_coordinator(self) -> LocalSTTDispatchCoordinator:
    with self._local_stt_executor_lock:
        if self._ingest_shutdown:
            raise ExecutorUnavailableError("Local STT is shutting down")
        if self._local_stt_executor is None:
            self._local_stt_executor = self._create_local_stt_executor()
        if self._local_stt_dispatch_coordinator is None:
            self._local_stt_dispatch_coordinator = LocalSTTDispatchCoordinator(
                self._local_stt_executor,
                on_dictation_idle=lambda: self._marshal_local_stt_call(
                    self._top_up_ingest_parse_pool
                ),
            )
        return self._local_stt_dispatch_coordinator
```

Use the `RLock` shown above because `_ensure_local_stt_dispatch_coordinator`
reuses `_ensure_local_stt_executor`; this preserves one lazy-construction lock
without a second lock-held helper or a self-deadlock.

- [ ] Add failing Library tests proving `_build_local_stt_dispatch` delegates its Parakeet branch to `resolve_parakeet_dispatch` while transcribe.cpp output stays unchanged.
- [ ] Add failing dispatch tests proving every Library `executor.submit` now goes through `coordinator.submit_library(source=FileAudioSource(Path(job.source_path)))` and preserves attempt/job/result callbacks.
- [ ] Add a top-up test where `dictation_reserved=True`: queued audio/video remains queued, a queued document still enters the parse pool, and no synthetic `ExecutorBusyError` is produced.
- [ ] Add a terminal-order test proving a batch callback cannot top up a second heavy job before pending dictation is submitted.
- [ ] Add an accessor concurrency/reentrancy test proving simultaneous Library/Console callers receive the same coordinator and executor and the nested executor accessor cannot self-deadlock.
- [ ] Add shutdown tests proving `coordinator.close()` is nonblocking and
  process-agnostic: it marks admission closed, resolves pending dictation as
  cancelled, and cooperatively cancels an active dictation without joining or
  closing the executor. `_shutdown_ingest_parse_pool` then detaches coordinator
  and executor, and the existing teardown thread closes the executor/process.
- [ ] Run `python -m pytest Tests/Library/test_library_ingest_runner.py Tests/App/test_submit_library_ingest_job.py -q` and confirm intended failures.
- [ ] Route Library submissions through the coordinator and add `coordinator.dictation_reserved` to the existing heavy-lane fullness condition; do not change the light parse pool or Library registry ordering.
- [ ] Merge `option_updates` from the shared Parakeet resolver into the per-job options copy; never mutate global configuration.
- [ ] Add `_create_console_dictation_service(**kwargs)` on the app. It returns
  `LazyLiveDictationService(**kwargs,
  transcription_service_factory=lambda: TranscriptionService(
  local_stt_dispatcher=self._ensure_local_stt_dispatch_coordinator()))` and
  performs both imports inside the method so importing/mounting Console does
  not import native STT packages.
- [ ] Update `_shutdown_ingest_parse_pool` to set `_ingest_shutdown`, call the
  coordinator's nonblocking `close`, detach coordinator/executor under the
  `RLock`, and pass the executor to the existing teardown thread; do not create
  a second shutdown thread or let the coordinator close a process it does not
  own.
- [ ] Re-run the two focused test files and commit: `feat(stt): share app-owned dispatch across batch and dictation`.

## Task 5: Adapt the public facade and live service without blocking its processing thread

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/transcription_service.py`
- Modify: `tldw_chatbook/Audio/dictation_service_lazy.py`
- Modify: `tldw_chatbook/Chat/console_voice_input.py`
- Modify: `Tests/STT/test_transcription_service_facade.py`
- Modify: `Tests/Audio/test_dictation_lazy_transcription.py`
- Modify: `Tests/Audio/test_dictation_segment_finalization.py`
- Modify: `Tests/Audio/test_dictation_stop_join.py`
- Modify: `Tests/Chat/test_console_voice_input.py`

**Interfaces:**
- Consumes: explicit app-owned coordinator dependency, historical `transcribe_buffer` signature, current Lazy service callbacks, and `ConsoleVoiceInputController` generation fencing.
- Produces: blocking public Parakeet buffer dispatch, async live capture dispatch, truthful streaming fallback, and a bounded retry token.
- Invariant: retained providers keep their current bridge calls and Parakeet never silently falls back to an in-process legacy model.

Change only the facade constructor, not the public method signatures:

```python
class TranscriptionService:
    def __init__(
        self,
        *,
        local_stt_dispatcher: LocalSTTDispatchCoordinator | None = None,
    ) -> None:
        self._bridge = LegacyTranscriptionBridge(_LegacyTranscriptionBackend)
        self._local_stt_dispatcher = local_stt_dispatcher

    @property
    def uses_deferred_local_stt_dispatch(self) -> bool:
        return self._local_stt_dispatcher is not None
```

`transcribe_buffer` behavior:

```python
effective_provider = provider or self.config["default_provider"]
if effective_provider == "parakeet-onnx":
    if self._local_stt_dispatcher is None:
        raise TranscriptionError(
            "Parakeet ONNX buffer transcription requires the shared local executor."
        )
    source = BufferAudioSource(audio_data, sample_rate, channels, sample_width)
    dispatch = resolve_parakeet_dispatch(
        model_id=model or PARAKEET_V2_MODEL,
        precision=str(
            kwargs.pop(
                "precision",
                get_cli_setting("transcription.default_precision", "int8"),
            )
            or "int8"
        ).strip().lower(),
        model_dir=kwargs.pop("model_dir", None),
    )
    return self._local_stt_dispatcher.transcribe_buffer(
        source=source,
        dispatch=dispatch,
        language=language or "en",
    )
return self._bridge.transcribe_buffer_legacy(
    audio_data,
    sample_rate,
    channels,
    sample_width,
    provider,
    model,
    language,
    **kwargs,
)
```

The retained bridge still receives the original `provider` argument for every
non-Parakeet call. `create_streaming_transcriber` resolves the same effective
provider and returns `None` before consulting the retained bridge when it is
`parakeet-onnx`.

Add one internal facade method used only by the live service:

```python
def begin_dictation_capture(
    self,
    *,
    capture_generation: int,
    model: str | None,
    language: str,
    sample_rate: int,
    channels: int,
    sample_width: int,
    on_logical_segment: Callable[[int, str], None],
) -> DictationCaptureHandle:
    if self._local_stt_dispatcher is None:
        raise TranscriptionError("The shared local executor is unavailable.")
    dispatch = resolve_parakeet_dispatch(
        model_id=model or PARAKEET_V2_MODEL,
        precision=str(
            get_cli_setting("transcription.default_precision", "int8") or "int8"
        ).strip().lower(),
        model_dir=None,
    )
    return self._local_stt_dispatcher.begin_dictation(
        capture_generation=capture_generation,
        dispatch=dispatch,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        language=language,
        on_logical_segment=on_logical_segment,
    )
```

`resolve_parakeet_dispatch(model_dir=None)` performs the existing configured-
directory/managed/verified-legacy lookup and validates precision as `int8` or
`f32`. The model fallback is the approved English v2 default; the precision
fallback is INT8 only when the first-run/config value is absent.

Inject the facade lazily into `LazyLiveDictationService`:

```python
def __init__(
    self,
    transcription_provider: str = "auto",
    transcription_model: str | None = None,
    language: str = "en",
    enable_punctuation: bool = True,
    enable_commands: bool = True,
    audio_backend: str | None = None,
    max_buffer_bytes: int | None = None,
    on_buffer_limit: Callable[[], None] | None = None,
    transcription_service_factory: Callable[[], Any] | None = None,
) -> None:
    self._transcription_service_factory = transcription_service_factory
    self._dictation_handle = None

@property
def transcription_service(self):
    if self._transcription_service is None:
        factory = self._transcription_service_factory or TranscriptionService
        self._transcription_service = factory()
    return self._transcription_service
```

For Parakeet with an injected dispatcher,
`LazyLiveDictationService.reserve_deferred_dictation(capture_generation)`
creates the `DictationCaptureHandle`. `ConsoleVoiceInputController._run_begin`
calls that method immediately after service construction and before
`_prepare_speech_model`, so the reservation exists before any Library top-up
can race the Mic press. A true `handle.waiting_for_executor` value emits the
approved busy event. `_prepare_speech_model` then skips the retained silence
warm-up for this exact deferred capability; `start_dictation` installs the
ordinary transcript callbacks and opens the recorder using the already-
reserved handle.

`_transcribe_segment_audio` emits `done=False`, calls
`handle.append_segment(audio_data)`, and returns without waiting. The handle's
ordered callback performs the current `_handle_partial_text` -> `done=True` ->
`_finalize_current_segment` flow for each logical segment; blank results still
emit `done=True` and `on_segment_no_final`. `LIMIT_REACHED` invokes the existing
one-shot buffer-limit callback, which posts onto the Console UI thread.

On stop, the processing thread only drains/seals PCM, so its existing bounded join completes quickly. The same off-UI stop worker then calls `handle.finish()` and `handle.wait()`. `abandon()` cancels the handle without joining. Do not change retained-provider synchronous segment behavior.

- [ ] Update facade signature tests for a keyword-only optional dispatcher while asserting every public method signature remains unchanged.
- [ ] Add facade tests: explicit Parakeet and omitted-provider/configured-
  Parakeet use the coordinator and return the compatibility dictionary;
  omitted precision uses INT8, persisted explicit F32 selects the F32 artifact,
  and an invalid configured precision fails without loading a model;
  missing dispatcher fails clearly; retained faster-whisper/MLX calls are
  byte-for-byte bridge forwards; Parakeet streaming returns `None` without
  bridge/native calls; and facade cleanup never closes the app-owned
  coordinator.
- [ ] Add Lazy service tests proving Parakeet creates one capture handle, segment processing returns before the fake inference is released, ordered callbacks preserve final/no-final events, and retained providers still call synchronous `transcribe_buffer` once per segment.
- [ ] Add a stop-join regression with a 50 ms join bound and a real event-delayed batch: processing thread completes within the bound, audio remains pending, and stop finishes after batch+dictation without `transcription_complete=False`.
- [ ] Add generation/cancellation tests proving a stale handle callback cannot mutate a later capture and abandon cancels pending/active dispatch without inserting text.
- [ ] Add Console controller tests proving the reservation exists before the warm-up decision, deferred Parakeet skips `warm_transcription_model` entirely, opens the microphone while batch is active, and emits `VoiceLocalSTTBusy("Local transcription busy — dictation will run next.")`.
- [ ] Add a test proving `auto`/faster-whisper retains the current warm-before-capture behavior; the skip is exact-provider/capability based, not global.
- [ ] Run `python -m pytest Tests/STT/test_transcription_service_facade.py Tests/Audio/test_dictation_lazy_transcription.py Tests/Audio/test_dictation_segment_finalization.py Tests/Audio/test_dictation_stop_join.py Tests/Chat/test_console_voice_input.py -q` and confirm intended failures.
- [ ] Implement explicit facade dispatch and mark the legacy Parakeet buffer branch `retained until TASK-605; unreachable from production Parakeet facade` without deleting it.
- [ ] Implement the async handle branch as a narrow alternative inside existing segment finalization/stop paths; do not rewrite the recorder or streaming regime.
- [ ] Add a `VoiceLocalSTTBusy` event and a retry-availability boolean on the existing sanitized `VoiceFailed`; keep native error objects out of UI events. A retryable executor failure remains on the bounded handle until stop, where `handle.wait()` raises one sanitized retryable failure through the existing blocking-call path; do not add a parallel mid-capture failure event.
- [ ] Store any retry callable/token only on the controller/session that owns the failed capture. The token replays the failed+not-yet-run logical segments through retained faster-whisper and excludes earlier successful Parakeet segments. Clear it on later start, discard, success, decline, cancellation, and abandon.
- [ ] In `_run_finish`, catch `RetryableDictationFailure` before the generic
  failure branch. When `faster-whisper` is installed, store one bounded retry
  callable on the controller, emit sanitized `VoiceFailed(retry_available=True)`,
  and release the stopped service; otherwise clear the retry buffer and use the
  ordinary non-retryable failure path. `ConsoleStreamingDictationSession`
  exposes read-only `retry_available`, `retry_with_faster_whisper()`, and
  `clear_retry()` methods instead of adding a second event/error channel.
- [ ] Re-run the five focused test files and commit: `feat(stt): defer live Parakeet buffers to shared executor`.

## Task 6: Wire Console busy, limit, retry, and explicit-resume behavior

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/dictation.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify: `Tests/UI/test_console_dictation_streaming.py`
- Modify: `Tests/UI/test_console_dictation.py`
- Modify: `Tests/UI/test_console_voice_chip.py`
- Modify: `Tests/UI/test_console_controller_wiring.py`
- Modify: `Tests/UI/test_console_hands_free_wiring.py`

**Interfaces:**
- Consumes: app-owned service factory, `VoiceLocalSTTBusy`, sanitized retry availability, current Mic/session state machine, and existing `ConfirmationDialog`.
- Produces: visible busy and limit copy, one explicit retry confirmation, normal caret insertion after success/retry, and no automatic capture restart.
- Invariant: UI loop never waits on executor/native inference and no success path sends automatically.

Add one optional named dependency to `ConsoleDictationController` and pass it through the existing session port:

```python
def _create_console_dictation_session(self) -> ConsoleStreamingDictationSession:
    return ConsoleStreamingDictationSession(
        on_event=self._emit_console_dictation_event,
        service_factory=self._dictation_service_factory,
        max_buffer_bytes=pcm_byte_limit(
            sample_rate=CONSOLE_DICTATION_SAMPLE_RATE,
            channels=CONSOLE_DICTATION_CHANNELS,
            sample_width=CONSOLE_DICTATION_SAMPLE_WIDTH,
        ),
    )
```

The constructor gains the keyword-only parameter
`dictation_service_factory: Callable[..., Any] = default_service_factory` and
stores it as `self._dictation_service_factory`; no other constructor dependency
changes.

`wiring.py` supplies a late-binding lambda to the app-owned factory so test monkeypatches and screen replacement remain valid:

```python
dictation_service_factory=lambda **kwargs: (
    screen.app_instance._create_console_dictation_service(**kwargs)
),
```

Retry stays inside `_stop_console_dictation`'s existing off-loop workflow. The
session's normal failure is still an exception; its read-only retry state tells
the UI whether to offer the bounded replay:

```python
except Exception as exc:
    if not session.retry_available:
        await asyncio.to_thread(session.discard)
        if self._console_dictation_session is session:
            self._notify_console_dictation_error(exc)
        return
    confirmed = await self.run_worker(
        self.app_instance.push_screen_wait(
            ConfirmationDialog(
                title="Parakeet transcription failed",
                message="Parakeet failed. Retry this audio with faster-whisper?",
                confirm_label="Retry",
                cancel_label="Keep draft",
            )
        ),
        exclusive=False,
        exit_on_error=False,
    ).wait()
    if not confirmed:
        await asyncio.to_thread(session.clear_retry)
        self._finish_failed_console_dictation(session)
        return
    transcript = await asyncio.to_thread(session.retry_with_faster_whisper)
```

Factor the existing successful transcript insertion/idle/pending-action tail into one private helper used by both the first Parakeet success and accepted retry. Do not duplicate caret spacing, realtime adoption, undo-history invalidation, hands-free routing, or no-auto-send logic.
`_finish_failed_console_dictation(session)` clears the owning session, origin,
partial text, timers, and retained retry state and returns the Mic to idle; it
does not mutate the draft. Prompt cancellation, retry failure, or unmount uses
that same cleanup rather than leaving the Console in `transcribing`.

- [ ] Add a wiring test proving production sessions receive the app-owned factory and existing tests/fakes may omit it.
- [ ] Add a mounted Console test with active batch: Mic reaches recording, chip/status shows `Local transcription busy — dictation will run next.`, batch completes, dictation runs, transcript inserts at the captured caret, and no message is sent.
- [ ] Add a limit test at the exact derived byte cap: all accepted PCM is transcribed, warning copy is exactly `Limit reached — press Mic to continue.`, state returns to idle, timers are cancelled, and no new session starts until a second Mic press.
- [ ] Add a hands-free limit regression proving the loop exits without `OpenCapture`/auto-send/reopen; retained text follows the ordinary insertion path and another physical Mic press is required.
- [ ] Add a whole-segment command test where coalesced `ordinary text` + `console stop` arrives from one executor request and still produces one `VoiceFinal` plus one `VoiceCommand`, never dictated command text.
- [ ] Add retry tests for confirm and decline: the prompt appears only for retryable Parakeet failures when faster-whisper is installed; confirm replays exact failed/pending boundaries once then inserts normally; decline leaves the draft unchanged.
- [ ] Add teardown tests proving prompt cancel, screen unmount, app shutdown, session discard, retry success, and retry failure each clear retained PCM and cannot insert into a newer capture generation.
- [ ] Add a non-retryable/missing-faster-whisper test proving the bounded normalized error is shown with no confirmation and no native exception detail.
- [ ] Run `python -m pytest Tests/UI/test_console_dictation_streaming.py Tests/UI/test_console_dictation.py Tests/UI/test_console_voice_chip.py Tests/UI/test_console_controller_wiring.py Tests/UI/test_console_hands_free_wiring.py -q` and confirm intended failures.
- [ ] Replace the current headroom-derived byte constant with
  `pcm_byte_limit(sample_rate=CONSOLE_DICTATION_SAMPLE_RATE,
  channels=CONSOLE_DICTATION_CHANNELS,
  sample_width=CONSOLE_DICTATION_SAMPLE_WIDTH)` and use
  `DICTATION_MAX_SECONDS` for the timer; keep one source of truth.
- [ ] Display busy through the existing composer voice chip/status functions; do not add another widget or button.
- [ ] Change limit handling to stop/transcribe and exit any hands-free auto-reopen path. This is the smallest implementation of the approved explicit-resume rule; do not add a new paused hands-free state.
- [ ] Use the existing `ConfirmationDialog`; do not create a retry modal or persistent retry queue.
- [ ] Re-run the five focused UI test files and commit: `feat(console): finish bounded shared-executor dictation`.

## Task 7: Focused verification, macOS evidence, and honest open gates

**Files:**
- Modify: `backlog/tasks/task-603 - Restore-bounded-Parakeet-ONNX-dictation-buffers.md`
- Create: `Docs/STT_Evaluation/task-603/README.md`
- Create: `Docs/STT_Evaluation/task-603/macos-evidence.json`
- Modify only directly affected STT/Console documentation if focused verification proves copy is stale.

**Interfaces:**
- Consumes: the focused tests above, installed macOS Parakeet v2 INT8 artifact, real Console Mic path, and the task's release-gate criteria.
- Produces: reproducible focused verification evidence, concise implementation notes, and only truthfully completed acceptance criteria.
- Invariant: no full suite, no unrelated CI wait, no Windows/Linux claim, no TASK-605 promotion/removal.

- [ ] Run the union of only the test files changed in Tasks 1–6. Use exact file paths; do not run bare `pytest`.
- [ ] Run Ruff only on changed Python files, `python -m compileall` only on changed package modules, and `git diff --check`.
- [ ] Run a placeholder scan on the plan and changed code for `TBD|TODO|FIXME|pass  #|NotImplemented`; distinguish pre-existing retained markers from newly introduced placeholders.
- [ ] Run a type/contract consistency review: `ExecutorRequest.source`, optional `job_id`, segment-frame units, callback sequence IDs, retry token ownership, and shutdown order must match across parent, worker, facade, Lazy service, and UI.
- [ ] Using the repository virtual environment's absolute Python path and an isolated profile, run a macOS CPU smoke with the real installed Parakeet v2 INT8 bundle: open Console, press Mic, dictate English, stop, and confirm insertion without send.
- [ ] During the same smoke, start one real local Parakeet Library batch item first, press Mic while it is active, confirm the busy copy, dictate, verify batch -> dictation -> batch ordering, then hit/induce the bounded limit and confirm another Mic press is required.
- [ ] If practical with bounded sample audio, induce one retryable Parakeet failure and confirm the faster-whisper prompt/replay. If not safely inducible without corrupting an installed artifact, record that exact limitation and rely on the focused injected-failure test rather than manufacturing a destructive live condition.
- [ ] Record interpreter, OS/architecture, package versions, exact artifact lease/model/precision, timestamps, commands, observed UI states, and redacted outcomes in `macos-evidence.json`; validate it with `python -m json.tool`.
- [ ] Self-review the complete diff for a second executor/model load, hidden download, PCM disk staging, unbounded references, callback-under-lock, Mic warm-up blocking, hands-free auto-reopen, native path/error leakage, accidental default promotion, and legacy deletion.
- [ ] Request code review using `superpowers:requesting-code-review`; address only verified findings and rerun the focused tests covering each changed fix.
- [ ] Rebase onto current `origin/dev`, resolve conflicts without discarding unrelated upstream work, and rerun only focused verification affected by the rebase.
- [ ] Add concise Implementation Notes to TASK-603, including ADR-025, exact files/behavior, focused test commands, macOS evidence, retained legacy code, and open Windows/Linux/TASK-605 gates.
- [ ] Check AC1–AC5 only when directly evidenced. Keep AC6 and TASK-603 status open if representative Windows/Linux release evidence remains unavailable; do not mark the task globally Done merely because the macOS implementation can merge.
- [ ] Commit evidence/task hygiene: `docs(stt): record TASK-603 focused evidence`.

## Plan self-review checklist

- [x] Every approved behavior appears in at least one implementation step and one focused test: shared executor, no true streaming, one pending/coalescing, bounded overrun, dictation-next/no preemption, busy copy, exact limit copy, explicit Mic resume, faster-whisper confirmation, caret insertion, voice commands, cancellation, shutdown, and open platform gates.
- [x] Every new type has one owner and one reason to exist:
  `ParakeetDispatch` (artifact identity), `ParakeetBufferResult` (normalized
  result plus logical boundaries), `LocalSTTDispatchCoordinator` (admission),
  `DictationCaptureHandle` (one capture), `DictationAppendStatus` (typed bound
  result), `RetryableDictationBuffer` (bounded explicit retry),
  `RetryableDictationFailure` (sanitized ownership transfer to the blocking
  caller), and `VoiceLocalSTTBusy` (existing Console event transport).
- [x] No step creates a parallel queue, downloader, executor, model cache, recorder, dialog class, or persistence layer.
- [x] File paths and test commands are exact; no `TBD`, `TODO`, ellipsis-only implementation instruction, or unspecified “add tests” remains.
- [x] Public compatibility is explicit: only the facade constructor gains a keyword-only dependency; public method signatures remain stable; retained providers remain unchanged.
- [x] The plan does not require a 30-second sleep, full-suite run, unrelated CI, Windows/Linux access, legacy deletion, or default promotion.
