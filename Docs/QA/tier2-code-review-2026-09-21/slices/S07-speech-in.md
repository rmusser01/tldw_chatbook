# S07 — `Audio/` + `STT/`

**Coverage:** files read in full: 3 | sampled: 26 | mechanical only: 26 (of 55).
Full: `Audio/realtime_mic_tap.py`, `Audio/wav_writer.py`, `Audio/__init__.py`. Sampled: 26 (large regions + all
candidate rows). Mechanical only: 26 — AST/grep sweeps for subprocess, sqlite, fsync, `os.replace`, `id()`-keys,
`optional_deps`, timestamps, and every candidate row.

## Findings

### P1 [D1] — The recorder's VAD gate silently discards the tail of every capture chunk that is not an exact multiple of the 20 ms VAD frame; a user-set `dictation.buffer_duration_ms` of 150 loses 10 ms of speech out of every 150 ms
- Where: `Audio/recording_service.py:511-543` — `for i in range(0, len(chunk) - frame_size + 1, frame_size)` with **no
  residual carry between calls**. Reached with `chunk_size = int(buffer_duration_ms * 16)` from
  `Audio/dictation_service_lazy.py:522`; the value is a free-form integer Input (100–2000, no step) at
  `UI/Dictation_Window_Improved.py:352`, persisted at `:605-609`.
- Evidence: driving the real `_process_audio_chunk` with an always-speech VAD and a full-size chunk:
  ```
  buffer_duration_ms=100 -> 0.00%      =150 -> 6.67% (10.0 ms of every 150 ms)
  =200 -> 0.00%                        =250 -> 4.00% (10.0 ms of every 250 ms)
  =300 -> 0.00%                        =333 -> 3.90% (13.0 ms of every 333 ms)
  =500 -> 0.00% (the default)          =750 -> 1.33%
  LiveDictationService (dictation_service.py:127, DEFAULT_CHUNK_SIZE=1024) -> 128 B = 6.25%
  ```
  Clean iff `buffer_duration_ms % 20 == 0`. The dropped tail is not even retained as pre-roll — `_preroll_frames` is
  fed only inside the loop.
  **The same repo already has the correct version:** `Audio/meeting_capture.py:518-548` keeps a `_gate_carry`
  bytearray across calls with the comment *"no audio is discarded from the dictation stream just because a chunk
  crosses a 640-byte boundary"*. `recording_service` is the drifted copy.
- Why it matters: a periodic gap at a fixed offset in every chunk, on the one path where the audio **is** the
  product. `recording_service.py:129-145` (`VAD_PREROLL_MS`) documents a live incident — parakeet transcribing
  "stop" as "dot"/"top", "send" as "and", leading consonant gone — diagnosed as an aggressiveness-3 onset problem
  and fixed with pre-roll. A periodic 10 ms hole is a **second, independent producer of exactly that symptom class**
  and was never ruled out.
- Recommended correction: give `_process_audio_chunk` the same `_gate_carry` residual `meeting_capture._gate` has
  (6 lines). Belt-and-braces: round `chunk_size` at `dictation_service_lazy.py:522` to a multiple of the frame.
- Size: S · Confidence: **verified**
- Pinning test: `Tests/Audio/test_recording_vad_preroll.py`, `test_dictation_vad_finalization.py`,
  `test_audio_integration.py` all feed **exact multiples of 640 bytes** ("20ms at 16kHz"), so none exercises the
  remainder. No test states the drop as a requirement.

### P1 [D1/D4] — Voiceprint writes (biometric-derived) are atomic but not durable: `Audio/voiceprint.py` re-rolls the atomic-write helper without its `fsync`, after TASK-32808.5 closed
- Where: `Audio/voiceprint.py:235-256` `_atomic_write` (mkstemp → write → `os.replace`, **no `os.fsync`, no directory
  fsync**), used at `:365` (the key file), `:532` (`save()`, the centroid envelope) and `:656` (`export()`).
- Evidence: `rg -n "fsync|atomic_file_ops" tldw_chatbook/Audio tldw_chatbook/STT` → **zero `os.fsync` in all 39,378
  lines of the slice**. `Utils/atomic_file_ops.py:103` does `f.flush(); os.fsync(f.fileno())` before its
  `os.replace`. `atomic_write_text(path, content, mode=0o600)` is a behaviour-for-behaviour drop-in **plus** the
  fsync — the caller need only supply `mode=0o600` (the helper defaults to `0o644` while `_atomic_write` relies on
  mkstemp's implicit `0o600`).
- Why it matters: `os.replace` publishes the *name* immediately; without fsync the *data* may still be in page
  cache. A power loss between the two leaves a zero-length or torn envelope at the live path. `load()` then hits
  `_read_envelope`'s `_EnvelopeError` and returns `LoadResult(voiceprint=None, …)` — the user's enrolled voiceprint
  is gone and they must re-enroll. **The key file (`:365`) is worse: a torn key makes every previously saved
  envelope permanently undecryptable.**
- Size: S · Confidence: verified
- Already covered: TASK-32808.5 is **Done** and this site was missed — the "already-handled, now shown insufficient"
  case. *(Lead's note: the shared helper still lacks a parent-dir fsync and the Darwin `F_FULLFSYNC` barrier — see
  the repo-wide finding in `report.md`. Adopting it here is a strict improvement, not a complete fix.)*

### P2 [D1] — `swiftc` is invoked with no `timeout=` inside the exclusive `meetings-prepare` worker; the sibling `pactl` call in the same function does pass one
- Where: `Audio/system_audio_tap.py:235-239` vs `:305` (`timeout=5`) — both inside `probe()`'s dispatch.
  Reached from `UI/Screens/meetings_screen.py:437` `@work(exclusive=True, group="meetings-prepare", thread=True,
  exit_on_error=False)` → `_owner.prepare()` → `system_audio_tap.probe()`.
- Why it matters: not on the event loop (good), but `exclusive=True` means **a wedged compiler holds the group
  forever**: the Meetings screen never leaves its prepare state, and no later prepare can run for the process
  lifetime. A Swift toolchain stalling on a code-signing/licence prompt is the realistic trigger on macOS.
- Recommended correction: `timeout=120`, catching `TimeoutExpired` alongside the existing `returncode != 0` branch
  (which already returns `None` → "helper unavailable, pick a virtual device"). · Size: S · Confidence: verified

### P2 [D1/D4] — Two ffmpeg conversions run with no `timeout=`; the repo majority passes one
- Where: `STT/parakeet_onnx.py:136-153` (`_prepared_wav`) and `STT/transcribe_cpp.py:284-301`.
- Evidence: **AST census of every `subprocess.run/call/check_call/check_output` under `tldw_chatbook/`** → 26 call
  sites, **14 pass `timeout=`, 12 do not**: these two, `Local_Ingestion/{audio_processing:1332,
  transcription_service:1171, video_processing:1140}`, `TTS/audio_service.py:412`,
  `Media/local_media_reading_service.py:4680`, `UI/CodeRepoCopyPasteWindow.py:358`,
  `Backup_Recovery/isolated_restore.py:397`, `app.py:{20887,21085,21177}`.
- Why it matters: ffmpeg on a malformed/truncated container can block indefinitely. `transcribe_cpp.py:284` runs
  inside the STT worker subprocess, so `STT/executor.py`'s containment can still kill the tree — that is the
  mitigation, not the design. `parakeet_onnx.py:136` is the ordinary transcription path.
- Recommended correction: add `timeout=` on both, following the repo's own precedent of a named module constant
  (`Media_Playback/stream_resolve.py:187 YTDLP_TIMEOUT_SECONDS`). · Size: S · Confidence: verified

### P2 [D3] — `dictation.buffer_duration_ms` is the only one of five config reads in the same constructor with no validation, and it is the one that feeds a device parameter
- Where: `Audio/dictation_service_lazy.py:324-326` (bare `get_cli_setting`, multiplied and passed as `chunk_size` at
  `:522`) vs its four siblings `_resolve_stop_join_timeout` `:352`, `_resolve_silence_threshold` `:386`,
  `_resolve_vad_aggressiveness` `:419`, `_resolve_vad_preroll_ms` `:455` — each type-checks, range-checks,
  `math.isfinite`-checks and logs a fallback, with docstrings naming the incidents that forced it ("`nan`/`inf` are
  valid TOML floats", "`int(float('inf'))` raises `OverflowError` not `ValueError`").
- Why it matters: a TOML string survives to `int("500" * 16)` (a 48-digit `chunk_size`); `0` or a negative gives
  `chunk_size <= 0`. Both land in `frames_per_buffer`/`blocksize`. `set_buffer_duration` at `:1437` **does** clamp
  (`max(100, min(2000, …))`) — so the same value is validated on the UI path and unvalidated on the config path.
- Size: S · Confidence: verified

### P2 [D3] — `Audio/dictation_metrics.py` (380 lines) is now fully orphaned: TASK-32810 deleted its only consumer
- Evidence: `rg -n "dictation_metrics|DictationPerformanceMonitor|get_performance_monitor" --glob '*.py' .` →
  **zero hits outside the file** (no source, no test, no fixture). Its sole importer was
  `Widgets/dictation_performance_widget.py`; `git log -3 -- …` → `dda82ba6dd fix(polish): Widget bundles .5/.6/.10 —
  delete dead widget … (task-32810)`. Two doc references only.
- Why it matters: 380 lines that also carry the slice's only naive-local-time timestamps (`:95, 179, 296, 305`).
- Size: S · Confidence: verified
- Already covered: TASK-32807 family but **on none of its sub-lists** — this is a *new orphan created by* 32810, so
  it postdates every list.

### P2 [D3] — `Audio/console_dictation.py` (252 lines) is dead code kept alive only by its own test; the module that replaced it says so in a comment
- Evidence: `rg -n "\bConsoleDictationSession\b|\bConsoleDictationError\b"` → only
  `Tests/Audio/test_console_dictation.py`. `CONSOLE_DICTATION_MAX_BYTES` is **redefined**, not imported, by the live
  owner (`UI/Console_Modules/dictation.py:169`). `UI/Console_Modules/dictation.py:567` reads: *"the one-shot backend
  this replaced (`Audio/console_dictation.py`)"*. `app.py:4756` builds `LazyLiveDictationService`, not this.
- Why it matters: it is the only remaining caller of
  `Local_Ingestion.parakeet_v2_installer.verify_parakeet_v2_bundle` in `Audio/`, keeping a stale model-resolution
  path looking live. · Size: S · Confidence: verified

### P3 [D3] — `Audio/dictation_service.py` (652 lines) is reachable only through a module TASK-32807.1 is already deleting
- Evidence: `rg -n "\bLiveDictationService\b"` (excluding `Lazy*`) → `Tests/Audio/test_property_based.py` only,
  which patches `tldw_chatbook.Widgets.voice_input_widget.LiveDictationService`; and
  `rg -n "voice_input_widget|VoiceInputWidget" tldw_chatbook/` → only the file itself (one of 32807.1's 21
  unreferenced widget modules). Nothing in `tldw_chatbook/` imports `Audio.dictation_service` by any spelling.
- Why it matters: it is the **second copy** of the dictation service and the one carrying the `use_vad=True` +
  `DEFAULT_CHUNK_SIZE=1024` combination the P1 above measures at 6.25% loss. Landing 32807.1 orphans it; it should
  go in the same PR rather than becoming the next `dictation_metrics.py`. · Size: S · Confidence: verified

### P3 [D1] — Downloaded ONNX diarizer models are `os.replace`d with no fsync, but the damage is loud, not silent
- Where: `Audio/diarizer_engine_onnx.py:562` (`_fetch_asset`). Unlike voiceprint, the byte stream is size-bounded and
  sha256-verified immediately before the replace (`:561`), and `load()` re-hashes both files on every open by
  default (`:869-871`, `verify_hashes: bool = True`; **no production caller passes `False`**). A torn write surfaces
  as `ValueError("model hash mismatch")` and `assets_to_fetch` refetches. · Size: S · Confidence: verified

### P3 [D3] — the two ruff fatals in `Audio/meeting_owner.py` are annotation-only and nothing introspects them
- **Promotion check run as asked:** `rg -n "get_type_hints|TypeAdapter|validate_call"` over the whole repo → the only
  `get_type_hints` in production is `MCP/gateway_runtime.py:515`, applied to MCP tool handlers; no pydantic adapter,
  dataclass, or `Protocol` runtime check touches `meeting_owner`'s signatures. **Not promoted.** Fix: add
  `MeetingCapture` to a `if TYPE_CHECKING:` block (the runtime import already lives at `:1184`). · Size: S

### P3 [D3] — dead per-sample `struct` fallback inside the PortAudio real-time callback
- Where: `Audio/recording_service.py:471-487` — the `else` branch of `if NUMPY_AVAILABLE and np is not None` inside
  `audio_callback`, doing a Python-level `struct.pack` per sample plus a function-body `import struct`.
- Evidence: `recording_service.py:184-194` — `__init__` raises `AudioRecordingError` unconditionally when
  `not NUMPY_AVAILABLE`, so **no instance can ever reach the branch**. It is also the single worst thing that could
  run on a PortAudio callback thread. · Size: S · Confidence: verified

### P3 [D1] — `id()`-keyed delivery map in `voice_process_lifetime` has no strong reference pinning the key, unlike its sibling in `voice_process_protocol`
- Where: `Audio/voice_process_lifetime.py:360` and `:391` — `self._deliveries[id(record)] = receiver.receive(...)`;
  popped at `:445`. The sibling `voice_process_protocol.py:1381` stores `self._owned[id(record)] = (lane, record)` —
  **the value holds the record**, so the id cannot be recycled — and `release()` re-validates `owned[1] is not
  record` before deleting.
- Size: S · Confidence: **inferred** — see UNVERIFIED.

## Candidate triage
**RETIRED with evidence — the 2026-09-17 seed entry:** `inline_path_check_no_pv` `STT/executor_worker.py:503`.
The inline block at `:484-503` is complete and in one respect **stricter** than the helper: it rejects
`PureWindowsPath` drives/roots/backslashes before joining, then `.resolve()`s and `is_relative_to`s.
`safe_join_path` (`path_validation.py:372`) + `validate_filename` (`:284`) does **not** check for `:` — a
drive-relative component survives `validate_filename` and `Path("/root") / "D:evil"` drops the base on Windows.
Worse, `safe_join_path` → `validate_path(..., allow_hidden=False)` **rejects any hidden component** (`:196-205`), so
adopting it would break a GGUF living under a dotted directory. **Adoption is a regression** unless
`validate_filename` gains a `:` check and the call passes `allow_hidden=True`. D4 style-only, P3, not a security gap.
**CONFIRMED:** `os_replace_no_atomic` `voiceprint.py:252` (P1 — no comment or task reference anywhere near it;
missed, not excluded); `diarizer_engine_onnx.py:562` (downgraded to P3, hash-verified).
**RETIRED:** `dotted_section_setting` ×6 — all the **2-arg** dotted form `get_cli_setting("dictation.x", DEFAULT)`,
which `config.py:8508-8531` documents TASK-1771 as having fixed; not the broken shape. `except_exception_pass` ×17 —
**all 17**, each documented best-effort teardown where raising is worse (`streaming_sink:817` literally says "Swallow
EVERYTHING: a raise here kills the PortAudio thread"; `diarizer_local`'s SIGTERM→SIGKILL ladder each has a `finally`
that still posts the sentinel). `except BaseException: pass` in `voice_turn_coordinator._fence_active` — swallowing
`CancelledError` in an `async def` is the usual hazard, but `_invoke_effect` never awaits inline (it
`ensure_future`s), so no await point exists inside the guarded blocks. `id_keyed_dict`
`voice_process_protocol.py:1381/1391/1398` — value holds a strong ref **and** `release()` re-validates identity;
textbook-correct. `mutable_class_attr` `recording_service.py:118` — `deque(maxlen=0)` whose `append()` is a no-op, a
documented `__new__`-safety fallback; `executor_process_tree.py` ×4 — `ctypes.Structure._fields_`.
`tempfile_no_secure` ×6 — `mkstemp` (0o600 by construction), `mkdtemp` + explicit `chmod(0o700)`, `NamedTemporaryFile`
+ explicit `chmod(0o600)`. One asymmetry noted: `parakeet_onnx.py:132` has **no** chmod where its twin
`transcribe_cpp.py:276-283` does — content is a 16 kHz WAV under the default 0600 mkstemp mode, so no exposure; fold
into the ffmpeg-timeout fix. `raw_mkdir` ×6, `raw_1024x1024` (`STT/contracts.py:23` is a named commented bound),
`strftime` (`meeting_owner.py:1221` is a human-facing recording **folder name**; `utc_now_iso()` is already adopted
for the stored `started_at` at `:1246`), `try_import_guard` ×17 (`recording_service.py:19/45/54/69` catch `Exception`
not `ImportError` **on purpose** — documented at `:33-44`: `sounddevice` calls `Pa_Initialize()` at import and raises
`PortAudioError`, a `RuntimeError`, which used to fail `import tldw_chatbook.app`).
**NONE IN THIS SLICE:** god modules >5k lines (max `voice_turn_coordinator.py`, 2287); sqlite or DB anywhere
(`rg -n "sqlite3|\.transaction\(|execute\(|_DB\b"` → 8 hits, all `ISOLATION_*_DB` **decibel** constants);
`_maybe_await` definitions or call sites (**no instance to give the lead**); `set_status_line` candidates
(`diarizer_local.py:556 _set_status` is an integer progress callback on a non-Textual class).
**`Audio/realtime_mic_tap.py` (the brief's first target): clean, no finding.** The `Condition`, per-thread
`_in_flight_by_thread`, reentrancy exclusion, bounded stop wait, and `stop_recording()`-outside-the-lock ordering
are each correct and each carry the review round that produced them; `use_vad=False` also keeps it clear of the P1.
One latent nit not worth filing: if `on_frames` raises inside `mark_ready`'s flush loop (`:266`), `_flushing` stays
`True` forever — but the only wired consumer is documented thread-safe and non-raising.

## D4 observations for repo-wide Phase 3
1. **VAD frame slicing, 2 copies, drifted, one loses audio.** `meeting_capture.py:518-548` (`_gate_carry`, correct)
   vs `recording_service.py:511-543` (no carry). The drift feeds the transcription wire, so it is a D1 (P1 above).
2. **Atomic-write, helper exists, ignored.** `Utils/atomic_file_ops.py` (49 importers, fsyncs the file) vs
   `Audio/voiceprint.py:235` and `Audio/diarizer_engine_onnx.py:562`. **`Audio/` is a TASK-32808.5 blind spot** —
   re-run that sweep with `rg -n "os\.replace|os\.rename"` rather than by module list.
3. **`subprocess.run` timeout — no helper, 12 non-adopters repo-wide** (full AST census above). There is no shared
   runner, and the 14 adopters each spell their own constant. A `Utils/run_bounded(argv, *, timeout)` is the obvious
   home; ADR-worthy only if the timeout budget needs to be per-family.
4. **`path_validation.safe_join_path` has two properties that block adoption where an inline check is currently
   used**: `validate_filename` does not reject `:` (Windows drive-relative), and `validate_path`'s default
   `allow_hidden=False` rejects dotted components. **Before Phase 3 recommends "adopt `safe_join_path`" anywhere,
   both need settling** — otherwise the adoption is a functional regression, as it would be at
   `STT/executor_worker.py:503`.
5. **Dead-code orphan chains.** TASK-32810's widget deletion orphaned `Audio/dictation_metrics.py` (380 lines) with
   no follow-up. TASK-32807.1's pending deletions will orphan `Audio/dictation_service.py` (652) the same way.
   A cheap guard: after each deletion batch, re-run a reachability pass rather than deleting from a pre-computed list.

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| The P1 audio drop is audible / measurably degrades transcription accuracy, not just byte-accounted | cannot run the app; no live microphone | set `dictation.buffer_duration_ms = 150` in a scratch `TLDW_CONFIG_PATH` profile, dictate a fixed sentence 10× with and 10× without the `_gate_carry` patch, diff WER |
| `Delivery` objects in `voice_process_lifetime._deliveries` do not retain their `record`, so `id()` reuse is possible | traced the set/pop pair but did not read `ReceiveStream.receive`'s return type end to end | `rg -n "class ReceiveStream" -A 60 tldw_chatbook/Audio/voice_process_protocol.py`, then `pytest Tests/Audio/test_voice_process_lifetime.py -q` with an assert that every `_deliveries` key is popped by a matching `release` |
| No caller anywhere constructs `AudioRecordingService(use_vad=True)` with a chunk size not enumerated | enumerated the 7 in-repo construction sites | `rg -n "AudioRecordingService\(|_mic_factory\(|recorder_factory\(" tldw_chatbook/ Tests/` and diff against the 7 |
| `console_dictation.py` / `dictation_metrics.py` are not re-exported through an `__init__` not read | read `Audio/__init__.py` in full (neither is in `_LAZY_EXPORTS`); did not audit every package `__init__` | `pytest --collect-only -q 2>&1 \| rg -c "console_dictation\|dictation_metrics"` and `rg -n "console_dictation\|dictation_metrics" --glob '**/__init__.py'` |
| The ffmpeg no-timeout calls can actually hang (vs always erroring) on this app's inputs | would need a crafted malformed container | `ffmpeg -i <truncated .m4a> -ar 16000 -ac 1 -c:a pcm_s16le out.wav` under `strace`/`dtruss` |
