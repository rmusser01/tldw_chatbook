# Console radio portrait qualification — partial

Date: 2026-10-23 (local). TASK-36550 remains **In Progress**. ADR-242 applies;
ADR-052 still governs the packaged ComfyUI image-edit workflow. The completed
implementation is in the isolated `codex/console-radio-portraits` worktree.
Runtime, package hashes and the final sink source digest are in `runtime.json`.

## Verified

- Installed wheel / actual TldwCli: six marked imported poses, explicit Apply,
  navigation to Console, automatic portrait timer paint from an issued synthetic
  test observation, exact session-owner rejection, reopen matching source/revision/
  frames, Remove and explicit Prepare/reapply. Original portrait bytes survive.
  No _tick call or patched character speech reader drives this journey. The
  observation is test data, **not live TTS evidence**. Unrelated localhost model
  discovery uses the existing offline test seam; the child network guard remains on.
- Native full-profile F9 create/inspect/isolated restore/open controls, using an
  installed wheel and real terminal suspend. A fresh application verifies six hashes,
  revision, canonical digest and original portrait. The durable authority UUID is
  preserved; an owner with a different exact DB-object scope is rejected. The
  original plain F9 control journey also passes.
- Actual Actor Pack archive bytes and character JSON dumps remain identical after
  adding a local performance. A real LocalChatbookService export omits performance
  records/frame digests. The targeted SQLite tests cover unchanged sync logs and
  soft-delete/restore behavior.
- Real MacBook Pro Speakers / sounddevice / CoreAudio: 24 kHz mono PCM streaming and
  the handler's complete eligible-WAV artifact path both submit the **entire known
  waveform** unchanged. The waveform is two 440 Hz periods separated by a nonzero
  sub-gate pause, followed by silence. WAV synthesis is supplied test data through
  the existing test service; no real TTS model is claimed.
- Automatic avatar timer output is timestamped at actual terminal driver flushes.
  Exact waveform sample boundaries are matched inside captured device-submitted PCM
  and mapped with the production observer's monotonic clock offset. Final scored
  transitions: PCM 7.056/40.298/32.893/32.776 ms; WAV
  29.799/62.801/14.240/47.012 ms (onset / nonzero-pause closure / onset / silence).
  Driver latency is 53.25 ms, reported separately. `analyze_audio.py` asserts all
  transitions are within 100 ms and completion follows both the native ACK and the
  last scheduled sample. Raw observations and summaries are beside this record.
- The real macOS afplay path reports playing/stopped while keeping timing invalid,
  mouth closed and the bar unqualified; it invents no amplitude or lip sync.

`calibration.wav` is the input. `pcm-device-submitted.wav` and
`wav-device-submitted.wav` capture buffers submitted to PortAudio, **not microphone
or physical-loopback recordings**. Terminal flushing is not a photon measurement.
No physical acoustic capture/loopback, perceptual real-speech recording, or other
OS/device/player qualification was available in this run.

## Measured correction and review

CoreAudio delivered NativeFinished about 24 ms before the last scheduled DAC sample.
The initial WAV handler could therefore release completion early. Deterministic
future-tail tests failed with and without portrait observation. The correction keeps
native acknowledgement mandatory and additionally respects the bounded valid final
output interval. It does not depend on amplitude preparation or a portrait channel.
Invalid/missing timing retains native acknowledgement; Stop/replacement remains
immediate. Deferred close stays tracked by the existing notification queue, including
a held-close regression. No extra thread, timer, player or provider was added.

Final device recheck: native ACK was 24.37 ms (PCM) / 24.25 ms (WAV) early; logical
completion was 130.70 / 28.96 ms after ACK, after the scheduled tail in both cases.
The PCM figure also includes native stream-close/notification settlement overhead.
See the TASK-36550 review and ADR-242 clarification.

## Commands and results

All pytest invocations used `-q -o addopts='' -p no:cacheprovider`, private
`--basetemp=/private/tmp/radio-36550-*`, `--tb=short --show-capture=no`.

- `Tests/ProductionApp/test_console_radio_portraits.py`: **1 passed** (journey6).
- `Tests/Backup_Recovery/test_portrait_performance_restore.py`: **1 passed** (restore2).
- Original `Tests/ProductionApp/test_backup_restore_end_to_end.py` `[plain]`:
  **1 passed** (f9-control).
- New Actor Pack case + character JSON file + LocalChatbookService export file,
  with `--import-mode=importlib`: **10 passed** (exports3).
- Focused envelope, presentation, repository/migration, generation, ownership and
  mounted Console/Personas authoring gate: **136 passed** (gate; before the tail
  correction, which changes only the sink).
- Final sink/pump/presentation, Console presentation, sentence speech, spoken
  feedback, Buddy guarded speech, speech admission and lazy-native-import files:
  **226 passed** (tail-reviewed). This includes early ACK, Stop/replacement,
  invalid timing, analysis failure, exact-owner and held-close controls.
- Explicit opt-in hardware probe copied temporarily to
  `Tests/Audio/_radio_live_probe.py`, marker `real_audio_device`, actual terminal:
  **3 passed** (live-reviewed). The temporary test is removed afterward. The probe
  source remains `live_audio_probe.py` beside this record; it is never normal CI.
- `python .../analyze_audio.py`: both local routes meet the target and final-tail
  assertions; real external fallback checks pass.
- New test/probe files pass Ruff lint/format. Modified legacy files have no added
  Ruff diagnostics; existing baseline style findings are retained. Edited ranges
  are formatted and `git diff --check` passes. Unit 8's style/timer/architecture
  evidence remains applicable because this unit changes no UI styles/geometry.

No full sweep ran. Earlier baseline limits are recorded in atomic reviews:
three obsolete architecture/bare-shell checks reproduced on pre-feature source,
six unchanged module-budget violations, and two Actor Pack golden-byte baselines.
They are not counted as new passing checks here.

## Still required

1. Real reference-image generation and one selected regeneration with reviewed
   identity/alignment at rail size. User specified **MOSS image creation** on
   `http://192.168.2.216:8188`. Direct `/system_stats` and `/object_info` requests
   repeatedly timed out, including a proxy-free recheck after clarification.
   **No image upload or generation job was submitted.** Server reachability and the
   MOSS workflow/node/model contract remain unknown. The current ComfyUI adapter is
   packaged H3 edit-only (ADR-052), so a MOSS path cannot be inferred or silently
   replaced by that graph. The pending user question requests reachability and the
   saved workflow name.
2. Real TTS synthesis, audible final syllables, perceptual capture, stop/replacement
   during a real speech tail, hands-free sentence/listening handoff and conversation
   changes with real speech. Known-waveform device tests and mocked ownership tests
   do not replace these obligations. MOSS was clarified as image creation, not TTS.
3. Complete the remaining acceptance criteria and final task review only after that
   live evidence exists. No push, PR or merge has been performed.
