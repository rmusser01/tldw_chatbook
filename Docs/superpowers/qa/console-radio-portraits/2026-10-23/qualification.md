# Console radio portrait qualification

Date: 2026-10-23 (local). TASK-36550 **Done** after the corrected artwork review. ADR-242 applies;
ADR-052 still governs the packaged ComfyUI image-edit workflow. The completed
implementation is in the isolated `codex/console-radio-portraits` worktree.
Runtime, package hashes and the final sink source digest are in `runtime.json`.

## Verified

**Artwork correction, 2026-10-24.** User review rejected v1 for exposure drift
and unmatched mouth/blink pairs. The user then explicitly authorized local
compositing. The corrected `paired-v2` set reuses one original base, each mouth
patch across both eye states, and one eyelid patch across all mouth states.
Full-size visual review and exact pixel assertions pass; the actual mounted app
journey passes on this corrected set. Prior v1 captures are historical evidence,
not approved final artwork. Audio and storage evidence remains valid.
See `imagegen-v1-rejection.json`, `paired-manifest.json`, `paired-review.png`
and `paired-observations.json`.

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

## Original ChatGPT artwork journey — v1 rejected

The user replaced unreachable ComfyUI/MOSS with ChatGPT image generation. Built-in
reference edits produced six individual Samira poses plus a selected `blink_small`
replacement. All seven originals are 1254×1254 PNGs. They are preserved byte-for-byte;
the production boundary rejected all seven at the unchanged 1-megapixel limit.
After explicit user authorization, Pillow LANCZOS resized separate copies to
1024×1024, without cropping. The historical six-file `selected/` set uses the replacement; use `paired-v2/selected/` for the correction.
[Exact prompts, original/prepared hashes and selected files](imagegen-manifest.json)
are recorded. Assets are in workspace `output/imagegen/console-radio-samira-2026-10-23/`
(ignored local deliverables, not shipped defaults).

`PYTHONPATH=. .venv/bin/python Docs/superpowers/qa/console-radio-portraits/2026-10-23/imagegen_import_probe.py`
passes with actual TldwCli, an owned private profile and the network guard intact.
Six imports, a selected replacement preserving the other five normalized hashes,
source/staged review, explicit Apply, six individual Console poses, idle timer,
reopen equality and Remove all pass. Original portrait bytes remain unchanged.
[Mounted observations](imagegen-observations.json) record the six distinct fitted
rasters, a fixed 23×12 cell box / 24×24 pixel mosaic, and revisions 1 → 2.
This probe uses normal offline preview controls and publishes no speech snapshots.

Actual editor/Console SVGs and their PNG renders are under `preview/` beside the
assets. SVG rendering used CairoSVG with its Homebrew library path and substituted
Menlo for unavailable Fira Code; the original SVG captures remain intact. Color and
monochrome journeys pass. Reviewed color rest/open/blink captures show stable head
and backdrop placement. Fine mouth/blink differences are subtle at the 24×24 mosaic
resolution; native graphics-terminal rendering was not exercised.

The initial private fixture's provider-setup overlay obscured the Console. Its
existing-profile onboarding flag and a visible fitted-avatar barrier now make the
capture witness the actual rail. An intermediate wait also observed an empty rail
during fit; the final probe waits for the currently visible widget rather than a
pre-fit child. No production visibility, ownership or allocation gate was bypassed.

These checks proved the external-art import workflow, but user review subsequently
rejected the actual v1 artwork. The corrected set below supersedes it. Neither run
qualifies live in-app adapter generation. No ChatGPT provider was added and no
ComfyUI job was submitted.

## Corrected paired artwork — 2026-10-24

After explicit authorization, `compose_portrait_matrix.py` composites only mouth
and eyelid patches from the preserved generated copies onto the original Samira
portrait. No new generation or dependency is involved. The output is six separate
1024×1024 PNGs in `output/imagegen/console-radio-samira-2026-10-23/paired-v2/selected/`.
The original rest frame is pixel-identical to the original source. Every pixel
outside the two disjoint feathered masks equals the source exactly; every mouth
is identical across its open/closed-eye pair, and eyelids match across mouth
states. Six raster hashes are distinct; all original input file hashes remain
unchanged. These are decoded-pixel assertions, not a subjective lighting score.

The labeled [2×3 face-detail comparison](paired-review.png) was visually inspected
for mouth/eyelid state, lighting, alignment and patch edges. It shows three mouth
states, each with open and closed eyes. The rest of the portrait remains untouched.
[Manifest](paired-manifest.json) records source/patch/final hashes and dimensions;
[v1 rejection measurements](imagegen-v1-rejection.json) retain the prior failure.
The compositor refuses to overwrite an existing correction.

`PYTHONPATH=. .venv/bin/python Docs/superpowers/qa/console-radio-portraits/2026-10-23/paired_artwork_probe.py`
passes through actual TldwCli in a guarded owned private profile. It imports the
corrected files, temporarily places a rejected v1 `blink_small` as a replacement
control, then replaces it with corrected `blink_small`. All six normalized draft
frames match the corrected matrix and the other five slots remain unchanged.
Explicit Apply, six normal timer-painted Console previews, idle, exact reopen,
Remove and original-byte preservation pass. [Observations](paired-observations.json)
record six distinct fitted rasters and revisions 1 → 2. Color rest/open/blink-open
actual-app captures were inspected; the fixed 23×12-cell / 24×24-pixel mosaic still
makes facial details subtle. The full-resolution matrix is the visual-pair witness.
This artwork-only run publishes no speech observations and makes no new TTS claim.

The dedicated probe reuses the original guarded artwork journey with checked
replacement anchors; it preserves that historical probe and its Pocket TTS caller.
Ruff lint/format, artifact SHA/dimension/size/JSON checks and `git diff --check` pass.
No production renderer, image-generation pipeline, default artwork or profile was
changed. Local compositing repaired this sample set; automatic generation still
requires reviewing all six poses before Apply. Existing speech/recovery results
remain applicable; no redundant device run or full suite was performed.

## Real Pocket TTS qualification

User explicitly selected Pocket TTS and authorized installation. Pocket TTS 3.3.0
is installed in isolated `output/tools/pocket-tts-venv/`; model assets are cached in
`output/tools/pocket-tts-hf-cache/`. The original app environment, default voice and
user profile were not modified. The local server used English (2026-09), the public
without-voice-cloning model and built-in `alba`, listening on 127.0.0.1:18765.
Install/serve follows the [official Pocket TTS documentation](https://github.com/kyutai-labs/pocket-tts/blob/main/README.md).

`PYTHONPATH=. .venv/bin/python Docs/superpowers/qa/console-radio-portraits/2026-10-23/pocket_tts_probe.py`
passes against real Pocket synthesis, actual TldwCli and MacBook Pro Speakers.
Its private profile configures the existing OpenAI-compatible native `/tts` dialect;
normal Console message snapshots, real character-authority context navigation,
admission, WAV repair, event delivery and native sink remain in place. No external
adapter, synthesis response, audio device, timer or speech snapshot is faked. Observers
delegate to the real backend and native callbacks. Microphone input is not opened.

- Native WAV: the entire 5.32-second source WAV is byte-identical to the captured
  device-submitted WAV. Normal, small and open mouth states appear automatically;
  measured silence closes the mouth. Natural completion is 34.770 ms after the
  final scheduled DAC interval. The existing known-waveform run separately proves
  nonzero quiet-background closure and the 100 ms paint target.
- Incremental transport: 39 real HTTP chunks from the same server feed their verified
  PCM16 payload directly to the existing `pump`, with complete source-to-device
  equality. Confirmed playing begins 200.079 ms before the HTTP response ends.
  This is a transport probe: the app's native Pocket adapter remains WAV-only and
  buffers the body to repair its header. No PCM-provider capability is inferred.
- Actual Console Stop succeeds near the last 0.4 seconds of pending output; a second
  message replaces the first voice, and changing conversations retires the exact
  replacement owner. Late ownership cannot animate the new context.
- The production hands-free sentence dispatcher and sequencer speak two real Pocket
  utterances with zero failures. The next sentence starts 276.150 ms after the prior
  scheduled tail; the listening/OpenCapture intent follows the final tail by 23.619 ms.
  Capture-end input is simulated and intents are observed; microphone/STT hardware
  and the complete acoustic hands-free loop are not qualified by this probe.
- Installed local faster-whisper `tiny.en` recognizes the complete original sentence
  from both source and device-submitted WAV, including “The final word is complete.”
  The streamed sample also retains that final word. This is automated speech-content
  evidence, not an independent human/acoustic audition. Playable WAVs are preserved.

[Summary and hashes](pocket-summary.json), [raw observations](pocket-observations.json),
[transcripts](pocket-transcripts.json), [native WAV](pocket-speech.wav) and
[streamed WAV](pocket-http-stream.wav) record the evidence. One native WAV is stored
because source and device captures are identical. The previous 53.25 ms device
latency calibration applies to this same local route; this probe does not claim a
new physical latency measurement. The reviewed production source commit is recorded
in the summary. Pocket protocol + Console presentation regressions: **26 passed**
using `--basetemp=/private/tmp/radio-pocket-final` and the targeted pytest flags above.
Local transcription used the installed `faster_whisper.WhisperModel('tiny.en',
device='cpu', compute_type='int8', download_root='output/tools/whisper-qa')`, with
`language='en', beam_size=5` on each captured WAV. Both new opt-in probes pass Ruff lint/format; JSON hashes and `git diff --check` pass.

## Qualification limits

No independent physical/acoustic loopback, microphone/STT hardware, other OS/device,
native graphics-terminal rendering or real in-app image-adapter generation is claimed.
User-selected external image generation is the approved artwork qualification method.
No full test sweep, push, PR or merge ran. All earlier unrelated baseline findings
remain recorded above and in the owning reviews.
