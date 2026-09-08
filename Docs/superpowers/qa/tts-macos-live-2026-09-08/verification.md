# Real local TTS on macOS ARM — 2026-09-08

TASK-32096 follows the delivery checks in PR #2520. Chatterbox CPU/MPS and
audio.cpp CPU/Metal now have real inference, whole-file decoding, physical audio
device completion, and independent transcription evidence. No model inference
or audio output was replaced with a fixture in these live runs.

## Defects found and repaired

- **audio.cpp Speak replies:** Speech Lab could generate successfully, but the
  Console destination check omitted the native capability reader and rejected
  an exact model with `TTSEffectiveResolutionError: catalog_unavailable` before
  synthesis. The handler now deliberately refreshes the native catalog before
  using the service's passive public capability snapshot API. This starts a
  stopped managed child and applies eligible staged settings while retaining
  exact model/voice validation and final endpoint authorization. Qodo found the
  stopped-child gap in the initial repair; eight new cases reproduced it before
  the follow-up fix. The original Console regression crosses destination resolution
  and admission for both an explicit voice and the server default voice; both
  cases failed before the repair and passed afterward.
- **Chatterbox installation:** an unbounded extra could resolve Chatterbox
  0.1.3 and fail building its obsolete `pkuseg` dependency. Even with 0.1.7,
  Perth 1.0.1 imports undeclared `pkg_resources`, suppresses its absence, and
  exposes a non-callable watermarker. Model loading then fails despite a
  successful Chatterbox import. The Chatterbox and all-tools extras now require
  `chatterbox-tts>=0.1.7` and `setuptools<82`. Core and build-system requirements
  are unchanged; watermarking is retained. The recovery guide explains the
  temporary cap and its removal condition.

## Runtime and source identity

- Host: Apple M5 Max, 128 GiB RAM, arm64, macOS 26.5.2; output device:
  **MacBook Pro Speakers**. Python 3.12.11 and Textual 8.2.8.
- Application base for the live runs:
  `27a7555c2e3333d4c5e38a48626cfdb18aa8226c`. The successful audio.cpp runs
  include this task's destination-reader repair. The initial failing run is
  retained separately.
- Managed cold-start follow-up runs use dev
  `7e81ed55db66ace04cb3dd1f8feb6d40a21f6f48` plus this task's repairs.
  The tested TTS runtime tree was unchanged by the rebase before that follow-up.
- Chatterbox 0.1.7, Torch/Torchaudio 2.6.0, Transformers 5.2.0, NumPy 1.26.4,
  Perth 1.0.1, setuptools 80.10.2, pydub 0.25.1, SoundFile 0.14.0 and
  sounddevice 0.5.6. Actual worker initialization logs identify CPU or MPS;
  MPS availability and tensor execution were checked outside the sandbox.
- Model: `ResembleAI/chatterbox`, revision
  `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`; downloaded checkpoint files are
  recorded in `chatterbox-model-provenance.json` under the evidence root below.
- Native runtime: `0xShug0/audio.cpp`, `release-0.5.1`, unchanged source commit
  `238ab6a9e321c17de8e120559f57efeedaeb1345`. Separate Release arm64 CPU and
  Metal binaries, six build jobs, custom `pocket_tts,supertonic` model set.
- Native model actually exercised: **Supertonic 3 F16**, preset **M1**,
  curated artifact `audio-cpp-supertonic-3-f16`, recipe
  `audio-cpp-0.5.1.supertonic.supertonic_3_f16` revision 1. Model repository
  `audio-cpp/audio.cpp-gguf`, revision
  `597048d9a920592808d7d4e2acd7b9c4596a143a`.
  GGUF: 312,784,196 bytes, SHA-256
  `b312b57797d40ac5c09d915893dbdbaf6405b7dc043f544776c5c95712dff88c`.
  Configuration and all ten voice styles are embedded in the package.
- CPU server SHA-256:
  `28afedc4a263bc6710cbd6985b77e7bf6d249bd9282c751b0e227d572dd1bf6f`.
  Metal server SHA-256:
  `d00827005f6b14ecbb6b343e77b5477d5cdc07d5b1ba8d4eb8bbe0c1e5d5aedb`.

## Live playback results

Each route mounts the production Speech Lab pane in a minimal Textual host,
presses Generate, and uses the production resolver, registry, backend, and
player. Two different Console messages then pass through the real auto-speak
decision with Speak replies enabled, trusted message snapshots, destination
authorization, TTS event handling, and audio-device drain.

| Route / WAV output | Speech Lab decoded / player time | Reply 1 decoded / device drain | Reply 2 decoded / device drain | Complete content |
| --- | --- | --- | --- | --- |
| Chatterbox CPU | 4.400 / 5.231 s | 5.080 / 5.134 s | 4.600 / 4.634 s | All 3 |
| Chatterbox MPS | 5.200 / 5.992 s | 5.080 / 5.134 s | 4.600 / 4.624 s | All 3 |
| audio.cpp CPU, Supertonic F16 | 6.252 / 7.256 s | 6.455 / 6.500 s | 6.446 / 6.492 s | All 3 |
| audio.cpp Metal, Supertonic F16 | 6.250 / 7.113 s | 6.453 / 6.476 s | 6.447 / 6.462 s | All 3 |

Speech Lab uses real `afplay` processes, all exit code 0. Its first three
timings come from application player-start/finish timestamps; the Metal run
records a monotonic timer. Console uses real sounddevice streams at the model's
sample rate: Chatterbox 24 kHz and Supertonic 44.1 kHz. Each reply reaches
`playing → stopped`; each run records two SinkStarted and two SinkDrained events
and no SinkFailed. Saved clips decode fully, contain finite non-silent samples,
and match their recorded SHA-256 hashes.

The three utterances exercise a Speech Lab sentence, a first automatic reply,
and a second reply ending with “playback complete successfully.” The complete
files were independently transcribed using cached faster-whisper English models.
The tiny recognizer read “Speak” as “Speed” in one MPS clip; the larger base.en
model correctly recovered the unchanged recording. Raw transcripts are retained.
Content checks allow punctuation/case, `SpeechLab` spacing, and the recognizer's
`complete`/`completes` verb-agreement variation; they require beginning, middle,
and end content. No audio was edited to satisfy those checks.

The primary verifier for MPS and native runs is `Systran/faster-whisper-base.en`
at `3d3d5dee26484f91867d81cb899cfcf72b96be6c`, CPU/int8. The Chatterbox CPU run
passed with tiny.en at `0d3d19a32d3338f10357c0889762bd8d64bbdeba`.

### Managed cold-start follow-up

The real app supervisor also owns separate CPU and Metal runs. Mounting leaves
the process stopped at generation 0; destination resolution starts generation 1.
Before each of two automatic replies, the harness explicitly shuts down the
child. The production request handler then starts generations 2 and 3, generates
the complete sentence and drains the real speaker stream. Service shutdown joins
all owned resources and returns to Stopped. These runs exercise Console speech;
the mounted Speech Lab pane receives no Generate action in this follow-up.

| Managed route | Reply 1 decoded / device drain | Reply 2 decoded / device drain | Complete content |
| --- | --- | --- | --- |
| CPU, Supertonic F16 | 6.455 / 6.506 s | 6.446 / 6.492 s | Both |
| Metal, Supertonic F16 | 6.453 / 6.494 s | 6.447 / 6.491 s | Both |

All four recordings pass the same full-file transcription and audio checks.
Together with the initial WAV and installed-wheel MP3 runs, the evidence covers
**19 real clips across seven configurations**.

## Installation verification

An unseeded Python 3.12 uv environment could not install the original wheel's
unconstrained Chatterbox extra because of the old `pkuseg` build. Selecting
Chatterbox 0.1.7 with the original wheel then installed setuptools 84.0.0 and
reproduced the missing-watermarker failure. Importing Chatterbox alone passed.

A second unseeded environment installed only the repaired wheel's `[chatterbox]`
extra from the populated package cache, without dependency overrides or shared
site-packages. All five checks passed: package isolation, Chatterbox import,
Perth resource import, callable watermarker plus bundled assets, and Chatbook
backend import. `uv pip check` passed for all 158 installed packages. Wheel
metadata checks cover both extras and the absence of these runtime constraints
from core. The large all-tools extra was not installed as a whole.

After preserving that extra-only evidence, a final **installed-wheel MPS/MP3**
run added sounddevice solely for playback observation and upgraded setuptools
to **81.0.0**, the newest version allowed by the compatibility cap. Python `-I`
kept the checkout off the import path. The wheel contains the original dev
runtime plus the repaired dependency metadata; this Chatterbox run does not
claim to exercise the separate audio.cpp source repair.

All three MP3 clips decoded and transcribed completely and played through real
afplay processes with exit code 0. Speech Lab decoded/player time was
4.880/5.899 s; the two automatic replies were 4.760/5.904 s and 4.840/5.892 s.
Both replies reached `playing → stopped`. These initial runs account for
15 clips across five configurations. The installed wheel's SHA-256 is
`26829fbcefc8d739bb40cd2a45961014b1b95b56e1f52e99a4e4c7b57bd4bacb`;
its exact runtime versions and separate run evidence are retained locally.

## Automated checks and review

- Native destination regression: **2 failed before / 2 passed after** the fix.
- Managed/Console/native/effective-selection/diagnostic cohort: **270 passed**.
  The managed follow-up adds **8 failed before / 11 passed after** cases covering
  cold start, restart, staged configuration, exact selection rejection, changes
  between preparation and passive validation, and old-consent rejection after a
  port change. Passive settings/profile discovery still does not launch a child.
- Chatterbox delivery and audio.cpp adapter cohort: **215 passed**, including
  the real Chatterbox subprocess protocol test. No optional backend skip.
- Independent follow-up review checked the deliberate preparation boundary,
  passive validation races and GET-only destination discovery; it found no
  additional correctness or security issue. Final synthesis retains its separate
  admitted-endpoint authorization check.
- All six derived-artifact preflight checks passed. All three changed Python files
  pass formatting and syntax/undefined-name checks. Full Ruff has the same
  57 existing handler, 5 existing Console-test and 3 existing managed-test
  diagnostics as the base, with no added diagnostics. The two targeted cohorts
  contain **485 distinct passing tests**.

No full repository test sweep was run. Initial tests reported a missing
pytest-timeout plugin; it was installed before the final backend cohort.
Upstream pydub/pkg_resources deprecations and the host's joblib semaphore warning
remain recorded in the logs.

## Evidence and reproduction

Local evidence root: `/private/tmp/tts-macos-live-validation`.

- `validate_live.py`: mounted application synthesis/playback harness.
- `run_native_validation.py`: loopback server ownership, readiness, real app run,
  and process termination/reaping. CPU uses port 18731; Metal uses port 18732.
- `validate_managed_live.py` and `verify_managed_content.py`: real managed cold
  start/restart playback and complete-file transcription; results are in
  `runs/audio_cpp-{cpu,metal}-wav-managed-cold/evidence.json`.
- `verify_content.py`: complete-file independent transcription and content checks.
- `live-summary.json`, `runtime-versions.json`, model provenance files, and
  `runs/{chatterbox-cpu-wav,chatterbox-mps-wav,audio_cpp-cpu-wav-fixed,audio_cpp-metal-wav}/evidence.json`.
  Each run retains its config, application log, and complete audio artifacts.
- `runs/chatterbox-mps-mp3-installed-wheel/evidence.json` and
  `installed-wheel-runtime-versions.json`: the separate installed-wheel MP3 run.
- `audio-cpp/setup.md`: exact source checkout, build flags, model download URLs,
  hashes, and server configuration. `audio-cpp/runtime-manifest.json` and
  `package-manifest.json` retain binary/package identity.
- `chatterbox-packaging-check/`: original/repaired wheels, isolated environments,
  probe code, metadata, install results, and failure/success logs.

The native config selects backend CPU or Metal, lazy loading, offline task TTS,
the absolute Supertonic GGUF/spec paths, and server-default M1. CORS and request
body logging are disabled. Externally launched servers were reaped; the managed
follow-up joined service shutdown after three owned generations per compute path.
Application config, data, caches and newly installed runtime/model assets are
isolated under this task's evidence root. The CPU transcription check reused
the earlier Kokoro validation's tiny.en model cache read-only, as recorded in
its evidence; the other recognizer assets were downloaded for this task.
The user's config hash matches its pre-validation value; the dirty main
checkout was not modified.

## Limits and platform observations

- These are real mounted production speech paths, not a complete manual GUI
  acceptance session or a human listening-quality assessment.
- This host lacks the offline `metal` compiler. The pinned runtime supports
  embedding shader source and compiling it at runtime; that build worked.
  Metal logs prove GPU selection, compiled compute pipelines and completed
  inference. Its optional tensor API probe fails and disables that optimization;
  ordinary Metal inference still succeeds.
- A temporary attempt to reuse another environment's site-packages exposed an
  incompatible TorchVision installation. That test harness contamination was
  removed before successful runs; all runtime environments are isolated.
- Sandboxed OpenMP initialization aborted a subprocess regression. The unchanged
  test passes outside the sandbox. Joblib reports the host's existing semaphore
  shortage and runs serially; disk free space was checked separately.
- Downloading PocketTTS Q8 does not qualify its runtime. Its named Alba voice
  asset returned HTTP 401, so the live native checks use the publicly available
  Supertonic package. No authenticated voice download was attempted.
- Other audio.cpp model families, Kokoro's PyTorch engine, Higgs, AllTalk and
  authenticated OpenAI/ElevenLabs inference remain outside this live matrix.
  Linux/Windows and CUDA were not exercised.

ADR required: no. Existing ADRs 023 and 050 govern the unchanged adapter and
native-runtime boundaries. The changes repair existing capability wiring and
installation requirements; no new runtime architecture or storage contract is
introduced.
