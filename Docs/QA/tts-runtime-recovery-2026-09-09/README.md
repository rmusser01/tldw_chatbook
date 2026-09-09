# Kokoro PyTorch and macOS native speech validation

TASK-32111 replaces Kokoro's incompatible placeholder PyTorch model and random
waveform path with the official runtime. TASK-32112 and TASK-32113 extend real
audio.cpp recovery and model qualification. The runtime decision is
[ADR-140](../../../backlog/decisions/140-official-kokoro-pytorch-runtime.md).

## Kokoro results

Every row below generated one clip through the mounted production Speech Lab
Generate control and two trusted Console messages with Speak replies enabled.
WAV replies crossed the real streaming sink and reached natural device drain;
MP3 replies and Lab artifacts used the real `afplay` process, which exited zero.
All runs awaited handler/service cleanup and verified unchanged source hashes.

| Installation / runtime | Device and format | Lab / reply durations | Result |
| --- | --- | --- | --- |
| Python 3.12, Kokoro 0.9.4, Torch 2.6.0, Transformers 4.57.6 | CPU, WAV | 6.425 / 6.850 / 6.100 s | Playback and content passed |
| Same runtime | MPS, WAV | 6.425 / 6.850 / 6.100 s | Playback and content passed |
| Fresh installed wheel, Python 3.12, Torch 2.14.0, Transformers 5.16.1 | MPS, WAV | 6.425 / 6.850 / 6.100 s | Playback and content passed |
| Same fresh wheel, `bf_emma`, speed 1.25 | CPU, MP3 | 5.175 / 5.600 / 5.000 s | Playback and content passed |
| Fresh installed wheel, Python 3.13, kokoro-onnx 0.6.1, ONNX Runtime 1.29.0 | CPU, WAV | 6.101 / 6.336 / 5.717 s | Playback and content passed |

Observers delegated every actual call to upstream `KModel.forward` or
`Kokoro.create_stream` and recorded the real class/device. MPS used the upstream
Fourier implementation on CPU through `_CpuSTFT`; the neural model stayed on
`mps:0`. The British run asserted the Lab request's voice and speed as well as
the speed received by all three real model calls. Earlier Lab rows used its
default `af_alloy` selection, while Console used `af_heart`.

Independent faster-whisper `base.en` transcription checked beginning, middle,
and final clauses of all 15 complete saved files. All passed this content gate.
This is not exact word-for-word certification: the verifier explicitly permits
case/punctuation, SpeechLab spacing, and `complete`/`completes` agreement; one
British transcript also began `The second reply` instead of `This second reply`.
Raw transcripts remain in the evidence. No microphone capture of room sound or
full Console shell navigation is claimed. Kokoro sink captures record the
complete source passed to the real device, not its hardware callback samples.

The initial fresh-wheel source hashes match production at `df254584d0`, before
the review follow-up below. Both actual `local_tts` installations succeeded. Python 3.13 excluded
`kokoro`, included `audioop-lts`, passed two installed-wheel dependency-guidance
tests, and completed real ONNX playback. Resolution alone was not treated as
installation evidence: generic macOS-target resolution selected older native
wheels than the actual macOS 26.5.2 host.

## Reproduced defects and repairs

- Official v1 weights failed to load into the old placeholder transformer; its
  later synthesis functions ignored learned output and returned random noise.
  The replacement loads upstream component weights and retains every waveform
  segment. Tests cover language aliases, real voice-pack dimensions, mixing,
  speed, invalid output, and failures after an upstream segment.
- Default Console requests ignored Global's PyTorch engine setting. The bridge
  now resolves the immutable applied engine before locking, while preserving an
  explicit request's selection. Direct initialization and ONNX fallback honor
  the configured PyTorch checkpoint; first use reaches deferred model loading.
- Upstream `torch.angle` failed on Torch 2.6 MPS. A numerical test proves that the
  CPU Fourier wrapper preserves upstream magnitude, phase, inverse values and
  shape exactly. No approximate inverse or process-wide fallback is enabled.
- Cancelled loaders could publish a model after close. Retained worker tasks now
  join initialization, download, voice loading and inference, including the
  timestamp path. Repeated cancellation/close probes observed no late model or
  remaining task.
- Misaki's English installer could raise `SystemExit`; Japanese/Chinese missing
  extras produced generic retries; oversized non-English segments were silently
  truncated upstream. Typed errors now retain fixed setup/length guidance in
  both UI surfaces. Native inference failures no longer masquerade as missing
  configuration. Non-English text above 510 phonemes per paragraph fails before
  synthesis, and newlines remain explicit boundaries.

Initial observer failures are retained and excluded from the passing matrix:
the Lab ONNX switch was initially left enabled, `mps:0` failed an overly literal
device assertion, and the first British Lab run retained its default voice and
speed. Corrected runs asserted the actual controls/model arguments. The initial
fallback created a duplicate shared checkpoint; its absence before the run and
exact hash/mtime were verified before removing only that task-created copy.

## Automated checks and provenance

Review follow-up closes named-voice traversal, absolute-path and symlink escapes
in both the wrapper and the actual backend download/load paths. A pre-existing
partial-download symlink also reproduced outside-file corruption; downloads now
use exclusively created temporary siblings and clean up only their own file.
Streaming and timestamps now share the same voice-prefix language map in both
engines, including Hindi, Italian, and Brazilian Portuguese. Public wrapper API
contracts have Google-style documentation.

The MPS review concern does not apply to supported Kokoro 0.9.4: `TorchSTFT` has
no parameters or registered buffers, and its window is a plain tensor moved to
the input device on each call. The numerical test now calls `.to("mps")` on the
wrapper before comparing transform, phase and inverse with upstream CPU values.

Windows CI exposed a deferred-focus race in an unchanged GGUF keyboard test.
The corrected test waits for actual selector focus, overlay focus, highlight,
and selected value while retaining real keypresses. Local observation captured
the old navigation button still focused after `mode.focus()` returned. The
diagnostic inventory failure was also PR-owned: removing the old NLTK/placeholder
logs left a stale pin. Statement review found six fewer diagnostic calls, one
fewer model-path candidate, and no new sinks; the replacement runtime log records
only the validated compute device. The inventory was regenerated after review.

The next Fast Lane run passed 754 tests and exposed a separate MCP Test Tool
teardown race. A visible raw editor could remain in Textual's compositor after
its component styles were cleared. TASK-32114 batches the awaited panel removal,
while still clearing the preview and posting nonce revocation first. A real
Escape/timer regression reproduced the exact missing-gutter `KeyError` before
the fix; both Escape tests and all 26 selected lifecycle/preview tests passed
afterward. Formatting passes and the two existing files add no Ruff findings.
The regression deliberately observes Textual's private lifecycle boundary;
production uses its public `batch_update()` API.

The pre-Buddy integration targeted dev
`c4a7b1911f14181faa471eb1ea209cfdeed98226`. At that revision, all measured TTS
source hashes matched the review playback wheel below; that wheel predates the
Library changes from dev and the MCP teardown fix. [CI follow-up](ci-followup.json)
records the failing job, deterministic reproduction and targeted repair checks.
The combined diagnostic inventory reproduces exactly, retaining dev's one new
redacted Library count diagnostic and all reviewed Kokoro changes.

The pre-Buddy review regression run passed 340 tests with one optional ONNX skip.
The diagnostic guard reproduced its inventory exactly. New/replaced-file Ruff
and all five review-modified Python files' formatting pass; the three existing
files retain their baseline Ruff findings with none added.

The review wheel was installed into the same isolated Python 3.12/3.13
environments and replayed after all review fixes. All nine complete clips passed
independent content checks: MPS WAV (`af_heart`), CPU MP3 (`bf_emma`, speed 1.25),
and Python 3.13 ONNX WAV, each through Speech Lab plus two Speak replies. The
installed Python 3.13 dependency-guidance checks also passed. Every measured
production hash matched the source qualified by that run, cleanup joined, and the
user's configuration hash was unchanged. These are additional to the 15 initial clips.
The same transcription qualifications above apply, including the British
`The second reply` and `completes` variants.

Pre-Buddy review wheel SHA-256:
`d0edef43e3b9690fe036fa2a9f9291c36e7f7a8f7b705e62084ff14e8fc3e430`.
[Review validation](review-validation.json) preserves that revision's source hashes,
runtime/device observations, audio hashes and complete raw transcripts.

The targeted Kokoro, registry/bridge, admission, diagnostics and architecture
run passed 301 tests with one optional ONNX test skipped in the PyTorch-only
environment. After the last language-error and optional-test correction, all
65 focused runtime/diagnostic tests passed. New/replaced Python files pass Ruff;
all seven changed Python files pass formatting and the diff passes whitespace
checks. Existing-file Ruff findings decreased from 129 to 126, with no added
code/message findings; this is not a claim of repository-wide lint cleanliness.
No full test-suite sweep was run.

Rebased verification against dev `80f29a9a1dcd9307662714233c65605e4c517b11`
passed 302 tests with one optional skip. The one failing census had counted the
wheel's generated `build/lib` copy as an extra runtime owner. Moving that owned
build tree outside the checkout made its rerun pass: all 303 targeted tests
passed across the run and this rerun, without changing production or the guard.

Official model repository: `hexgrad/Kokoro-82M`, revision
`f3ff3571791e39611d31c381e3a41a3af07b4987`. Checkpoint SHA-256:
`496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`.
The qualified wheel SHA-256 is
`f7d50fcc3772252acaa86badf764be1213a59052a30f26623b3118de71f3f583`.

[Evidence summary](evidence-summary.json) records runtime versions, audio and
source hashes, raw transcripts, delivery/cleanup observations, native results,
model provenance and lint comparison. Task-local raw logs, WAV/MP3 files,
observer scripts, isolated installations and download manifests remain under
`/private/tmp/tts-runtime-recovery-validation`.

## Remaining coverage

Japanese/Chinese dependency failures are tested; their real language models and
non-English pronunciation are not qualified here. CUDA, Windows/Linux,
long-duration stress, exhaustive voices/blends and real Kokoro mid-inference
cancellation remain outside the live matrix. Kokoro cancellation is covered by
controlled real worker-thread tests; the real native fault matrix below covers
audio.cpp. This work does not qualify cloud credentials or other untested engines.

The additional native results and their unresolved upstream/content limitations
are recorded in [native-validation.md](native-validation.md).
