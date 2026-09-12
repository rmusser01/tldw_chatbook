# Live Kokoro validation

These opt-in scripts qualify real Kokoro synthesis, complete audio playback,
cancellation and bounded repetition. They use the mounted Speech Lab pane and
the trusted Console destination/admission/handler path. They do not navigate
the complete Console shell or record acoustic loopback from a microphone.

Run one invocation at a time, with other model/audio work paused. Ordinary
pytest collection and either script's `--help` do not run speech or ASR.

## Supply assets and choose the code under test

Prepare the optional runtime and assets yourself before running. The runner
does not install packages or download weights or language data. PyTorch needs
the official checkpoint, adjacent `config.json`, and the selected local `.pt`
voice pack. English also needs the installed `en_core_web_sm` package. Other
languages need their Misaki language extras/resources. ONNX needs a model and
voices binary. WAV uses the app's real sink and file player; MP3 also needs
FFmpeg for independent complete-file decoding. The output directory must be
new and its parent must exist.

Use an interpreter with the relevant runtime already installed. The supplied
`--expected-package-root` is checked against the package actually imported by
the private worker. For source work, expose that checkout explicitly:

```sh
export PYTHONPATH="/abs/checkout:/abs/checkout/packages/tldw_profile_core/src"
/abs/venv/bin/python -B /abs/checkout/scripts/validate_live_tts.py \
  --engine pytorch --device cpu \
  --model /abs/assets/kokoro-v1_0.pth \
  --voice-dir /abs/assets/voices --voice af_heart \
  --expected-package-root /abs/checkout/tldw_chatbook \
  --output /abs/evidence/new-cpu-run --play-audio
```

For MPS, use the same command with `--device mps` and a different output
directory. The report records the actual model device and Fourier module;
the production MPS implementation may place Fourier operations on CPU.

For CUDA, use `--engine pytorch --device cuda` with a CUDA-enabled PyTorch
installation and a new output directory. The worker explicitly rejects
unavailable CUDA before model loading. `pytorch_device` records the concrete
selected device (for example `cuda:0`), GPU name, total memory, compute
capability, PyTorch version and its CUDA build version. Each native call checks
the actual `model.device` against that selected device; CPU fallback or a
different GPU fails qualification. Device ordinals follow `CUDA_VISIBLE_DEVICES`.
Retain the host's NVIDIA driver version and physical GPU identity separately
with your environment manifest; the CUDA build version is not the driver version.
ONNX validation remains CPU-only.

Device families also pass the shared input validator before output creation,
including programmatic admission that bypasses argparse. PyTorch is loaded
through the optional-dependency helper after private worker profile setup;
a missing runtime reports the `local_tts` install extra. Help and module import
remain inert and never load that runtime.

For ONNX:

```sh
/abs/venv/bin/python -B /abs/checkout/scripts/validate_live_tts.py \
  --engine onnx --device cpu \
  --model /abs/assets/kokoro-v1.0.onnx --voices /abs/assets/voices-v1.0.bin \
  --voice af_heart --expected-package-root /abs/checkout/tldw_chatbook \
  --output /abs/evidence/new-onnx-run --play-audio
```

For an installed wheel, use its interpreter with `-I`, omit source
`PYTHONPATH`, and point `--expected-package-root` at that interpreter's
`site-packages/tldw_chatbook`. The worker preserves isolated mode. The report
contains imported package paths and source hashes, distribution versions,
actual loaded runtime module hashes, and hashes of the model/config/voice.
A Git revision is included only when the actual package is in a checkout.

## Select scenarios and language

The default `--scenarios playback,cancel,repeat` runs one mounted Speech Lab
generation and Play action, a Console warmup, cancellation during an observed
native call, a full successor, and six repeated Console replies through the
same service/model. Select a subset explicitly or use `--repeats 1` through
`--repeats 12`. Use `--format mp3` to qualify the encoded Console file-player
path. `--phase-timeout`, `--cleanup-timeout` and `--run-timeout` set finite
deadlines; the whole-run maximum is two hours.

To qualify a language, supply UTF-8 text and the matching voice. The explicit
`--language` must match the production language inferred from its voice
prefix, for example `--language es --voice ef_dora --text-file /abs/spanish.txt`.
Speech Lab receives that explicit language selection; Console follows its
normal voice-derived language path. For non-English PyTorch, keep individual
sentences below 510 phonemes and use newlines for explicit segments.

Both engines use Kokoro's model-aligned Misaki frontends for Japanese and
Mandarin. In the same environment as the selected TTS engine, install
`misaki[ja]` for Japanese or `misaki[zh]` for Mandarin. Japanese additionally
requires the UniDic dictionary data: run `python -m unidic download` with that
environment's interpreter before validation. The UniDic Python package alone
does not contain the dictionary. Missing extras or dictionary data produce
setup guidance; the application does not install them automatically.

ONNX passes the resulting Japanese/Mandarin phonemes to the selected ONNX
model, without loading a PyTorch speech model. French `fr` and `fr-fr` both
use the upstream `fr-fr` locale. These frontend rules follow
[ADR-142](../../../backlog/decisions/142-kokoro-east-asian-phonemization-across-engines.md).
Include the optional package versions and dictionary file hashes in language
qualification receipts alongside the model and voice hashes.

Cancellation defaults to four copies of the selected text separated by
newlines. Supply `--cancel-text-file /abs/longer.txt` if the model completes
before Stop can overlap a call. An interval that misses inference fails; the
runner does not insert sleeps inside the model or manufacture a passing gate.
Observed PyTorch `KModel.forward` and ONNX `InferenceSession.run` calls delegate
their original arguments/results unchanged. Cancellation means the request
stops while a call is active and joins its native work; it does not claim hard
preemption of that native computation.

CUDA observations synchronize the selected device before native entry and
after the original forward call returns or raises. `host_returned_at` records
the Python return boundary; `exit_at` is recorded only after the completion
barrier succeeds, alongside `cuda_synchronized: true`. A failed barrier leaves
the call incomplete and records `synchronization_exception`, so cleanup cannot
pass with an unverified GPU completion. Barriers run without the observation
ledger lock, allowing Stop to be recorded while completion is pending. They
cover all streams on the selected device and add observation overhead; these
intervals establish native-call overlap and completion, not kernel-level
profiling or proof that a GPU kernel occupied every instant in the interval.

## Read the evidence

Each run owns a private profile, data/cache directories, original/decoded
audio, application/worker logs and `evidence.json`. It does not change HOME or
the user's configuration. The worker forbids network connections and runtime
installation; missing assets fail explicitly. Audio and text remain in the
run directory, so use synthetic text when publishing evidence.

`runtime_passed_content_pending` means every requested runtime scenario
passed. Per-request records separate real inference, encoded/decoded audio,
physical playback, lifecycle, and resource ownership. They include complete
frame counts and hashes, file-player exit/elapsed time or real sink drain and
device-frame counters. A zero player exit alone cannot pass a shortened clip.
Device callback hashes include its padding/silence and describe rendered
buffers, not an acoustic recording.

Cancellation records require Stop between native entry/exit, no further
native call or audio for that cancelled request, joined production ownership,
and an unchanged old record after successful successor playback. Shutdown
checks retained backend work, actual active native calls, handler tasks,
service operations/responses/leases and file-player processes. An early
ownership release while native work continues fails. A parent-enforced timeout
terminates only its private worker process group and records `failed_cleanup`;
forced termination is never joined-cleanup evidence.

After each settled request, the report records current RSS and peak RSS
separately, thread count, ownership, and available MPS memory counters. The
observer keeps callback counters/hashes, writes source audio to disk, and
retains no run-wide list of waveform arrays. Read warmup and subsequent
settled samples as bounded retention observations. They are not a universal
memory-leak threshold or a long-duration soak. The harness does not force
garbage collection or empty GPU caches to improve those numbers.

CUDA runs also synchronize before the pre-model, per-request settlement and
post-cleanup memory snapshots. Each includes `cuda_device`,
`cuda_allocated_bytes`, `cuda_reserved_bytes`, `cuda_peak_allocated_bytes` and
`cuda_peak_reserved_bytes`, with `cuda_synchronized: true`. Peaks are cumulative
allocator peaks for this worker; they are not reset between phases. Allocated
memory measures live tensors, while reserved memory includes the caching
allocator's retained blocks. These counters omit allocations outside PyTorch's
allocator and do not establish total GPU process memory. CPU/MPS and ONNX runs
do not initialize CUDA just to collect these counters.

## Verify full content separately

Use an environment with Chatbook and its optional faster-whisper runtime,
plus an already-present faster-whisper model directory. An English-only model
cannot qualify non-English content; supply a multilingual snapshot and the
desired ASR language (or omit `--language` for detection).

```sh
/abs/asr-venv/bin/python -B /abs/checkout/scripts/verify_live_tts_content.py \
  --evidence /abs/evidence/new-cpu-run/evidence.json \
  --model /abs/assets/faster-whisper-model \
  --language en --output /abs/evidence/new-cpu-run/content.json
```

For custom or multilingual text, add at least three `--anchor` arguments in
the order they occur near the beginning, middle and end. Every successful
phase, including repetitions and the successor, is transcribed in full.
Cancelled clips remain in the runtime report and are excluded from the
full-success content denominator. Missing or changed success audio fails
verification. The separate report is bound to the runtime report/audio hashes
and records ASR model hashes, runtime versions, raw text and segment times.

Only case, punctuation and whitespace are normalized. Both ordered anchors
and exact normalized text must match for `content_passed` (exit 0).
`content_review_required` (exit 1) preserves lexical ASR differences for review;
it does not rewrite them or imply a model defect. The runtime report remains
unchanged.

Both commands apply the shared path validator before filesystem work. The
content verifier validates the evidence schema and unique phase IDs, derives
audio authority from the selected evidence file's parent, and rejects audio
outside that run, symlinks and nonregular files. ASR consumes an opened,
hash-verified stream with no-follow descriptor traversal; this guarded reader
currently requires macOS or Linux support. Windows qualification remains in
TASK-32156. Optional dependency lookup uses a disposable private profile and
restores its environment afterward. Invalid input or missing dependencies
produce a CLI error (exit 2), before transcription.

## Cheap regression checks

```sh
python -m pytest Tests/TTS/test_live_validation_harness.py Tests/TTS/test_live_tts_input_validation.py -q
```

The targeted module uses synthetic records and small local WAVs. It checks
inert imports/help, asset/opt-in admission, inference interval and cleanup
negative controls, full-duration playback, complete WAV decoding, hash binding,
and multilingual content coverage. No live model, audio device, ASR, download,
or server is needed by these tests.

CUDA controls use a fake runtime and bounded thread gates to check admission,
fallback rejection, synchronization ordering on normal/error exits and
settled allocator samples. They validate the observation tooling; a real
NVIDIA host is still required to qualify CUDA inference and playback.

The harness follows ADR-023 (TTS operation ownership), ADR-039 (Global/Studio
settings ownership), and ADR-040 (Speech Lab audition). It is a test-only tool;
production defects discovered by a run must be fixed and verified separately.
