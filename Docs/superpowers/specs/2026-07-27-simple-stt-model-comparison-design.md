# Simple STT Model Comparison

## Goal

Run one indicative macOS comparison of the proposed Parakeet v2 and v3 INT8
ONNX models against their F32 forms and faster-whisper. The result is evidence
for the next routing decision, not a reusable qualification framework.

## Scope

Create one script at
`Helper_Scripts/Benchmarks/stt_model_comparison.py`.

The script accepts:

- a local JSONL case file;
- local directories for Parakeet v2 INT8, v2 F32, v3 INT8, and v3 F32;
- a local faster-whisper model directory; and
- an output JSON path.

Each case contains an ID, local WAV path, reference text, language, and tag.
The curated set has one clean clip for English and every currently routed v3
language, plus a few English accent/noise, silence, and long-form cases. Audio
is supplied locally and must already be PCM signed 16-bit, 16 kHz, mono WAV.
The script does not download, extract, convert, or publish corpus data.

Each model is loaded once. The required execution matrix is:

- Parakeet v2 INT8 and F32 run every English case and the silence case.
- Parakeet v3 INT8 and F32 run every declared supported non-English case and
  the silence case.
- faster-whisper runs every case, including silence.

Silence is recorded separately because WER and CER have no non-empty reference
denominator.

## Measurements

Use a small Unicode-aware normalizer implemented with the Python standard
library: Unicode normalization, case folding, punctuation-to-space, and
whitespace collapse.

For every scheduled model/case pair, record:

- normalized reference and hypothesis;
- word and character edit counts and reference-unit counts;
- elapsed transcription seconds;
- audio duration and real-time factor; and
- any execution error.

The summary reports micro-averaged WER and CER from aggregate edit/reference
counts, total elapsed time, aggregate real-time factor, and silence output.
No bootstrap, confidence interval, RSS sampler, timestamp validator, automatic
threshold, promotion decision, or cross-platform claim is produced.

## Output and failure behavior

Write one JSON report containing the case identities, model labels, environment
summary, raw rows, and aggregate summaries. Write through a temporary sibling
file and replace the destination only after every required run finishes.

Invalid input, a missing model, or a failed required transcription exits
nonzero and does not publish a complete report. The console identifies the
failed model and case without printing credentials.

The report is labeled `indicative_macos_comparison`. Its README states that it
does not by itself open the production promotion gate.

## Verification

One focused test module covers Unicode normalization, WER/CER edit counts,
case validation, aggregate calculations, and atomic report publication using
fake model callables. Real model execution is performed manually on macOS and
the resulting JSON plus a short Markdown interpretation are committed as
evidence.

## Deferred work

Reusable multi-engine suites, cloud-provider evaluation, integration smoke
tests, reusable corpus preparation, bootstrap statistics, and automated
promotion gates are deferred to TASK-1023.

## ADR check

ADR required: no.

Reason: this is a one-shot diagnostic script that does not change production
routing, artifact ownership, runtime boundaries, storage, or service contracts.
ADR-025 remains the governing decision for the eventual routing change.
