# TASK-593 Indicative macOS STT Comparison

> **Evidence label: `indicative_macos_comparison`.** This is a small,
> single-host diagnostic. It is not a production promotion gate and does not
> authorize a routing, artifact-ownership, or legacy-provider change.

## Sources and scope

The run used CPython 3.12.11 on macOS 15.6 arm64 with `onnx-asr==0.12.0`,
`onnxruntime==1.27.0`, `faster-whisper==1.2.1`, and
`ctranslate2==4.8.1`. Inference used CPU providers.

Exact corpus rows, derived-case commands, model-file hashes and sizes, and run
identity are recorded in [provenance.json](provenance.json).

The uncommitted local corpus contains 29 signed 16-bit, 16 kHz, mono PCM WAV
cases:

- the first streamed FLEURS test row for English and each of the 24 routed v3
  languages, from
  [`google/fleurs`](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd)
  at revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`;
- one synthetic Indian-English sample generated with the macOS `Rishi` voice;
- one deterministic white-noise derivative of the clean English FLEURS case,
  measured at -12.76 dB whole-file SNR;
- one 63.36-second English case made by concatenating that clean clip and its
  reference six times; and
- one 5-second artificial silence case with an empty reference.

The model snapshots were downloaded to explicit local directories and loaded
offline:

- [`istupakov/parakeet-tdt-0.6b-v2-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/tree/0bbb45a3365852604aef28b538a8f066f4ccaa85),
  revision `0bbb45a3365852604aef28b538a8f066f4ccaa85`;
- [`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/tree/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce),
  revision `8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`; and
- [`Systran/faster-whisper-base`](https://huggingface.co/Systran/faster-whisper-base/tree/ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66),
  revision `ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66`.

Each Parakeet directory contains exactly the requested seven-file bundle. The
complete pinned faster-whisper snapshot contains `config.json`, `model.bin`,
`tokenizer.json`, and `vocabulary.txt`; `preprocessor_config.json` and
`vocabulary.json` are not present at that revision.

## Command

Run from the repository root:

```bash
.venv/bin/python \
  Helper_Scripts/Benchmarks/stt_model_comparison.py \
  --cases Docs/STT_Evaluation/task-593/cases.jsonl \
  --v2-int8 .benchmarks/stt-task-593/models/parakeet-v2 \
  --v2-f32 .benchmarks/stt-task-593/models/parakeet-v2 \
  --v3-int8 .benchmarks/stt-task-593/models/parakeet-v3 \
  --v3-f32 .benchmarks/stt-task-593/models/parakeet-v3 \
  --faster-whisper .benchmarks/stt-task-593/models/faster-whisper-base \
  --output Docs/STT_Evaluation/task-593/report.json
```

The one real run exited 0. All 89 scheduled model/case rows completed and the
report records no execution errors.

## Results

WER and CER are Unicode-normalized micro averages and exclude silence. Model
load time is excluded; elapsed time and real-time factor (RTF) cover
transcription. Scope differs by family, so faster-whisper's all-case aggregate
is not directly comparable with the routed Parakeet subsets.

| Model | Scheduled/successful | WER | CER | Elapsed (s) | Audio (s) | RTF | Silence output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Parakeet v2 INT8 | 5/5 | 16.87% | 12.74% | 2.102 | 94.931 | 0.0221 | empty |
| Parakeet v2 F32 | 5/5 | 15.06% | 9.86% | 3.617 | 94.931 | 0.0381 | empty |
| Parakeet v3 INT8 | 25/25 | 17.45% | 4.99% | 7.601 | 275.300 | 0.0276 | empty |
| Parakeet v3 F32 | 25/25 | 10.43% | 2.42% | 7.845 | 275.300 | 0.0285 | empty |
| faster-whisper base INT8 | 29/29 | 43.24% | 14.46% | 24.516 | 365.231 | 0.0671 | `you` |

The raw v2 aggregate shows 41.90% lower RTF for INT8, but that number is
first-call-sensitive. Models ran once in fixed order with INT8 first, no
warm-up, and no repetitions. INT8 was faster on 3 of 5 matched v2 cases: the
first timed case took 0.202 seconds for INT8 and 1.554 seconds for F32. Across
the remaining four matched cases, INT8 was 7.92% faster. V2 timing is
therefore inconclusive. Its WER was 1.81 percentage points higher and CER was
2.88 points higher than F32.

The raw v3 INT8 aggregate RTF was 3.11% lower than F32, while WER was 7.02
points higher and CER was 2.57 points higher. That timing remains indicative
because it comes from the same fixed-order, single-run design.

### Derived stress cases

Word results are shown as edits/reference units (WER). The severe artificial
noise case does not represent typical background-noise conditions or support
generalization to other noise types or levels.

| Case | Parakeet v2 INT8 | Parakeet v2 F32 | faster-whisper base INT8 |
| --- | ---: | ---: | ---: |
| Synthetic `Rishi` accent | 0/14 (0.00%) | 0/14 (0.00%) | 0/14 (0.00%) |
| White noise, -12.76 dB SNR | 18/19 (94.74%) | 18/19 (94.74%) | 26/19 (136.84%) |
| Six-copy, 63.36 s long form | 6/114 (5.26%) | 6/114 (5.26%) | 29/114 (25.44%) |
| Artificial silence output | empty | empty | `you` |

Both Parakeet v2 variants preserved all six long-form repetitions but
pluralized the final `year` in each one. Faster-whisper emitted only five of
the six repeated clauses. Both v3 variants also returned empty output on
silence.

## Observed failures and conclusion

There were zero execution errors: no model-load or transcription exception was
recorded in any of the 89 scheduled rows. Quality failures still occurred. The
noise case had high WER for all three English-capable runs, faster-whisper
omitted one repeated clause in the long case, and faster-whisper produced
`you` on silence. All Parakeet variants returned empty silence output.

F32 was more accurate than INT8 for both Parakeet families. V2 performance is
inconclusive because a large first-call difference dominates its aggregate and
the run had no warm-up or repetition. V3 INT8 had a slightly lower raw
aggregate RTF but a materially higher WER. The stock v3 INT8 result does not
provide strong evidence for default promotion. Faster-whisper had the highest
mixed-scope aggregate WER, but it also handled every case, so this run does not
invalidate its broader fallback role. No production routing decision follows
from these observations.

## Limitations

- Each natural-language result comes from only the first streamed FLEURS test
  row for one locale; there is no speaker, topic, utterance-length, or
  difficulty balance.
- The accent case is a synthetic macOS voice, the noise case uses one fixed
  severe white-noise mixture at -12.76 dB whole-file SNR, the long case repeats
  one utterance and reference six times rather than using natural long-form
  speech, and silence is artificial.
- There was one run on one Apple-silicon macOS host. It does not characterize
  variance, model-load cost, peak memory, or Linux/Windows behavior.
- Models ran once in a fixed order, with no warm-up and no repetitions.
  First-call initialization can dominate short-case timing, as observed for
  v2, so timing differences are not stable benchmark estimates.
- Parakeet v2, Parakeet v3, and faster-whisper follow different routing scopes.
  Cross-family aggregate comparisons are therefore directional, not
  apples-to-apples.
- The committed report is reproducible only when the uncommitted local audio
  and pinned model snapshots described above are reacquired.
