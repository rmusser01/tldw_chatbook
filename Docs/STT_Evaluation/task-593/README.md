# TASK-593 Indicative macOS STT Comparison

> **Evidence label: `indicative_macos_comparison`.** This is a small,
> single-host diagnostic. It is not a production promotion gate and does not
> authorize a routing, artifact-ownership, or legacy-provider change.

## Sources and scope

The run used CPython 3.12.11 on macOS 15.6 arm64 with `onnx-asr==0.12.0`,
`onnxruntime==1.27.0`, `faster-whisper==1.2.1`, and
`ctranslate2==4.8.1`. Inference used CPU providers.

The uncommitted local corpus contains 29 signed 16-bit, 16 kHz, mono PCM WAV
cases:

- the first streamed FLEURS test row for English and each of the 24 routed v3
  languages, from
  [`google/fleurs`](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd)
  at revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`;
- one synthetic Indian-English sample generated with the macOS `Rishi` voice;
- one deterministic white-noise derivative of the clean English FLEURS case;
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
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
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
| Parakeet v2 INT8 | 5/5 | 16.87% | 12.74% | 2.184 | 94.931 | 0.0230 | empty |
| Parakeet v2 F32 | 5/5 | 15.06% | 9.86% | 2.991 | 94.931 | 0.0315 | empty |
| Parakeet v3 INT8 | 25/25 | 17.45% | 4.99% | 5.709 | 275.300 | 0.0207 | empty |
| Parakeet v3 F32 | 25/25 | 10.43% | 2.42% | 7.923 | 275.300 | 0.0288 | empty |
| faster-whisper base INT8 | 29/29 | 43.24% | 14.46% | 25.304 | 365.231 | 0.0693 | `you` |

Relative to the matching F32 run, v2 INT8 reduced aggregate RTF by 26.98% while
increasing WER by 1.81 percentage points and CER by 2.88 points. V3 INT8 reduced
RTF by 27.94% while increasing WER by 7.02 points and CER by 2.57 points.

## Observed failures and conclusion

There were no model-load or transcription exceptions. The observable
silence-case failure was faster-whisper producing `you`; all Parakeet variants
returned empty output.

On this corpus, INT8 was consistently faster than matching F32, but F32 was
more accurate for both Parakeet families. The v2 quality gap was smaller in WER
than the v3 gap; the stock v3 INT8 result in particular does not provide strong
evidence for default promotion. Faster-whisper was slowest and had the highest
mixed-scope aggregate WER, but it also handled every case, so this run does not
invalidate its broader fallback role. No production routing decision follows
from these observations.

## Limitations

- Each natural-language result comes from only the first streamed FLEURS test
  row for one locale; there is no speaker, topic, utterance-length, or
  difficulty balance.
- The accent case is a synthetic macOS voice, the noise case uses one fixed
  white-noise mixture, the long case repeats one utterance, and silence is
  artificial.
- There was one run on one Apple-silicon macOS host. It does not characterize
  variance, model-load cost, peak memory, or Linux/Windows behavior.
- Parakeet v2, Parakeet v3, and faster-whisper follow different routing scopes.
  Cross-family aggregate comparisons are therefore directional, not
  apples-to-apples.
- The committed report is reproducible only when the uncommitted local audio
  and pinned model snapshots described above are reacquired.
