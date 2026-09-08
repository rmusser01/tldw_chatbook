# TASK-31827 diarizer bake-off: ONNX vs SpeechBrain (spec §7)

- Commit: `737173f6e093d45dda0b9c7a2b219180776b8393`
- Machine: Apple M5 Max, 18 cores, macOS-26.5.2-arm64-arm-64bit, Python 3.12.11
- sherpa-onnx: 1.13.7
- Corpus: 3 files (3 VoxConverse dev + 0 AMI dev), 18 minutes of audio
- Collar: 0.25 s. Live window: 3.0 s. max_speakers: 8.

## Model hashes

| asset | sha256 (first 12) | bytes |
| --- | --- | --- |
| pyannote-segmentation-3-0.onnx | `220ad67ca923` | 5992913 |
| pyannote-segmentation-3-0-int8.onnx | `d582f4b4c6b4` | 1540506 |
| nemo_en_titanet_small.onnx | `ad4a1802485d` | 40257283 |
| 3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx | `c59158379255` | 26485263 |
| wespeaker_en_voxceleb_resnet34.onnx | `5ef208a9da14` | 26534365 |
| 3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx | `357a834f702b` | 29596978 |

## Go/no-go (spec §7) -- best cell per embedder vs the ECAPA baseline

Baseline (SpeechBrain/ECAPA): DER --, purity --, RTF --, latency -- ms, separation --.

Gates: DER within 0.02 absolute, purity within 0.03, RTF <= 0.15, embed latency <= 150 ms (M-series) / 300 ms (runner), separation within 0.05.

| embedder | DER | live purity | RTF | embed latency | separation | best thresholds |
| --- | --- | --- | --- | --- | --- | --- |
| campplus_en | 0.220 (n/a) | -- (n/a) | 0.059 (PASS) | -- ms (n/a) | 0.560 (n/a) | cluster 0.80 / live -- |
| eres2net_en | 0.044 (n/a) | -- (n/a) | 0.102 (PASS) | -- ms (n/a) | 0.528 (n/a) | cluster 0.90 / live -- |
| titanet_small | 0.040 (n/a) | -- (n/a) | 0.045 (PASS) | -- ms (n/a) | 0.538 (n/a) | cluster 0.95 / live -- |
| wespeaker_resnet34 | 0.177 (n/a) | -- (n/a) | 0.076 (PASS) | -- ms (n/a) | 0.226 (n/a) | cluster 0.60 / live -- |

## Stop-pass DER, RTF and peak worker RSS

| engine | embedder | segmentation | via | cluster threshold | DER | RTF | peak RSS (MB) | files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| onnx | titanet_small | float | worker | 0.95 | 0.040 | 0.045 | 2092.7 | 3/3 |
| onnx | titanet_small | float | inprocess | 0.95 | 0.040 | 0.045 | 1855.0 | 3/3 |
| onnx | titanet_small | int8 | inprocess | 0.95 | 0.057 | 0.062 | 1543.2 | 3/3 |
| onnx | eres2net_en | float | worker | 0.90 | 0.044 | 0.102 | 1897.8 | 3/3 |
| onnx | eres2net_en | float | inprocess | 0.90 | 0.044 | 0.089 | 1938.1 | 3/3 |
| onnx | eres2net_en | int8 | inprocess | 0.90 | 0.044 | 0.094 | 1494.7 | 3/3 |
| onnx | wespeaker_resnet34 | float | worker | 0.60 | 0.177 | 0.076 | 847.5 | 3/3 |
| onnx | wespeaker_resnet34 | float | inprocess | 0.60 | 0.177 | 0.078 | 830.4 | 3/3 |
| onnx | wespeaker_resnet34 | int8 | inprocess | 0.60 | 0.299 | 0.082 | 799.5 | 3/3 |
| onnx | campplus_en | float | worker | 0.80 | 0.220 | 0.059 | 689.4 | 3/3 |
| onnx | campplus_en | float | inprocess | 0.80 | 0.220 | 0.055 | 785.0 | 3/3 |
| onnx | campplus_en | int8 | inprocess | 0.80 | 0.405 | 0.068 | 693.8 | 3/3 |

## Live purity / coverage and per-window embed latency

| engine | embedder | live threshold | purity | coverage | clusters vs speakers | latency median (ms) | p95 (ms) | windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Self-match separation (VoxConverse speakers)

| engine | embedder | cluster threshold | self cos | best other cos | separation | recommended voice_match_threshold |
| --- | --- | --- | --- | --- | --- | --- |
| onnx | titanet_small | 0.95 | 0.968 | 0.430 | 0.538 | 0.301 |
| onnx | eres2net_en | 0.90 | 0.977 | 0.449 | 0.528 | 0.287 |
| onnx | wespeaker_resnet34 | 0.60 | 0.975 | 0.755 | 0.226 | 0.132 |
| onnx | campplus_en | 0.80 | 0.850 | 0.289 | 0.560 | 0.431 |

