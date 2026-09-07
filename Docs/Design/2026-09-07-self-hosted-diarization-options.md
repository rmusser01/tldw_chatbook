# Self-hosted speaker diarization options (research for TASK-31827)

**Date:** 2026-09-07. **Purpose:** ground the "MOSS / server diarizer backend" design
(TASK-31827) in what actually exists and how each option fits the `Diarizer` seam the
Meetings feature already has. This is research, not a decision.

## 1. What a backend has to fit

The seam (`Audio/meeting_session.py::Diarizer`, phase-2 spec §3.2/§3.3):

| Call | When | What the session expects |
|---|---|---|
| `assign(pcm, sr, seq) -> id \| None` | after every finalised ASR segment (near-live) | a stable live cluster id within the assign budget; `None` = keep the coarse label |
| `diarize(wav, start, end) -> segments` | once, at Stop | segments whose ids are already reconciled onto the live ids |
| `pin`, `centroids`, `close` | rename / Stop | pinning a named cluster; centroids for reconciliation |

Two consequences the survey keeps running into:

1. **The current design is "embedding model + our own online clusterer"** (`diarizer_cluster.py`).
   Any backend that yields a per-window speaker **embedding** slots in with almost no change
   to the session, the reconciliation, or the voiceprint feature (which matches centroids
   inside the worker). A backend that yields **labels only** (end-to-end diarizers, joint
   ASR+diarization models) needs its own reconciliation story and loses voiceprint matching
   unless an embedding pass is added beside it.
2. **The base install has no torch.** Live labels today require the `diarization` extra
   (torch + speechbrain). A torch-free option changes who gets the feature, not just how
   fast it runs.

Other constraints: Linux/macOS/Windows; the worker subprocess model (warm-up ≤ 120 s,
bounded assign, crash → one restart); privacy (server backends send meeting audio off the
device; nothing is written outside the meeting folder and the encrypted voiceprint store).

## 2. Options

### 2.1 Current: SpeechBrain ECAPA-TDNN + online clusterer (local, torch)

- Embedding-only model; language-agnostic; any speaker count; near-live + Stop pass;
  centroids → voiceprint works. CPU is fine for the live path.
- Cost: the `diarization` extra (torch, speechbrain, sklearn ≈ 2 GB); Stop-pass clustering
  quality is ours to tune; no overlap handling.

### 2.2 sherpa-onnx (k2-fsa) — ONNX runtime, torch-free

- Offline diarization = `pyannote/segmentation-3.0` (ONNX, 6.6 MB) + a 3D-Speaker
  embedding extractor (ERes2Net / CAM++ ≈ 28 MB) + clustering (`num_clusters` auto,
  `threshold`, `min_duration_on/off`). `OfflineSpeakerDiarization` processes a whole
  file; output is `(start, end, speaker_k)` segments.
- The same package exposes the speaker-embedding extractor on its own (its model zoo
  also carries WeSpeaker VoxCeleb and NeMo TitaNet exports), so the **live path can keep
  our online clusterer** with an ONNX embedder instead of speechbrain — i.e. the seam,
  reconciliation and voiceprint matching survive unchanged, with no torch.
- Wheels for Linux/macOS/Windows incl. arm64; 12 language bindings; runtime is Apache-2.0.
  Model licences vary (pyannote segmentation is MIT; 3D-Speaker weights come via
  ModelScope — verify per model). Training data for CAM++/ERes2Net is Chinese-centric,
  but speaker embeddings transfer across languages far better than ASR does.
- Fit: **best candidate for a torch-free default**. Risk: accuracy of the ONNX pair vs
  ECAPA on our recordings — needs a bake-off on real meeting audio.

### 2.3 pyannote.audio 4 + `speaker-diarization-community-1`

- Library MIT; model CC-BY-4.0 but **gated on Hugging Face** (accept terms + token) — a
  real friction for a desktop app. Pipeline = segmentation + WeSpeaker embedding + VBx.
  DER: AMI 17.0 %, DIHARD-3 20.2 %, AliMeeting 20.3 % (vs 3.1: 18.8 / 21.4 / 24.5).
  31 s per audio hour on an H100; no CPU figure published (expect roughly real-time or
  slower on CPU for an hour-long file). Batch only; torch.
- Fit: a strong **Stop-pass** replacement if the gated download is acceptable; nothing for
  the live path. Its segmentation model is exactly what sherpa-onnx already ships as ONNX.

### 2.4 diart — streaming diarization on pyannote models

- MIT; rolling 5 s buffer stepped every 500 ms, latency configurable 0.5–5 s; per-step
  cost 12/26 ms (segmentation/embedding) on CPU. Real live labels with overlap handling.
- Uses the **older** gated pyannote models and pins `pyannote.audio<3.1`; RxPY streams.
  Maintenance is slow. It keeps its own incremental clustering (centroids are internal —
  an adapter could read them).
- Fit: a genuine live diarizer, but it drags in torch, gated models and a stale pin. Not
  worth adopting over 2.2 unless overlap-aware live labels become a requirement.

### 2.5 NVIDIA streaming Sortformer (`diar_streaming_sortformer_4spk-v2`/v2.1)

- End-to-end streaming diarizer, 117 M params, CC-BY-4.0. Latency presets from 0.32 s
  ("ultra low") to 30 s; DER 6.6 % CALLHOME 2-spk, 13.2 % DIHARD-3 (1–4 spk). **Max 4
  speakers** (degrades at 5+); English-centric; speaker cache keeps labels stable.
- Two runtimes: NeMo (heavy, torch, git install) or **NeMo-Speech.cpp** (ggml C++
  runtime, GGUF q8 = 147 MB, CPU works, multiple backends) — the torch-free route.
- Fit: labels only (no embeddings), so reconciliation is by time overlap and voiceprint
  matching needs a side embedding pass. Attractive for a low-latency, small-room live
  labeller; the 4-speaker cap rules it out as the only backend. Worth a spike once
  NeMo-Speech.cpp has stable Python bindings.

### 2.6 MOSS-Transcribe-Diarize 0.9B (the task's namesake)

- Joint transcription + diarization + timestamps in one pass: Whisper-Medium encoder +
  Qwen3-0.6B-style decoder; Apache-2.0; released 2026-07-09; 50+ languages; up to 90 min
  per pass; output `[start][S01] text [end]` at segment level. Won MLC-SLM 2026.
  cpCER 7.4 % (podcast), 15.8 % (AISHELL-4), 22.2 % (AliMeeting); RTFx 294 on the
  open-ASR leaderboard (GPU).
- **Batch only, CUDA only** (CUDA 12/13, SGLang Omni or vLLM *nightly*; eager attention
  OOMs on long files — needs flash-attn/SDPA). No CPU or Apple-Silicon path. No speaker
  embeddings exposed. Not present in tldw_server today.
- Fit: it is not a `Diarizer` — it replaces the transcriber *and* the diarizer together.
  The honest shape is a **server-hosted post-meeting pass** ("re-transcribe this meeting
  with speaker labels" behind tldw_server, or a local CUDA option), producing a second,
  speaker-attributed transcript. Live labels stay with 2.1/2.2. Voiceprint would need the
  embedding side-pass over MOSS's segments.

### 2.7 tldw_server streaming endpoint (the "server" backend that already exists)

- `WS /api/v1/audio/stream/transcribe` returns `speaker_id`/`speaker_label` on final
  segments plus a `diarization_summary` when `diarization.enabled` — implemented server-side
  with Silero VAD + `speechbrain/spkrec-ecapa-voxceleb` (the same approach as 2.1). The file
  endpoint (`POST /api/v1/audio/transcriptions`) has **no** diarization. Server is GPLv3.
- Fit: the cheapest server backend — same embedding family, ids per final segment — but it
  couples diarization to the server's transcription stream (the client would send audio
  and receive both), and the privacy model (LAN vs remote, `store_audio`) is the design
  work. No MOSS there yet.

### 2.8 Others surveyed (not recommended as the backend)

- **Reverb diarization v2 (Rev):** pyannote-3.0 + WavLM, −22 % WDER vs pyannote 3.0.
  Inference code Apache-2.0, but the **weights are "license: other"** (Rev's non-commercial
  terms per the announcement) — a licence risk for an app users may run commercially.
- **FunASR / 3D-Speaker (CAM++ 7.2 M, ERes2Net):** MIT toolkit, per-model weight licences;
  Chinese-centric; offline; torch. Useful as the *source* of the ONNX embedders in 2.2 and
  as MOSS's serving adapter, not as a backend by itself.
- **DiariZen:** pruned WavLM-Large + Conformer powerset + VBx; state-of-the-art offline;
  research-grade code and a large encoder; GPU-oriented. Watch, don't adopt.
- **WhisperX / whisper-diarization / docker-whisper / Speakr:** pipelines around
  pyannote 3.x + faster-whisper; nothing new beyond 2.3 for our seam.
- **Voxtral Transcribe 2 (Mistral):** diarization only in the batch "Mini Transcribe V2";
  the open-weights Apache-2.0 release is the *Realtime* model, which has no diarization;
  ≥ 16 GB VRAM via vLLM nightly. Check the diarizing variant's weight status before
  considering it.
- **Commercial on-device SDKs (e.g. Picovoice Falcon):** closed; out of scope.

## 3. Comparison

| Option | Live? | Stop pass? | Embeddings (voiceprint) | torch-free | Platforms | Speakers | Licence notes |
|---|---|---|---|---|---|---|---|
| 2.1 ECAPA + ours (current) | yes | yes | yes | no | all | any | Apache/MIT |
| 2.2 sherpa-onnx | yes (embedder + ours) | yes | yes | **yes** | all incl. arm64 | any | runtime Apache; models vary |
| 2.3 pyannote community-1 | no | yes | partial | no | all (GPU-fast) | any | CC-BY-4.0, **gated** |
| 2.4 diart | yes | no | internal | no | all | any | MIT + gated models, stale pin |
| 2.5 streaming Sortformer | yes (0.3–30 s) | no | no | via NeMo-Speech.cpp | CPU/GPU | **≤ 4** | CC-BY-4.0; English-centric |
| 2.6 MOSS 0.9B | no | yes (joint ASR) | no | no (CUDA only) | Linux+NVIDIA | any | Apache-2.0 |
| 2.7 tldw_server WS | yes (server) | via server | on the server | on the client | client: all | any | server GPLv3 |
| 2.8 Reverb v2 | no | yes | partial | no | GPU | any | weights non-commercial |

## 4. What this suggests for the TASK-31827 design (input, not a ruling)

1. **Split the task's two halves.** "MOSS" and "server backend" are different things:
   MOSS is a batch joint-transcript model that needs a CUDA host; the server backend that
   exists today is tldw_server's ECAPA streaming diarization. Design them as (a) a
   `ServerDiarizer` over the WS endpoint with an explicit off-device-audio consent model,
   and (b) a post-meeting "speaker-attributed re-transcription" action (MOSS, server-hosted
   or local CUDA) that writes a second transcript, not a live `Diarizer`.
2. **Consider a torch-free local backend first.** sherpa-onnx gives the base install live
   labels and a Stop pass without the 2 GB extra, keeps the seam and the voiceprint design
   intact, and is the option most users could actually run. It needs a bake-off (ONNX
   ERes2Net/CAM++ vs ECAPA on real meeting audio) before it replaces anything.
3. **Keep streaming Sortformer as a spike**, gated on NeMo-Speech.cpp bindings and on the
   4-speaker cap being acceptable for a "small room" mode.
4. **Do not pick** pyannote community-1 as the default (gated download) or Reverb (weight
   licence) for a shipped desktop app.

## 5. Sources

- MOSS-Transcribe-Diarize: https://github.com/OpenMOSS/MOSS-Transcribe-Diarize ,
  https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize
- pyannote: https://huggingface.co/pyannote/speaker-diarization-community-1 ,
  https://github.com/pyannote/pyannote-audio , https://www.pyannote.ai/blog/community-1
- NVIDIA Sortformer: https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2 ,
  https://github.com/NVIDIA/NeMo-Speech.cpp
- sherpa-onnx: https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html ,
  https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-speaker-diarization.py
- diart: https://github.com/juanmc2005/diart , https://joss.theoj.org/papers/10.21105/joss.05266
- Reverb: https://github.com/revdotcom/reverb , https://huggingface.co/Revai/reverb-diarization-v2
- FunASR / 3D-Speaker: https://github.com/modelscope/FunASR , https://github.com/modelscope/3D-Speaker
- DiariZen: https://arxiv.org/abs/2604.21507
- Voxtral Transcribe 2: https://mistral.ai/news/voxtral-transcribe-2/
- tldw_server audio API: https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Audio_Transcription_API.md
