# OmniVoice ONNX (int8hq) Standalone TTS Backend — Design

Status: Approved design, pending implementation plan
Date: 2026-09-23
Model: <https://huggingface.co/ct03/omnivoice-onnx-int8hq>

## Context and Problem

`ct03/omnivoice-onnx-int8hq` is a CPU-only ONNX int8 export of k2-fsa/OmniVoice: a
multilingual (646-language) TTS built on a Qwen3-0.6B diffusion LM, with zero-shot
voice cloning from reference audio and 24 kHz output. It runs on `onnxruntime` plus a
Python sampling loop — no PyTorch, no native binary. Components:

| Component | Role | Size |
|---|---|---|
| `omnivoice_lm_int8_hq/` | diffusion LM (int8, audio head fp32) | ~611 MB |
| `audio_tokenizer_decoder_int8/` | codec tokens → 24 kHz waveform | ~82 MB |
| `audio_tokenizer_encoder_int8/` | reference audio → codec tokens (cloning) | ~378 MB |
| `tokenizer.json`, `config.json` | Qwen3 BPE tokenizer + config | small |

Chatbook can already run OmniVoice through the audio.cpp runtime (GGUF and
safetensors recipes, `tts`/`clone`/`design` capabilities). The user wants this ONNX
export as a **standalone backend separate from audio.cpp**: a pip-installable
`onnxruntime` engine with no native runtime supervision. RTF is ~3–7 on desktop CPUs,
so the backend is strictly offline/batch — it never joins streaming-PCM or duplex
voice paths.

Licensing facts shape the work — there are **three layers**:

- **Code**: the k2-fsa/OmniVoice code (and the `pip install omnivoice` package) is
  Apache-2.0. We may read and port the sampler/prompt logic with attribution.
- **LM weights**: k2-fsa states the upstream weights are **CC-BY-NC** (training-data
  constraints, e.g. Emilia); the ct03 ONNX export is a quantized derivative, so the
  restriction carries through. Chatbook is AGPL but never redistributes weights —
  acquisition is download-on-user-consent — so this is workable, but the consent step
  must state the non-commercial restriction and the card's anti-impersonation
  disclaimer explicitly.
- **Audio tokenizer**: from Boson AI's Higgs Audio 2 under a Llama-3-style community
  license. The repo already ships a Boson-derived `higgs` TTS backend, so there is
  precedent; the consent step surfaces this too.

The reference ONNX implementation (github.com/AFun9/Omnivoice-onnx) is
**unlicensed** — reference-only. We do not copy code from it. It also depends on
torch/transformers at inference (it monkey-patches the PyTorch `generate()` loop),
so it is not vendorable even if licensed.

## Goals

- OmniVoice ONNX int8hq as the 8th built-in TTS provider (`omnivoice`), with the same
  standing as kokoro/higgs: selectable in Speech Lab and TTS profiles.
- Zero-shot voice cloning using the existing ADR-051 canonical clone-reference assets.
- Managed artifact acquisition (~1.1 GB, three components) through the existing
  `Model_Artifacts` curated registry — consent-gated, resumable, integrity-checked —
  with an explicit config-path fallback (higgs parity).
- A pure-Python engine: `onnxruntime` + `tokenizers` + numpy. No torch, no
  transformers, no subprocess.

## Non-goals (v1)

- Console voice output / assistant reply narration (RTF 3–7 makes it useless there).
- Streaming or chunked synthesis; duplex/realtime voice paths.
- Voice **design** (instruct-text timbre control). The model supports it and the
  adapter shape must not block adding it later, but no v1 UI or plumbing.
- Variants other than int8hq (int4/fp16/fp32 exports), and no GGUF/safetensors
  handling — that is audio.cpp's lane.
- Audiobook- or character-specific features beyond what TTS profiles already give
  every backend.

## Design Overview

Approach A (chosen): an in-process backend surfaced through the legacy bridge —
exactly the kokoro/higgs shape — plus a curated artifact catalog entry. The
alternatives (native ADR-023 adapter, external-runtime wrapper) are covered in
"Alternatives Considered".

```
[OmniVoiceSettings] config ──┐
Model_Artifacts active ──────┼─► resolution ─► OmniVoiceOnnxTTSBackend (TTS/backends/omnivoice.py)
explicit model_root ─────────┘                    │  3× ORT sessions (LM / decoder / encoder)
                                                 │  tokenizer.json via `tokenizers`
                                                 │  prompt construction (pinned template)
                                                 │  numpy diffusion sampling loop (CFG)
                                                 ▼
                                     legacy_bridge ─► ADR-023 registry ─► Speech Lab / profiles
```

## Components

### Engine — `TTS/backends/omnivoice.py`

`OmniVoiceOnnxTTSBackend(LocalTTSBackend)` (base contract: `initialize()`,
`generate_speech_stream()`, `close()`, `load_model()`; this backend implements the
stream method as a single final chunk — whole-utterance batch semantics, matching
how non-streaming siblings behave).

- **Sessions**: lazily created on first use — LM, decoder, and (only when cloning is
  requested) encoder. `intra_op_num_threads` configurable.
- **Tokenizer / prompt construction**: `tokenizer.json` loads via the `tokenizers`
  library (Qwen2 BPE — no torch/transformers). Verified against the export's
  `tokenizer_config.json`: there is **no chat template**; prompts are composed from
  `extra_special_tokens` — `<|text_start|>`/`<|text_end|>`, `<|instruct_start|>`/
  `<|instruct_end|>`, `<|denoise|>`, and language markers — the same token family
  the in-repo higgs backend already drives. Exact composition (token order,
  language-tag placement, encoder-token interleaving for clones) is verified against
  the upstream `generate()` at implementation time, with unit tests asserting exact
  token sequences for known prompts (plain, multilingual, clone-prefixed,
  instruct-framed).
- **Sampling loop**: numpy implementation of the upstream diffusion sampler — CFG via
  the block-diagonal `[2B, 1, S, S]` audio mask (conditional + unconditional in one
  LM call), 8 codebooks × 1025 logits per step, `num_step` iterations, then decoder →
  24 kHz waveform. Step semantics (remasking schedule across codebooks × steps) are
  derived faithfully from the Apache-2.0 k2-fsa upstream — **not** improvised, and
  **not** copied from the unlicensed AFun9 repo. The implementation plan must contain
  a written step-by-step description of the sampler before code is written.
- **Execution model**: single-flight (`_generation_lock`, higgs pattern); all
  synthesis off the UI thread; whole-utterance output. **Progress**: per-step
  progress (`step k of N`, elapsed, rough ETA from observed RTF) reported through
  the base class's existing `ProgressCallback` seam — minute-long generations are
  unusable without it. **Cancellation** is cooperative, checked between diffusion
  steps; an individual LM forward cannot be interrupted — that granularity is
  documented and accepted.
- **Timeouts**: `generation_timeout` is computed from input length × RTF headroom ×
  safety factor rather than a flat constant (RTF 3–7 means a long utterance can
  legitimately take minutes); config override available.
- Provider declares capabilities `("tts", "clone")`, non-streaming.

### Voice manager — `TTS/omnivoice_voice_manager.py`

Extends `VoiceManagerBase` (profile CRUD, audio validation, import/export come
free), modeled on `higgs_voice_manager`. Clone flow mirrors the upstream API
`generate(text=..., ref_audio=..., ref_text=...)`: the profile's ADR-051 canonical
clone reference supplies both the audio and its transcript (`reference_text` —
already a validated, bounded field) → audio materialized and resampled to 24 kHz
mono PCM → encoder session → codec tokens → acoustic prompt prefix for the LM.
`max_reference_duration` (default ~30 s) enforced with a clear error.

### Artifact catalog — `TTS/omnivoice_artifact_catalog.py`

Follows `audio_cpp_artifact_catalog`: one `ArtifactDescriptor` covering the full repo
layout (each component is `model.onnx` + `model.onnx_data` sidecar), per-file sha256,
HF URLs as sources; registered in `Model_Artifacts.curated_registry()` alongside the
parakeet and audio.cpp entries. This buys consent gating, resumable downloads,
disk-space preflight, staging→active promotion, and model-browser visibility. The
consent step runs the ADR-080 machine-memory fit check (the engine wants ~1.2 GB+
RAM) and surfaces the full license stack: CC-BY-NC LM weights (non-commercial use;
anti-impersonation disclaimer) and the Boson Higgs Audio 2 Community License for
the tokenizer.

### Model resolution

1. Explicit `[OmniVoiceSettings] model_root` path (validated against the expected
   component layout — wrong-variant or partial dirs are `model_invalid`).
2. Else the active managed artifact.
3. Else provider state `not_configured`, with a recovery action pointing at the
   model browser. No silent auto-download.

### Provider identity and settings

`omnivoice` joins `BUILT_IN_TTS_PROVIDER_IDS` as the 8th entry. The touch-point
surface is wider than the tuple alone — enumerating it honestly (verified by
tracing every module the `higgs` provider ID reaches):

- **TTS package**: `provider_ids`, `legacy_bridge` (spec list, `_STATIC_ROUTES`
  `local_omnivoice_default`, display-name and backend-prefix maps),
  `legacy_catalogs`, `legacy_request_builder`, `request_admission`, `profile_types`,
  `studio_preferences`, `TTS_Backends`, `optional_deps`.
- **Speech Lab UI (per-provider hardcoded, not data-driven)**:
  `speech_catalog_mixin` (settings-pane visibility branch + voice-profile choices),
  `speech_param_group` (an omnivoice parameter group — `num_steps` and any sampler
  knobs the upstream exposes), the `speech_settings_*`/`speech_playground_*`
  cluster, `stts_playground_catalog`, `speech_runtime_status`,
  `lab_speech_status`.
- **Settings**: `settings_speech_tts.py` validation/guidance (the sanctioned way
  the bounded provider set grows), `settings_screen.py` guidance tables,
  `settings_search_index.py`, the `speech_tts_settings_panel` widget. Settings live
  under the existing `speech-tts` category; no new category.
- **Voice Cloning window**: `UI/Voice_Cloning_Window.py` hardcodes a backend
  dropdown and per-backend voice managers — omnivoice gets an entry wired to
  `omnivoice_voice_manager`.

The implementation plan enumerates the exact edit in each; the spec's claim is only
that all of the above are bounded, pattern-following edits.

Config — `[OmniVoiceSettings]` (higgs-style `get_cli_setting`):

| Key | Default | Notes |
|---|---|---|
| `model_root` | unset | path override; wins over managed artifact |
| `num_steps` | 32 | diffusion steps; quality/speed dial |
| `intra_op_threads` | 0 (auto) | ORT session threads |
| `max_reference_duration` | 30 | seconds, cloning cap |
| `generation_timeout_factor` | computed | length-scaled timeout override |
| `seed` | unset | reproducible sampling for tests/demos |

### Dependencies

New pyproject extra `omnivoice_tts`: `onnxruntime`, `tokenizers` (numpy is base;
`onnxruntime` already serves two other extras). `Utils/optional_deps.py` gains a
matching feature gate; absence surfaces as `dependency_missing`, never an import
crash.

## Error Handling

All failures map onto existing `TTSOperationError` codes — no new codes:

| Condition | Code |
|---|---|
| onnxruntime/tokenizers missing | `dependency_missing` |
| no model resolved | `not_configured` |
| artifact layout/sha mismatch, wrong variant | `model_invalid` |
| length-scaled timeout exceeded | `generation_timeout` |
| loop/session failures | `generation_failed` |

Provider-neutral safe-message discipline per the registry contract; recovery actions
point at settings or the model browser.

## Testing

Proportionate to risk, per `lessons-testing-evidence.md`:

- **Sampling loop**: fake ORT sessions returning deterministic logits — CFG batch
  shape, 8-codebook token selection, stop behavior, `num_step` accounting, cancellation
  between steps. No model files needed.
- **Prompt construction**: exact token-sequence assertions for known prompts
  (plain, multilingual, clone-prefixed, instruct-framed).
- **Progress & cancellation**: step-progress callbacks fire per diffusion step;
  cancellation between steps aborts promptly and releases the generation lock.
- **Catalog & resolution**: descriptor layout/sha tests; resolution order (config
  path → managed artifact → `not_configured`) against temp dirs.
- **Settings/provider surface**: extend the existing `settings_speech_tts` and
  provider-catalog tests to the 8-provider world.
- **Real-model integration**: one env-gated test (skipped unless the artifact is
  installed) doing a short synthesis and a clone.
- **Live verification** per `lessons-live-verification.md` before any RTF/quality
  claims land in docs, including the correctness gate: outputs compared
  qualitatively against the model card's published demo outputs.

## ADR Check

ADR required: yes — `backlog/decisions/180-omnivoice-onnx-standalone-tts-backend.md`
(number to confirm at creation; 179 is referenced by in-flight hosted-provider work).
Reason: a second OmniVoice runtime alongside audio.cpp is a provider/runtime boundary
decision future contributors will question; the unlicensed-reference-implementation
constraint and the pure-Python engine choice belong on record. To be created before
implementation begins and linked from the Backlog task, plan, and implementation
notes. Existing ADRs 023 (registry), 051 (clone assets), 039 (settings ownership),
080 (memory fit) apply and are followed, not amended.

## Alternatives Considered

- **Use the existing audio.cpp OmniVoice recipes (no new code).** Rejected by the
  user: the ONNX export is wanted as a standalone, pip-only runtime path separate
  from the native audio.cpp supervision stack.
- **Native ADR-023 adapter (audio.cpp shape).** More operational-state rigor
  (capability snapshots), but that shape exists for process supervision an
  in-process engine doesn't need; it would also be the only model-running backend
  off the legacy bridge. Revisit if the backend ever gains process isolation.
- **External-runtime wrapper around AFun9/Omnivoice-onnx.** Rejected: the repo is
  unlicensed (not vendorable) and drags torch + transformers at inference, defeating
  the point of the ONNX export.

## References

- Model card: <https://huggingface.co/ct03/omnivoice-onnx-int8hq>
- Upstream weights & code: <https://huggingface.co/k2-fsa/OmniVoice> (Apache-2.0
  code, CC-BY-NC weights) and its GitHub repo — the canonical source to port the
  sampler and prompt logic from; paper: arXiv:2604.00688
- ADR-023 (TTS adapter registry), ADR-051 (clone reference assets), ADR-039 (TTS
  settings ownership), ADR-080 (machine-memory fit)
- Prior art in-repo: `TTS/backends/higgs.py`, `TTS/backends/kokoro.py`,
  `TTS/audio_cpp_artifact_catalog.py`, `Model_Artifacts/curated_registry.py`
