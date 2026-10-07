# Image and video generation

This document describes the media-generation packages: the adapter registries, secrets precedence, the single validation choke point (`worker.run_generation`), chat integration (`/generate-image`, `/generate-video`), persistence of results, and the ephemeral video store. Video mirrors image per ADR-044.

## Authoritative files

| File | Role |
| --- | --- |
| `Image_Generation/adapter_registry.py` | `ImageAdapterRegistry`, `DEFAULT_ADAPTERS`, singleton `get_registry()` |
| `Image_Generation/adapters/` | Nine shipped adapters: `stable_diffusion_cpp`, `swarmui`, `openrouter`, `novita`, `together`, `modelstudio`, `gemini`, `fal`, `comfyui` |
| `Image_Generation/config.py` | `[image_generation]` TOML; secrets precedence; frozen `ImageGenerationConfig` |
| `Image_Generation/worker.py` | `run_generation()` — the single validation choke point; `build_request()` → frozen `ImageGenRequest` |
| `Video_Generation/adapter_registry.py` | `DEFAULT_ADAPTERS`: `minimax`, `comfyui`, `stable_diffusion_cpp` (sd.cpp video is registered but its adapter module does not exist yet — planned, not implemented) |
| `Video_Generation/config.py` | `[video_generation]` globals, retention/store policy, `get_video_store_policy()` |
| `Video_Generation/worker.py` | `run_generation()` — same choke-point pattern + result container triple-check |
| `Video_Generation/video_store.py` | `VideoStore` — ephemeral message-keyed video file store (task-3401.4) |
| `Video_Generation/video_formats.py`, `video_metadata.py`, `video_templates.py`, `workflows/` | Closed MP4/WebM mapping, metadata, templates, ComfyUI workflows |
| `Chat/console_generate_image.py` | `/generate-image`: arg parsing, style tokens, LLM context prompt, `run_generation_batch` |
| `Chat/console_generate_video.py` | `/generate-video`: off-UI-loop generation + VideoStore save |
| `UI/Console_Modules/image.py` | Console image UI: offload, regenerate, variant browse |

## Configuration and secrets

Nested TOML: `[image_generation]` globals plus `[image_generation.<backend>]` (same shape for video). **Secret precedence: env > config > keyring** (`_resolve_secret`; keyring namespace `tldw_chatbook_imagegen`; per-backend env names like `OPENROUTER_API_KEY`, `FAL_KEY`, `GEMINI_API_KEY`).

Image globals: `default_backend` (sd_cpp), max width/height/pixels (1024), `max_steps` 50, `max_prompt_length` 1000, `inline_max_bytes` 4,000,000 (the DB inline BLOB cap), `max_variants_per_message` 8, context-LLM keys. Video globals: `retention` (`session` default | `ttl`), `retention_ttl_hours`, `max_store_mb`, `download_max_mb`, `max_reference_assets`, `confirm_cost_estimate`; minimax `allow_uploads` (default off).

## The choke point

Every generation request funnels through `worker.run_generation` (image and video each have one):

1. Resolve the backend (a not-enabled backend fails with an error naming `[image_generation].enabled_backends`).
2. Capability checks (reference-image support).
3. Request validation — bounds, per-backend `extra_params` allowlist, reference-image mime/size/content.
4. Adapter dispatch. Adapters are synchronous and must run off the UI loop (`asyncio.to_thread` at the chat seam).

Video adds cooperative cancel (threaded only to adapters that support it) and a **result container triple-check**: `request.format == result.container == container_for_mime`, else the result is rejected — a mislabeled container can never reach playback (ADR-044 rev 3).

## Chat integration

- `/generate-image` parses args and style tokens; an optional LLM-composed context prompt (kill-switch + keyword-extractor fallback; refusal is a first-class outcome) precedes `prepare_generation_request`. `run_generation_batch` enforces the identical-image guard (explicit seed only on variant 0; later variants random) and collects per-variant failures without aborting the batch; ComfyUI edits require count == 1.
- Results persist through the chat store: image bytes + metadata land as message attachments (inline up to 4 MiB) plus a generation-metadata sidecar whose `position` ties it to the attachment index; multiple images form the per-message **variant set** browsed with variant-previous/next (see [chat-pipeline.md](./chat-pipeline.md)).
- `/generate-video` runs generation off the UI loop and saves bytes to the `VideoStore` keyed by the new message id; video metadata goes to the messages' local-only metadata JSON column — deliberately **not** the image sidecar (ADR-044 decision 7). Video bytes never enter the database.

## The video store

`VideoStore` is an ephemeral, message-keyed file store under `<user_data_dir>/generated_videos/<message_id>/<slug>.<validated-ext>`:

- Message rows carry a `[video] <slug>` marker parsed back by `parse_video_marker`; a missing file is a **normal state** rendered as a named tombstone with a regenerate affordance, not an error.
- Retention: session wipe or TTL; capacity eviction is oldest-by-mtime with a store-size cap; a cross-process `portalocker` root lease (5 s timeout) serializes maintenance; symlinks/reparse points are refused; an oversized single artifact has a sole-oversized adoption exception.

## Dataflow

1. `/generate-image` or `/generate-video` (or the demo screen) resolves styles/templates and optionally an LLM context prompt.
2. The chat seam offloads to a thread; `build_request` produces the frozen contract.
3. `run_generation` validates and dispatches to the adapter (HTTP poll APIs or a local subprocess).
4. Artifacts persist: images → attachments + sidecar → inline card / variant set; videos → `VideoStore.save` → marker + metadata → card or tombstone.

## Failure behaviors

| Case | Behavior |
| --- | --- |
| Backend not enabled | `ImageGenerationError`/`VideoGenerationError` naming the enabled-backends key |
| Validation failure | Refused before any adapter call |
| One variant fails in a batch | Collected per-variant; batch continues |
| LLM context prompt refuses | `GenerationRefusal` — surfaced as a refusal, not an error |
| Video container mismatch | Result rejected at the triple-check |
| Video file missing at playback | Named tombstone + regenerate |
| Store over capacity | Oldest-first eviction; sole-oversized adoption exception |

## Governing decisions

ADR-044 (`backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md`, rev 3) — video mirrors image, ephemeral storage, marker protocol. Related: ADR-052 (ComfyUI H3 image-edit provider boundary). Settings categories `image_generation` / `video_generation` in the Settings screen.

## Verified gotchas

1. Video `stable_diffusion_cpp` is registered in `DEFAULT_ADAPTERS` (and is the default backend name) but its adapter module does not exist yet — the registry entry is planned, not implemented.
2. Image inline BLOB cap is 4 MiB (`inline_max_bytes`); larger artifacts must not be inlined into message rows.
3. Video metadata deliberately avoids the image generation sidecar — the two persistence paths differ by design.
4. Adapters are synchronous; every call site must offload (`asyncio.to_thread`) or the UI loop stalls.

## Related docs

- [chat-pipeline.md](./chat-pipeline.md) — variant sets and message persistence
- [database-layer.md](./database-layer.md) — attachments and metadata columns
