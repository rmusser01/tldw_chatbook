# Image Generation — Multi-Provider Foundation (Phase 1)

**Status:** Design approved, ready for implementation planning
**Date:** 2026-07-22
**Program:** In-chat image generation (Console conversations + character canvas), mirroring tldw_server's webui
**This spec covers:** Phase 1 of the program — the backend layer only. No chat card yet.

---

## 1. Background & motivation

We want image generation available inside chat/conversations and the character/roleplay surface, the way tldw_server's webui does it: a user triggers generation and the result appears as a rich "Image Generation" card (prompt, negative, seed, backend, style, prev/next variants, regenerate, TTS) — as in the reference mockup.

This is a multi-layer feature. Rather than spec it all at once, it is decomposed into phases (§9). **This spec is Phase 1: the multi-provider backend foundation.** It builds and proves the image-generation engine with no chat UI. The visible chat card, slash command, character integration, variants, and TTS are later phases.

### What already exists in tldw_chatbook (and how we treat it)

- A **complete but orphaned SwarmUI stack** lives under `tldw_chatbook/Media_Creation/` (`swarmui_client.py`, `image_generation_service.py`, `generation_templates.py`) plus a `SwarmUIWidget` and an event layer with a DB-save TODO stub pointing at a `media_generations` table that was never built. It is mounted only in a *legacy* settings sidebar the active Console chat does not use, and `[media_creation]` config never existed, so it is wired to nothing.
- **We do not touch or delete this code in Phase 1** (finding A1). Deleting `swarmui_client.py` / `image_generation_service.py` would break live imports in `swarmui_widget.py`, `swarmui_events.py`, and `settings_sidebar.py`. Instead we build the new package cleanly alongside it. Two pieces are explicitly **kept for reuse in later phases**: `generation_templates.py` (14 "Style" presets → the reference card's "Style: Anime Base") and `ImageGenerationService.extract_context_from_messages` (chat-context → prompt). A separate follow-up task will remove the dead SwarmUI chain once the new package supersedes it.

### The source we port from

tldw_server's `tldw_Server_API/app/core/Image_Generation/` (~3,207 lines) is a clean, self-contained Python package: **zero FastAPI/starlette/pydantic imports**, plain frozen dataclasses, and **fully synchronous adapters** (`def generate(self, request) -> ImageGenResult`) using a sync `httpx.Client`, `subprocess` for the local backend, and `time.sleep` polling for async-job backends. Its only cross-package couplings are a config loader, a sync HTTP helper, egress guards, and a media-DB-backed reference-image resolver. This makes it portable almost verbatim.

---

## 2. Goals & non-goals

### Phase 1 goals
- A self-contained `tldw_chatbook/Image_Generation/` package supporting **6 backends**: `stable_diffusion_cpp` (local subprocess), `swarmui` (fronts ComfyUI), `openrouter` (gpt-image-1), `novita`, `together`, `modelstudio`.
- Backend selection via a registry with lazy adapter imports.
- Request validation, prompt refinement, and model listing (`is_configured` per backend).
- A `[image_generation]` TOML config surface re-homed onto this app's conventions (TOML + env + keyring), with **identical flat field names** to the server dataclass so adapters need no edits.
- A thin sync `fetch_json` / `create_client` HTTP shim on `httpx.Client`, plus a **light egress guard** (real SSRF hardening deferred to task-485).
- A **throwaway in-app demo panel** (command-palette screen) to generate one image end-to-end and eyeball the result + raw metadata.
- Tests: mocked per-adapter, request-validation bounds, registry logic, prompt-refinement table, config-loader round-trip + key precedence, opt-in live integration.

### Phase 1 non-goals (deferred)
- The Console chat "Image Generation" card and the `/generate-image` slash command (Phase 2).
- **Any DB schema change** (finding A3) — no persistence of generation metadata; the demo panel renders in memory / to a temp file only. This deliberately avoids colliding with the concurrent Console-branching program's `v22→v23` migration slot.
- Character-canvas generation, per-mood reaction images, persona visual packs.
- Variant navigation (prev/next), regenerate, TTS.
- Reference-image resolution (`reference_images.py` is dropped; ModelStudio's `reference_image` degrades to `None`).
- Full SSRF/egress protection (tracked in **task-485**).
- Deleting the orphaned `Media_Creation` SwarmUI code (separate cleanup task).

---

## 3. Architecture

### 3.1 Package layout

New package `tldw_chatbook/Image_Generation/` (dir name matches the `Media_Creation` / `RAG_Search` / `LLM_Calls` codebase convention; the config section is lowercase `[image_generation]` matching `[chat.images]` / `[embeddings]`):

```
tldw_chatbook/Image_Generation/
  __init__.py              # import-light; NO eager adapter/Pillow imports (finding B8)
  exceptions.py            # port verbatim
  prompt_refinement.py     # port verbatim
  capabilities.py          # port verbatim (ResolvedReferenceImage, capability checks)
  request_validation.py    # port verbatim
  listing.py               # port; is_configured per backend
  adapter_registry.py      # port; repoint DEFAULT_ADAPTERS module strings
  config.py                # RE-HOME: nested TOML + env + keyring -> flat dataclass
  http_client.py           # NEW: sync fetch_json/create_client on httpx.Client + light egress guard
  image_format_utils.py    # port; re-home only fetch_image_bytes onto local http_client
  adapters/
    __init__.py
    base.py                # port verbatim (ImageGenRequest/ImageGenResult/protocol)
    stable_diffusion_cpp_adapter.py
    swarmui_adapter.py
    openrouter_image_adapter.py
    novita_image_adapter.py
    together_image_adapter.py
    modelstudio_image_adapter.py
```

### 3.2 Port map

| Server file | Action in port |
|---|---|
| `exceptions.py`, `prompt_refinement.py`, `capabilities.py`, `adapters/base.py`, `request_validation.py` | **verbatim** — `loguru` is already this app's logging library (CLAUDE.md key deps), so **no logger swap is needed** |
| `image_format_utils.py` | port; re-home only `fetch_image_bytes` onto local `http_client` |
| `adapter_registry.py`, `listing.py` | port; repoint only the `DEFAULT_ADAPTERS` module strings (loguru kept as-is) |
| `config.py` | **re-home** (see §4); keep flat field names identical |
| `adapters/{sd_cpp,swarmui,openrouter,novita,together,modelstudio}.py` | port; only change the `fetch_json`/`create_client` import source |
| `reference_images.py` | **drop** — verified safe: `capabilities.py` defines `ResolvedReferenceImage` itself (no import of `reference_images`), and modelstudio guards every reference path on `request.reference_image is not None`, so `reference_image=None` degrades to a no-op |

### 3.3 Core contracts (unchanged from server)

- `ImageGenRequest` (frozen): `backend, prompt, negative_prompt, width, height, steps, cfg_scale, seed, sampler, model, format, extra_params, request_id=None, reference_image=None`.
- `ImageGenResult` (frozen): `content: bytes, content_type: str, bytes_len: int`.
- `ImageGenerationAdapter` — a structural `typing.Protocol` with `name`, `supported_formats`, and **synchronous** `generate(request) -> ImageGenResult`.
- Registry: lazy `DEFAULT_ADAPTERS` name→"module.Class" strings; `resolve_backend()`; `enabled_backends` gates availability (empty list ⇒ nothing enabled); no-arg adapter constructors that each read config themselves; cached singletons.

---

## 4. Config re-home (the main port work — finding B4)

The server's `ImageGenerationConfig` is a **single flat frozen dataclass** (~50 prefixed fields, e.g. `swarmui_base_url`, `openrouter_image_api_key`) loaded from INI strings via `configparser`. We keep the **exact flat field names** so no adapter changes, but expose a **nested** TOML surface and re-home the loader.

### 4.1 TOML surface

```toml
[image_generation]
default_backend = "swarmui"          # finding B7 (server default was sd_cpp)
enabled_backends = ["swarmui"]        # finding B7 (server default was [] = nothing)
max_width = 1024
max_height = 1024
max_pixels = 1048576
max_steps = 50
max_prompt_length = 1000
inline_max_bytes = 4000000

[image_generation.stable_diffusion_cpp]
binary_path = ""                      # local `sd` CLI; empty = backend unusable
diffusion_model_path = ""             # OR model_path
model_path = ""
vae_path = ""
lora_paths = []
device = "auto"
default_steps = 25
default_cfg_scale = 7.5
default_sampler = "euler_a"
timeout_seconds = 120
allowed_extra_params = []

[image_generation.swarmui]
base_url = "http://127.0.0.1:7801"
default_model = ""
timeout_seconds = 120
allowed_extra_params = []
# swarm_token: secret, resolved via key precedence (§4.3), not stored plaintext here

[image_generation.openrouter]
base_url = "https://openrouter.ai/api/v1"
default_model = "openai/gpt-image-1"
timeout_seconds = 120
allowed_extra_params = []

[image_generation.novita]
base_url = "https://api.novita.ai"
default_model = "sd_xl_base_1.0.safetensors"
timeout_seconds = 180
poll_interval_seconds = 2
allowed_extra_params = []

[image_generation.together]
base_url = "https://api.together.xyz/v1"
default_model = "black-forest-labs/FLUX.1-schnell-Free"
timeout_seconds = 120
allowed_extra_params = []

[image_generation.modelstudio]
base_url = ""                         # region-derived if empty
default_model = "qwen-image"
region = "sg"                         # sg|cn|us
mode = "auto"                         # sync|async|auto
poll_interval_seconds = 2
timeout_seconds = 180
allowed_extra_params = []
```

### 4.2 Loader responsibilities

A new loader in `Image_Generation/config.py` that:
1. Reads each nested subsection via the app's **canonical nested accessor** — `get_cli_setting("image_generation", "<subsection>", {})` returns the subsection dict, or read the raw `load_settings().get("image_generation", {}).get("<subsection>", {})`. (Note: the dotted form `get_cli_setting("image_generation.swarmui", ...)` also works — the earlier "dotted lookup is flat/broken" concern was **stale**; `get_cli_setting` was fixed on 2026-07-16 (`b983f2548`) to walk nested TOML tables. No workaround needed.)
2. **Flattens** nested keys to the server's flat field names (`[image_generation.swarmui].base_url` → `swarmui_base_url`, `[image_generation.openrouter].default_model` → `openrouter_image_default_model`, etc.) so the ported adapters/`request_validation`/`listing` read unchanged.
3. Applies the server's `DEFAULT_*` constants as fallbacks (documented defaults above).
4. Retains the `_coerce_*` / `_parse_list` helpers (TOML is typed, but env-var overrides arrive as strings).
5. **Resolves each secret via §4.3 and writes the result into the config object's `*_api_key` / token field.** This is load-bearing: `listing.py:51-81`'s `is_configured` reads `getattr(cfg, "<backend>_api_key")` *first* (then env), and the adapters read the same field — so populating it from the full precedence (incl. keyring) makes a keyring-only backend correctly report as configured *and* usable, with no change to `listing.py` or the adapters.
6. Caches like the server (`get_image_generation_config(reload=False)` + reset hook).
7. **The `[image_generation]` default block (§4.1) must be added to the app's default config template** (`config.py`) so the section materializes for users to edit — the old `[media_creation]` section never existed, which is why the orphaned SwarmUI stack silently fell back to hard-coded defaults. Don't repeat that.

This loader and its precedence are the highest-value test target.

### 4.3 Secret handling & the OpenRouter/Together collision (finding A2)

OpenRouter and Together are **both LLM providers and image backends** in this app. Reusing `get_api_key("openrouter")` would silently share the LLM key. Image-backend secrets therefore resolve on an **image-specific precedence**, independent by default but reuse-capable:

1. Backend env var — `OPENROUTER_API_KEY`, `NOVITA_API_KEY`, `TOGETHER_API_KEY`, `DASHSCOPE_API_KEY`/`QWEN_API_KEY`; SwarmUI token env if present.
2. `[image_generation.<backend>].api_key` (config, honoring config-encryption).
3. Keyring (image-backend-scoped account name, distinct from LLM-provider accounts).
4. *Optional* fallback to the shared provider key via `get_api_key()` — only if explicitly opted in, so it never surprises.

Secrets are **never logged** (matches `[API]` handling per CLAUDE.md security rules). Because the loader stores resolved secrets on the in-memory config object (§4.2 step 5), that object must never be logged/serialized wholesale — dump non-secret fields only if debugging.

---

## 5. HTTP shim & egress (findings §5 + task-485)

New `Image_Generation/http_client.py` provides the sync surface the ported adapters expect:
- `fetch_json(method, url, *, headers=None, json=None, params=None, cookies=None, timeout=None)` → parsed JSON, on `httpx.Client`.
- `create_client(timeout=None)` → configured `httpx.Client` (used by `fetch_image_bytes`).
- Redirect helpers: `DEFAULT_MAX_REDIRECTS`, `_resolve_redirect_url` (urljoin).
- **Light egress guard** `_validate_egress_or_raise(url)`: reject non-`http(s)` schemes; enforce max redirects. It stays **permissive for user-configured `base_url`s** so local backends (127.0.0.1 SwarmUI, local sd.cpp) keep working. It does **not** block private/metadata ranges for API-returned URLs — that is the deferred hardening.
- `evaluate_url_policy(url)` stub for ModelStudio → simple host allowlist (`aliyuncs.com` / base host), preserving the adapter's built-in check.

Each adapter's own per-backend URL checks are preserved (SwarmUI same-origin resolution; ModelStudio host allowlist). **Full SSRF protection for API-returned image URLs is task-485**, and this light guard is its explicit placeholder.

---

## 6. Concurrency model (finding §4)

The ported adapters are **fully synchronous and blocking** (sync httpx; `subprocess.run` for sd.cpp; `time.sleep` polling for Novita/ModelStudio). The app therefore **never calls `generate()` on the asyncio/UI loop**. A thin wrapper runs generation on a Textual **thread worker** (`@work(thread=True)` / `run_worker(..., thread=True)`), consistent with the app's known "sync-under-async blocks the UI loop" rule. The demo panel (and later the chat card) both go through this wrapper.

**Cancellation caveat (finding B6):** a blocking `subprocess.run` / `time.sleep` will not cancel promptly, so in-flight generation cannot be cleanly aborted in Phase 1. Mitigations: sane default `timeout_seconds` per backend (as configured), and a clear "generating…" state. A proper cancel/timeout story is deferred to the chat-card phase.

**Request construction:** `ImageGenRequest.format` is a *required* field (no dataclass default). The wrapper that builds the request supplies a default output `format` of `"png"` (overridable). Note `inline_max_bytes` (default 4 MB) caps `validate_and_convert_image_output`; a large PNG can exceed it, so the wrapper surfaces that error clearly and `webp`/`jpg` remain available as smaller-output formats.

---

## 7. Throwaway demo panel (finding B5)

An `ImageGenDemoScreen` reached via a **command-palette action** ("Image Gen (dev)") — least invasive in a tab-based app, trivial to delete in Phase 2. It is explicitly labeled temporary.

Contents:
- Backend `Select` populated from `listing.list_image_models_for_catalog()`, showing which backends are `is_configured`; a clear inline message when the chosen backend is enabled-but-not-configured, pointing at config.
- Prompt + negative-prompt text areas; a few param inputs (size, steps, seed).
- Generate button → the §6 thread worker → on completion, render the result.
- **Rendering uses the low-level `textual-image` / `rich-pixels` primitives directly**, NOT the Console transcript render path (which is coupled to message IDs / view-state). This keeps the throwaway decoupled.
- Raw `ImageGenResult` + request metadata dumped below the image for verification.

The demo panel is the manual proof surface; it does not persist anything.

---

## 8. Testing

- **Per-adapter unit tests** with mocked `fetch_json` (and mocked `subprocess.run` for sd.cpp): request payload shape, response extraction (b64 / data URL / http URL paths), error handling, session refresh (SwarmUI), poll loop terminal states (Novita/ModelStudio).
- **`request_validation` bounds suite**: prompt length, width/height/pixels, steps, cfg_scale positivity/finiteness, extra_params allowlist (incl. ModelStudio `mode` exemption and `cli_args` list rule).
- **Registry tests**: `resolve_backend` enabled/disabled/default logic, lazy import, singleton caching, unknown backend.
- **`prompt_refinement` table tests**: off/basic/auto modes, no-double-append, max-length guard.
- **Config-loader tests (priority)**: nested TOML → flat field mapping, `DEFAULT_*` fallbacks, `_coerce_*` on string env overrides, the §4.3 key precedence (env > config > keyring > optional shared), and that a **keyring-only** secret populated by the loader makes `listing.is_configured` report that backend configured (§4.2 step 5).
- **Opt-in live integration tests**: one per backend, skipped unless creds/servers/binary are present.
- Cold-start guard: assert importing `tldw_chatbook.Image_Generation` does not import adapters or Pillow (finding B8).

---

## 9. Phase roadmap (context, not in scope here)

- **Phase 1 (this spec):** multi-provider backend foundation + demo panel.
- **Phase 2:** the Console chat "Image Generation" card (matching the reference), the `/generate-image [:backend] <prompt>` trigger, DB storage of generation metadata + variants (introduces the schema migration deferred here), variant prev/next, regenerate, TTS, and the "Style" field wired to `generation_templates.py`.
- **Phase 3:** character-canvas generation — auto-prompt from character appearance, per-mood reaction images.
- **Cleanup task (parallel):** remove the orphaned `Media_Creation` SwarmUI chain once superseded.
- **task-485 (parallel):** port the real egress/SSRF protections.

---

## 10. Locked decisions & remaining detail

**Locked** (from brainstorming): package `Image_Generation/` with `[image_generation]` config; 6 backends; port the server core; command-palette demo screen; light egress guard now with real SSRF port tracked in task-485; `default_backend = "swarmui"` / `enabled_backends = ["swarmui"]` first-run default; no schema migration in Phase 1; orphaned `Media_Creation` SwarmUI code left in place (separate cleanup task).

**Remaining implementation detail** (settle during planning, not a blocker): the exact keyring account-naming scheme for image-backend secrets — it must be namespaced so it cannot collide with LLM-provider keyring accounts (e.g. an `imagegen:` prefix), per §4.3.
