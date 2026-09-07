# Video Generation, Playback, and Streaming — Brainstorm / Design

Date: 2026-08-07
Status: Brainstorm (post-review revision 1; pre-ADR implementation)
Scope: Expand the existing image-generation feature to video (MiniMax H3 official API, ComfyUI, stable-diffusion.cpp); add ephemeral, name-referenced video storage; add in-app video playback and URL streaming in the Console.

Revision 1 (same-day review): corrected the metadata store (v31 `messages.metadata_json`, no migration — not a v26 sidecar extension), re-keyed the video store by `message_id` (not console session id), added store size-cap/LRU, closed the ffmpeg/yt-dlp redirect egress hole, scoped streaming v1 to progressive streams, added MiniMax cancel-on-stop, paused-by-default previews, in-flight guard, and single-demux audio. See §10.

ComfyUI workflow supersession (2026-08-09): the workflow-asset examples and
output assumptions in §3.2 are superseded by the approved
[ComfyUI MiniMax H3 design](2026-08-09-comfyui-minimax-h3-video-workflows-design.md).
TASK-3401.6 ships Base and Spectrum H3 text-to-video graphs, not Wan/SVD.

---

## 1. Context and goals

The app already has a mature image-generation pipeline: adapter Protocol + registry + config/secrets + validation choke point + off-thread worker + `/generate-image` Console command + generation cards persisted as DB BLOB attachments with a local-only metadata sidecar (schema v25).

Goal: replicate that success for **video**, under one explicit product constraint from the user:

> Videos are NOT stored permanently. Conversations refer to them by name; there is no guaranteed durable link once the file is gone or the app restarts.

And one follow-up capability:

> A user can **stream** any video they would normally download, playing it without saving it.

Reference implementations for playback:

- `rmusser01/textual-video` — in-process Textual widget (PyAV decode → `textual-image` display modes → `textual-canvas` track). Silent.
- `anshupriyan/TermTube` — subprocess pipeline (yt-dlp → ffmpeg RGB frame pipe → ANSI halfblock/ASCII render; ffplay audio with clock-driven A/V sync, drift stats, auto-reconnect).

---

## 2. What exists today (inventory to build on)

| Piece | Location | Reuse for video |
|---|---|---|
| Adapter Protocol + frozen request/result dataclasses | `Image_Generation/adapters/base.py` | Mirror as `VideoGenRequest`/`VideoGenResult` |
| Lazy-loading registry (`DEFAULT_ADAPTERS`, enable list) | `Image_Generation/adapter_registry.py` | Same pattern, video registry |
| Nested TOML config, env→config→keyring secrets, unknown-key warnings | `Image_Generation/config.py` | New `[video_generation]` section, same mechanics |
| Single validation choke point before adapter dispatch | `Image_Generation/worker.py` + `request_validation.py` | Same; add duration/fps/ratio bounds |
| Blocking adapters driven via `asyncio.to_thread` | `Chat/console_generate_image.py` (`run_generation_batch`) | Same offload pattern |
| Command grammar `:backend` / `@style` tokens | `Chat/console_command_grammar.py`, `console_generate_image.py` | `/generate-video` sibling |
| Generation cards + variant browse/keep/regenerate | `Widgets/Console/console_generation_card.py` | Video card (player + details) |
| Image metadata sidecar (local-only, position-aligned WITH attachments) | v25 migration `message_generation_metadata` | NOT reused for video — see §4 |
| **Local-only per-message JSON metadata** | v31 migration `messages.metadata_json` (task-2364) | **Video metadata store — no migration needed** |
| SSRF egress policy for outbound URLs | `Utils/egress.py` | Gate streaming + CDN download URLs |
| Optional-dep gating | `Utils/optional_deps.py` (`av` slot already exists, `rich_pixels` core) | New `video_playback` extra |
| Async poll-loop adapters (precedent for task-based APIs) | `novita`, `modelstudio`, `fal` image adapters | MiniMax task polling |
| Subprocess-binary adapter (precedent for sd.cpp) | `stable_diffusion_cpp_adapter.py` | sd.cpp video flags |
| Inline image render (pixels/graphics modes, fit-to-cell) | `Chat/console_image_view.py`, `rich_pixels` | Video frame rendering |
| Per-session in-flight generation guard | `chat_screen.py` `_console_imagegen_inflight_sessions` | Mirror for `/generate-video` |

Schema note (verified 2026-08-07): `chachanotes` head is **v32**; v26 is taken (`conversation_context_summary`). Any future migration would be v32→v33 — but this design needs none (§4).

Legacy parallel path: `Media_Creation/image_generation_service.py` (SwarmUI-only, file-output based). Do NOT extend it; the Console-native pipeline is canonical. Consider it for eventual retirement (separate task).

---

## 3. Video generation backends

### 3.1 MiniMax H3 official API (`minimax` adapter)

Facts verified against platform.minimax.io docs (2026-08-07):

- `POST https://api.minimax.io/v2/video_generation`, model `MiniMax-H3`.
- Multimodal `content[]`: exactly one `type=text` (the prompt/motion description, ≤ 7000 chars); optional `image_url` items with `role` = `first_frame` / `last_frame` / `reference_image`; `video_url` (`reference_video`); `audio_url`. Mixed input cap 12 files.
- Modes fall out of inputs: text-only → T2V; +first_frame → I2V; first+last frame → FL2V; reference assets → R2V.
- `duration` 4–15 s integer; `resolution` 768P / 2K; `ratio` common ratios or `adaptive` (mandatory adaptive for I2V; T2V requires an explicit ratio).
- Async task API: returns `task_id` → poll `GET /v2/query/video_generation/{task_id}` (~10 s interval) → on success the response carries the **download URL directly** (`task.content.url`). Terminal failure states: `failed` / `cancelled`.
- **Cancellation**: on user stop, the adapter stops polling AND calls the MiniMax task cancel/delete endpoint — a stopped cloud task must not bill to completion.
- Download URLs are **expiring CDN URLs** — download immediately on success; never persist the URL. This aligns with (and partly motivates) the ephemeral-storage product decision.
- Input media must be **URLs**. For local inputs (e.g. animating an image the app just generated) we must upload first (`/v1/files/upload` multipart) or host; uploading generated-image bytes to MiniMax is a privacy-relevant step gated by `allow_uploads` (default off).
- Config: `[video_generation.minimax]` `api_key` (env `MINIMAX_API_KEY` → config → keyring), `base_url`, `default_model` (`MiniMax-H3`), `poll_interval_seconds` (default 10), `timeout_seconds` (default ~600 — videos take minutes), `allowed_extra_params`.
- Cost note: H3 is priced per generated second; a 15 s 2K clip is real money. Add a settings toggle `confirm_cost_estimate` that prints the est. cost on the card before dispatch (default on for video).

### 3.2 ComfyUI adapter (`comfyui`)

- Talks to a user-run ComfyUI server (default `http://127.0.0.1:8188`) over HTTP + WebSocket:
  - `POST /prompt {prompt: <workflow JSON>, client_id}` → `prompt_id`
  - `WS /ws?clientId=…` for progress events (or poll `GET /history/{prompt_id}` — prefer polling v1 to avoid a new `websockets` dependency; WS is an optimization)
  - Outputs enumerated from history (`filename`, `subfolder`, `type`); bytes via `GET /view?...`
  - Local input images via `POST /upload/image`
- TASK-3401.6 ships exactly the Base and opt-in Spectrum MiniMax H3 API graphs
  described by the superseding H3 design. Users can still drop in their own
  confined workflow JSON; exact title conventions remain the generic control
  seam.
- The H3 assets terminate at `SaveVideo` and return MP4 descriptors under the
  live-observed history shape. Generic workflows may still use the other
  output classes supported by the adapter.
- Config mirrors SwarmUI's (`base_url`, optional auth header, `timeout_seconds`, `default_workflow`, `allowed_extra_params`). The SwarmUI adapter is the closest in-repo cousin — reuse its session/retry structure.
- Egress: ComfyUI base URL is a user-configured trusted origin (same treatment as SwarmUI today).

### 3.3 stable-diffusion.cpp adapter (extend `stable_diffusion_cpp` or sibling `stable_diffusion_cpp_video`)

Verified against leejet/stable-diffusion.cpp README (fetched 2026-08-07):

- Video models now supported in sd.cpp: **Wan2.1 / Wan2.2** (2025-09-06), **Wan2.1 VACE** (2025-09-14), **LTX-2.3**, **HunyuanVideo 1.5**, **LingBot-Video**, and — landed 2026-08-04, three days ago — **Day-1 MiniMax-H3** support. Docs: `docs/wan.md`, `docs/minimax_h3.md`, `docs/ltx2.md`, `docs/hunyuan_video.md`; CLI surface: `examples/cli/README.md`.
- SVD (the 2023 img2vid model) is effectively superseded; ggml's 4-D tensor limit made conv3d painful and the ecosystem moved to Wan/LTX/H3.
- The existing adapter already shells out to the configured `sd` binary with an argv list in a temp dir (`subprocess.run`, `shell=False`, timeout, output-file-readback). Video extends this: same binary, video-model paths (`--diffusion-model` + VAE/LLM/clip companions per model doc), frame/fps flags, longer timeout (default much higher; video on CPU is minutes-to-hours).
- **Pin the exact CLI flags against the user's sd.cpp build at implementation time** — upstream warns the CLI changes frequently. The adapter feature-detects a minimum binary version at init.
- Same MiniMax-H3 model can thus be run cloud (adapter 3.1) or local (this adapter) — the config's per-backend `default_model` makes that an explicit user choice.
- Local guardrails: conservative default duration/resolution/fps for local backends so a default invocation on CPU/Metal finishes in minutes, not hours.

### 3.4 Shared semantics across the three

- `VideoGenRequest`: prompt, negative_prompt (sd.cpp/ComfyUI), duration_seconds, fps, width/height or ratio, seed, model, `first_frame`/`last_frame`/`reference_assets` (reuse `ResolvedReferenceImage` mechanics for local files), extra_params (per-backend allowlist).
- `VideoGenResult`: content bytes + content_type (`video/mp4`, `image/webp`), width/height/duration/fps, resolved_seed/resolved_model (only when verifiable — same rule as task-558).
- All three are **long-running and async-shaped** (task polling, WS progress, or a minutes-long subprocess). The worker must support cooperative cancellation (stop button) and emit progress events (`Preparing/Queueing/Processing/Encoding`) for the card.
- Validation choke point mirrors images: duration/fps/dimension bounds, per-backend `extra_params` allowlist, reference-asset mime/size/count checks (MiniMax caps: image ≤ 30 MB, video ≤ 50 MB, audio ≤ 15 MB).

---

## 4. Ephemeral storage + name references (the user's core constraint)

Images today persist as DB BLOBs (`message_attachments.data`, 4 MB inline cap). Videos (5–100 MB) must not.

Proposed model — **message-keyed ephemeral video store**:

1. On generation, bytes land in `<user_data_dir>/generated_videos/<message_id>/<slug>.mp4` (never in the DB). Keyed by the **stable message id**, NOT the console session id: console session ids are ephemeral, so a session-keyed directory would orphan `ttl`-retained files on every restart (new session → new directory → old files unfindable). Message-keyed paths make resolution a direct path construction with no directory scan. `Utils/paths.get_user_data_dir()` + `path_validation` for all joins; slugs are app-generated, prompt-derived, filesystem-safe, collision-suffixed (`-2`, `-3`).
2. The message persists only:
   - a content marker, e.g. `[video] dusk-over-neon-tokyo` — a human-readable **slug name** mirroring `[image] <prompt>`;
   - generation metadata in the **local-only `messages.metadata_json` column** (v31, task-2364 precedent: engine provenance / interrupted flag live there today): prompt, backend, model, seed, duration, resolution, slug name, source-image message id (i2v) — **no path, no URL, no migration required**.
   - The v25 `message_generation_metadata` sidecar is deliberately **NOT** extended: its `position` column is defined by index alignment with the message's attachments (`GenerationVariantMeta` docstring), and a video message has no attachments — a position-0 row without attachment 0 would break that invariant for every generic reader (`get_generation_metadata_for_messages`, variant regenerate, the image card builder).
3. A runtime **VideoStore** resolves `(message_id, slug) → file path` for the live session and reports missing after restart or expiry. The card resolves through it.
4. Disk bounds: a total store size cap (config `[video_generation] max_store_mb`, default ~2 GB) with **LRU eviction** applies in ALL retention modes — evicting tombstones the oldest live videos even within a running session (a long session generating 100 MB clips would otherwise grow unbounded between restarts).
5. Ephemerality policy (config `[video_generation] retention = "session" | "ttl"`):
   - `session` (default, strictest match to the request): dir wiped on app start; after a restart the card renders **"video expired"** with its metadata and a ♻ regenerate action.
   - `ttl`: files survive restarts up to N hours (still under the size cap), with no durability promise; expired/missing → same expired card.
6. Deliberate consequences to document in the ADR:
   - Export/copy of a video message exports the **name + metadata**, not bytes (offer "Save a copy…" as an explicit card action that copies the live file to a user-chosen path — escaping ephemerality is a user act, never automatic). Export paths must render video messages as marker+metadata text and never error on missing bytes.
   - Image-generation readers never consume video messages: the image card path keys off the `[image] ` marker, the video path off `[video] ` — marker discrimination plus the separate metadata column keeps the two pipelines isolated by construction.
   - MiniMax's expiring CDN URL is never persisted; if the local file is gone, the message cannot re-download (server-side retention is short anyway). Regenerate is the recovery path.
   - Sync: nothing video-related syncs (`metadata_json` is already local-only per the v31 migration note; same rule as v19/v24/v25/v26).

This gives exactly the requested semantics: conversations *mention* videos by name; links/paths are best-effort and silently degrade to a named tombstone.

---

## 5. Playback design (from the two reference repos)

### What each reference contributes

| | `rmusser01/textual-video` | `anshupriyan/TermTube` |
|---|---|---|
| Architecture | In-process Textual widget | Subprocess pipeline |
| Decode | PyAV (`av`) | ffmpeg CLI → raw RGB frames |
| Render | `textual-image` (SIXEL/TGP/HALFCELL/UNICODE) | ANSI truecolor halfblock / ASCII ramp |
| Audio | none | ffplay, clock-driven sync + drift stats |
| Streaming | file paths only | yt-dlp URL resolution + ffmpeg `-reconnect` |
| Scrubber | `textual-canvas` track | n/a |
| License | verify before vendoring (fork of fi-res/textual-video) | MIT |

Note: `textual-image` is already a **core** dependency; `rich_pixels` already drives inline images; `av` already has an `optional_deps` slot; TermTube's author-independent insight is that audio+sync belongs in a subprocess, not in the Textual loop.

### Recommended hybrid — two playback surfaces

**A. Transcript preview (silent, in-card).** textual-video-style: decode with PyAV on a thread, push frames into a `rich_pixels`-backed widget at ~12–15 fps, HALFCELL-class rendering, hard caps (≤ 30 s clips, ≤ preview resolution, pause when scrolled off-screen, one active preview per screen). **Previews default to paused** — playback starts only on explicit user action, so a transcript full of cards never burns CPU/GPU unprompted. Dependencies: `av`, `textual-canvas` (progress track) in a new `video_playback` extra.

**B. Full player screen (audio, streaming-capable).** TermTube-style pipeline behind a modal screen (`player_screen.py`):

- **Single-demux pipeline**: ONE ffmpeg process per source demuxes both the video frame pipe AND the audio PCM pipe (raw s16le to ffplay stdin, or an equivalent audio sink). Never let ffplay open the URL/source itself — for streams that would fetch the same source twice (double bandwidth, double server load, divergent clocks).
- Master clock = audio; video timer chases it with drift correction + dropped-frame stats in a status line (TermTube's approach, surfaced as a footer hint).
- Controls per repo keybinding conventions (single-letter, htop-style; no terminal-reserved keys): space pause, `s` stop, `←/→` seek (restart decode with `-ss`), `+/-` volume, `q` close. Footer advertises only implemented actions.
- Terminal capability detection: prefer sixel/kitty graphics when available, fall back to halfcell pixels; degrade to ASCII ramp on dumb terminals (TermTube's two-style fallback).
- While the modal player is active, suspend competing workers/timers (precedent: splash-screen exclusivity) so the compositor keeps frame pacing.

Both surfaces live in a new `Media_Playback/` package behind a shared `VideoFrameSource` protocol (`AvFileSource`, `FfmpegPipeSource`), with binary detection (`ffmpeg`, `ffplay`, `yt-dlp`) via `shutil.which` at init, reported in Diagnostics.

---

## 6. Streaming (play-any-URL instead of download)

- Entry points: `/stream-video <url>` Console command and a card action on video messages; also reusable from the Media tab later.
- Resolution pipeline: user-typed URL → **egress policy first** (`Utils/egress.py`; user-typed URL seeds trusted-origin intent) → if not a direct media URL, `yt-dlp -g`/JSON resolves to a direct stream (yt-dlp subprocess, never imported as a library — keeps licensing and dependency weight clean) → **the fully resolved final URL is validated through egress AGAIN before reaching ffmpeg** (redirect hops and `yt-dlp -g` output included — ffmpeg follows redirects internally, so validating only the typed URL would leave a policy hole) → ffmpeg reads the URL with `-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5` (TermTube's flags) → player screen B.
- Streaming deliberately sidesteps `MAX_FETCH_BYTES_MEDIA` (500 MB): no byte cap on streams; sessions are time-boxed and user-terminated. State this residual-risk decision in the egress doc/ADR.
- **v1 scope: progressive single-URL streams only.** Sites where yt-dlp yields separate audio/video DASH or HLS renditions need ffmpeg dual-input muxing — documented as a follow-up, not v1.
- Seek support depends on the source honoring HTTP range requests; when not seekable, disable seek keys (and say so in the footer).
- No caching to disk by default (`stream` ≠ `download`); an explicit "save while watching" toggle can tee to a file later — phase 4+, not v1.

---

## 7. Security, privacy, policy

- All outbound media URLs (MiniMax CDN download, ComfyUI base, user stream URLs AND their resolved final forms) pass `Utils/egress.py`; trust is seeded only at user-intent boundaries (typed URL, configured base_url), threaded down — per the web-fetch-hardening design.
- I2V from a locally generated image uploads that image to MiniMax when using the cloud adapter — settings toggle `[video_generation.minimax] allow_uploads` (default off; off = cloud I2V refused with a clear message, local backends unaffected).
- Secrets: `MINIMAX_API_KEY` env → config → keyring, same mechanics as image backends; never logged.
- Temp files under user data dir; `path_validation` for all path joins; filenames are app-generated slugs, never raw prompt text.
- License check before vendoring any code from the reference repos (TermTube is MIT; textual-video's license must be verified — otherwise reimplement from its architecture).

## 8. Dependencies and packaging

- `video_generation` extra: no new deps (httpx is core; ComfyUI v1 polls `/history` so no `websockets` requirement).
- `video_playback` extra: `av`, `textual-canvas`; runtime binaries `ffmpeg`/`ffplay` (required for screen B), `yt-dlp` (streaming only) detected with `shutil.which`.
- `optional_deps.py`: fill the existing `av` slot; add `textual_canvas`; AREA_MEDIA_CREATION / new AREA_PLAYBACK messaging with install hints.

## 9. Phasing sketch (backlog-ready; epic TASK-3401)

- **Phase 0 — ADR + spikes** (task-3401.1): ADR-044 done; spikes pinning sd.cpp video CLI flags, MiniMax-H3 request/response against a real key, ComfyUI workflow JSON contract.
- **Phase 1 — Core + MiniMax** (.2–.5): `Video_Generation/` package (base/registry/config/validation/worker), `minimax` adapter (T2V first), ephemeral VideoStore + `metadata_json` message metadata (no migration), `/generate-video`, video card v1 (metadata + poster frame + "open in full player"/"save a copy"), cost-confirm toggle, in-flight guard, palette provider.
- **Phase 2 — Local/self-hosted backends** (.6–.8): `comfyui` adapter + shipped workflow assets; sd.cpp video support; I2V from kept image variants (first_frame bridging — the image→video killer feature, gated by `allow_uploads` for cloud).
- **Phase 3 — Playback** (.9–.10): `Media_Playback/`, transcript preview widget (paused by default), full player screen with single-demux audio sync, capability detection.
- **Phase 4 — Streaming** (.11): `/stream-video`, yt-dlp resolution, final-URL egress validation, reconnect/drift polish, seek semantics; progressive streams v1, DASH/HLS mux follow-up.
- **Phase 5 — Parity** (.12): Settings page (mirror image-gen panel), `@style` video templates, Diagnostics, user-guide docs, eval hooks.

Each phase is independently shippable; 1+2 deliver generation without playback (external player fallback), 3+4 are pure client features.

## 10. Open questions / risks

1. **sd.cpp CLI drift** — H3 support is 3 days old; pin a minimum binary version and feature-detect at adapter init.
2. **Local generation time** — Wan/H3 on CPU/Metal can be minutes-to-hours; honest progress + cancellation + conservative local default guardrails are required, not optional.
3. **MiniMax URL expiry window** — undocumented; download immediately on task success, never lazily.
4. **A/V sync inside Textual** — the compositor owns the screen; full-player screen must suspend other workers/timers while active (precedent: splash screen exclusivity).
5. **Terminal graphics fragmentation** — sixel vs kitty vs halfcell vs ASCII; keep halfcell the baseline, graphics opt-in.
6. **ComfyUI workflow versioning** — shipped JSON can rot against user's installed nodes; validate required node classes at adapter init and name the missing ones in the error.
7. **Storage naming** — slugs must be unique per message and stable for the message's lifetime; collisions get `-2`, `-3`.
8. ~~Metadata shape~~ — **resolved in review**: `messages.metadata_json` (v31 precedent), no migration; the v25 sidecar's position↔attachment invariant is preserved by not touching it.
9. ~~Audio double-fetch for streams~~ — **resolved in review**: single-demux pipeline (§5B); ffplay never opens the source itself.
10. ~~ffmpeg internal redirects bypass egress~~ — **resolved in review**: validate the fully resolved final URL before handing to ffmpeg (§6).
11. **Merge-time board hygiene** (from lessons-backlog-hygiene): task IDs 3401.x were CLI-auto-assigned; sweep all remote refs for collisions before merging, and check for in-flight PRs by task id before starting each implementation.

## 11. ADR checklist (required per repo rules)

- Storage/ephemerality + data-ownership policy for generated video: **ADR-044 (written, revised same-day)**.
- Provider/runtime boundary (third media backend family, first WS use): covered in ADR-044.
- Dependency/tooling choices (`av`, `textual-canvas`, ffmpeg/ffplay/yt-dlp as runtime binaries): ADR-044.
- Reference to this brainstorm from the ADR and each phase's backlog task: done.
