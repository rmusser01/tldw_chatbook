# Image-gen backends: fal.ai, Gemini (AI Studio), Fireworks — design

Date: 2026-07-26. Status: approved pending user review.
Adds three backends to `tldw_chatbook/Image_Generation/` on the established adapter contract. API shapes verified against live vendor docs 2026-07-26 (the task-620 stale-model lesson): fal queue API at `queue.fal.run` with `Authorization: Key`; Gemini `generateContent` with `x-goog-api-key` returning base64 `inlineData` (Imagen is deprecated, shutdown 2026-08-17 — not used); Fireworks workflows `text_to_image` returning raw bytes with Bearer auth.

## Decisions (user-approved)

- Backend ids: `fal`, `gemini`, `fireworks`. Labels: "fal.ai", "Gemini (AI Studio)", "Fireworks".
- Defaults: fal `fal-ai/flux/schnell`; gemini `gemini-2.5-flash-image` (nano-banana); fireworks `flux-1-schnell-fp8`.
- Live UAT with user-provided keys for all three after merge-readiness; opt-in live tests (`TLDW_LIVE_*` pattern) regardless.
- Per-service adapters (no generic REST abstraction, no OpenAI-compat shims — response shapes differ too much; codebase precedent is per-service).

## Architecture

One adapter file per service under `Image_Generation/adapters/` implementing the existing base contract (`generate(request) -> ImageGenResult`, sync/blocking, worker-driven). Everything else is table rows:
- Registry lazy-map entry per backend.
- `Image_Generation/config.py`: defaults constants, `ImageGenerationConfig` fields, `_SECRETS` rows (`fal`: `fal_image_api_key`, env `["FAL_KEY"]`, keyring `fal`; `gemini`: `gemini_image_api_key`, env `["GEMINI_API_KEY", "GOOGLE_API_KEY"]` (order = precedence, the modelstudio two-var pattern), keyring `gemini`; `fireworks`: `fireworks_image_api_key`, env `["FIREWORKS_API_KEY"]`, keyring `fireworks`), `_NON_SECRET` rows (`base_url`, `default_model`, `timeout_seconds` each; fal also `poll_interval_seconds`), `key_sources` picks the new rows up automatically.
- Defaults: fal base `https://queue.fal.run`, poll interval 2s, timeout 120s; gemini base `https://generativelanguage.googleapis.com/v1beta`, timeout 120s; fireworks base `https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models`, timeout 120s.
- `listing.py`: is_configured per backend (key present; same non-critical-exception envelope as siblings).
- Settings (`settings_image_gen_defaults.py`): `BACKEND_IDS` += 3 (canonical order: existing six, then `fal`, `gemini`, `fireworks`), `BACKEND_LABELS`, `FIELD_SCHEMA` rows mirroring openrouter's (`base_url`[url], `default_model`[text], `timeout_seconds`[int,min 1], `api_key`[secret]); probe table below. The panel, draft/save, key UX, and drift test pick the rows up with no panel code changes (that is the point of the schema-driven design; any test asserting six backends updates honestly to nine).

## Adapters

**fal (`fal_image_adapter.py`)** — queue + poll, the novita/modelstudio precedent:
1. Submit `POST {base}/{model_path}` with header `Authorization: Key {api_key}`, JSON body: `prompt` (negative prompt appended as a suffix line, the openrouter precedent), `seed` when set, size mapping when width/height set (fal `image_size` enum or `{width,height}` object — implementer verifies exact field from the model schema and uses the generic form).
2. **Poll URLs are SELF-BUILT, never response-followed**: the submit response's `request_id` is combined with the CONFIGURED base into `{base}/{app_id}/requests/{request_id}/status` and `/requests/{request_id}` — where **`app_id` is the first two segments of the model path** (`fal-ai/flux/schnell` → app `fal-ai/flux`; fal's queue addresses requests by app, not full model path — a naive full-path build 404s for our own default model). The response's `status_url`/`response_url` fields are NEVER followed (following server-provided URLs while attaching `Authorization` is a credential-exfiltration primitive) but ARE used as a cross-check: assert the response-provided status_url matches the self-built URL (origin + path); on mismatch fail loudly with a clear "fal queue URL shape changed" error so vendor drift surfaces as a diagnosis, not silent 404s. `request_id` is validated (UUID-ish charset) before URL use.
3. Poll with the auth header at `poll_interval_seconds` until COMPLETED/timeout; result `images[0].url` (fal CDN, different origin) is fetched via the shared `fetch_image_bytes` as an ENFORCED-untrusted URL (no trusted_origins, no credentials — exactly the openrouter image-link pattern).
- Self-built submit/poll URLs carry `trusted_origins=origin_set(url)` (allows operator-pointed private bases; metadata IPs still hard-blocked).

**gemini (`gemini_image_adapter.py`)** — single call:
1. `POST {base}/models/{model}:generateContent`, key ONLY in the `x-goog-api-key` header — never the `?key=` query form Gemini also accepts (URLs land in logs; headers do not).
2. Body: `contents=[{parts:[{text: prompt(+appended negative)}]}]`, `generationConfig.responseModalities` including `"IMAGE"` (implementer verifies the exact current requirement for `gemini-2.5-flash-image` from the docs and pins it in a test).
3. Response: iterate ALL `candidates[].content.parts[]`; first `inlineData` part wins → base64-decode with its `mimeType`. **No-image responses are a first-class error path**: safety blocks / text-only answers (`promptFeedback.blockReason`, `finishReason` SAFETY etc.) map to a clear sanitized error naming only the reason CATEGORY (never response text, never the prompt) — no index crashes on missing parts.
4. Response size: the base64 image arrives inside the JSON body (~3MB+ at 1024²). OpenRouter already receives base64 data-URLs through the same guarded `fetch_json` path, so this works today — the implementer CONFIRMS no byte-cap in the guarded path truncates image-sized JSON rather than assuming (and raises a clear error if a cap is hit).

**fireworks (`fireworks_image_adapter.py`)** — single call, bytes back:
1. `POST {base}/{model}/text_to_image`, `Authorization: Bearer {api_key}`, `Content-Type: application/json`, `Accept` derived from `ImageGenRequest.format` (`png`→`image/png`, `jpeg`→`image/jpeg`; pinned by test); body maps `prompt`/`negative_prompt`/`width`/`height`/`steps`/`cfg_scale`/`seed` directly (implementer verifies current field names from the API reference).
2. Response is RAW IMAGE BYTES on success; error responses arrive as JSON on the same endpoint — the adapter distinguishes by status + content-type and surfaces sanitized errors (never raw response bodies; a 404 gets the task-620 enriched message naming the attempted model id and `[image_generation.fireworks] default_model`).

**Shared requirements (all three):**
- **Model-id shape validation before URL construction** — NEW attack surface unique to these adapters: the model id becomes part of the URL path (existing six carry it in the JSON body). Charset allowlists per backend: gemini/fireworks `[A-Za-z0-9._-]+`; fal additionally allows `/` (path-shaped ids) but rejects `..` segments, leading/trailing `/`, and any of `?#%\s`. Violations raise the adapter's config-error type naming the offending id.
- All HTTP through the package's guarded helpers; the task-620 enriched not-found messaging pattern applied per backend (404/400-model-unknown names the model id + config key).
- `api_key` never logged, never in error text; negative prompt appended not dropped; `ImageGenResult` populated per contract (resolved_model only where genuinely reported — do not fabricate).

## http_client addition

`fetch_bytes_via_post(url, *, headers, json, timeout, trusted_origins) -> tuple[bytes, str]` (bytes + content-type): fireworks' shape (POST returning bytes) fits neither `fetch_json` (JSON-only) nor `fetch_image_bytes` (GET-shaped). Same discipline as both: egress `check_url_or_raise` with trusted_origins pre-request, no auto-redirects, per-hop revalidation with `same_origin`-gated credential stripping on any redirect, explicit timeout honoring `timeout=0`, size-bounded read. Unit-tested with the same fake-client matrix as the existing helpers (incl. the cross-origin cred-strip and redirect-to-private cases).

## Settings probes

- gemini: `GET {base}/models` with `x-goog-api-key` → 2xx `Reachable` / 401,403 `Auth failed` / other mapping — full auth-verified probe (cheap list endpoint exists).
- fireworks: implementer checks for the OpenAI-compat `GET .../inference/v1/models` with Bearer on the api.fireworks.ai host; if confirmed cheap+authed, full probe; else reachability-only → `Reachable (auth unverified)`.
- fal: reachability-only on the configured base (no confirmed cheap authed endpoint) → `Reachable (auth unverified)`.
- All through the existing probe module's closed badge set and sanitization; no new badge strings.

## Testing

- Per-adapter unit suites (`Tests/Image_Generation/test_{fal,gemini,fireworks}_adapter.py`) with fake clients patched on the real module: payload shape incl. auth-header placement (gemini: header-not-query pinned), model-id validation matrix (incl. fal `..` rejection), response parsing (gemini all-parts iteration + no-image/safety mapping; fireworks bytes-vs-JSON-error branch; fal poll lifecycle with SELF-BUILT URL assertion — the fake asserts the poll URL was constructed from config base + request_id, NOT taken from the response's status_url), error sanitization, egress trust threading (self-built trusted; extracted enforced).
- `fetch_bytes_via_post` helper suite as above.
- Config loader: new `_SECRETS`/`_NON_SECRET` rows, key_sources for the gemini two-var precedence.
- Settings: the existing drift test must pass unmodified (it validates schema-vs-loader automatically); backend-count-sensitive tests updated to nine; probe tests for the two/three new probe branches.
- Opt-in live tests per backend in `test_live_backends.py` style (`TLDW_LIVE_FAL_API_KEY` etc., markers integration/optional/slow).
- Live UAT (post merge-readiness, user keys): per backend — Settings probe, then Console `/generate-image :backend <prompt>` end-to-end with card verification; the OpenRouter UAT recipe.

## Reference images (engine seam — user-approved scope addition)

The P1 port left a dormant seam that this program opens at the ENGINE level for the three new backends (Console attach UX is a follow-up spec — engine first, surface second):
- `ImageGenRequest.reference_image: ResolvedReferenceImage | None` already exists (`capabilities.py`: bytes `content`, `mime_type`, dims) — the request shape needs no change. `worker.build_request` gains an optional `reference_image` param that populates it.
- **Per-backend capability flags**: `supports_reference_image` — True for `fal`, `gemini`, `fireworks`; False for the existing six (modelstudio's dormant per-model map stays dormant; per-model gating remains v2). Exposed via the existing `resolve_backend_reference_image_capability` seam so callers can query it.
- **Fail loudly, never silently ignore**: `run_generation`'s validation choke point rejects a request carrying a reference image for a backend whose flag is False ("backend X does not support reference images") — the silent no-op contract of the dormant seam ends with this program.
- **Validation at the choke point**: mime allowlist (png/jpeg/webp), size cap `IMAGE_GEN_REFERENCE_MAX_BYTES = 10 * 1024 * 1024`, non-empty content bytes required (file_id/temp_path variants are NOT accepted by the engine — bytes-in-memory only, preserving the P1 no-media-DB-coupling decision).
- **Per-adapter encoding**:
  - gemini: an `inlineData` part (`{inline_data: {mime_type, data: <b64>}}`) alongside the text part — nano-banana's native editing input.
  - fal: `image_url` field as a base64 data URI (`data:{mime};base64,...`) — fal accepts data URIs, avoiding any fal-storage upload scope. When a reference is present the model is used as-is (choosing an image-capable fal model is the caller's job; a model that ignores image_url is vendor behavior, not adapter error).
  - fireworks: implementer verifies the current image-input field/route from the API reference (Kontext-family workflows; possibly a distinct `image_to_image`/kontext endpoint) — if the configured model's workflow does not accept an image input the adapter surfaces the vendor error sanitized; if a distinct route exists the adapter selects it when a reference is present.
- **Tests**: per-adapter encoding shape (gemini part structure; fal data-URI; fireworks field/route selection); choke-point refusal for unsupported backends (all six legacy ids); mime/size validation matrix; capability-flag exposure.
- **Live UAT addition**: one nano-banana edit case (tiny reference image + edit instruction) alongside the text-to-image cases.

## Non-goals

Console/UI reference-image attach UX (follow-up spec); per-MODEL reference gating (the dormant `reference_image_supported_models` map stays dormant); reference images for the six existing backends; fal websocket/realtime; LoRA/ControlNet params; per-backend `allowed_extra_params` curation beyond the existing config-only mechanism; Vertex-AI service-account auth for Gemini (AI-Studio API keys only, per the request).

## Risks

- Vendor model-id drift (mitigated: enriched errors, live tests, Settings model field with resolved-default placeholder).
- fal queue URL shape is vendor-documented-but-unversioned; the self-built-URL decision trades a hypothetical shape change (loud, easy fix) for closing a real credential-exfiltration channel — correct trade.
- Settings backends list grows to nine rows; fine at ≥120 cols (sub-120 remains task-653).
