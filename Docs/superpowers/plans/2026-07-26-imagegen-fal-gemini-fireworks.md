# fal.ai / Gemini / Fireworks Image Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three new image-generation backends (`fal`, `gemini`, `fireworks`) on the established adapter contract, plus the engine-level reference-image seam, wired through config, listing, registry, and the Settings page.

**Architecture:** One adapter file per service; everything else is rows in existing tables (`_SECRETS`/`_NON_SECRET`, registry lazy-map, `BACKEND_IDS`/`FIELD_SCHEMA`, probe dispatch). One new guarded HTTP helper (`fetch_bytes_via_post`) for Fireworks' bytes-returning POST. Reference images open the dormant `ResolvedReferenceImage` seam with per-backend capability flags and fail-loudly choke-point validation. Spec: `Docs/superpowers/specs/2026-07-26-imagegen-fal-gemini-fireworks-design.md` — read it before any task.

**Tech Stack:** Python 3.11+, httpx (sync, via the package's guarded helpers), pytest with fake clients.

## Global Constraints

- All HTTP through the package's guarded helpers (`http_client.fetch_json`, `image_format_utils.fetch_image_bytes`, and Task 1's `fetch_bytes_via_post`): egress `check_url_or_raise` pre-request, self-built URLs carry `trusted_origins=origin_set(url)`, response-extracted URLs are fully enforced (no trust, no credentials), no auto-redirects.
- **Never follow server-provided URLs with credentials attached** (fal poll URLs are self-built; response URLs are cross-check only).
- **Model ids are validated before URL construction**: gemini/fireworks `^[A-Za-z0-9._-]+$`; fal additionally allows `/` but rejects `..` segments, leading/trailing `/`, and any of `?#%` or whitespace.
- API keys never in logs, error text, or URLs (gemini key goes in the `x-goog-api-key` HEADER, never `?key=`).
- 404/400-model-unknown errors name the attempted model id and the config key (`[image_generation.<backend>] default_model`) — the task-620 pattern; copy `openrouter_image_adapter.py`'s enrichment shape.
- Negative prompt is appended to the prompt text (openrouter precedent), never dropped.
- `ImageGenResult.resolved_model`/`resolved_seed` only where the backend genuinely reports them — never fabricate.
- Defaults (exact): fal base `https://queue.fal.run`, model `fal-ai/flux/schnell`, poll 2s, timeout 120; gemini base `https://generativelanguage.googleapis.com/v1beta`, model `gemini-2.5-flash-image`, timeout 120; fireworks base `https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models`, model `flux-1-schnell-fp8`, timeout 120.
- Env precedence: `FAL_KEY`; `GEMINI_API_KEY` then `GOOGLE_API_KEY`; `FIREWORKS_API_KEY`. Keyring ids: `fal`, `gemini`, `fireworks`.
- Reference images: bytes-in-memory only (`ResolvedReferenceImage.content`), mime allowlist `{image/png, image/jpeg, image/webp}`, cap `IMAGE_GEN_REFERENCE_MAX_BYTES = 10 * 1024 * 1024`, unsupported-backend requests REFUSED at the `run_generation` choke point.
- Where the spec says "implementer verifies from the API reference" (fal image_size field, gemini responseModalities requirement, fireworks body field names + image-input route, fireworks/fal probe endpoints): verify against the LIVE vendor docs via WebFetch during the task, record the verified shape + doc URL in your report, and pin it in a test. Do not guess from training data (task-620 lesson).
- Tests FOREGROUND only (background pytest notifications never reach subagents). venv: `source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate`; run from the worktree. TDD red-then-green per behavior. `ruff check` touched files; `python -c "import tldw_chatbook.app"` clean. Explicit-path staging.
- Baseline flake (NOT yours): `test_console_conversation_browser_search_ignores_stale_results`.

---

### Task 1: `fetch_bytes_via_post` guarded helper

**Files:**
- Modify: `tldw_chatbook/Image_Generation/http_client.py`
- Test: `Tests/Image_Generation/test_http_client.py`

**Interfaces:**
- Consumes: existing module internals — `_validate_egress_or_raise`, `create_client`, `egress.same_origin`/`egress._hop_headers`, `DEFAULT_MAX_REDIRECTS`, `_DEFAULT_TIMEOUT` (read `fetch_json` at :126 end-to-end first; mirror its structure).
- Produces: `fetch_bytes_via_post(url: str, *, headers: dict | None = None, json: Any = None, timeout: float | None = None, trusted_origins: frozenset = frozenset(), max_bytes: int = 32 * 1024 * 1024) -> tuple[bytes, str]` — returns `(body_bytes, content_type_header_value)`. Raises the module's existing egress/HTTP error types exactly like `fetch_json`.

- [ ] **Step 1: Failing tests** (append to `test_http_client.py`, reusing its fake-client fixtures — read the `hc` fixture and the fake-response helpers first):

```python
def test_fetch_bytes_via_post_returns_body_and_content_type(monkeypatch, hc):
    """Happy path: POST returns bytes + the content-type header."""
    # fake client returns 200, content=b"\x89PNG...", headers={"content-type": "image/png"}
    body, ctype = hc.fetch_bytes_via_post("https://api.example.com/gen", json={"prompt": "x"})
    assert body.startswith(b"\x89PNG") and ctype == "image/png"

def test_fetch_bytes_via_post_validates_egress_first(monkeypatch, hc):
    # private IP without trusted_origins -> egress error, fake client NEVER called

def test_fetch_bytes_via_post_strips_credentials_on_cross_origin_redirect(monkeypatch, hc):
    # 307 to another host: Authorization absent on hop 2; same-origin keeps it
    # (mirror test_fetch_json_strips_authorization_on_cross_origin_redirect's fake shape)

def test_fetch_bytes_via_post_redirect_to_private_ip_blocked(monkeypatch, hc):
    # 307 Location -> 10.0.0.1 raises egress error

def test_fetch_bytes_via_post_max_bytes_exceeded_raises(monkeypatch, hc):
    # content longer than max_bytes -> clear error naming the cap, not a truncated return

def test_fetch_bytes_via_post_respects_explicit_zero_timeout(monkeypatch, hc):
    # timeout=0 passed through (the task-497 lesson), None -> _DEFAULT_TIMEOUT
```

- [ ] **Step 2: Run** `python -m pytest Tests/Image_Generation/test_http_client.py -q` → new tests FAIL (attribute missing).
- [ ] **Step 3: Implement** by mirroring `fetch_json`'s manual redirect loop verbatim in structure: per-hop `_validate_egress_or_raise(current, trusted_origins=...)`, `same_origin` computed against the FIRST url, `egress._hop_headers(headers, is_same_origin)` on hops, json body re-sent per the module's documented redirect semantics, `raise_for_status()`, then length check against `max_bytes` before returning `(resp.content, resp.headers.get("content-type", ""))`. Google docstring stating the fireworks-shaped use case and the cap semantics.
- [ ] **Step 4: Run** the full file green; `ruff check tldw_chatbook/Image_Generation/http_client.py`.
- [ ] **Step 5: Commit** `feat(imagegen): guarded fetch_bytes_via_post helper`.

---

### Task 2: Config + listing rows for the three backends

**Files:**
- Modify: `tldw_chatbook/Image_Generation/config.py` (defaults constants ~line 40s; `_SECRETS` ~:64; `_NON_SECRET` ~:76; `ImageGenerationConfig` fields ~:190s; builder ~:380s), `tldw_chatbook/Image_Generation/listing.py`, `tldw_chatbook/config.py` (shipped default-TOML example block for `[image_generation]` — add commented sections for the three)
- Test: `Tests/Image_Generation/test_config_loader.py`, `Tests/Image_Generation/test_listing.py` (or the file that tests is_configured — find it)

**Interfaces:**
- Produces (exact names later tasks rely on): constants `DEFAULT_FAL_IMAGE_BASE_URL/MODEL/POLL_INTERVAL_SECONDS/TIMEOUT_SECONDS`, `DEFAULT_GEMINI_IMAGE_BASE_URL/MODEL/TIMEOUT_SECONDS`, `DEFAULT_FIREWORKS_IMAGE_BASE_URL/MODEL/TIMEOUT_SECONDS` (values from Global Constraints); cfg fields `fal_image_base_url/api_key/default_model/poll_interval_seconds/timeout_seconds`, `gemini_image_base_url/api_key/default_model/timeout_seconds`, `fireworks_image_base_url/api_key/default_model/timeout_seconds`; `_SECRETS` rows `("fal", ("fal_image_api_key", ["FAL_KEY"], "fal"))` etc. per Global Constraints (use the table's existing tuple shape — note swarmui's row also carries a config-key name after the #901 fix; new rows use `"api_key"` there); `_NON_SECRET` rows mapping `("fal","base_url")→"fal_image_base_url"` etc.; `key_sources` covers the three automatically (verify).
- `listing.py`: `_is_fal_configured`/`_is_gemini_configured`/`_is_fireworks_configured` following the sibling shape (key present ⇒ configured), registered wherever the sibling functions are dispatched.

- [ ] **Step 1: Failing tests**: loader round-trip per backend (nested `[image_generation.fal] api_key/default_model/...` → flat fields + `key_sources["fal"]=="config"`); env precedence (`GEMINI_API_KEY` beats `GOOGLE_API_KEY`; each env var name surfaces as `env:<VAR>`); defaults when unset (exact constants); listing is_configured true/false per key presence for all three.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** the rows + listing functions + the commented default-TOML sections. **Step 4:** `Tests/Image_Generation/ -q` green (the #901 drift test in `Tests/UI/test_settings_image_gen_defaults.py` will fail ONLY after Task 7 adds schema rows — do NOT touch settings files here; run that file to confirm it still passes since `BACKEND_IDS` is unchanged so far).
- [ ] **Step 5: Commit** `feat(imagegen): config + listing rows for fal/gemini/fireworks`.

---

### Task 3: Reference-image engine seam

**Files:**
- Modify: `tldw_chatbook/Image_Generation/capabilities.py` (per-backend `supports_reference_image` — read `resolve_backend_reference_image_capability` at :97 first and extend ITS mechanism rather than adding a parallel one), `tldw_chatbook/Image_Generation/request_validation.py`, `tldw_chatbook/Image_Generation/worker.py` (`build_request` optional param; `run_generation` already calls `validate_image_generation_request` at :82 — the new checks land inside the validator)
- Test: `Tests/Image_Generation/test_request_validation.py` (or the validator's existing test file), `Tests/Image_Generation/test_worker.py`

**Interfaces:**
- Consumes: `ResolvedReferenceImage` (capabilities.py:17 — `content: bytes | None`, `mime_type`, `bytes_len`).
- Produces: `REFERENCE_IMAGE_CAPABLE_BACKENDS: frozenset[str] = frozenset({"fal", "gemini", "fireworks"})` (in capabilities.py); `IMAGE_GEN_REFERENCE_MAX_BYTES = 10 * 1024 * 1024` and `REFERENCE_IMAGE_ALLOWED_MIMES = frozenset({"image/png", "image/jpeg", "image/webp"})` (in request_validation.py); `build_request(..., reference_image: ResolvedReferenceImage | None = None)`; validator issues (exact user-facing strings): `"backend '<id>' does not support reference images"`, `"reference image mime '<mime>' is not supported (png/jpeg/webp)"`, `"reference image exceeds the 10MB limit"`, `"reference image has no content bytes"`.

- [ ] **Step 1: Failing tests**: refusal for each of the six legacy backend ids with a reference set; acceptance for the three new ids (validator passes; adapter not reached); mime matrix (webp ok, image/gif refused); oversize (10MB+1) refused; `content=None` refused; `build_request(reference_image=...)` threads to `ImageGenRequest.reference_image`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** — capability set + validator checks (only when `request.reference_image is not None`), keeping every existing validation behavior untouched. **Step 4:** validator + worker files green; full `Tests/Image_Generation/ -q` green.
- [ ] **Step 5: Commit** `feat(imagegen): reference-image capability flags + choke-point validation`.

---

### Task 4: Gemini adapter

**Files:**
- Create: `tldw_chatbook/Image_Generation/adapters/gemini_image_adapter.py`
- Modify: `tldw_chatbook/Image_Generation/adapter_registry.py` (row: `"gemini": "...adapters.gemini_image_adapter.GeminiImageAdapter"`)
- Test: `Tests/Image_Generation/test_gemini_adapter.py` (new)

**Interfaces:**
- Consumes: `fetch_json` (http_client), cfg fields from Task 2, `ImageGenRequest`/`ImageGenResult`/adapter base contract (copy `openrouter_image_adapter.py`'s class skeleton, config access, and error types — it is the closest sibling), `ResolvedReferenceImage`.
- Produces: `GeminiImageAdapter.generate(request) -> ImageGenResult`; module-level `_validate_model_id(model: str) -> str` (raises the adapter config-error type on charset violation — regex `^[A-Za-z0-9._-]+$`).

Behavior (spec §gemini, all pinned by tests):
- URL `{base}/models/{validated_model}:generateContent`; headers `{"x-goog-api-key": key, "Content-Type": "application/json"}` — assert the key appears in NO url and NO query param.
- Body: `{"contents": [{"parts": [<optional inlineData part>, {"text": prompt_with_negative}]}], "generationConfig": {"responseModalities": [...]}}` — VERIFY the exact responseModalities requirement for `gemini-2.5-flash-image` from https://ai.google.dev/gemini-api/docs/image-generation via WebFetch; record in report; pin in test.
- Reference image (when set): `{"inline_data": {"mime_type": ref.mime_type, "data": base64(ref.content)}}` part BEFORE the text part.
- Response: iterate all `candidates[*].content.parts[*]`; first part with `inlineData`/`inline_data` → b64decode. No image → error mapping: `promptFeedback.blockReason` present → `"Gemini blocked the prompt (<blockReason>)"`; candidate `finishReason` not STOP → `"Gemini returned no image (<finishReason>)"`; else `"Gemini returned no image"`. Never include response text or the prompt.
- 400/404 naming the model → task-620 enrichment: message contains the model id and `[image_generation.gemini] default_model`.
- Self-built URL is the only URL — `trusted_origins=origin_set(url)` on the fetch_json call.

- [ ] **Step 1: Failing tests** (fake `fetch_json` patched on the adapter module, capturing kwargs): payload shape incl. header-not-query pin + responseModalities; negative-prompt appended; reference-image part shape + ordering; parts iteration across a text-then-image response; blockReason/finishReason/no-parts error matrix (sanitization: a response containing marker text must not leak into the error); model-id validation matrix (`good-model_1.0` ok; `../evil`, `a?b`, `a b` refused); 404 enrichment; trusted_origins passed.
- [ ] **Step 2: Run** → FAIL (module missing). **Step 3: Implement.** **Step 3b (spec §gemini.4):** CONFIRM `fetch_json`'s guarded path has no byte-cap that truncates image-sized JSON (~3-8MB base64): read the redirect loop + response handling for any size limit; if one exists, ensure it accommodates image payloads or raises a CLEAR error (never a silent truncation) — record the finding in your report (OpenRouter's base64 data-URLs already traverse this path, so expected answer is "no cap"; verify, don't assume). **Step 4:** file + `Tests/Image_Generation/ -q` green; registry resolves `get_adapter("gemini")`.
- [ ] **Step 5: Commit** `feat(imagegen): Gemini (AI Studio) image adapter`.

---

### Task 5: Fireworks adapter

**Files:**
- Create: `tldw_chatbook/Image_Generation/adapters/fireworks_image_adapter.py`
- Modify: `tldw_chatbook/Image_Generation/adapter_registry.py` (row `"fireworks"`)
- Test: `Tests/Image_Generation/test_fireworks_adapter.py` (new)

**Interfaces:**
- Consumes: Task 1's `fetch_bytes_via_post` (this is its consumer); cfg fields; base contract (openrouter skeleton); Task 3's reference validation guarantees (adapter may assume a present reference is mime/size-valid).
- Produces: `FireworksImageAdapter.generate(request) -> ImageGenResult`; `_validate_model_id` with `^[A-Za-z0-9._-]+$`.

Behavior:
- VERIFY from https://docs.fireworks.ai/api-reference/generate-a-new-image-from-a-text-prompt via WebFetch: exact body field names (prompt/negative_prompt/width/height/steps/cfg_scale/seed spellings) and the image-input route for reference images (Kontext-family or `image_to_image` workflow); record in report, pin in tests.
- URL `{base}/{validated_model}/text_to_image` (or the verified reference route when `request.reference_image` is set); headers `Authorization: Bearer {key}`, `Content-Type: application/json`, `Accept` from `request.format` (`png`→`image/png`, `jpeg`/`jpg`→`image/jpeg`, else `image/png`).
- Response: `(bytes, content_type)` from `fetch_bytes_via_post`; content_type starting `image/` → success (`ImageGenResult` with those bytes + content type); JSON content_type or non-2xx → sanitized error (status + short category; never raw body), 404 → task-620 enrichment naming model + `[image_generation.fireworks] default_model`.
- `trusted_origins=origin_set(url)` (self-built only).

- [ ] **Step 1: Failing tests**: Accept mapping matrix; body field spellings (as verified — the test IS the pin); bytes-success path; JSON-error-on-same-endpoint branch (sanitized); 404 enrichment; model-id matrix; reference-image route/field selection when set; trusted_origins.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4:** green + registry resolves. **Step 5: Commit** `feat(imagegen): Fireworks image adapter`.

---

### Task 6: fal adapter

**Files:**
- Create: `tldw_chatbook/Image_Generation/adapters/fal_image_adapter.py`
- Modify: `tldw_chatbook/Image_Generation/adapter_registry.py` (row `"fal"`)
- Test: `Tests/Image_Generation/test_fal_adapter.py` (new)

**Interfaces:**
- Consumes: `fetch_json` (submit/poll), `image_format_utils.fetch_image_bytes` (result image, ENFORCED — no trust, no credentials), cfg fields incl. `fal_image_poll_interval_seconds`; Task 3's reference guarantees.
- Produces: `FalImageAdapter.generate(request) -> ImageGenResult`; `_validate_model_path(path: str) -> str` (allows `/`; rejects `..` segments, leading/trailing `/`, `?#%` and whitespace); `_app_id(model_path: str) -> str` (first two `/`-segments; raises the config-error type if fewer than two segments).

Behavior (spec §fal):
- Submit `POST {base}/{validated_model_path}` with `Authorization: Key {key}`, body `{"prompt": prompt_with_negative, "seed": ... when set, "image_size": {"width": w, "height": h} when both set}` — VERIFY the image_size generic-object form against https://docs.fal.ai (queue + any flux schnell schema page) via WebFetch; when `request.reference_image` set add `"image_url": "data:{mime};base64,{b64}"`.
- Extract `request_id` (validate `^[A-Za-z0-9-]+$`). Self-build `status_url = {base}/{app_id}/requests/{request_id}/status` and `result_url = {base}/{app_id}/requests/{request_id}`. **Cross-check**: if the response carries `status_url`, compare against the self-built value (exact string or parsed origin+path); mismatch → error `"fal queue URL shape changed — expected <self-built>, vendor sent a different location"` (include NEITHER credentials nor the vendor URL beyond its origin). NEVER request the vendor-provided URL.
- Poll `status_url` with the auth header + `trusted_origins=origin_set(...)` every `poll_interval_seconds` until status COMPLETED (or IN_PROGRESS/IN_QUEUE continue; anything else → sanitized error) within `timeout_seconds` total; then GET `result_url` (auth, trusted) → `images[0].url` → `fetch_image_bytes(url)` with NO trusted_origins and NO auth header.
- 404 on submit → task-620 enrichment naming the model path + `[image_generation.fal] default_model`.

- [ ] **Step 1: Failing tests** (fake `fetch_json`/`fetch_image_bytes` capturing every call): submit payload shape (+data-URI reference; +image_size); **the self-built-URL pin** — fake submit response carries a DIFFERENT `status_url`; assert the poll call used the self-built URL and the adapter errored on the cross-check (two tests: matching status_url → polls fine; mismatching → loud error, zero polls to the vendor URL); app-id derivation (`fal-ai/flux/schnell` → polls hit `/fal-ai/flux/requests/...`; single-segment model refused); poll lifecycle (IN_QUEUE→IN_PROGRESS→COMPLETED), timeout path, FAILED status sanitized; result image fetched WITHOUT auth/trust (assert the fake fetch_image_bytes got no Authorization and no trusted_origins); request_id charset refusal; model-path matrix (`fal-ai/flux/schnell` ok, `../x` refused, `a//b` handling per the validator, trailing `/` refused); 404 enrichment.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (poll sleep via `time.sleep` — blocking adapter, worker-driven; make the sleep injectable/monkeypatchable for tests). **Step 4:** green + registry resolves + full `Tests/Image_Generation/ -q` green.
- [ ] **Step 5: Commit** `feat(imagegen): fal.ai queue image adapter with self-built poll URLs`.

---

### Task 7: Settings rows, probes, live tests

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_image_gen_defaults.py` (`BACKEND_IDS` → nine, canonical order: existing six then `fal`, `gemini`, `fireworks`; `BACKEND_LABELS` += `"fal": "fal.ai"`, `"gemini": "Gemini (AI Studio)"`, `"fireworks": "Fireworks"`; `FIELD_SCHEMA` rows per backend: `base_url`[url], `default_model`[text], `timeout_seconds`[int,min 1], `api_key`[secret]; probe dispatch for the three), `Tests/Image_Generation/test_live_backends.py` (three opt-in live suites)
- Test: `Tests/UI/test_settings_image_gen_defaults.py` (extend), `Tests/UI/test_settings_image_gen_panel.py` (update any nine-backend-count assertions honestly)

**Interfaces:**
- Consumes: everything from Tasks 2–6; the probe module's existing badge set and `_guarded_get` (NO new badge strings).
- Produces: probes — gemini: `GET {base}/models` with `x-goog-api-key` → 2xx `Reachable` / 401,403 `Auth failed` / other `Unreachable: HTTP <n>`; fireworks: VERIFY via WebFetch whether `https://api.fireworks.ai/inference/v1/models` (OpenAI-compat, Bearer) exists as a cheap authed list — if yes, full auth probe; if unverifiable, reachability-only → `Reachable (auth unverified)` (record which in the report); fal: reachability-only on the configured base → `Reachable (auth unverified)`.

- [ ] **Step 1: Failing tests**: drift test passes UNMODIFIED once schema rows land (it round-trips secret+non-secret keys through the real loader — Task 2 made the loader ready; if it fails, the schema/loader keys drifted — fix the ROWS, never the test); `build_backend_rows` returns nine with the new labels/key_sources; probe branches per backend (fake `_guarded_get`: gemini 200→Reachable, 401→Auth failed; fal → auth-unverified; fireworks per verified route); panel tests asserting six updated to nine (each updated assertion listed in the report).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** rows + probe branches + live tests (`TLDW_LIVE_FAL_API_KEY`+`TLDW_LIVE_FAL_MODEL?`, `TLDW_LIVE_GEMINI_API_KEY`, `TLDW_LIVE_FIREWORKS_API_KEY` — mirror the existing per-backend gating in `test_live_backends.py`, markers integration/optional/slow; each live test: tiny prompt, asserts non-empty image bytes; gemini additionally a reference-image edit case gated on the same key).
- [ ] **Step 4:** `Tests/UI/test_settings_image_gen_defaults.py` + `test_settings_image_gen_panel.py` + `Tests/Image_Generation/ -q` green; `python -c "import tldw_chatbook.app"`.
- [ ] **Step 5: Commit** `feat(settings): fal/gemini/fireworks rows, probes, opt-in live tests`.

---

## Post-plan (controller)

Final whole-branch review (most capable model; secret-lifecycle + spec audit + the fal self-built-URL property end-to-end), then live UAT with the user's three keys (Settings probe + Console `/generate-image :backend` per backend + one nano-banana reference-image edit), then PR per the finishing flow. Backlog task filing for the program + follow-ups at PR time, IDs swept across all refs at assignment.
