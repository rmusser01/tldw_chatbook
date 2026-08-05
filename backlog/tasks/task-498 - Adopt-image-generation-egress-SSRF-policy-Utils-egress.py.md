---
id: TASK-498
title: Adopt image-generation egress/SSRF policy (Utils/egress.py)
status: Done
assignee: []
created_date: '2026-07-22 11:32'
updated_date: '2026-07-25 06:58'
labels:
  - image-generation
  - security
  - followup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
RESCOPED 2026-07-24: dev now ships its own SSRF policy — `Utils/egress.py` (PR #822: per-hop redirect revalidation, private/link-local/metadata blocking, cross-origin cred-strip, byte caps) — so this task is no longer a port from tldw_server. Instead, ADOPT `Utils/egress.py` inside `tldw_chatbook/Image_Generation/http_client.py`, replacing the Phase-1 light guard (`_validate_egress_or_raise` scheme-check placeholder). The original intent stands: image URLs RETURNED by remote backends (OpenRouter/Novita/ModelStudio) must be validated before fetch, while user-configured local backend base_urls (127.0.0.1 SwarmUI, local sd.cpp) keep working. Also wire ModelStudio's dead `allowlist` local (built but only partially enforced) through the adopted policy, and cover `fetch_json`'s manual redirect loop + `image_format_utils.fetch_image_bytes`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `Image_Generation/http_client.py`'s egress guard delegates to `Utils/egress.py` policy (no parallel/duplicate SSRF logic): API-returned image URLs are blocked for private/link-local/loopback/metadata ranges; user-configured backend `base_url`s (e.g. `http://127.0.0.1:7801`) continue to work.
- [x] #2 Every hop of `fetch_json`'s manual redirect loop and `fetch_image_bytes` re-validates through the adopted policy.
- [x] #3 ModelStudio's host allowlist (aliyuncs + base host) is enforced via the adopted policy; the dead `allowlist` local in `modelstudio_image_adapter._is_allowed_remote_image_url` is wired or removed.
- [x] #4 Unit tests with SSRF payloads (private IPs, metadata IP, scheme abuse, redirect-to-private) pass and local-backend generation does not regress.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rewrite Image_Generation/http_client.py to delegate to Utils/egress.py: _validate_egress_or_raise(url, *, trusted_origins=frozenset()) calls egress.check_url_or_raise (catching EgressBlockedError -> ImageGenerationError); evaluate_url_policy(url, *, allowed_hosts=None, trusted_origins=frozenset()) calls egress.evaluate_url_policy for SSRF then layers the existing host-allowlist check on top; fetch_json gains a trusted_origins param threaded into each redirect-hop's _validate_egress_or_raise call. Public names/module path stay stable for adapter imports.
2. Thread trusted_origins through adapters/image_format_utils.py fetch_image_bytes (same per-hop revalidation pattern).
3. Update each backend adapter to pass trusted_origins = Utils.egress.origin_set(base_url) for every request it builds itself from the configured base_url (session/generate/submit/poll calls), while leaving fetch_image_bytes calls on API-RETURNED image URLs untrusted (default frozenset()) so they get full SSRF enforcement: swarmui_adapter (session+generate+image fetch, image fetch is same-origin-checked already so trust base_url host), novita/openrouter/together adapters (submit/poll trusted, extracted image URL untrusted), modelstudio_image_adapter (sync/async submit+poll trusted; repoint _is_allowed_remote_image_url's dead local evaluate_url_policy allowlist check through the adopted policy, using the SAME trusted_origins={base_host} for both the allowlist gate and the subsequent fetch_image_bytes call so the two checks can't disagree).
4. Update Tests/Image_Generation/test_http_client.py: the "local backend URL passes with no trusted_origins" assertion was pinning the old light guard's permissiveness -- flip it to assert 127.0.0.1 is now BLOCKED by default and only allowed when trusted_origins includes it; add SSRF payload coverage (private 10.x/192.168.x/172.16.x, loopback when untrusted, link-local 169.254.x, metadata 169.254.169.254, scheme abuse, redirect-to-private) plus a configured-trusted-local-base_url-still-works case.
5. Run the Image_Generation suite + ruff + app import check; update task ACs/notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Adopted Utils/egress.py (the app-wide SSRF policy) as the sole egress authority for Image_Generation, removing the Phase-1 scheme-only placeholder.

Trust-distinction mechanism: Utils/egress.py already exposes exactly the primitive this task needed -- a `trusted_origins: frozenset[str]` param threaded through check_url_or_raise/evaluate_url_policy (and the guarded_fetch_* helpers used elsewhere in the app). A hostname in `trusted_origins` may resolve to a private/link-local IP; cloud metadata IPs are blocked regardless of trust. `Utils.egress.origin_set(url)` derives the single-host trusted set from a URL.

http_client.py: `_validate_egress_or_raise`/`evaluate_url_policy`/`fetch_json` now delegate to `egress.check_url_or_raise`/`egress.evaluate_url_policy`, each gaining a `trusted_origins` kwarg (default empty = fully enforced); fetch_json's manual redirect loop re-validates every hop with the same trusted_origins. `image_format_utils.fetch_image_bytes` got the same kwarg, threaded into its own per-hop loop. Public names/signatures kept backward-compatible (new kwargs only) so adapter call sites needed additive changes, not rewrites.

Per adapter, the rule applied: URLs the adapter BUILDS ITSELF from its configured base_url (session/generate/submit/poll requests) pass `trusted_origins=origin_set(url_or_base_url)`; URLs EXTRACTED FROM an API response body (image links) are never trusted, so they get full SSRF enforcement. swarmui_adapter threads trusted_origins explicitly through its session/generate/image-fetch call chain (its image URL is already same-origin-gated against base_url, so trusting that host is safe). novita/openrouter/together adapters trust only their own submit/poll fetch_json calls. modelstudio_image_adapter's `_is_allowed_remote_image_url` (previously wired to a local no-op stub) now calls the adopted `evaluate_url_policy`, and a new `_image_trusted_origins()` computes trusted_origins={base_host} ONCE and threads the identical value into both the allowlist gate and the subsequent fetch_image_bytes call, so the two checks can never disagree (a bug I caught mid-implementation: using different trusted_origins in each place would let the gate approve a private base_host image that the fetch then re-blocks).

Tests: rewrote Tests/Image_Generation/test_http_client.py -- the old "127.0.0.1 passes with no trust" assertion was pinning the light guard's permissiveness; flipped to assert it's blocked by default and allowed only with trusted_origins. Added an SSRF payload matrix (10.x/192.168.x/172.16.x/127.0.0.1 loopback, 169.254.x link-local, 169.254.169.254 metadata -- including "still blocked even if trusted", file:// and gopher:// scheme abuse, redirect-to-private-IP re-validation) plus positive trusted-origin cases. Added adapter-level tests: swarmui trust-wiring for its own base_url calls and image fetch; novita/openrouter/together each get a new test proving an API-returned private-IP image URL is blocked; modelstudio gets a positive test (private base_url host + matching returned image URL succeeds, trusted_origins observed on the fetch) and a negative test (returned URL on a different, non-allowlisted private host is blocked before any fetch is attempted).

Files: tldw_chatbook/Image_Generation/http_client.py, tldw_chatbook/Image_Generation/adapters/{image_format_utils,swarmui_adapter,novita_image_adapter,openrouter_image_adapter,together_image_adapter,modelstudio_image_adapter}.py; Tests/Image_Generation/{test_http_client,test_swarmui_adapter,test_novita_adapter,test_openrouter_adapter,test_together_adapter,test_modelstudio_adapter}.py.

Verification: Tests/Image_Generation full suite 71 passed / 6 skipped (skips are opt-in live-backend tests, unrelated); ruff check clean; `python -c "import tldw_chatbook.app"` clean; Tests/Utils/test_egress.py + Tests/Subscriptions/test_subscription_egress_wiring.py (unmodified, sanity re-run) both green, confirming Utils/egress.py itself was untouched.

Fix round 1 (post-review): `fetch_json`'s manual redirect loop and `fetch_image_bytes`'s redirect loop resent the caller's `headers`/`cookies` unchanged on every hop, including cross-origin ones -- since the SSRF policy allows public hosts, a provider redirecting to an attacker-controlled public host would forward the `Authorization: Bearer <api_key>` header. Fixed by tracking the original request's host and, on any hop whose host differs, stripping `Authorization`/`Cookie`/`Proxy-Authorization` via `Utils.egress._hop_headers` (reused, not reimplemented) and dropping `cookies` outright -- mirrors what `Utils.egress.guarded_fetch_httpx` already does for the app's other egress consumers. `fetch_json` also now only applies `params` on the first hop (the redirected URL already carries the server's own query), matching the equivalent `guarded_fetch_httpx` convention rather than inventing new redirect semantics; method/JSON-body downgrade on 301/302 was considered but left unchanged since egress's own helpers are GET-only and have no precedent for it, and no current adapter needs it. Added red/green tests pinning: cross-origin hop strips Authorization+cookies, same-origin hop (e.g. a local backend's own redirect) still carries them.
<!-- SECTION:NOTES:END -->
