# ADR-079: Network TLS Trust Policy (corporate DPI networks)

**Status:** Accepted
**Date:** 2026-08-29
**Spec:** Docs/superpowers/specs/2026-08-29-network-tls-trust-policy-design.md

## Context

Corporate TLS-inspection/DPI networks re-sign outbound HTTPS with a corporate
root CA that lives in the OS trust store. None of this app's transports
(requests, httpx, aiohttp, websockets) consult the OS store — they verify
against certifi/OpenSSL default paths only — so every HTTPS call fails with
`SSLCertVerificationError` in intercepted networks, and no setting exists to
express trust. ~50 inline client constructions across four transports plus one
OpenAI-SDK site; no shared seam today.

## Decision

One global config knob, `[network] ssl_verify = true | false | "/path/ca.pem"`:

- `true` (default): verify against the default bundle.
- `false`: verification disabled — insecure escape hatch, warned loudly.
- path: **additive** trust — the custom CA is trusted *in addition to* certifi,
  never as a replacement (selective interception is the common corp topology).

Implementation: shared helper `tldw_chatbook/Utils/tls_trust.py` (normalization,
additive SSL contexts, merged-PEM cache for requests, client factories,
warn-once + metrics). Shared seams (Console gateway, tldw_api client, image
gen, TTS, model catalog, evals) adopt factories; the long tail threads
`session.verify` / `ssl=` / injected `http_client=`. `tldw_api/client.py` stays
standalone (Apache-2.0): it gains an `ssl_verify` constructor param and the app
passes the resolved policy in. F9 Settings gains a Network category. The
transport-level spelling of "disabled" is an unverified `ssl.SSLContext`
(`check_hostname=False`, `verify_mode=CERT_NONE`), not bare `False`, because
websockets ≥14 rejects bare `False` for wss:// connections (amended during the
Task 8 review, verified against websockets 16; aiohttp treats the context
identically to `ssl=False`).

**Fail-safe direction is always verification-on**: invalid value, missing/
unreadable file, corrupt PEM, or bundle-write failure → default verification,
with an error log stating the remedy.

## Considered and rejected

- **Global startup injection** (env vars / `ssl.SSLContext` monkeypatching):
  requests respects `REQUESTS_CA_BUNDLE`, httpx uses certifi regardless, aiohttp
  and websockets ignore both; no stack supports disabling verification via env.
  Monkeypatching silently alters TLS for out-of-scope libraries.
- **Replace-semantics custom CA**: breaks every non-intercepted public endpoint
  under selective interception. (A team wanting "corp CA only" can export a
  bundle containing only the corp root — same knob.)
- **Per-provider granularity / insecure-host allowlist**: interception is
  network-wide; per-host verification is unsupported natively by any of the
  four stacks (would need mounted adapters / event hooks in each).
- **`truststore` (OS trust store) now**: new dependency; deferred as an
  additive follow-up mode if users still struggle to obtain the corp CA.
- **Unifying Subscriptions feeds into the global knob**: the existing per-feed
  `ssl_verify` flag keeps working; revisit later.

## Consequences

- With `false`, API keys and conversation content are interceptable by anyone
  on the network path — surfaced as a settings warning and a once-per-process
  log + metric.
- Policy binds at client construction; loop-cached clients (Console gateway)
  pick up changes after restart. Per-call sessions pick changes up immediately.
- New outbound-HTTP code should use `build_httpx_async_client` /
  `build_httpx_client` / `build_requests_session` so the policy applies by
  construction.
