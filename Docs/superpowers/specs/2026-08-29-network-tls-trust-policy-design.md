# Network TLS Trust Policy (corporate DPI / TLS-interception networks)

**Date:** 2026-08-29
**Backlog:** task to be filed at plan time (per AGENTS.md, no forward task references here)
**Branch:** `feat/network-tls-trust` (plan confirms; line citations below are as of this date)

## Problem

Corporate networks that run TLS inspection / DPI re-sign outbound HTTPS with a
corporate root CA. That CA lives in the OS trust store (which is why the user's
browser works), but none of this app's HTTP stacks consult the OS store:

- `requests` (17 files) verifies against **certifi** only.
- `httpx` (59 files) builds its default context from **certifi** only.
- `aiohttp` and `websockets` use the stdlib/OpenSSL default paths, which on
  macOS/Windows installs do not include the user's system store.

Result: every HTTPS call in an intercepted network fails with
`SSLCertVerificationError`, and the app offers **no setting anywhere** to
express trust. There is also no single seam — roughly 50 client constructions
are spread inline across four transport stacks (`requests`, `httpx`, `aiohttp`,
`websockets`) plus one OpenAI-SDK construction site.

Precedent: Subscriptions/Watchlists already solve this narrow case per feed
(`ssl_verify` column, honored at `Subscriptions/monitoring_engine.py:1017` and
`:1702`, warned via `Utils/egress.py:775 warn_insecure_ssl`). Nothing exists
for LLM calling or content fetching.

## User decisions (brainstorm record)

| Question | Decision | Rationale |
|---|---|---|
| Coverage scope | **B: LLM calls + main content-fetching surfaces** | Covers what corp users actually hit; MCP/Playwright/HF/git are an incremental follow-up, not a redesign |
| Setting shape | **Ternary ladder** `true \| false \| "/path/to/ca.pem"` | Users who can export the corp CA keep verification on; verify-off remains the escape hatch; no new dependency (`truststore` deferred) |
| Granularity | **One global value** | Corp interception is network-wide; per-provider merge logic and per-host verification (unsupported natively by any stack) buy little |
| Surface | **config.toml + F9 settings screen** | Discoverable for corp users who don't read docs; canonical surface is `UI/Screens/settings_screen.py` |
| Custom-CA semantics | **Additive** (certifi + custom both trusted) | Appliances intercept *selectively*; replace semantics would break every non-intercepted public endpoint |
| Implementation | **Factories at shared seams + explicit threading for the long tail** | Traffic concentrates in a few constructions; the factory is the seam future code uses. Global startup injection (env vars / monkeypatching) rejected — inconsistent across stacks and no supported way to turn verification *off* |

## Design

### 1. Config schema

New section in the default config template (`tldw_chatbook/config.py`, next to
`[web_security]` at line 3729):

```toml
[network]
# TLS trust for outbound HTTP/HTTPS/WebSocket:
#   true                  verify against the default bundle (default)
#   false                 DISABLE certificate verification (insecure; last resort)
#   "/path/to/ca.pem"     ALSO trust this CA bundle (corporate root CA) — additive
# Windows paths: use a literal string ('C:\certs\corp.pem') to avoid backslash escapes.
ssl_verify = true
```

Read via `get_cli_setting("network", "ssl_verify", True)`. **No env-var
override in v1** (candidate follow-up; not part of this design). Normalization
follows the repo's `_config_enabled` leniency:

| Config value | Effective |
|---|---|
| `true` / `"true"` / `"1"` / `"on"` | verify on (default bundle) |
| `false` / `"false"` / `"0"` / `"no"` / `"off"` | verification **off** |
| any other string | CA-bundle path; must be an existing readable file, and must parse as a PEM bundle at context-build time |
| anything else (int, list, …) | **fail safe**: `logger.error` with remedy, effective = verify on |

Fail-safe direction is always **verification on** — the setting can never
silently disable verification.

`[web_security]` is unchanged (SSRF egress policy); TLS trust is a separate
concern with its own section.

### 2. Helper module — `tldw_chatbook/Utils/tls_trust.py` (new)

Deliberately not in `egress.py` (that module owns SSRF policy), but follows its
patterns (config reads, fail-safe, metrics, warn-once). Public API:

- `tls_verify_setting() -> bool | str` — normalized setting with the fail-safe
  above.
- `requests_verify() -> bool | str` — value for `session.verify` / `verify=`
  (requests natively accepts `bool | path`).
- `ssl_context_for_transport() -> None | False | ssl.SSLContext` — for aiohttp
  (`TCPConnector(ssl=…)`) and websockets (`connect(ssl=…)`). `None` = default
  verification, `False` = disabled, else an **additive** context:
  `ssl.create_default_context(cafile=certifi.where())` then
  `load_verify_locations(cafile=custom)`.
  *Amended during implementation (Task 8 review, empirically verified against
  websockets 16):* the disabled mode returns an UNVERIFIED `ssl.SSLContext`
  (`check_hostname=False`, `verify_mode=CERT_NONE`) rather than bare `False` —
  websockets ≥14 raises `ValueError: server_hostname is only meaningful with
  ssl` when handed bare `False` for wss://, while aiohttp treats the context
  identically to `ssl=False`.
- `build_httpx_async_client(**kw)` / `build_httpx_client(**kw)` /
  `build_requests_session(**kw)` — constructors that inject the policy unless
  the caller passed an explicit `verify`. These are the seam all **new** client
  code should use. For httpx the injected `verify` is `bool` or the additive
  `SSLContext` — **never a bare custom-CA path**, which httpx would load as the
  *only* trusted bundle, silently reverting to replace semantics.
- `warn_tls_policy()` — once per process per mode: warning log + metrics
  counter (`network_tls_verify_off`, `network_tls_verify_custom_ca`) when the
  effective policy is off or custom-CA. Copy names the stakes: with
  verification off, API keys and conversation content can be intercepted by
  **anyone on the network path**, not just the corporate proxy.

**Merged-bundle cache (the requests side of additive semantics).** requests
cannot take an `SSLContext`, so when a custom CA is configured the helper
concatenates certifi's PEM + the custom PEM into
`<user_data_dir>/cache/merged-ca-bundle.pem` and `requests_verify()` returns
that path. Write is atomic (tmp + `os.replace`); regeneration is keyed on both
sources' `(mtime_ns, size)` so certifi upgrades and CA edits are picked up.
Any failure writing/parsing → fail safe to default verification (logged).

### 3. Adoption inventory (scope B)

**Factory adoption — shared seams** (one edit each, where traffic concentrates):

| Seam | Site (as of 2026-08-29) |
|---|---|
| Console gateway owned client | `Chat/console_provider_gateway.py:1271 _new_owned_http_client()` → `build_httpx_async_client` |
| Image-gen shared client | `Image_Generation/http_client.py:119` → `build_httpx_client` |
| TTS shared client | `TTS/base_backends.py:120` → `build_httpx_async_client` |
| Model-catalog refresh | `LLM_Provider_Catalog/openai_compatible_model_discovery.py:822` → `build_httpx_async_client` (plan sweeps the package for further provider fetchers) |
| Evals word-bench capture | `Evals/word_bench/capture_client.py:205` → `build_httpx_async_client` |

**tldw_api client — constructor param, not an import.** `tldw_api/client.py`
is Apache-licensed and deliberately standalone; it gains
`ssl_verify: bool | str = True` used when its `httpx.AsyncClient` is built
(`client.py:1145`), and app-side construction sites pass `tls_verify_setting()`.

**Explicit threading — long tail** (one-liners; in the summarization libs set
`session.verify = requests_verify()` right after each `requests.Session()` —
none of the `.post()` calls pass their own `verify`):

- `LLM_Calls/LLM_API_Calls.py` (direct `requests.post` at :475 + session posts)
- `LLM_Calls/hosted_chat.py` (also covers `moonshot.py` / `zai.py`, which route
  through it), `LLM_Calls/qwencloud*.py` (own `requests` usage)
- `LLM_Calls/Summarization_General_Lib.py`, `LLM_Calls/Local_Summarization_Lib.py` (~40 inline sessions)
- `Web_Scraping/WebSearch_APIs.py`, `Tools/web_tool_impls.py` (httpx)
- aiohttp: `Web_Scraping/Article_Scraper/crawler.py`, `Media_Creation/swarmui_client.py` → `ssl=ssl_context_for_transport()`
- websockets: `LLM_Calls/realtime/transport.py:113` → `connect(..., ssl=ssl_context_for_transport())`
- OpenAI SDK (only one construction site in the app): `Local_Ingestion/OCR_Backends.py:826` → `http_client=build_httpx_client(...)`
- `Chat/local_server_discovery.py:499,549` (existing `http_client or …` seam → factory default)

**Out of scope v1** (documented follow-ups, not behavior changes): MCP client,
Playwright browsing, `Model_Artifacts/` HF downloads, Subscriptions feeds (keep
the existing per-feed flag and today's behavior), Notes git push, `Web_Server`'s
own outbound calls, `truststore`/OS-trust-store mode.

**Runtime-change semantics.** The policy binds at client construction:
per-call sessions pick up a settings change immediately; loop-cached clients
(Console gateway per-loop cache) keep the old policy until recreated/restart.
Documented, accepted.

### 4. Settings UI (F9)

New **Network** category in `UI/Screens/settings_screen.py`, following the
category/detail-pane `SettingsRegion` pattern (class at :2284):

- A `Select` with three options: *Verify certificates (default)* /
  *Disable verification* / *Custom CA bundle*.
- A conditional path `Input` shown when Custom is chosen; inline validation
  (existing readable file, or the value is not saved and an error shows).
- Save writes `[network] ssl_verify` through the normal config-write path.
- A **persistent warning label** whenever the effective policy is off or
  custom-CA, naming the API-keys/conversation-content stakes.
- A hand-edited invalid config renders an explicit error row
  ("invalid value — using default verification") rather than silently
  displaying the default state.

### 5. Error handling

- Config read: unknown type / lenient-string handling per the table above;
  always fail safe to verification on.
- Context/bundle build: missing, unreadable, **or unparseable** (corrupt PEM)
  custom CA → `logger.error` with remedy, default verification. Fail-safe wraps
  `ssl_context_for_transport()` and the merged-bundle writer, not just the
  config read.
- Runtime TLS failures surface exactly as today (no new swallowing); the
  feature only changes *what is trusted*, never error reporting.

### 6. Testing

Per `backlog/docs/lessons-testing-evidence.md` — assertions on behavior, not
smoke tests:

- **Unit (`Utils/tls_trust.py`):** the full coercion table (incl. corrupt PEM
  and garbage types → fail-safe verify-on); merged-bundle regeneration keying
  (regenerates on certifi or custom change); factory injection (policy applied;
  caller's explicit `verify=` wins); `ssl_context_for_transport()` shapes
  (`None`/CERT_NONE-context/additive-context; the additive context's `get_ca_certs()` contains
  every certifi cert **plus** the custom bundle's certs — asserted by comparing
  DER sets against a certifi-only context); merged-bundle file contains both
  PEMs concatenated and regenerates when either source's `(mtime_ns, size)`
  changes.
- **Seam tests:** Console gateway and tldw_api client honor the policy
  (monkeypatched config + `MockTransport`); settings region renders the three
  states, rejects a bad path, shows the invalid-config error row.
- **Long tail:** grep-based completeness check during plan execution/review
  (adopted modules contain no bare `requests.Session()` /
  `httpx.AsyncClient(` without policy), plus targeted spot tests
  (`hosted_chat` session attribute; one summarization lib).
- **Optional live verification** (per `lessons-live-verification.md`): manual
  mitmproxy re-signing probe — default config fails closed, custom-CA config
  succeeds, verify-off succeeds with warning logged.

## ADR

`backlog/decisions/079-network-tls-trust-policy.md` — **079** because 078 is
taken and 077 is reserved by the in-flight TASK-19610 renumber (duplicate ADR-076
→ 077). Records: ternary + global-only + additive-CA + fail-safe-on decisions;
rejected alternatives (global startup injection, replace semantics, per-provider
granularity, insecure-host allowlist, `truststore` now); scope-B boundary and
follow-ups. Linked from the backlog task, this spec, and the implementation
plan.

## Files touched (summary)

- New: `tldw_chatbook/Utils/tls_trust.py`, `Tests/Utils/test_tls_trust.py`,
  `backlog/decisions/079-network-tls-trust-policy.md`, settings Network region.
- Modified: `config.py` (default template + docs), the seam/long-tail files in
  §3, `tldw_api/client.py` (constructor param), settings screen + its tests.
