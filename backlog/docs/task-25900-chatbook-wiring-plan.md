# TASK-25900 — chatbook wiring half (design scaffolding)

**Status:** scaffolding only. No code touches the `mcp_unified.federation` import
surface yet — this is the dep-independent prep so the build is fast once
mcp-unified 0.3.0 is pinnable.

**Dependency gate.** The server half shipped in tldw_server PR #2861 (Streamable
HTTP + SSE `ExternalFederationTransport` impls, `ExternalServerDefinition`
gains `streamable_http`/`sse` + a static `headers` field, `create_external_transport`
dispatch, version bumped to 0.3.0). mcp-unified 0.3.0 publishes to PyPI
automatically on the next tldw_server **main** release + version bump (the bump
is already on tldw_server dev). Until 0.3.0 is on PyPI, chatbook's pin
(`pyproject.toml`: `mcp-unified==0.2.1`, two occurrences — lines ~148 and ~435)
cannot move: a bump to `==0.3.0` fails CI install. **Do not open the wiring PR
against a pin that will not resolve.**

The transports live in `mcp_unified.federation` (`create_external_transport`
returns Streamable HTTP / SSE / stdio impls of the `ExternalFederationTransport`
protocol). chatbook today imports only `mcp_unified.gateway` (`serve_stdio`,
gateway_runtime) — the federation import is net-new for this task.

## AC map

- AC#1/#2: add + connect + invoke a Streamable-HTTP / SSE remote server from Console.
- AC#3: URL-based record persists distinctly from a command-based one, round-trips.
- AC#4: remote servers pass the same permission gate, definition-hash rug-pull
  guard, and execution log as stdio — verified by tests.
- AC#5/#6 (readiness): connect / TLS / auth failures each get a distinct, honest
  readiness state (not a generic error).
- AC (implicit): stdio path untouched and extra-free; remote requires the `mcp` extra.

## 1. URL server record schema (`MCP/local_store.py`)

Today `LocalExternalMCPProfile` (frozen dataclass, keyed by `profile_id`) models a
**command-based** server only: `command`, `args`, `env_placeholders` /
`env_literals` / `legacy_env_literals`, timestamps. It has no `transport`,
`url`, or `headers`. `save_profile` persists it; `to_storage_dict` /
`from_storage_dict` are the round-trip seam.

Sketch (keeps stdio byte-identical; additive, defaulted):

- Add `transport: str = "stdio"` (values: `stdio` | `streamable_http` | `sse`).
- Add `url: str = ""` (required + validated when transport in the URL set;
  must be empty for stdio — mirror the server-side model_validator).
- Add `headers_placeholders: dict[str,str]` and `headers_literals: dict[str,str]`,
  reusing the EXISTING env secret-guards (`_is_secret_bearing_env_key`,
  `_looks_like_raw_secret_value`, placeholder-vs-literal split). A bearer token
  is secret-bearing → must be a `$NAME` placeholder, never a stored literal.
  This is the single most important reuse: it keeps AC#3's "never persist a raw
  secret" guarantee that stdio env already enforces.
- `__post_init__`: when `transport == "stdio"`, forbid `url`/headers and keep the
  current command/env validation; when URL-based, forbid `command`/`args`/env,
  require `url`, run the header guards. Cross-field, same shape as the server's
  `ExternalServerDefinition._validate_transport_fields`.
- `to_storage_dict`/`from_storage_dict`/`to_input_dict`/`from_input_dict`: add the
  new keys with stdio-safe defaults so OLD records (no `transport`) load as
  `stdio` unchanged (AC: stdio round-trips byte-identically).

Migration: `LocalMCPStore` persists JSON, not a SQL schema, so no DB migration —
but confirm the reader tolerates a missing `transport` key (defaults to stdio)
and that `save_profile` of a legacy record does not spuriously add URL keys.

**Decision to lock at build time:** extend `LocalExternalMCPProfile` in place
(one type, transport-tagged) vs. a sibling `LocalRemoteMCPProfile`. Recommend
in-place + transport tag — the hub, permission store, and execution log all key
on `profile_id`, and a second type would fork every one of those call sites.

## 2. Connect-path wiring (`MCP/client.py`, `MCP/local_control_service.py`)

Today `client.py` connects stdio-only (`asyncio.create_subprocess_exec`), and
`local_control_service._get_client()` is the single client-creation site (it
already wires `_server_request_dispatcher_factory` per server_id — the sampling/
elicitation seam from 27019). The federation manager
(`mcp_unified.federation`'s `ExternalFederationManager`, transport_factory =
`create_external_transport`) is the drop-in for URL servers.

Plan:
- At `_get_client()`, branch on the record's `transport`: stdio → today's path
  unchanged; URL → build an `ExternalServerDefinition` from the stored record and
  hand it to the federation manager / transport. Resolve `$NAME` header
  placeholders from the environment at connect time (never persist resolved
  values), exactly as stdio env placeholders are resolved.
- Keep the `_server_request_dispatcher_factory` wiring for URL servers too, so
  sampling/elicitation policy (27019) applies uniformly.
- stdio import path must not pull `mcp_unified.federation` — gate the federation
  import behind the URL branch (and the `mcp` extra), so the stdio path stays
  extra-free.

## 3. Gate / hash / log reuse (AC#4)

These are keyed on `profile_id` / tool identity, NOT on transport, so remote
servers ride them unchanged **as long as** the URL record flows through the same
add → discover → permission-store → execute path:
- Permission gate: `permission_store.py` (`allow|ask|deny` + `definition_hash`).
- Rug-pull: `hub_test_execution.py` recomputes `definition_hash` over the tool
  set; a remote server's discovered tools must feed the SAME hash input as stdio
  (name + input_schema), so a changed remote definition trips the same guard.
- Execution log: the hub's audit path (`OPEN_AUDIT`) — remote invocations must
  emit the same audit rows.
AC#4's tests: parametrize an existing stdio gate/hash/log test over a URL record
(fed by a fake `ExternalFederationTransport`, mirroring the server-side
`FakeExternalTransport`) and assert byte-identical gate/hash/audit behavior.

## 4. Readiness mapping (AC#5/#6)

The transports raise reason-coded, secret-free errors:
`auth_required`, `tls_failed`, `connect_failed`, `request_timeout`,
`connection_closed`, `insecure_url`, `invalid_endpoint`, `upstream_http_error`.

`MCP/readiness.py` today has `ReasonCode` {NOT_CONFIGURED, AUTH_MISSING,
RUNTIME_UNAVAILABLE, PREFLIGHT_FAILED, UNREACHABLE, DISCOVERY_FAILED,
CONFIG_CHANGED, DISCOVERY_NOT_RUN, NO_TOOLS_RETURNED, CATALOG_EXPIRED,
PARTIAL_CAPABILITY} → `ReadinessState` via `REASON_TO_STATE`, ordered by
`REASON_PRIORITY`, with `REASON_TO_ACTIONS` driving the hub's fix buttons.

AC#6 wants connect / TLS / auth to read DISTINCTLY, not collapse to one
"unreachable". Proposed additive mapping (new `ReasonCode`s → existing states +
actions; no new `ReadinessState` needed except possibly an auth one):

| transport reason_code | new ReasonCode        | ReadinessState   | primary action        |
|-----------------------|-----------------------|------------------|-----------------------|
| auth_required         | REMOTE_AUTH_REQUIRED  | NEEDS_SETUP      | OPEN_CREDENTIALS      |
| tls_failed            | REMOTE_TLS_FAILED     | NEEDS_ATTENTION  | EDIT_CONFIG / VIEW    |
| connect_failed        | REMOTE_CONNECT_FAILED | NEEDS_ATTENTION  | CONNECT (retry)       |
| request_timeout       | REMOTE_TIMEOUT        | NEEDS_ATTENTION  | CONNECT (retry)       |
| connection_closed     | REMOTE_CONNECT_FAILED | NEEDS_ATTENTION  | CONNECT (retry)       |
| insecure_url          | REMOTE_INSECURE_URL   | NEEDS_SETUP      | EDIT_CONFIG           |
| invalid_endpoint      | REMOTE_INVALID_ENDPOINT | NEEDS_ATTENTION | EDIT_CONFIG          |
| upstream_http_error   | UNREACHABLE (reuse)   | NEEDS_ATTENTION  | VIEW_DETAILS          |

- Distinct primary MESSAGE per code is the point (AC#6 "honest"); several can
  share a `ReadinessState` bucket + action as long as the message differs.
- Add the new codes to `REASON_PRIORITY` ABOVE generic UNREACHABLE so a specific
  cause wins the display.
- `auth_required` maps to the credentials action, not a generic error — that is
  the specific AC#6 example.

Open question for build time: whether `insecure_url` (a config mistake caught
before any network call) deserves NEEDS_SETUP vs NEEDS_ATTENTION — leaning
NEEDS_SETUP because the fix is editing the URL/headers, not retrying.

## 5. Test plan (build time)

- URL record round-trips through `local_store` distinctly from stdio (AC#3);
  legacy stdio record without `transport` still loads as stdio unchanged.
- Header secret placeholders enforced (bearer literal rejected; `$NAME` accepted).
- Gate/hash/log parametrized over a URL record via a fake federation transport
  (AC#4) — byte-identical to stdio.
- Each transport reason_code → its distinct readiness state + message (AC#6),
  table-driven.
- stdio connect path unchanged; `mcp_unified.federation` NOT imported on the
  stdio path (import-graph assertion, mirrors the ADR-097 boot ratchet style).

## 6. Sequencing once 0.3.0 is on PyPI

1. Bump both `mcp-unified==0.2.1` → `==0.3.0` pins.
2. Land §1 (schema) + tests → §2 (connect wiring) → §3 (gate/hash/log tests) →
   §4 (readiness) — each behind the URL branch so stdio stays extra-free.
3. Live-verify one Streamable-HTTP and one SSE remote from the Console (AC#1/#2).
