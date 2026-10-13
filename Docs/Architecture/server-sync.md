# Server sync and the tldw API client

This document describes the client side of the companion tldw server: the typed REST client and its safety semantics, the Sync v2 outbox machinery, the `sync_log` change feed (and why it is *not* the outbox), runtime identity and local↔server flips, and the repeatable LOCAL|SERVER scope-service pattern.

## The API client

`tldw_api/client.py::TLDWAPIClient` is a typed httpx client (roughly a thousand generated-feeling async methods over ~54 schema modules). The package `__init__` is a lazy facade, but importing the client itself still pulls the whole schema surface (~450 ms) — call sites defer the import for exactly this reason.

Safety semantics (verified):

- **Auth**: `token` → `X-API-KEY` (default mode); optional bearer mode. Because httpx does not strip `X-API-KEY` on cross-host redirects, redirects are refused outright (301/302/303/307/308; 304 deliberately excluded) — credentials are never forwarded.
- **Timeouts**: 300 s overall default (long reads are legitimate — uploads/transcription), with a 15 s connect ceiling because screens fetch during mount and a blackholed connect would freeze the app. NaN/negative rejected at construction.
- **Endpoint safety**: every request path passes an unsafe-endpoint check (query/fragment delimiters, `..` traversal).
- **Errors**: 401 → `AuthenticationError`; 422 → `APIRequestError` with field-joined detail; other non-2xx → `APIResponseError` (preferring structured `{detail: {code, message, retryable}}`); transport → `APIConnectionError`; 204/205 → `{}`.
- **No client-internal retry** — retry/idempotency belongs to callers (e.g., sync push idempotency keys).

Config: `[tldw_api] base_url` / `auth_token` in config.toml (shipped localhost placeholders); `build_runtime_api_client` accepts endpoint/token aliases and an auth mode; TLS trust resolves through the tls-trust helper. An empty endpoint raises.

## Sync v2 (`Sync_Interop/`)

| File | Role |
| --- | --- |
| `server_sync_service.py` | policy-gated server transport (`sync.changes.{launch,observe}.server`), profile bootstrap (requires the *complete* organization domain group), v2 dry-run, personal-context link, recovery bundles; storage-admission leases per execution |
| `local_first_sync_service.py` | `sync_once` push+pull+apply; push chunking to the server-advertised batch size; per-envelope idempotency keys; partial-failure records and conflict reviews instead of aborting |
| `notes_organization_sync_service.py` | projects immutable ChaChaNotes organization intents (folders/keywords/collections) into the outbox |
| `notes_outbox_producer.py` | converts successful local writes into envelopes at mutation time |
| `sync_state_repository.py` | durable SQLite state (`tldw_chatbook_sync_state.db`): the `sync_v2_local_outbox` table, profile/mirror state |
| `manual_sync_control.py` | preview / run-once / organization adoption; reads `sync_log` heads as durable dependency ids |
| `runtime_policy/server_context.py`, `server_credentials.py` | active-server context + credential store (keyring-backed) |

**The active sync does not read `sync_log`**: Sync v2 producers write envelopes into the outbox at mutation time; the local-first service pushes/pulls envelopes and applies them. v1 send/get-changes remain exposed only as transport wrappers.

## `sync_log` is a ledger, not the outbox

SQL triggers in ChaChaNotes, Prompts, and the Media DB append `(entity, entity_id, operation, timestamp, client_id, version, payload)` rows on create/update/delete/undelete. Frontier-pruning triggers keep only the latest version per entity and purge on hard deletes — that trigger layer is the live retention mechanism. Verified consumers: manual-sync durable dependency ids, organization inventory deletions, targeted note purges, import-executor inserts, and the recovery/backup schema (which recreates the trigger set on restored DBs). There is no `Sync_Client` module anywhere — the name survives only in historical docs. Conflating `sync_log` with the Sync v2 outbox is the most likely documentation error in this area.

## Runtime identity and local↔server flips

`runtime_policy/` owns the app-wide mode:

- `RuntimeSourceState` (`active_source` local|server, server id/configured/reachability/auth + timestamps) persisted to a **profile-scoped** `runtime_policy.json` beside the active config (private-path atomic IO; a scratch profile cannot flip the default profile's mode).
- Reachability/auth observations older than a 5-minute freshness window normalize to `unknown`.
- `handle_runtime_backend_changed` is the flip entrypoint: validate → commit (revision-checked, owner-thread asserted) → invalidate the server context (or rebind config) → re-wire notes sync → re-activate collections authority → notify the active screen. A failed flip keeps the previous source active.
- Every screen snapshot (and the Console prompt-target projection) is keyed by `RuntimeIdentity`; incompatible identities **invalidate silently** — nothing is migrated across a flip.

## The scope-service pattern

Every domain repeats one contract (`Evaluations_Interop`, `Meetings_Interop` (server-only), `Subscriptions/watchlist_scope_service`, `Prompt_Management/prompt_scope_service`, `Notes/notes_scope_service`, `Media/media_reading_scope_service`, `Research_Interop`, `Sync_Interop`):

1. A `mode` parameter (local|server) normalizes and selects a service; the unavailable side raises "backend is unavailable".
2. Each call enforces a policy action id `<domain>.<action>.<mode>` against the runtime state (offline/unauthenticated server ⇒ `PolicyDeniedError`).
3. Both sides advertise typed unsupported-capability contracts (reason codes, user messages, affected actions) — e.g., local evaluations cannot deliver webhooks; the server has no session CRUD for research runs; **automatic local/server mirroring is intentionally not enabled** (the sync scope advertises `sync_engine_missing`; only transport + manual/local-first flows exist).
4. Server services hold a client provider and build lazily — server wiring degrades to "unavailable" rather than raising at startup; a failed server health check is never cached as capabilities.

### A server prompt fetch (dataflow)

1. A controller reaches `app.prompt_scope_service` (built with a lazy server service only when a server binding is configured).
2. `mode="server"` selects the server service; policy `prompts.list.server` evaluates the runtime state.
3. The service builds/caches `TLDWAPIClient` from `[tldw_api]`, calls `list_prompts` → `_request("GET", "/api/v1/prompts")` with the endpoint check, redirect refusal, and shared error mapping.
4. Records normalize into the domain shape (backend-tagged); capabilities come from a health projection.

## Failure behaviors

| Case | Behavior |
| --- | --- |
| Server offline / unauthenticated | Policy denial before any request; capabilities degrade, never cached as failures |
| Envelope push partial failure | Per-envelope failure records + conflict reviews; batch continues |
| Profile bootstrap with a partial domain group | Raises — adoption requires the complete group |
| Flip failure | Previous source stays active + warning |
| Redirect response | Refused (credentials never forwarded) |

## Governing decisions

ADR-008 (`008-sync-v2-client-m1-contract-alignment.md`), ADR-059 (notes folder import/sync ownership), ADR-073 (`073-notes-sync-round-trip-and-interoperability-constraints.md`), ADR-102 (personal context authority/sync/encryption), ADR-005 (server readiness / local RAG mirroring). Specs: `2026-05-26-chatbook-sync-v2-completion-roadmap-design.md`, `2026-06-12-sync-v2-client-m1-conformance-design.md`, `2026-07-31-application-sync-runtime-provider-design.md` (under `Docs/superpowers/specs/`). `Docs/Development/Sync-v2-client.md` is historical. (A sync-log-retention ADR existed historically but is not present on this branch — the trigger layer is the retention mechanism.)

## Verified gotchas

1. The client import is expensive (~450 ms) despite the lazy package facade — defer importing `TLDWAPIClient` at call sites.
2. The 300 s overall timeout is per-operation by design; the connect cap is the freeze guard. Don't "fix" the default.
3. `runtime_policy.json` is profile-scoped, not global.
4. Snapshot invalidation on flips is silent — screens must rebuild state, not expect migration.
5. `ServerResearchService.supports_run_delete = True` contradicts the research scope contract (see [research-workspace.md](./research-workspace.md)).
