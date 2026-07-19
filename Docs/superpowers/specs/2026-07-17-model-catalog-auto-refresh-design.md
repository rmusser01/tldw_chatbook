# Model Catalog Auto-Refresh — Design

Date: 2026-07-17
Status: Approved design (pre-planning)
Amends: ADR-002 (`backlog/decisions/002-openai-compatible-model-discovery.md`)

## Problem

Model lists for LLM providers live in `[providers]` in `config.toml` and only change when a
user (or a release) edits them by hand. New upstream models never appear in the selectors
unless the user runs the manual "Discover models" flow in Settings and explicitly saves the
results. Users want fresh model lists for cloud providers without manual intervention.

Providers in scope: **OpenRouter, Moonshot, Z.AI, OpenAI, Anthropic, Mistral** (MistralAI in
config keys).

## Existing Foundation (ADR-002 / task-78)

The `tldw_chatbook/LLM_Provider_Catalog/` package already implements a manual discovery
pipeline: fetch → in-memory cache → merge into selectors → explicit user save to config.
Key pieces this design extends rather than replaces:

- `openai_compatible_model_discovery.py` — async `GET <endpoint>/v1/models` client,
  Bearer auth, 10s timeout, `validate_url` boundary, metadata scrubbing, typed errors.
- `local_llm_provider_catalog_service.py` — endpoint/key resolution, `ModelDiscoveryCache`
  (in-memory, no TTL, no disk), `merge_saved_and_discovered_models()`,
  `persist_discovered_models_to_settings()` (explicit-save path).
- `UI/Screens/provider_model_resolution.py` — merged saved+discovered options for selectors,
  discovered entries labeled "runtime discovered".
- Settings → Providers & Models → "Model discovery" subsection (`settings_screen.py`).

Current gaps blocking the six providers: `_DEFAULT_OPENAI_COMPATIBLE_ENDPOINTS` only covers
openai/openrouter; the endpoint path policy rejects OpenRouter's `/api/v1` and Z.AI's
`/api/paas/v4`; Anthropic needs `x-api-key` + `anthropic-version` headers instead of Bearer;
Z.AI has no `[providers]` entry or `[api_settings.zai]` default at all.

## Decisions (from brainstorming)

- **Hybrid update mode**: fetched models auto-merge into selectors via a durable cache
  (default), subject to `SELECTOR_MERGE_CAP` — oversized catalogs stay saved-list-only in
  the dropdown and are reachable via a new search picker. Per-provider opt-in additionally
  appends new models into `[providers]` in `config.toml` (write-through).
- **Trigger**: on app startup, in a non-blocking background worker, per-provider staleness
  check (default 24h). No periodic timer, no refresh-on-select.
- **Control**: global toggle (default ON) + per-provider opt-out.

## Architecture

All new logic lives in `tldw_chatbook/LLM_Provider_Catalog/`, with small touch-points in
`app.py`, `config.py`, settings UI, and chat selectors.

### 1. Provider coverage

- `local_llm_provider_catalog_service.py`: extend
  `_DEFAULT_OPENAI_COMPATIBLE_ENDPOINTS` with:
  - `anthropic` → `https://api.anthropic.com/v1`
  - `mistralai` → `https://api.mistral.ai/v1`
  - `moonshot` → `https://api.moonshot.ai/v1`
  - `zai` → `https://api.z.ai/api/paas/v4`
- `openai_compatible_model_discovery.py`:
  - **Auth profiles**: per-provider header builder. Bearer remains the default; Anthropic
    sends `x-api-key: <key>` and `anthropic-version: 2023-06-01`. Selected by provider key,
    injectable for tests.
  - **Path policy**: two code sites change — add `/api/v1` and `/api/paas/v4` to
    `_EXPLICIT_OPENAI_COMPATIBLE_ENDPOINT_PATHS`, and teach
    `_models_path_for_endpoint_path()` to map them to their models routes
    (today it returns `None` for both, so allowlisting alone is insufficient).
  - Response parsing is OpenAI-shaped (`{"data": [{"id": …}]}`) for all six, with one
    exception: **Anthropic paginates** (`limit` default 20, range 1–1000,
    `has_more`/`first_id`/`last_id` fields; cursor params are `after_id`/`before_id` —
    verified against the official anthropic-sdk-python source, *not* OpenAI's
    `starting_after`). Anthropic requests pass `?limit=1000`; if `has_more` is true,
    follow with `after_id=<last_id>` until the catalog is complete, capped at 10 pages as
    a safety bound against a misbehaving endpoint. Other providers are single-GET.
  - All six models routes were probe-verified live (2026-07-17): OpenRouter returns 200
    **without auth** (public catalog); the other five return 401 pre-auth, confirming the
    routes exist. Z.AI's authenticated response shape is assumed OpenAI-shaped — one
    live smoke check with a real key during implementation confirms it.
- `config.py` (`CONFIG_TOML_CONTENT`): add `ZAI = [...]` to `[providers]` and an
  `[api_settings.zai]` section (`api_key_env_var = "ZAI_API_KEY"`,
  `api_base_url = "https://api.z.ai/api/paas/v4"`), matching the pattern of the other
  providers, so Z.AI has an anchor for key/list resolution. No config migration is needed
  for existing users: `load_settings()` deep-merges the user's file over the parsed
  defaults (`config.py:2753`, `config.py:697-699`), so new default sections appear in
  memory for old configs automatically; the first write-through save materializes the key
  into the user's file, which is the normal settings-save behavior.

### 2. Durable TTL cache

New `model_discovery_disk_cache.py` in `LLM_Provider_Catalog/`:

- JSON file at `get_user_data_dir() / "model_catalog_cache.json"` (follows the
  audio-history JSON precedent).
- Shape: `{ "<provider_key>|<endpoint_fingerprint>": {"models": [...], "fetched_at": "<iso8601>"} }`.
  `fetched_at` is timezone-aware UTC; staleness comparisons never mix naive and aware
  datetimes.
- Loaded into the existing `ModelDiscoveryCache` at startup; written after every
  successful fetch (atomic write: temp file + rename).
- Missing/corrupt file → treated as empty cache, never a crash.
- On each successful refresh, prune entries for providers/endpoints no longer configured,
  so orphaned `provider|old_fingerprint` keys don't accumulate.
- Staleness: `fetched_at` older than `stale_after_hours` (default 24). A value of `0`
  means always-stale — refetch on every launch.

### 3. Startup refresher

A function on the local catalog service, e.g.
`refresh_stale_configured_providers() -> RefreshReport`, invoked from a `@work(exclusive=True)`
worker in `app.py` immediately after `_init_providers_models()`:

1. Load `[model_catalog]` settings. If `auto_refresh_enabled` is false → no-op.
2. Load the disk cache into `ModelDiscoveryCache` **synchronously during startup** (before
   or with `_init_providers_models()`, not inside the background worker), so selectors can
   already show last-known discovered models, even offline.
3. For each of the six providers (defined as a constant list of provider keys), skip when:
   - listed in `auto_refresh_disabled`, or
   - not ready (API key resolution via the existing readiness path fails) — **except
     OpenRouter, whose catalog is public**: it refreshes without credentials (no auth
     header) when no key is configured, or
   - cache fresh (`fetched_at` within `stale_after_hours`).
   (`auto_refresh_disabled`/`write_to_config` hold exact-cased `[providers]` list keys like
   `"ZAI"`/`"MistralAI"`; the refresher maps them to normalized keys via the existing
   `resolve_provider_list_key` before endpoint/key resolution.)
4. Otherwise run the existing discovery client (existing timeout, URL validation,
   fingerprinting, metadata scrubbing).
5. On success: compute the new-ID set (in the fresh fetch, but not in the *previous cache
   entry* — the entry under the same `provider|endpoint_fingerprint` key prior to this
   fetch — and not already saved) **before** overwriting the cache, then update in-memory +
   disk cache. (The previous entry is also empty after an endpoint-fingerprint change or
   cache corruption/deletion; the baseline guard below keeps that safe.) If the provider is
   in `write_to_config`, append the new IDs into `[providers]` for the exact-cased key —
   **append-only, never removes user entries** — via one `save_settings_to_cli_config()`
   call per provider, **with one guard**: if there was no previous cache entry and the
   catalog exceeds `SELECTOR_MERGE_CAP` (50), treat the fetch as a baseline and append
   nothing — this prevents dumping OpenRouter's 300+ model catalog into `config.toml` on
   first run. Small catalogs (≤ cap) backfill in full on first run, since their entire list
   is "new". Re-read current settings immediately before each save (the file may have
   changed since startup; atomic rewrite is not a merge), and after any write-through,
   re-run `_init_providers_models()` so `app.providers_models` doesn't serve a stale
   startup snapshot for the rest of the session.
6. Emit one consolidated `app.notify()` at the end (via `call_from_thread`, since the
   worker runs off the UI thread): updated providers with new-model counts (the diff is
   reported even when the baseline guard suppressed the config write, phrased as
   "cached", not "saved"), or failures with "using cached list". If the active chat
   selector's provider gained models, refresh its options; otherwise no UI churn.

### 4. Settings UI

In Settings → Providers & Models → existing "Model discovery" subsection
(`settings_screen.py`):

- "Auto-refresh model lists on startup" checkbox (master switch).
- "Refresh after (hours)" numeric input (default 24).
- Per-provider rows for the six providers: `auto-refresh` checkbox (default on),
  `save to config` checkbox (default off; hint: "appends new models to config.toml —
  large catalogs like OpenRouter only add newly released models after a first baseline").
- Persist via the existing settings save path.

### 5. Selector consumption

- **Merge cap.** A constant `SELECTOR_MERGE_CAP = 50` governs whether discovered-but-
  unsaved models are merged into the chat model dropdown. The cap is measured on the
  provider's **total cached discovered catalog**, regardless of saved overlap:
  - Total discovered catalog ≤ cap → merge (saved-first, "runtime discovered"
    labels), via `resolve_provider_model_options()`.
  - Total discovered catalog > cap (e.g. OpenRouter's ~300+) → dropdown shows the saved
    `[providers]` list only; the full fetched catalog stays in cache and is reachable via
    the search picker (below) and the existing Settings discovery UI.
  The rule is evaluated at options-build time, so a catalog crossing the cap between
  refreshes deterministically adds/removes merged entries on the next rebuild.
- Point `UI/Screens/chat_screen.py` provider-change handler (currently re-reads only
  `get_cli_providers_and_models()`) at `resolve_provider_model_options()` so merged
  saved+discovered models appear there under the cap rule.
- **Selection preservation.** Textual's `Select.set_options()` resets the selection, so
  every options rebuild (startup refresh, provider change, picker insertion) must
  re-apply the active model ID afterward — re-adding a transient picker option first if
  the active ID isn't in the rebuilt options.
- **Searchable model picker** (new widget, e.g. `Widgets/model_search_picker.py`, placed
  next to the model Select in the chat sidebar): a filter `Input` + `OptionList`. Typing
  substring-filters the *full* catalog for the current provider (saved + all cached
  discovered, no cap) by model ID — OpenRouter IDs embed the upstream provider prefix
  (`anthropic/claude-3.7-sonnet`), so provider search works naturally. On selection, the
  chosen ID is **inserted into the model `Select` as a transient option if absent, then
  selected** (Textual's `Select` rejects values outside its options, so setting the value
  directly would raise). This deliberately puts one user-chosen entry into the dropdown
  even for over-cap providers. The picked ID persists as the provider's active model
  exactly like a dropdown selection; on restore (next launch), if it is absent from the
  rebuilt options it is re-inserted as a transient option rather than falling back to the
  first available model.
- Other sidebars keep reading config lists (only the primary chat selector is in scope).

### Config schema

```toml
[model_catalog]
auto_refresh_enabled = true
stale_after_hours = 24                 # 0 = always stale (refetch every launch)
auto_refresh_disabled = ["ZAI"]        # per-provider opt-out (provider list keys)
write_to_config = ["OpenRouter"]       # per-provider write-through opt-in
```

## Data Flow

```
startup → load [model_catalog] settings → load disk cache → selectors show last-known models
       → worker: per provider (enabled? opted-out? ready? stale?) → GET <endpoint>/models
       → success → memory cache + disk cache (+ append-new to config if write-through)
       → done → one consolidated notification; refresh active selector if affected
```

## Error Handling

Every failure degrades to "keep using cached/saved models"; startup is never blocked.

- Offline / timeout / 5xx → skip provider, keep stale cache, log with context, include in
  the end-of-refresh notification.
- 401/403 → treat as not-ready; skip quietly (log line only — no per-launch key nagging).
- Corrupt/missing disk cache → empty cache; next trigger refetches.
- Write-through failure → cache still updated, config untouched, error logged + notified.
- Upstream removals are never propagated: write-through is append-only; users prune their
  own `[providers]` lists.
- All network calls keep going through `validate_url`, endpoint fingerprinting, and
  metadata scrubbing. API keys are resolved via the existing readiness path and never
  logged.

## Security

- Reuses the existing credential resolution (config key → configured env var → default
  env var) and the discovery client's credential-metadata scrubbing.
- The disk cache stores model IDs and timestamps only — never keys or request headers.
- New default endpoints are validated with `validate_url` before any request, as today.

## Testing

Mirror `Tests/LLM_Provider_Catalog/`; mocked `httpx`, no real network. (Plus a one-time
manual smoke check during implementation: one authenticated GET against each of the six
routes — all six were probe-verified to exist on 2026-07-17 — confirming Z.AI's response
shape and Anthropic's pagination against live APIs.)

- Auth profiles: Anthropic headers vs Bearer default; wrong-provider profile never leaks.
- Path policy: `/api/v1` and `/api/paas/v4` resolve to correct models URLs; other paths
  still rejected.
- Disk cache: round-trip, corrupt-file recovery, TTL staleness boundaries
  (fresh at 23h59m, stale at 24h01m with default).
- Refresher: skips for disabled / opted-out / not-ready / fresh; fetches stale; aggregates
  report; consolidated notification content.
- Write-through: appends only new IDs, never removes existing entries, preserves exact key
  casing, single save call, no-op when nothing new, failure leaves config untouched;
  first-fetch baseline guard (empty previous cache + catalog > 50 → append nothing;
  ≤ 50 → full backfill); diff-append on subsequent fetches.
- Merge cap: provider with ≤ 50 discovered models merges into selector options; > 50 stays
  saved-list-only in the dropdown while the full catalog remains searchable.
- Search picker: substring filter over saved + full discovered catalog (including
  provider-prefix matches like `anthropic/` for OpenRouter); selecting a result inserts a
  transient Select option and sets it as the active model; picked ID persists across
  restarts via re-insertion on restore; empty/no-match states.
- Selection preservation: options rebuilds (refresh, provider change, picker insertion)
  re-apply the active model ID and never silently reset it.
- Selectors: `resolve_provider_model_options` surfaces refreshed models with
  "runtime discovered" label; chat provider-change handler shows merged list under the cap.
- Settings UI: toggles render, persist, and gate the refresher.

## Governance

- **ADR required: yes** — this amends ADR-002's "explicit user persistence only" decision.
  Create `backlog/decisions/020-automatic-model-catalog-refresh.md` (cache-first default,
  opt-in write-through, append-only semantics) before implementation; link from the task.
- **Backlog task**: create one task for the implementation with the acceptance criteria below.

## Acceptance Criteria

- [ ] On startup (global toggle on, provider ready, cache stale), each of the six providers'
      model lists is fetched in the background without blocking app start.
- [ ] Fetched models from small catalogs (≤ 50) appear in the chat model selector (labeled
      "runtime discovered") without any manual step, and survive restarts via the disk cache.
- [ ] Large catalogs (> 50, e.g. OpenRouter) never flood the dropdown; their full fetched
      list is reachable via the type-to-filter search picker and the Settings discovery UI.
- [ ] The search picker filters by substring over the full catalog and sets the active
      model on selection (via a transient Select option), including models not present in
      the dropdown; picked models survive restarts.
- [ ] Providers with `save to config` enabled get new model IDs appended to `[providers]`
      in config.toml; existing entries are never removed or reordered; an oversized
      first-ever fetch establishes a baseline without appending.
- [ ] Global toggle off, per-provider opt-out, missing API key, or fresh cache each
      reliably skip fetching.
- [ ] Offline/failure at refresh time leaves the app fully functional on cached/saved
      models, with at most one consolidated notification.
- [ ] Anthropic fetches use `x-api-key`/`anthropic-version`; all other providers use Bearer.
- [ ] Z.AI ships with `[providers]` and `[api_settings.zai]` defaults.
- [ ] New ADR linked; backlog task completed per repo DoD; tests for the above pass.

## Non-Goals

- Periodic refresh while the app runs; refresh-on-provider-select.
- Fetching pricing, context-window, or capability metadata (existing capability pattern
  system stays as-is).
- Auto-removal of upstream-deprecated models from config.
- Server-scope (tldw_server) auto-refresh — local scope only, as with ADR-002.
- Merging discovered models into every sidebar; only the primary chat selector.
