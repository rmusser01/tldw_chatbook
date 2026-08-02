# Console Cost Ticker & Cache-Break Alert — Design

- **Date:** 2026-08-01
- **Status:** Approved design, pending implementation plan
- **Scope:** Console (native ChatScreen) only
- **Program shape:** 3 PRs, foundation-first (Approach A, user-approved)

## Problem

The Console gives users no visibility into what a conversation costs, and no
warning when an action they are about to take (editing history, changing the
system prompt, switching model, or simply idling past the provider's cache
TTL) will invalidate the provider-side prompt cache and make the next send
substantially more expensive.

Today the app has **none** of the required substrate:

- Providers return a `usage` block, but `ConsoleProviderGateway.
  normalize_provider_response` discards it; the Anthropic streaming handler
  never captures it at all (`LLM_API_Calls.py` has a commented-out
  `# usage_anth = ...` placeholder). Nothing is persisted — the `messages`
  table has no usage column.
- There is no real pricing table (only a stale, blended $/1K list used by
  Evals).
- No request ever sets `cache_control` — for Anthropic there is currently no
  cache to break. OpenAI-style providers cache automatically but the app
  never accounts for it.

## Decisions (user-approved during brainstorming)

| Question | Decision |
|---|---|
| Cache-alert grounding | **Add real caching**: `cache_control` on Anthropic console requests + implicit-cache accounting on OpenAI-style providers |
| Cost basis | **Real usage from API responses** (streaming + non-streaming); local estimates only for projections and legacy rows |
| Persistence | **Per-message usage** via schema migration; tokens + provider + model stored, dollars always computed at read time (retroactively correctable) |
| Placement | **8th chip** in the Console control-bar chip row |
| Alert style | **Passive chip alert** (color + tooltip); never interrupts sending |
| Pricing coverage | **All cloud providers seeded** (Anthropic, OpenAI, Google, Mistral, Cohere, Groq), config-overridable, dated; local providers $0 |

## Reference facts (verified against current API docs, 2026-08)

- Anthropic caching: `cache_control: {"type": "ephemeral"}` breakpoints
  (max 4/request); prefix is a **byte-exact** match over render order
  `tools → system → messages`. Writes bill at **1.25×** input (5-min TTL;
  1-h TTL is 2×, not used here), reads at **~0.1×**. The 5-min TTL is
  sliding — it refreshes on every cache read. Minimum cacheable prefix is
  model-dependent (~1024 tokens on most current models; higher on some
  older ones); below it the API silently doesn't cache. Each breakpoint
  looks back ≤20 content blocks for a prior entry. Usage fields:
  `input_tokens` (uncached remainder — **excludes** cached tokens),
  `cache_creation_input_tokens`, `cache_read_input_tokens`.
- Invalidation hierarchy: tool definitions / model change → everything;
  system prompt content → system + messages; `tool_choice` / images /
  thinking toggle → messages only. Sampling params (`temperature`,
  `max_tokens`, `top_p`) are **not** part of the cached prefix.
- OpenAI: automatic prefix caching for ≥1024-token prefixes; no write
  premium; cached input discounted (typically 0.25–0.5× depending on
  family); `usage.prompt_tokens` **includes** `prompt_tokens_details.
  cached_tokens`; eviction is load-dependent (~5–10 min, undocumented).
  Streaming usage requires `stream_options: {"include_usage": true}`.

---

## PR1 — Usage capture + pricing foundation

UI-invisible plumbing. Everything else stands on this.

### Usage capture

- **Provider layer** (`LLM_Calls/LLM_API_Calls.py`):
  - Anthropic streaming: capture usage from `message_start` (input +
    cache buckets) and `message_delta` (output tokens).
  - OpenAI-style streaming: `stream_options: {"include_usage": true}` is
    added **per provider function, opt-in** (`chat_with_openai` first;
    DeepSeek/Groq/OpenRouter/Moonshot/Z.ai only after verifying support)
    — there is no shared builder, and unknown compat servers can 400 on
    the parameter. Same degrade rule as cache_control: a 4xx naming
    `stream_options` → retry once without it + diagnostic. Local
    providers are left untouched in PR1: usage is **captured if present,
    never required** — absent usage falls back to estimated display.
  - Non-streaming paths already return usage; keep passing it through.
- **Gateway** (`Chat/console_provider_gateway.py`):
  `normalize_provider_response` stops discarding usage. The gateway
  accumulates it during the stream and exposes one normalized
  `ProviderUsage` record on the run result the controller already
  consumes at completion (not as a yielded stream event — the chunk
  contract is unchanged).
- **Controller** (`Chat/console_chat_controller.py`): attaches the
  `ProviderUsage` to the assistant message it persists.

### `ProviderUsage` — disjoint buckets (normalization rule)

Four **disjoint** token buckets, so cross-provider math is well-defined:

```
uncached_input | cache_read | cache_write | output
```

Per-provider adapters map into them:

- Anthropic: fields are already disjoint — direct mapping.
- OpenAI: `uncached_input = prompt_tokens − cached_tokens`,
  `cache_read = cached_tokens`, `cache_write = 0`.
- Others: `uncached_input = prompt_tokens`, output as reported.

Each adapter is unit-tested. Cost = Σ bucket × per-model rate.

Edge cases:

- **Aborted stream**: persist whatever usage arrived, mark the record
  `partial: true`; missing output is estimated from generated text at
  display time. Failed sends produce no usage row.
- **Regenerations/variants and soft-deleted messages** carry their own
  usage rows and **count toward conversation cost** — the total means
  "money actually spent", never just the active path.
- **Ephemeral (unsaved) sessions**: usage attaches to the in-store message
  object and inherits the message's own persistence behavior
  (`persist=self.store.persistence is not None`); live cost works
  unsaved, and nothing new leaks to the DB.

### Persistence

- Migration adds a nullable `usage` JSON column to `messages`
  (schema version bump + migration file; **re-verify the version number at
  merge time** — concurrent-session drift is routine in this repo).
- Stored: the four buckets + provider + model (+ `partial` flag).
  **Never dollars.**
- **Read half (hydration)**: the conversation-load path
  (`_console_messages_from_conversation_tree`) maps the usage column back
  into `ConsoleChatMessage`, so reopened conversations show their real
  cost, not $0/estimated. Round-trip test required.

### Pricing catalog

New module `LLM_Calls/pricing_catalog.py`, copying the
`model_capabilities.py` pattern: seeded defaults → `[pricing]` config
overrides → pattern fallback → cached singleton.

- Entry shape: provider, model pattern/name, input / output /
  cache-read / cache-write $-per-Mtok, `as_of` date. Cache-write rate is
  `null` where the concept doesn't exist (everything except Anthropic).
  Rates are **explicit per model**, not global multipliers.
- Seeded for Anthropic, OpenAI, Google, Mistral, Cohere, Groq. Local
  providers resolve to $0.00. Unknown model → `None` → UI shows token
  counts plus a hint to add a `[pricing]` override.
- The `as_of` date is surfaced in the PR3 tooltip (staleness defense).

### PR1 testing

Catalog resolution/override tests; per-adapter normalization tests;
gateway usage-extraction tests against stub provider payloads (streaming +
non-streaming, with/without cache fields); migration + hydration
round-trip on real in-memory SQLite.

---

## PR2 — Prompt-caching enablement

Only the **native Anthropic console path** changes requests; every other
provider's requests are untouched. `cache_control` is injected by the
console gateway — non-console callers of `chat_with_anthropic` are
unaffected (pass-through test asserts legacy paths emit no
`cache_control`).

### Anthropic explicit caching

- **Breakpoints (2 of 4 allowed)**: one on the last system-prompt block,
  one on the last content block of the newest message turn. The per-turn
  breakpoint makes the conversation prefix reusable and grows it
  incrementally each turn.
- **TTL**: 5-minute ephemeral only. (1-hour doubles the write premium —
  wrong trade for interactive chat; not in v1.)
- **Config**: `[caching] anthropic_enabled`, default **on** (console chat
  is multi-turn; 5-min-TTL caching breaks even after two sends inside the
  window).
- **Fallback**: if an Anthropic request carrying `cache_control` fails
  with a 4xx naming the parameter (odd proxies), retry once without
  breakpoints and log a diagnostic. Caching must never break sends.
- Sub-minimum prefixes silently don't cache (no error) — PR3 shows this
  honestly as "no cache" from usage ground truth.
- Implementation note: the gateway extracts leading system rows into the
  provider call's `system_message` string parameter, so
  `chat_with_anthropic` must convert that string into a system **block
  array** to carry `cache_control` — the byte-stability test covers the
  conversion.

### Prefix-stability audit (part of PR2)

- System prompt: verified byte-stable — `_leading_system_message()`
  passes `self.system_prompt` verbatim, no interpolation. Keep it that
  way; add a test asserting two consecutive payload builds are
  byte-identical except for the appended turn.
- The gateway's leading-system-row extraction must produce byte-identical
  system blocks across turns — dedicated test.
- Skill substitution and chat dictionaries rewrite **only the final user
  turn**, at/after the last breakpoint — cache-safe, no change needed.
- Staged workspace sources appear only in the context-snapshot/inspector
  path, not the send payload — no cache risk (re-verify at
  implementation).
- **Known, accepted instability — the vision image budget**
  (`_provider_message_payloads`, newest-first reservation): once a
  conversation holds more images than the budget, each new image-bearing
  message rotates an older message's images out (image block → text
  placeholder), changing earlier history bytes and invalidating the cache
  from that point. Changing the budget policy is **out of scope**; the
  PR3 ticker models and reports it honestly instead.

### OpenAI implicit caching

Zero request changes. Start reading `prompt_tokens_details.cached_tokens`
into the PR1 `cache_read` bucket.

### PR2 testing

Payload-shape tests via the stub provider (breakpoint placement/count);
byte-identical consecutive-build test; leading-system stability test;
per-provider cache-field accounting tests; legacy-path no-`cache_control`
pass-through test. Live acceptance signal:
`cache_read_input_tokens > 0` on the second consecutive real send.

---

## PR3 — Cost chip, break detection, alert

### State model

Frozen `ConsoleCostState` dataclass (house equality-guarded `sync_state`
pattern): conversation cost, token totals, `pricing_known`,
`has_estimated_entries`, cache state (`NONE`/`WARM`/`EXPIRED`), TTL
seconds remaining, `break_pending` + reason + projected delta (`~`),
pricing `as_of` date. All state is **per console session**.

### Chip (8th chip in `#console-control-chip-row`, ≤34 cells)

| State | Render | Notes |
|---|---|---|
| Warm cache, no pending break | `$0.4821 ●` (dim) | Tooltip: bucket breakdown, TTL remaining, "prices as of \<date\>" |
| Actionable break pending | `$0.48 ⚠ +$0.13` (`console-chip-alert`) | Tooltip names the cause |
| Cache lapsed by time | `$0.48 ○` (neutral "cold") | Not alert-colored — a TTL lapse isn't undoable; delta lives in tooltip |
| No pricing data | `12.3k tok` | Tooltip points at `[pricing]` override |
| Local provider | `$0.00` | |

Chip shows 2–3 significant figures; exact 4-decimal amounts in
tooltip/modal. Click opens a small **cost-breakdown modal**: per-message
rows (role, model, buckets, cost), totals including variants, `~` markers
on estimated entries. Cold-glyph/alert CSS goes in source `.tcss` files,
never the generated bundle.

The chip row lives in a control bar hard-capped at 2 rows: the cost chip
is placed **last**, drops to a compact form (`$0.48⚠`, no delta) below a
width threshold, and PR3's live verification includes a narrow terminal
(80×24) to confirm nothing clips.

### Cost computation

Sum of usage rows priced through the catalog. Rows without usage
(pre-migration history, aborted partials) fall back to the local
estimator and set `has_estimated_entries`.

### Break detection — revision counter + lazy fingerprint

- The store/settings layer bumps a per-session **payload-affecting
  revision counter** at its few mutation choke points. **No event
  enumeration** — the chip's state builder, which already runs at every
  existing control-bar sync site (~12 event-driven call sites + the
  streaming tick), recomputes the fingerprint **only when the revision
  differs from the memoized one**. Lazy and self-healing.
- The fingerprint is computed from **the real payload builder**
  (`_provider_messages_for_session()`: system + post-image-budget
  history, in-memory, no draft) so it cannot diverge from what a send
  would transmit. Variant switches, `skip_failed` transitions, deletions,
  and image-budget rotation are covered for free. Per-message content
  hashes are memoized, and attachments contribute a stored **digest**
  rather than re-hashing base64 image bytes → recompute cost ∝ what
  changed, never ∝ attachment size.
- Component hashes: (provider+model, system, tools/config, ordered
  message-content hashes). Component-wise diff yields the reason;
  priority when several changed: **model > tools > system > history**
  (tooltip lists the rest). Dedicated reason for the image-budget case
  ("older images rotated out of the vision budget").
- Recorded baseline: at every successful send, from the payload actually
  sent. **Recompute is suppressed while a run is active** (streaming
  mutates the assistant message per chunk); the fresh baseline is
  recorded once at run completion.
- Sampling-param changes (temperature, max_tokens) deliberately do **not**
  alert. Typing in the composer never triggers anything (the new turn
  appends after the prefix). The alert is **self-clearing**: reverting an
  edit restores the hash and clears it.
- **Gating**: the break alert renders only while a warm cache exists (per
  last-send ground truth: cache fields nonzero). First sends,
  sub-minimum conversations, and expired sessions never alarm.

### Cache state & TTL

- "Is caching working?" comes from **ground truth**: last send's usage.
  Zero cache read *and* zero cache write → `NONE` ("no cache" tooltip).
- Warm-until = last successful Anthropic send + 5 min (sliding TTL
  refreshes on every send), tracked on a **monotonic clock**
  (`time.monotonic()`) so NTP/DST wall-clock jumps can't flip cache
  state. Cache state is **derived from the clock at sync time**, so switching to a background session tab immediately shows
  its true state; a 10-second UI timer — registered with the timer-leak
  audit, running only while the active session is WARM — exists solely to
  repaint the active tab's WARM→EXPIRED flip. Torn down on unmount.
- **TTL countdown and the timed EXPIRED flip are Anthropic-only.** OpenAI
  eviction is undocumented/load-dependent: OpenAI sessions get warm/cold
  from last-send ground truth plus fingerprint break alerts, no countdown.
- App restart: warm-until is in-memory → shows cold even if the provider
  cache is still warm. Under-promises, never lies. Accepted.

### Projection math (always shown with `~`)

- Anthropic: estimated prefix tokens × (cache-write − cache-read rate).
- OpenAI: estimated cacheable span × (input − cached-input rate).

### Providers without cache accounting

Google/Mistral/Cohere/Groq/local: cost display only — no cache glyph, no
break alerts. (Google `cachedContentTokenCount` mapping is a cheap
follow-up.)

### Error handling

Catalog failure → tokens-only display. Fingerprint failure → log
diagnostic, suppress alert. **The ticker never blocks or delays a send.**
No mid-stream cost animation — the chip updates at message completion.

### PR3 testing

Fingerprint matrix (each mutation flips exactly its component + right
reason); image-budget rotation case; TTL flip with mocked clock; chip
contract tests (alert class, cold glyph, width cap, no-pricing mode);
breakdown-modal data test; stub-provider integration with cache-bearing
usage; estimated-fallback for NULL-usage rows; per-session state
isolation across session tabs.

---

## Out of scope / possible follow-ups

- 1-hour TTL config option; cache pre-warming.
- Google implicit-cache accounting (`cachedContentTokenCount`).
- Vision image-budget policy changes (prefix-friendly eviction).
- Spend budgets / confirm-on-send thresholds; historical cost reporting
  UI; per-workspace aggregation.
- The screen footer's dead `Tokens: --` slot on Console (wire it to the
  same state or hide it — cosmetic consistency, not part of this program).
- Auto-disable heuristic for caching when the observed hit rate stays
  zero (slow-cadence users; see Risks).

## Risks

- **Slow-cadence caching economics**: a user who always waits longer than
  the 5-min TTL between sends pays the 1.25× write premium on every send
  and never gets a read — up to +25% on input cost with zero benefit. The
  ticker itself makes this visible (write costs appear in the breakdown),
  and `[caching] anthropic_enabled = false` is the escape hatch; an
  auto-disable heuristic is listed as a follow-up.
- **Pricing staleness**: mitigated by `as_of` date in tooltip + config
  overrides; never fabricate a price for unknown models (tokens-only
  display instead).
- **Estimator error** in projections and legacy rows: always labeled `~`.
- **Schema-version collision** with concurrent sessions: re-verify at
  merge.
- **Provider API drift** (usage field names/semantics): adapters are
  small, unit-tested seams; unknown shapes degrade to estimated, never
  crash.
