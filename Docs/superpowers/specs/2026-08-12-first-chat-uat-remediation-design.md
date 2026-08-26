# First-Chat UAT Remediation Design

**Date:** 2026-08-12
**Status:** Approved in design review; awaiting implementation plan
**Base:** `origin/dev` at `5414d811b`
**Scope:** Compatibility-first remediation of the fresh-install provider setup,
Settings, Console, launch, and diagnostic issues found during the August 2026
llama.cpp and custom OpenAI-compatible UAT.

## Context

The UAT replay used a clean profile, a mock llama.cpp endpoint, and a mock custom
OpenAI-compatible endpoint. Both providers eventually completed chats, including a
custom-provider chat after restart, but the path exposed contradictory readiness,
split provider/model state, missing manual endpoint entry in first-run setup, stale
Console defaults, retry duplication risk, browser splash corruption, and several
information-architecture and diagnostic-trust problems.

The application already has an accepted ownership boundary:

- Settings owns durable defaults.
- Each Console session owns its active provider, model, and generation settings.
- Typed handoffs cross screen boundaries.

This design preserves that boundary. It does not introduce named provider
connections or migrate existing configuration.

## Goals

- Make a fresh user's first provider setup and first chat truthful and complete.
- Accept both API-base and full chat-completions endpoint input.
- Derive chat and models URLs through one shared, provider-aware contract.
- Remove false or contradictory readiness and test verdicts.
- Keep default provider/model state coherent without overwriting active sessions.
- Make failed sends and retries lossless and non-duplicating.
- Improve first-run and Settings information architecture and compact-layout use.
- Render splash content correctly in terminal and browser modes.
- Make optional-subsystem and metrics diagnostics accurate and privacy-conscious.
- Preserve all existing config files without conversion.

## Non-Goals

- A named provider-connection registry.
- Multiple credentials or endpoints per provider identity.
- Migrating provider configuration to the server's connection model.
- Persisting connectivity claims across process restarts.
- Silently applying new defaults to user-owned Console sessions.
- Treating a successful `/v1/models` call as proof that generation works.

## Design Principles

1. One canonical interpretation of endpoint input.
2. Configuration completeness, endpoint connectivity, and model confirmation are
   separate facts.
3. A UI verdict must never contradict its supporting evidence.
4. Saving defaults affects new conversations; active sessions change only through
   an explicit, targeted action.
5. Preflight refusal creates no history. Accepted work creates one durable turn.
6. Template values are not user configuration until explicitly accepted.
7. Optional features stay quiet until they are relevant.
8. Browser and terminal rendering are release-equivalent surfaces.

## 1. Shared Provider Endpoint Contract

Add one pure provider endpoint contract used by first-run setup, Settings model
discovery and testing, and Console execution. A likely home is
`tldw_chatbook/Chat/provider_endpoint_contract.py`; the implementation plan will
pin the final module boundary after checking all existing endpoint helpers.

The resolved value is a typed record containing:

- provider identity;
- normalized user input;
- provider-specific persisted endpoint value;
- effective chat-completions URL;
- derived models URL;
- safe display values;
- endpoint form (`origin`, `api_base`, `chat_url`, `models_url`, or legacy local);
- warnings and validation errors.

### Accepted endpoint forms

For custom OpenAI-compatible providers, accept:

- `http://host:1234`
- `http://host:1234/v1`
- `http://host:1234/v1/chat/completions`
- `http://host:1234/v1/models`
- equivalent forms beneath a proxy prefix, such as
  `https://host/proxy/v1/chat/completions`

The proxy prefix is preserved when deriving sibling routes. Custom-provider
configuration is persisted in the full chat-completions form expected by legacy
callers.

For llama.cpp, accept the same OpenAI-compatible forms plus the legacy
`/completion` form. Persist the normalized llama.cpp root expected by its existing
gateway and derive `/v1/chat/completions` and `/v1/models` for compatible calls.

Provider-specific adapters may override persistence or route rules when an
existing provider uses a different established key or endpoint shape.

### URL safety

- Scheme-less input is accepted only for localhost.
- Remote HTTP is accepted only when explicitly entered and carries a warning.
- Embedded username/password is rejected; credentials belong in credential fields.
- Query strings and fragments are rejected unless a provider-specific adapter
  explicitly supports them.
- Encoded, ambiguous, or repeated API suffixes are rejected.
- Raw endpoint credentials, response bodies, and exception text never enter UI
  copy or normal logs.

## 2. Readiness and Test Evidence

Readiness is a structured snapshot rather than one boolean.

### Facets

**Configuration**

- `incomplete`
- `configured`

**Endpoint**

- `not_tested`
- `testing`
- `reachable`
- `unreachable`
- `model_listing_unavailable`

**Model**

- `missing`
- `confirmed`
- `unconfirmed`

The UI computes one verdict from these facets:

- **Verified**: endpoint reachable and selected model confirmed.
- **Configured, connection not tested**.
- **Configured, model listing unavailable**: chat may work, but the selected model
  and chat endpoint are unconfirmed.
- **Incomplete**: names the exact missing field.
- **Connection failed**: names one bounded category such as timeout, refused,
  unauthorized, or HTTP status.
- **Changed since test**.

A models-route `404` is not a chat failure and not a successful chat test. The
result says exactly: model listing is unavailable and the chat endpoint has not
been tested.

### Evidence lifecycle

Test evidence records canonical endpoint identity, credential source kind, an
in-memory credential-field revision, the returned model IDs, and a draft
generation token. It never records a raw secret or a secret-derived hash. A late
result cannot attach to a newer draft.

Changing to a model found in the tested result does not invalidate endpoint
evidence. Changing provider, canonical endpoint, or credential source does.

Saving the exact tested semantic draft preserves evidence after all writes
succeed. Trimming or equivalent route normalization does not make it stale.
Partial persistence or concurrent edits invalidate it. Evidence is process-local;
after restart the state is **Configured, connection not tested**. Request-time
readiness always re-resolves environment credentials because they may change
outside the application.

### Explicit configuration provenance

Template endpoints alone are not considered configured. Add a minimal,
backward-compatible boolean mapping under `provider_setup.confirmed`, keyed by
normalized provider ID, recording that the provider endpoint was explicitly
accepted. It contains no endpoint, model, or credential value. Existing configs
without this metadata continue through legacy heuristics. Explicit provider reset
or endpoint clearing removes that provider's confirmation. This is provenance
metadata, not a connection registry.

## 3. Persistence and Provider/Model Ownership

Keep the current configuration tree.

- `api_settings.<provider>` owns that provider's endpoint, credential source, and
  remembered preferred model using the provider's established key mapping.
- `chat_defaults.provider/model` owns the pair used by genuinely new Console
  conversations.
- `ConsoleSessionSettings` owns the active conversation's provider, model,
  endpoint resolution, and generation values.

Saving provider/model setup performs one serialized configuration mutation that
updates provider settings, `chat_defaults`, and setup provenance. In-memory config
updates only after the complete disk mutation succeeds.

When legacy values disagree and `chat_defaults.provider` matches the selected
provider, non-empty `chat_defaults.model` wins. The provider-specific compatibility
value is repaired only when the user explicitly saves that provider/model or
completes setup, never during unrelated saves.

Every new tab and new conversation snapshots the latest `chat_defaults`. Existing
user-owned sessions remain unchanged.

An existing session is eligible for automatic first-run refresh only when it:

- was created from defaults;
- has no messages;
- has no composer draft or staged attachments;
- has no user-edited system prompt, generation settings, or other session values.

If the active session is not eligible, **Start chatting** creates a new session.
Rerunning setup never mutates an existing session implicitly.

## 4. First-Run Experience

### Provider step

- URL-based providers show an Endpoint field immediately.
- llama.cpp shows **Authentication (optional)** as progressive disclosure rather
  than an irrelevant required API-key field.
- Custom OpenAI-compatible credentials are clearly optional.
- Manual endpoint entry is always available.
- **Detect local servers** presents all discovered candidates and never overwrites
  typed input without confirmation.
- **Test connection** uses the exact draft endpoint and credential source.

### Model step

- Discovery uses the exact draft connection from the provider step.
- Discovery caches are scoped by provider, canonical endpoint, and credential
  identity.
- A models-route failure leaves manual entry visible and explains whether listing
  is unavailable or the connection failed.
- Placeholder rows can never become selected model IDs.

### Navigation and summary

A pinned footer exposes **Back**, **Continue**, and **Skip this step**. Keyboard
shortcuts remain secondary hints. The welcome screen uses **Skip setup**. Exiting
mid-flow uses **Exit setup** with committed-step recovery copy.

The final first-run actions are:

- Primary **Start chatting** when provider/model configuration is complete.
- Primary **Review provider setup** when incomplete.
- Secondary **Explore Home**.
- Tertiary **Review settings**.

Ambiguous simultaneous **Finish** and **Finish later** actions are removed.
First-run setup uses inline progress and error status; redundant welcome and setup
toasts are suppressed until the modal closes.

## 5. Settings Information Architecture

### Overview

Lead with user tasks:

- **Configuration**: saved default provider/model and completeness.
- **Last connection test**: verified this process, not tested, or changed.
- Storage and privacy.
- Sync when configured.

Runtime ownership, server bindings, handoffs, and detailed diagnostics remain
available through Advanced/Diagnostics disclosures and deep links.

### Providers and Models

Organize the category into:

1. **Connection**: provider, endpoint, optional auth, effective chat URL, models
   URL, and one test verdict.
2. **Models**: endpoint-scoped discovery, default model, manual entry, and model
   confirmation.
3. **Generation defaults**: capability-driven Common, Reasoning, Advanced
   sampling, and provider-specific groups.
4. **Context capacity**: model window, source, effective conversation capacity,
   and expandable calculation details.

Provider selection becomes searchable and grouped while preserving saved unknown
provider IDs and an **Enter provider ID** path.

Unsupported generation fields are not shown and are not silently cleared. Group
reset applies only to the selected provider/model profile and states its scope.

When capacity inputs are unknown, show **Capacity unavailable** with a corrective
action. Do not present fallback values such as `8,001` as measured capacity. The
calculation has one visible source precedence and exposes provider cap, response
reserve, safety margin, and mandatory input only in expanded detail.

Saving reports **Saved for new conversations**. A separate **Apply to current
conversation** action is enabled only after save and while the target session is
idle. It uses a revisioned, session-targeted, secret-free handoff carrying the
session ID, provider/model/profile identities, and config revision. If the session
or revision changed, Console refuses the handoff without mutation.

## 6. Console Send Transactions

Console settings always load the active session's provider/model pair, never a
stale provider compatibility value.

Sending has three boundaries:

1. **Preflight refused**: persist no history; preserve composer text and staged
   attachments.
2. **Accepted**: persist the user turn exactly once; transfer attachments to it;
   clear the submitted composer snapshot and staging.
3. **Accepted request fails**: retain the one user turn; add one failed assistant
   attempt with **Retry**; do not restore submitted text into the composer.

Retry reuses the stored turn and attachments and never duplicates the user row.
Text entered after submission is never overwritten by a late result. Cancellation
retains the user turn and records a cancelled assistant attempt, distinct from
failure. Queued turns follow the same ownership rules.

A successful first chat may confirm usability for the active Console session but
does not persist permanent connectivity evidence.

## 7. Splash and Startup Communication

Splash effects produce a typed `SplashFrame` with `plain`, `ansi`, or `rich`
content. Plain content renders with markup disabled. ANSI content passes through a
single sanitizer/converter before `Text.from_ansi`. Structured Rich content is
passed directly. Existing string-returning effects cross a registry adapter with a
declared frame kind; the renderer never guesses a frame's format from its content,
and every effect does not need to be rewritten in the first slice.

Animation starts only after the display has valid dimensions and its first frame
renders. Frame generations fence resize and close races. The duration timer starts
after first paint. Late frames after close are ignored.

Every registered card supplies a tested static fallback. Missing or unsupported
fallbacks use a deterministic minimal ASCII-safe brand card. Compact browser or
terminal layouts may use that fallback when the selected card cannot fit. Reduced
motion always uses static content.

Startup notifications are bounded, generation-tagged, and deduplicated. Critical
setup failures remain inline. Deferred messages are discarded when their startup
generation is no longer relevant.

## 8. Diagnostic Trust and Metrics Privacy

Subsystem startup returns typed outcomes rather than logging inside a helper and
then logging unconditional success at the caller. Supported states are
`Started`, `Already running`, `Disabled`, `Unavailable`, `Degraded`, and `Failed`
as appropriate to the subsystem.

Metrics behavior changes to:

- externally served metrics disabled by default;
- explicit configuration or `METRICS_PORT` required to opt in;
- loopback binding unless a remote bind is explicitly configured;
- one exporter path instead of simultaneous Prometheus servers;
- idempotent initialization;
- exactly one truthful startup outcome.

Required TTS distribution resources receive built-wheel tests using
`importlib.resources`. Optional/downloadable resources are checked lazily and
reported as **Not installed** only on relevant TTS surfaces. Downloads require an
explicit user action and do not occur during unrelated startup.

User-facing diagnostics show status and recovery. Raw exceptions, secrets,
credential-bearing URLs, and response bodies remain in appropriately redacted
logs only.

## 9. Error Handling and Security

- All endpoint errors use bounded categories; raw transport text is never shown.
- Draft keys and environment credential values never enter evidence records.
- Endpoint display masks userinfo defensively even though new userinfo input is
  rejected.
- Probe and discovery workers are latest-generation-wins and cancellable.
- No test performs external network access; an egress guard permits only mock
  endpoints.
- Config writes remain serialized and atomic at the application mutation boundary.
- Apply-to-current handoffs carry identities and revisions, never credentials or
  endpoint values.

## 10. Delivery Order

### Slice 1: Provider contract and persistence

Endpoint interpretation, route derivation, readiness facets, probe evidence,
provider-model key accessors, explicit setup provenance, and atomic save mutation.

### Slice 2: First-run and default handoff

Manual endpoints, optional auth, multi-result discovery, model discovery, visible
navigation, summary hierarchy, and untouched-session eligibility.

### Slice 3: Settings and Console

Task-oriented Settings, searchable provider selection, grouped generation fields,
context copy, targeted apply intent, and send/retry/cancel transaction fixes.

### Slice 4: Launch and diagnostics

Typed splash frames, browser/terminal fallbacks, notification timing, metrics
initialization, TTS packaging checks, and diagnostic outcome cleanup.

Each slice is independently reviewable and testable. Later slices depend on the
shared contracts from earlier slices.

## 11. UAT Finding Acceptance Matrix

| ID | Finding | Required resolution and evidence |
|---|---|---|
| UAT-01 | No custom llama.cpp endpoint in first run | Manual Endpoint field, optional auth, base/full URL tests, first-run live replay. |
| UAT-02 | False readiness for unreachable template localhost | Explicit setup provenance and separate configuration/connectivity facets; fresh-profile regression. |
| UAT-03 | Provider test says passed and unreachable | One computed verdict; custom models `404` case; no contradictory text assertion. |
| UAT-04 | Saved defaults do not reach new Console sessions | New tab/conversation snapshots latest defaults; untouched-session and user-owned-session tests. |
| UAT-05 | `chat_defaults` and provider model disagree | Atomic paired save, provider-key accessor, legacy precedence and explicit-repair tests. |
| UAT-06 | Restart splash renders raw/fragmented markup | Typed frame renderer, terminal/browser restart screenshots, text and bounds assertions. |
| UAT-07 | Blocked send creates history while composer permits duplicate retry | Preflight/accepted/failure transaction tests and one-user-row retry assertion. |
| UAT-08 | Exact Save immediately marks successful test stale | Semantic evidence identity preserved through successful atomic save. |
| UAT-09 | Welcome has no visible Next | Pinned visible navigation tested at compact, normal, and browser sizes. |
| UAT-10 | Model discovery disconnected from custom llama setup | Discovery receives exact draft connection; cache isolation and manual fallback tests. |
| UAT-11 | Custom provider is difficult to find | Searchable grouped picker, saved-unknown provider, and manual-ID tests. |
| UAT-12 | Settings overview is system-oriented | Configuration/test/storage/privacy-first overview visual and content assertions. |
| UAT-13 | Generation settings are an undifferentiated scroll | Capability groups, scoped reset, focus help, and compact-layout visual checks. |
| UAT-14 | Summary exit hierarchy is ambiguous | State-dependent primary action and unique secondary/tertiary actions. |
| UAT-15 | Notifications/tooltips obscure setup | Inline setup status, deferred notification fencing, focus-help checks. |
| UAT-16 | Context-capacity language conflicts | Unknown-state behavior, source precedence, calculation tests, revised copy assertions. |
| UAT-17 | Missing TTS mappings and contradictory metrics logs reduce trust | Built-wheel resource test, lazy optional status, typed idempotent metrics outcomes, startup log assertion. |

## 12. Verification Strategy

### Automated

- Pure endpoint contract table tests, including proxy prefixes and malformed forms.
- Probe tests for models `200`, `404`, `401`, timeout, refused, malformed payload,
  and authenticated access.
- Atomic config and legacy-compatibility tests.
- Textual Pilot tests for first-run, Settings, targeted handoff, and Console send
  transactions.
- Splash frame and lifecycle tests.
- Built-wheel installation/resource test.
- Startup metrics outcome and idempotence tests.
- Egress-guard tests proving only local mocks are contacted.

### Live replay

Use a disposable profile and two local mocks:

- llama.cpp with a working models route and chat route;
- custom OpenAI-compatible endpoint whose models route returns `404` while chat
  succeeds.

Exercise base and full chat URL entry, setup, Save, new Console sessions, both
chats, failure/retry, cancellation, and restart. Repeat in terminal and browser
modes. Capture text assertions plus screenshots at compact, standard, and wide
viewports; screenshots alone are not acceptance evidence.

## 13. Risks and Mitigations

- **Endpoint normalization changes a legacy caller:** inventory every endpoint
  consumer and retain provider-specific persistence adapters.
- **Model state repair changes user intent:** repair only on explicit provider/model
  save and preserve active sessions.
- **Discovery cache leaks across endpoints:** include canonical connection identity
  in cache keys or bypass shared cache for unsaved drafts.
- **Targeted apply races with session changes:** include session ID and config
  revision; fail closed.
- **Splash compatibility regresses an effect:** require typed frames and a static
  fallback for every registered effect.
- **Metrics default change surprises monitoring users:** retain explicit
  `METRICS_PORT` opt-in and document the loopback/default behavior.
- **Scope becomes one unreviewable patch:** deliver in the ordered slices above.

## 14. Deferred Named Connections

Named provider connections remain a follow-up design and migration. When resumed,
inspect `tldw_server` first because it may already implement connection ownership,
endpoint/credential/model grouping, testing, and selection semantics that the
client should reuse or align with. The follow-up must define local/server identity,
sync, migration, conflict, and offline behavior before changing this config model.
