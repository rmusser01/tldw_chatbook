# ADR-222: Per-thread provider HTTP session reuse; payloads by reference

Status: Accepted
Date: 2026-10-06
Task: [TASK-34418](../tasks/task-34418%20-%20Provider-HTTP-session-reuse-and-payload-deepcopy-removal-ADR-214.md)
Plan: [Non-console efficiency remediation](../../Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md), Task 6 (F8b)
Numbered 222 at creation: the brief's provisional 214 was only a placeholder; the highest canonical ADR at authoring time was [ADR-221](221-prompt-injection-cold-start-caches.md) (verified per `backlog/docs/lessons-backlog-hygiene.md`).

## Decision

Hosted provider HTTP calls stop paying two per-turn costs that bought
nothing: a fresh `requests.Session` (TCP + TLS handshake per call, zero
keep-alive across turns) and a `deepcopy` of the entire normalized message
history per POST attempt.

### 1. Per-thread session registry (`LLM_Calls/provider_sessions.py`)

- `get_session(key: str, factory: Callable[[], requests.Session]) ->
  requests.Session`: a `threading.local` holds one `dict[key, Session]` per
  thread; first request for a key builds via `factory()` (which **is**
  `create_default_session()` or a thin closure over it), later requests on
  the same thread return the cached session object.
- Provider call sites import the registry when making a request. Importing
  provider modules during application startup does not load this registry,
  preserving the ADR-097 boot-module budget.
- `close_all_for_current_thread()` closes and drops every session in the
  calling thread's registry; `close_session(key)` does surgical cleanup of
  one key (for a future auth-change or endpoint-change path).
- Why per-thread: `requests.Session` is not thread-safe (documented
  connection-pool and cookie-jar races). One registry per thread needs no
  locks — a thread executes one call at a time, and distinct threads never
  see each other's sessions. A test pins that two threads get two distinct
  session objects for the same key.
- Why the thread is the right lifetime: every LLM/summarization call in
  this app runs either inside a Textual thread worker or an
  `asyncio.to_thread` hop, and Textual 8.x thread workers are dispatched
  with `loop.run_in_executor(None, ...)` (`textual/worker.py`, `Worker._run`
  for `thread=True`) — the event loop's **default ThreadPoolExecutor**,
  shared with every `asyncio.to_thread` hop (documented in
  `app_feature_glue.py`'s SubscriptionsDB note). Those pooled threads are
  created once and reused for the app's lifetime, so "per thread" is
  "per pooled worker thread" — turn N and turn N+1 of a conversation land
  on a warm thread and now share the session's keep-alive pool. Registry
  size is bounded by (pooled threads) × (active provider keys).

### 2. Key granularity

Key = `(provider, base_url)` plus every config-derived value the factory
bakes into the session **that the call site relies on**:

- TLS trust: sites that do not pass per-request `verify=` rely on the
  session's `verify` set by `create_default_session()` from
  `[network] ssl_verify`; they append the current trust value (via
  `provider_sessions.trust_setting_fragment()`) so a trust-policy change
  yields a new key (and a new session) instead of a silently stale one.
  The fragment read is guarded: it funnels into the same config bootstrap
  the factory's own read uses, and cold/unreconciled test sandboxes refuse
  that bootstrap — there the fragment degrades to a constant, which is
  harmless because such sandboxes patch out the factory, and in production
  the factory's identical read surfaces any real bootstrap error exactly
  as the pre-ADR-222 per-call sessions did. Cost parity holds:
  `create_default_session()` already calls `requests_verify()` once per
  call today.
- Default timeout: the one swapped site that omits an explicit `timeout=`
  (OpenAI embeddings) relies on `DefaultTimeoutSession`'s config-driven
  default, so it appends `provider_sessions.default_timeout_fragment()`
  (same guarded-read pattern) to the key.
- Retry adapters: sites that mount a `Retry` adapter whose parameters come
  from provider settings (`api_retries`, `api_retry_delay`) append the
  effective `(total, backoff_factor)` to the key. A settings change gets a
  new session instead of reusing one whose adapter still carries the old
  retry budget. Config-independent zero-retry mounts (hosted engine,
  qwencloud) need nothing extra — the mount is identical every call and
  moves inside the factory.
- The factory is resolved **through the calling module's
  `create_default_session` global at call time**, so the existing test seam
  (`monkeypatch.setattr(module, "create_default_session", ...)`) keeps
  working unchanged.

### 3. Lifecycle and teardown

- A registry session lives for its thread's lifetime. There is **no
  per-worker close hook, by design**: workers share pooled executor
  threads, so closing a session "when a worker exits" would close it
  before the next worker scheduled onto that same thread — destroying
  exactly the reuse this ADR exists to create. Process exit (and the
  default executor's shutdown) bounds every session; idle pooled sockets
  die with the process. `close_all_for_current_thread()` exists for test
  isolation — `Tests/conftest.py` gains an autouse fixture
  (`reset_provider_session_registry`) calling it so a session cached under
  one test's monkeypatched factory can never leak into the next test —
  and for any future explicit teardown path.
- Streams own their **response**, never the shared session:
  `OwnedSSEStream` / `QwenCloudStream` take an optional session (None from
  the swapped call sites) and close only the response. Closing the
  response releases the connection back to the pool for reuse; closing the
  session would have cleared the pool. The non-streaming paths in the
  hosted engine and qwencloud likewise stop closing the session in their
  `finally` blocks (the per-attempt response closes remain).
- Cookie parity: `get_session` clears the session's cookie jar on every
  hit. Today every call uses a fresh session whose jar is empty, so a
  `Set-Cookie` from provider call N never reached call N+1; the clear
  preserves that. Within a call, the retry loop keeps today's behavior
  (same jar across attempts of one call).
- `trust_env`, `verify`, headers, and hooks on a shared session are never
  mutated by swapped call sites. The only mutator in the codebase is the
  OpenAI recovery layer (see §5), which keeps its own per-call sessions.

### 4. Payload by reference (`hosted_chat.owned_json_post`)

`json=deepcopy(dict(payload))` becomes `json=payload`. `requests` passes
the `json` argument to `json.dumps` when preparing the request and never
mutates it; the hosted retry loop re-serializes the same mapping on every
attempt and nothing between attempts writes to it. The response side keeps
its `deepcopy(dict(result))` — that copy protects callers from sharing
mutable structure with the parsed body, a different concern. A
mutation-probe test pins the payload deep-equal to a pre-call snapshot
after a successful call and after a call that exhausts retries.

### 5. Call sites converted — and the deliberate exclusions

Converted: hosted engine (`hosted_chat.owned_json_post`), qwencloud
(`chat_with_qwencloud`), the summarizer transport
(`Summarization_General_Lib._post_with_retry`), and in `LLM_API_Calls`:
OpenAI embeddings, Anthropic, Cohere, Google Gemini, and both HuggingFace
paths.

Excluded, with reasons:

- **OpenAI chat streaming/non-streaming (`LLM_API_Calls.py`)**: every POST
  goes through `recovery_review.openai_post`, which calls
  `operation.own(session)` — the guarded recovery `_Operation` closes
  every owned resource at operation end, and in recovered-connection mode
  additionally sets `session.trust_env = False`. A shared session here
  would be closed after each call anyway (no reuse gained) **and** would
  leak that `trust_env` mutation into later ordinary calls on the same
  thread. These two sites keep per-call sessions until the recovery layer
  grows an explicit non-owning transport mode.
- **Local providers (`LLM_API_Calls_Local.py`, `Local_Summarization_Lib.py`)**:
  localhost endpoints; no TLS handshake and no cross-turn win worth the
  registry entries.

## Context

Wave 3 of the non-console performance remediation. Task 5
(TASK-34669, commit 028c4494bc) removed the per-chunk deepcopy chain in
the hosted streaming engine; this task removes the per-turn costs: the
`deepcopy(dict(payload))` on every POST attempt — the payload carries the
entire normalized message history — and the fresh
`create_default_session()` at every provider call site, each paying
TCP + TLS handshake with no connection reuse across turns.

## Alternatives

- **One global session shared by all threads**: `requests.Session` is not
  thread-safe; concurrent sends from parallel workers would race the pool
  and the cookie jar. Rejected.
- **A session pool keyed by URL with locking**: reimplements what
  urllib3's pool already does per session, adds a lock on every call, and
  still needs per-thread cookie discipline. Rejected.
- **Close sessions per call, rely on urllib3 pooling only**: pools die
  with the session; there is nothing to keep alive. This is the status
  quo being removed.
- **`httpx.AsyncClient` / aiohttp migration**: a provider-stack rewrite,
  explicitly out of scope for this remediation wave.
- **Keep the payload deepcopy "for safety"**: no code path mutates the
  payload between attempts; the copy cost scales with conversation length
  and is paid per attempt. The mutation-probe test pins the invariant the
  copy was guessing at.

## Consequences

- Back-to-back calls to the same `(provider, base_url)` on one pooled
  thread reuse one TCP+TLS connection (subject to server keep-alive
  policy): 3 calls that opened 3 connections now open 1 (evidence in the
  task report; measured against a local counting HTTP server).
- A settings change (TLS trust, retry counts/delays, default timeout)
  starts a new session under a new key; the previous session object stays
  in that thread's registry until thread/process death — bounded by
  pooled threads × distinct configs, and functionally inert.
- Tests that pin "session closed once per call" semantics
  (`test_hosted_chat.py` ownership tests) now pin the registry contract:
  one factory invocation per key per thread, responses closed per
  attempt, session close only via the registry.
- The OpenAI chat paths remain per-call-session (see §5); a future task
  would need a non-owning seam in `recovery_review` first.
