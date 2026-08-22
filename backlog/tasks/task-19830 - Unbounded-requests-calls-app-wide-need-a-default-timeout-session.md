---
id: TASK-19830
title: Unbounded requests calls app-wide need a default-timeout session
status: Done
assignee:
  - '@claude'
created_date: '2026-08-21'
updated_date: '2026-08-22 08:18'
labels:
  - performance
  - reliability
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while fixing task-19560, which named two unbounded `requests.post` calls
in the summarization libraries. An audit of those two modules found 29; an
audit of the whole package found the problem is systemic.

Measured on dev with an AST sweep (`requests.Session()` constructions, and
`post`/`get`/`put`/`delete`/`patch`/`request` calls with no `timeout=`):

    requests.Session() construction sites: 56
    timeout-less request calls:            40
    files involved:                        13

A `requests` call with no timeout waits forever on a half-open connection. On
the LLM paths that means a chat or summarization that never returns and cannot
be cancelled; several of these also run on the event loop, so the whole TUI
stops. `requests` has no per-session default timeout, which is why every site
has to remember one individually -- and 40 of them do not.

Files: `LLM_Calls/` (LLM_API_Calls, LLM_API_Calls_Local, Local_Summarization_Lib,
Summarization_General_Lib, hosted_chat, qwencloud), `Embeddings/Embeddings_Lib`,
`Character_Chat/local_character_persona_service`, `Local_Inference/ollama_model_mgmt`,
`TTS/backends/kokoro`, `Utils/egress`, `Web_Scraping/Confluence/confluence_auth`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A shared session factory applies a default connect/read timeout to every request that does not pass one explicitly, and an explicit `timeout=` still wins.
- [x] #2 The LLM_Calls package uses it everywhere -- no bare `requests.Session()` and no timeout-less request call remains there.
- [x] #3 An architecture guard test fails when a new bare `requests.Session()` or timeout-less request call is added to the covered area, so this cannot silently regress.
- [x] #4 Any file NOT yet converted is listed explicitly in that guard's exemption set, so the remaining work is visible rather than implied.
- [x] #5 The default is config-driven, consistent with the existing `api_timeout` settings the OpenAI path already reads.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
`Utils/egress.py` is the natural home: it already owns HTTP policy for this
app and already carries the house default (`guarded_fetch_requests(timeout:
float = 30.0)`).

Shape: a `requests.Session` subclass overriding `request()` to fill in
`timeout` when the caller omitted it, plus a factory. Then swap the
construction sites. A call that already passes `timeout=` is untouched by
construction, which keeps every deliberate per-provider timeout intact.

Sequence it so each step is independently reviewable:
1. factory + guard test scoped to `LLM_Calls/`, other files exempted and listed;
2. burn down the exemption list a subsystem at a time.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Factory** (`tldw_chatbook/Utils/egress.py`): `DefaultTimeoutSession(requests.
Session)` overrides only `request()` (every verb funnels through it), filling
`kwargs["timeout"]` with `default_session_timeout()` -- a config-driven
`(connect, read)` tuple read from `[web_security] request_connect_timeout_
seconds`/`request_read_timeout_seconds` (falls back to `10.0`/`30.0`) -- only
when the caller supplied neither the `timeout=` keyword nor a 7th+ positional
argument. `create_default_session(*, timeout=None)` is the factory function
every call site now imports. A tuple, not one number: `requests` re-arms the
READ half per chunk on a streamed response, so a slow-but-progressing stream
is only killed by a stall between chunks, never total elapsed duration --
several converted call sites stream.

**Re-derived the AST sweep** rather than trusting the task's numbers: 56
`Session()` sites (exact match), 34 timeout-less calls / 12 files (vs the
task's 40/13 -- explained by drift since filing: `Character_Chat/local_
character_persona_service.py`, named in the description, no longer imports
`requests`; `Web_Scraping/WebSearch_APIs.py`, not named, had real findings).

**Converted all of `LLM_Calls/`**: 49 `Session()` sites across 6 files, plus
2 bare `requests.post` calls wrapped in `with create_default_session() as
session:` (OpenAI embeddings in `LLM_API_Calls.py`; `summarize_with_local_llm`
in `Local_Summarization_Lib.py`). One bare `requests.post` (`summarize_with_
anthropic`, `Summarization_General_Lib.py`) deliberately NOT wrapped in a
session -- it streams (`stream=streaming`), and closing a session mid-read
risks tearing down the connection pool under an in-flight response; got
`timeout=default_session_timeout()` added directly instead, preserving the
`allow_redirects=False` credential-forwarding comment (task-19557) verbatim.
Verified: zero remaining bare `Session()`/timeout-less calls anywhere in
`LLM_Calls/`.

**Guard** (`Tests/Architecture/test_default_timeout_session_guard.py`): an
AST walker scans every file under `tldw_chatbook/` that imports `requests`
(module-level or a lazy in-function import), tracking session-variable data
flow to catch `session.post(...)` as well as `requests.post(...)`. 3
assertion tests (`LLM_Calls/` fully clean; whole-package unexempted-violation
check; exemption-set freshness) + 8 scanner self-tests. `EXEMPT_FILES`:
`Embeddings/Embeddings_Lib.py`, `Local_Inference/ollama_model_mgmt.py`,
`TTS/backends/kokoro.py`, `Web_Scraping/Confluence/confluence_auth.py`,
`Web_Scraping/WebSearch_APIs.py`, `Utils/egress.py` (its own internals).
Red-proofed by hand: appended a bare `requests.Session()` to `qwencloud.py`,
confirmed both assertion tests failed and named the file+line, removed it,
confirmed green again.

**The dominant cost of this task**: converting a construction symbol
(`requests.Session()` -> `create_default_session()`) silently disconnected
every test that monkeypatched the OLD symbol by name. Found and fixed 91
such mock call sites across 12 test files (8 in `Tests/LLM_Calls/`, 5 in
`Tests/Chat/` -- `test_chat_functions.py` alone had 26, none of which
`Tests/LLM_Calls/ -q` would have surfaced). Every fix retargets the SAME
fake/response the test already built at `<module>.create_default_session`
instead of `<module>.requests.Session`/`.post`. Left untouched (verified
safe): `requests.Session.post`/`.close` class-level method patches (still
work -- `DefaultTimeoutSession` doesn't override them), and mocks targeting
files this task doesn't convert (`ollama_model_mgmt.py`) or call sites this
task deliberately left bare (`summarize_with_anthropic`'s `requests.post`).
Full A/B evidence and the per-file breakdown are in
`.superpowers/sdd/19830-report.md`; a generalized lesson is recorded in
`backlog/docs/lessons-testing-evidence.md`.

**Gate**: `Tests/Architecture/` 173 passed / 5 pre-existing failures (both
before and after, byte-identical failing set, confirmed via `git checkout
HEAD --`). `Tests/LLM_Calls/` 1009 passed / 7 pre-existing failures (5
diagnostic-privacy ledger + 2 anthropic-redirect-credential-leak, both
reproduced against a fully unmodified checkout). Broad `-k "summariz or llm
or api_call"` sweep across `Tests/`: pre-existing collection errors from
missing optional deps (`numpy`/`jsonschema`/`playwright`) in this venv,
unrelated to this task; all touched modules import cleanly.

Files: `tldw_chatbook/Utils/egress.py`, `tldw_chatbook/config.py` (sample
config keys), the 6 `LLM_Calls/` modules, `Tests/Utils/test_egress.py` (10
new behavioural tests), `Tests/Architecture/test_default_timeout_session_
guard.py` (new), and the 12 test files listed above.
<!-- SECTION:NOTES:END -->
