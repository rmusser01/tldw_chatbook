# tldw-pydle 1.1.0.post1 implementation plan

Status: Foundational task ready; behavioral amendments under PR review
Reassessed: 2026-09-10
Start here: [PTO handoff](2026-09-10-network-chat-pto-handoff.md)
Amendment: [ADR-149](../../../backlog/decisions/149-network-chat-handoff-reliability-amendments.md)
Date: 2026-08-23
Design: [Network Chat IRCv3 and tldw-pydle design](../specs/2026-08-23-network-chat-ircv3-and-tldw-pydle-design.md)
Decision: [ADR-148](../../../backlog/decisions/148-network-chat-ircv3-and-tldw-pydle-boundary.md)
Scope: Dedicated fork and first release artifact only; no Chatbook runtime integration or screen

> **For agentic workers:** Use superpowers:executing-plans to execute one claimed task at a time. This PR delivers planning only.

**Goal:** Deliver an independently tested IRC client fork before Chatbook adopts it.

**Architecture:** One connection owns transport, ordered reduction and bounded delivery; Chatbook owns application lifecycle and UI behind an adapter.

**Tech Stack:** Python 3.11+ fork; Python 3.12+ and Textual 8.x for later Chatbook integration.

**Spec:** [Design](../specs/2026-08-23-network-chat-ircv3-and-tldw-pydle-design.md).

## Global constraints

- Canonical Codeberg base remains `4efcc3b5096536668dfe772461f19a17f1ddd84e`; never silently track develop.
- Separate distribution `tldw-pydle`, import `tldw_pydle`; no vendored source.
- Direct verified TLS; PLAIN only over verified TLS; no implicit reconnect.
- Memory-only transcripts; no raw wire or payload-bearing diagnostics.
- Fork Python 3.11–3.14 Linux gates; Windows/macOS transport and packaging coverage on 3.12.
- Behavioral changes require the corresponding ADR-149 amendments accepted first.

## ADR check

ADR required: yes

ADR path: `backlog/decisions/148-network-chat-ircv3-and-tldw-pydle-boundary.md` (approved boundary); `backlog/decisions/149-network-chat-handoff-reliability-amendments.md` (proposed corrections)

Reason: The work creates a maintained runtime dependency and defines transport,
authentication, privacy, lifecycle, protocol, packaging, and cross-repository
interfaces shared by Chatbook and approved tldw client roles.

## Objective

Publish `tldw-pydle==1.1.0.post1` as a provenance-preserving, independently
installable Python 3.11+ fork of canonical Codeberg pydle. The release must
apply IRC protocol state in wire order, keep content and credentials out of
diagnostics, provide verified TLS and TLS-gated SASL PLAIN, own its async
lifecycle honestly, stream LIST within bounds, preserve the first modern IRCv3
semantics, and pass deterministic, AgentIRC, and pinned-Ergo artifact gates.

## Task sequence

| Order | Task | Outcome | Depends on |
| --- | --- | --- | --- |
| 1 | [TASK-21601](../../../backlog/tasks/task-21601%20-%20Establish-tldw-pydle-fork-provenance-and-isolated-package.md) | Full-history fork, isolated namespace, buildable wheel/sdist | None |
| 2 | [TASK-21602](../../../backlog/tasks/task-21602%20-%20Remove-IRC-content-from-diagnostics-and-harden-outbound-transport.md) | Content-free diagnostics and one ordered bounded writer | 21601 |
| 3 | [TASK-21603](../../../backlog/tasks/task-21603%20-%20Own-tldw-pydle-connection-lifecycle-and-secure-TLS.md) | Explicit lifecycle, honest close and verified direct TLS | 21602 |
| 4 | [TASK-21604](../../../backlog/tasks/task-21604%20-%20Implement-ordered-IRC-protocol-reduction-and-correlated-replies.md) | Wire-ordered state and publish-before-send correlation | 21603 |
| 5 | [TASK-21605](../../../backlog/tasks/task-21605%20-%20Implement-CAP-302-and-TLS-gated-SASL-PLAIN.md) | CAP 302 state machine and dependency-free PLAIN | 21604 |
| 6 | [TASK-21606](../../../backlog/tasks/task-21606%20-%20Add-bounded-streaming-IRC-LIST-sessions.md) | Incremental bounded LIST with honest local cancellation | 21604 |
| 7 | [TASK-21607](../../../backlog/tasks/task-21607%20-%20Add-bounded-IRCv3-message-and-history-semantics.md) | Tags, time, echo, BATCH and draft history semantics | 21605 |
| 8 | [TASK-21608](../../../backlog/tasks/task-21608%20-%20Gate-and-publish-tldw-pydle-1-1-0-post1.md) | Reproducible compatibility-gated release | 21601–21607 |

TASK-21605 and TASK-21606 are code-independent after TASK-21604, but they
should still be executed serially unless the user explicitly requests parallel
agent work. TASK-21608 does not begin until both branches have landed with
TASK-21607.

## Repository and branch boundaries

Use a dedicated checkout of `rmusser01/tldw-pydle` outside Chatbook source. Choose an authorized local path; the original author's Windows directory is not required. A sibling path may require filesystem approval in an agent session. Do not place the fork under Chatbook as a
vendored directory, submodule, or long-lived temporary checkout.

Remote roles:

```text
origin    https://github.com/rmusser01/tldw-pydle.git
upstream  https://codeberg.org/shiz/pydle.git
```

The default branch is `develop`. Feature branches use the repository's normal
`codex/` prefix. Each Backlog task maps to one reviewable fork PR. Chatbook task
records, ADR links, and plan notes are updated in a separate Chatbook docs
commit without copying fork source into this repository.

Before every task:

- [ ] Re-read its Backlog file and ADR-148.
- [ ] Confirm no existing branch, worktree, or PR already owns the task.
- [ ] Move only that task to In Progress and add its implementation plan before
   changing fork behavior.
- [ ] Rebase or update from the accepted prior task tag/branch.
- [ ] Derive the focused test list from the files and public behavior being
   changed.
- [ ] For an ad-hoc full local sweep, ask and obtain explicit opt-in. This does
   not suspend the reviewed unattended fork CI workflow.

## TASK-21601 — fork provenance and isolated package

### Target files in the fork

```text
UPSTREAM_BASE
NOTICE
FORK_GOVERNANCE.md
pyproject.toml
README.md
LICENSE.md
tldw_pydle/**
tests/**
scripts/check_namespace.py
tests/test_distribution_artifacts.py
```

Exact upstream filenames may differ. Preserve canonical files rather than
creating duplicates when a file already has the same responsibility.

### Steps

- [ ] Verify the approved SHA and v1.1.0 ancestor exist, then branch explicitly from `4efcc3b5096536668dfe772461f19a17f1ddd84e`. Develop moved to `e27b6a2138c81061c2e6a937526fceaedb94d31a` by 2026-09-10; this does not block starting from the pin. Audit the eight-commit delta separately before changing the base.
- [ ] Check for an existing fork first. If absent, an authorized owner creates GitHub empty, without a template. Clone full Codeberg history, retain Codeberg as `upstream`, add GitHub as `origin`, and create fork develop at the approved SHA. Push only that branch and explicitly reviewed provenance tags; never mirror-push unrelated refs.
- [ ] Make `develop` the GitHub default branch and protect it according to the
   available repository policy. Do not enable workflows until their permissions
   and action pins are reviewed.
- [ ] Add `UPSTREAM_BASE` containing the exact SHA, plus NOTICE and governance
   documentation. Preserve upstream `LICENSE.md` verbatim even if its text has
   an unusual placeholder.
- [ ] Commit the provenance/governance boundary before renaming any package.
- [ ] In a second mechanical commit, rename `pydle/` to `tldw_pydle/` and rewrite
   internal imports, tests, console scripts, examples, README commands, source
   URLs, issue URLs, and build metadata.
- [ ] Set distribution `tldw-pydle`, version `1.1.0.post1`, Python floor 3.11, and
   classifiers matching the tested range. Preserve existing SASL dependency metadata in this mechanical task; TASK-21605 removes pure-sasl and obsolete mechanisms when the replacement passes tests. Do not publish this intermediate package as production-ready.
- [ ] Add a namespace guard that inspects both source and built artifact. Explicit
   upstream attribution strings are allowlisted; runtime `import pydle`, a
   top-level `pydle/`, and upstream console-script names fail.
- [ ] Build wheel and sdist, inspect their members, install each into its own clean
   environment, co-install upstream pydle, and prove both import identities are
   distinct.
- [ ] Compare the namespace commit against the pinned upstream tree after
    reversing only path/import/metadata substitutions; any protocol behavior
    delta is removed or moved to a later task.

### Focused verification

```text
python -m pytest tests/test_distribution_artifacts.py -q
python scripts/check_namespace.py
python -m build
python -m ruff check tldw_pydle tests scripts
python -m ruff format --check tldw_pydle tests scripts
```

Artifact tests must open the wheel and sdist directly. Reading only
`pyproject.toml` is not packaging evidence.

### Commit boundaries

- [ ] `chore(fork): record canonical upstream provenance`
- [ ] `refactor(package): isolate tldw_pydle namespace`
- [ ] `test(package): verify built distribution identity`

## TASK-21602 — diagnostics and outbound transport

### Target files

```text
tldw_pydle/client.py
tldw_pydle/connection.py
tldw_pydle/protocol.py
tldw_pydle/diagnostics.py          # only if a narrow new owner is warranted
tests/test_diagnostics_privacy.py
tests/test_outbound_transport.py
tests/support/scripted_server.py
```

### Steps

- [ ] Inventory every logger call, warning, exception wrapper, repr, and test
   artifact reachable from send, receive, parse failure, authentication,
   timeout, and disconnect. Record which currently carries raw content.
- [ ] Replace raw-frame logging with a helper or fixed call pattern that accepts
   only opaque connection ID, command/capability name, numeric/reply code,
   encoded size, duration, and reason category. Never redact a fully formatted
   raw frame after the fact.
- [ ] Remove invalid-message warnings that interpolate `message._raw`. Convert
   exception logging to fixed exception type/category where the original
   message may contain transport data.
- [ ] Build synthetic fixtures containing a plaintext secret, its exact base64
   representation, channel key, private endpoint, nick/channel/target, topic,
   and message body. Capture every field and rendered logger output.
- [ ] Introduce one writer with a queue bounded by entries and encoded bytes. Keep FIFO order for admitted application commands and reserve protocol-control capacity for PONG/negotiation. Reject admission on exhaustion, without an unbounded waiter backlog; the reducer never waits for application queue capacity.
- [ ] Validate structured command parts and encoded bytes before acquiring the
   writer. Reject CR/LF/NUL and unsafe targets. Enforce protocol/tag budgets
   after encoding, not by character count.
- [ ] Review upstream commit `7c7f5c73bf14b8207a551a6cbf38627984cb7f4e` (PR 196 is now merged) and adapt its stalled-write fix with provenance to the pinned base, namespace and disconnect result. Add a deterministic
   writer/drain double that stalls without sleeping the whole test.
- [ ] Prove a timeout publishes exactly one failure/disconnect outcome and leaves
   no second writer or unobserved drain task.

### Focused verification

```text
python -m pytest tests/test_diagnostics_privacy.py tests/test_outbound_transport.py -q
python -m ruff check tldw_pydle tests/test_diagnostics_privacy.py tests/test_outbound_transport.py
```

Mutation checks:

- restore either raw debug call and see the privacy test fail;
- remove the outbound lock/queue and see concurrent ordering fail;
- remove the drain bound and see the stalled-writer test fail.

## TASK-21603 — lifecycle and secure TLS

### Target files

```text
tldw_pydle/client.py
tldw_pydle/connection.py
tldw_pydle/features/tls.py
tldw_pydle/lifecycle.py
tests/test_client_lifecycle.py
tests/integration/test_tls_transport.py
tests/support/certificates/**
```

Prefer generated test certificates from a deterministic test helper over
committing private-key material unless the repository has a reviewed fixture
policy. Any committed fixture key is synthetic, conspicuously named, and
excluded from secret scanners through a narrow documented rule.

### Steps

- [ ] Define and review exact typed lifecycle signatures, connection states and
   terminal close/failure results before dependent work starts. Keep state
   transitions monotonic and reject new work after closing begins.
- [ ] Replace implicit `RECONNECT_ON_ERROR` and delayed reconnect timers with
   emitted unexpected-disconnect state. Preserve an explicit compatibility
   note instead of silently reconnecting old subclasses.
- [ ] Own the reader, writer, timers, callbacks, and pending queries in one client
   lifecycle registry. Every creation site registers before scheduling.
- [ ] Implement connect cancellation and one-generation readiness storage.
   `wait_ready()` supports multiple awaiters observing one terminal result.
- [ ] Implement idempotent `aclose()` with cleanup owned independently of its
   waiters; cancelling one waiter cannot cancel shared cleanup. Block new work, fail/settle queries,
   best-effort bounded QUIT, close the writer, await `wait_closed()`, observe
   tasks with bounded `asyncio.wait`, cancel, observe again, and drain done
   exceptions.
- [ ] Add force-abort for a broken transport after graceful bounds. Do not use
   `wait_for(gather(...))` as the sole shutdown bound.
- [ ] For callbacks ignoring cancellation but still yielding, return typed incomplete close with opaque retained-task identity. Revoke public command/publication authority and use a fresh Client for a replacement connection. Callbacks are trusted Python, not a sandbox: code blocking the loop cannot have an in-loop time guarantee.
- [ ] Make the effective client-level TLS default verified. Use
   `ssl.PROTOCOL_TLS_CLIENT`, hostname verification, certificate-chain
   verification, SNI, and a typed caller-provided trust context.
- [ ] Build loopback TLS fixtures for matching trusted, hostname mismatch,
   untrusted chain, required-TLS/no-plaintext-fallback, cancellation, and
   writer close.

### Focused verification

```text
python -m pytest tests/test_client_lifecycle.py tests/integration/test_tls_transport.py -q
python -m ruff check tldw_pydle tests/test_client_lifecycle.py tests/integration/test_tls_transport.py
```

Mutation checks restore implicit reconnect, omit `wait_closed`, accept an
untrusted cert, allow plaintext fallback, and let an old callback publish after
close. Each mutation must make a named test fail.

## TASK-21604 — ordered reduction and correlation

### Target files

```text
tldw_pydle/client.py
tldw_pydle/features/**
tldw_pydle/reducer.py
tldw_pydle/correlation.py
tldw_pydle/events.py
tests/test_protocol_ordering.py
tests/test_query_correlation.py
tests/test_callback_delivery.py
tests/support/scripted_server.py
```

New modules are justified only if they create narrow ownership. Do not create
a generic framework that obscures pydle's existing feature composition.

### Steps

- [ ] Inventory every `on_raw_*`, `_sync_*`, user/channel mutation, query future,
   and public callback. Classify each as parser, ordered reducer, correlation
   completion, or post-commit application callback. Add a ratchet so new raw
   handlers cannot bypass classification.
- [ ] Replace one-detached-task-per-message dispatch with monotonic inbound
   envelopes and one ordered reduction path.
- [ ] Split reducer-safe internal state work from public/user callbacks. Complete
   state commit and correlation resolution before enqueueing the immutable
   event view.
- [ ] Ensure capability-selection hooks used during negotiation are configured
   synchronously before connection. Arbitrary async application work is not
   invoked from the reducer.
- [ ] Implement a bounded ordered callback/event queue. Define exact coalescing
   identity for presence/state events. Never coalesce chat messages or query
   terminal results.
- [ ] Full non-coalescible callback queues initiate one typed slow-consumer close; the independent lifecycle result settles queries. Never await callback capacity from the reducer: a callback may be awaiting WHOIS. Test that cycle at capacity. Coalescing removes the old event and appends the new state at its latest receive sequence.
- [ ] Add a correlation registry whose `prepare` step creates result storage and
   future before the writer can send. Convert WHOIS first and then every query
   included in the release.
- [ ] Reject ambiguous duplicate same-target/same-command operations where IRC
   replies cannot be distinguished. Do not claim labeled-response concurrency
   before it is implemented.
- [ ] Settle callers exactly once, but retain a bounded tombstone for abandoned unlabelled queries until their terminal wire reply or teardown. Reject correlation-key reuse during that interval. Use ISUPPORT CASEMAPPING for target keys, never generic Unicode casefold. Late replies cannot resolve replacement operations.
- [ ] Script back-to-back JOIN/NICK/WHOIS/BATCH-like frames while callbacks block
    or fail. Assert state and events exactly match receive order.

### Focused verification

```text
python -m pytest tests/test_protocol_ordering.py tests/test_query_correlation.py tests/test_callback_delivery.py -q
python -m ruff check tldw_pydle tests/test_protocol_ordering.py tests/test_query_correlation.py tests/test_callback_delivery.py
```

Mutation checks restore detached dispatch, move future creation after send,
resolve after callback, remove the queue bound, and permit an ambiguous
duplicate query.

## TASK-21605 — CAP 302 and SASL PLAIN

### Target files

```text
tldw_pydle/features/ircv3/cap.py
tldw_pydle/features/ircv3/sasl.py
tldw_pydle/features/ircv3/readiness.py
tests/test_cap302.py
tests/test_sasl_plain.py
tests/test_registration_readiness.py
tests/support/scripted_server.py
```

### Steps

- [ ] Replace the current capability dictionaries/sets with explicit offered,
   supported-policy, requested, pending, negotiated, and semantic states.
- [ ] Parse CAP 302 continuation form and capability values without lowercasing or
   otherwise changing opaque capability names.
- [ ] Accumulate every LS row before computing requests. Use a stable feature
   registration order and split REQ chunks by encoded line budget.
- [ ] Associate ACK/NAK with pending request chunks. Each REQ is accepted or
   rejected atomically; distinguish outcomes across separately sent chunks,
   not partial acceptance within one REQ. Process NEW/DEL after registration
   without re-running CAP END.
- [ ] Make CAP END an idempotent registration action that can occur exactly once
   after required capability/SASL policy resolves.
- [ ] Emit ready once on `001` only after mandatory SASL/capabilities succeed; refuse premature welcome otherwise. Allow legacy no-CAP registration when none is mandatory. MOTD never gates readiness; cancelling one waiter must not cancel the shared result.
- [ ] Implement PLAIN with standard-library base64 over
   `authzid NUL authcid NUL password`. Chunk encoded bytes into 400-byte
   AUTHENTICATE parameters and send `+` after an empty or exact-multiple final
   response.
- [ ] Gate PLAIN on the actual SSL transport and validated trust context, not a caller boolean. Reject embedded credential NULs and nonempty/malformed PLAIN challenges. Remove pure-sasl metadata/imports with passing replacement tests.
- [ ] Replace all SASL timers with lifecycle-owned handles/tasks, including the
   continuation path that currently passes a coroutine object to `call_later`.
- [ ] Minimize retained secret references and prove all negotiation terminal
    paths release them from client state and exclude them from diagnostics.

### Focused verification

```text
python -m pytest tests/test_cap302.py tests/test_sasl_plain.py tests/test_registration_readiness.py -q
python -m ruff check tldw_pydle tests/test_cap302.py tests/test_sasl_plain.py tests/test_registration_readiness.py
```

Tests include multiline LS/LIST form, values, trailing space, split requests,
ACK/NAK, NEW/DEL, one END, multiple readiness waiters, exact 400-byte boundary,
multiple chunks, malformed challenge, success/rejection/abort/timeout, missing
required SASL, and TLS refusal.

## TASK-21606 — bounded streaming LIST

### Target files

```text
tldw_pydle/features/rfc1459/list.py
tldw_pydle/models.py
tests/test_channel_list_stream.py
tests/support/scripted_server.py
```

Use the existing feature layout if it already has a clearer LIST owner. Do not
copy Codeberg PR 172's shared arrays or send-before-future ordering.

### Steps

- [ ] Define immutable `ListEntry` and `ListResult` models plus an async LIST
   session/iterator. Keep raw numerics private.
- [ ] Register the session and bounded queue before sending LIST. Reject a second
   session while the first response remains ambiguous.
- [ ] Parse `321`, `322`, `323`, documented error numerics, standard replies, and
   TRYAGAIN into exact session transitions.
- [ ] Count received, delivered, and dropped rows independently. Validate user
   counts and preserve bounded topic content for the caller without logging it.
- [ ] On local limit or cancellation, stop caller delivery and enter drain mode.
   Continue reducing unrelated frames while discarding LIST rows through 323.
- [ ] Bound drain duration. Missing 323 settles the caller and releases buffered rows but retains one bounded LIST tombstone. Reject another LIST until 323 or teardown; unrelated chat remains usable. Terminal counters snapshot caller settlement, not later discarded rows.
- [ ] Accept server-side ELIST filters only when supported; document that the
   iterator is local flow control, not universal IRC pagination.

### Focused verification

```text
python -m pytest tests/test_channel_list_stream.py -q
python -m ruff check tldw_pydle tests/test_channel_list_stream.py
```

Mutation checks make the queue unbounded, create state after send, stop
reducing unrelated chat during drain, and omit the drain deadline.

## TASK-21607 — IRCv3 messages and history

### Target files

```text
tldw_pydle/features/ircv3/tags.py
tldw_pydle/features/ircv3/server_time.py
tldw_pydle/features/ircv3/batch.py
tldw_pydle/features/ircv3/chathistory.py
tldw_pydle/models.py
tests/test_message_tags.py
tests/test_server_time_and_echo.py
tests/test_batch.py
tests/test_chathistory.py
```

### Steps

- [ ] Implement tag escaping/unescaping and the IRCv3 byte budgets, including the
   separate tag and unchanged message portions. Reject or classify oversized
   input before it reaches semantic handlers.
- [ ] Retain bounded unknown tags as untrusted protocol metadata. Provide
   immutable views and prevent them from becoming log fields automatically.
- [ ] Parse server-time into a validated timezone-aware value. Preserve receive
   time separately for ordering and fallback.
- [ ] Preserve authoritative server echoes and actual correlation metadata. A write is not delivery acknowledgement. Without labels/usable identity, never merge by text or time: identical legitimate messages remain distinct. Local optimistic-send state stays explicitly unconfirmed.
- [ ] Implement bounded BATCH state with ID, type, parameters, parent, open/close order and member sequence. History cannot mutate live membership/topic state. Do not negotiate event-playback without separate semantics/tests. Support unknown batch types and honest
   incomplete termination.
- [ ] Implement negotiated `draft/chathistory` commands and correlated `chathistory`
   batches. Honor the server's CHATHISTORY/MSGREFTYPES declarations where
   present and preserve timestamp/msgid anchors.
- [ ] Validate server over-return, mismatched batch type/target, empty batch,
   standard replies, error numerics, cancellation, timeout, and disconnect.
- [ ] Return immutable history results. Leave paging policy, cross-page
   deduplication, transcript retention, unread state, and persistence to the
   embedding adapter/application.

### Focused verification

```text
python -m pytest tests/test_message_tags.py tests/test_server_time_and_echo.py tests/test_batch.py tests/test_chathistory.py -q
python -m ruff check tldw_pydle tests/test_message_tags.py tests/test_server_time_and_echo.py tests/test_batch.py tests/test_chathistory.py
```

Mutation checks remove a tag bound, scramble batch member order, accept a
mismatched history batch, and merge receive-time with server-time authority.

## TASK-21608 — compatibility gate and release

### Target files

```text
.github/workflows/ci.yml
.github/workflows/compatibility.yml
.github/workflows/release.yml
tests/integration/test_deterministic_oracle.py
tests/integration/test_agentirc.py
tests/integration/test_ergo.py
tests/integration/test_release_artifact.py
scripts/download_pinned_ergo.py
scripts/check_release_artifacts.py
CHANGELOG.md
```

### Steps

- [ ] Run Python 3.11–3.14 on Linux, plus 3.12 transport/TLS/packaging on Windows/macOS. Verify installer availability before patch-pinning interpreters. Pin every third-party action by full commit
   SHA with a nearby version comment. Use minimal top-level and job-level
   permissions.
- [ ] Ensure pull-request workflows use neither publishing credentials nor
   `pull_request_target` execution of untrusted code.
- [ ] Build wheel/sdist once in the release-artifact job. Downstream jobs download
   that artifact rather than rebuilding or editable-installing the checkout.
- [ ] Port the deterministic scripted oracle to the final public API and assert
   registration, CAP, SASL, PING/PONG, JOIN/channel/DM, WHOIS, LIST, tags,
   history, disconnect, and lifecycle outcomes.
- [ ] Pin an exact AgentIRC release or commit and an exact Ergo release. Verify the
   Ergo archive checksum before execution. Run two-client chat and cleanup
   against both using only synthetic fixture content.
- [ ] Prove normal/cooperative zero-task shutdown on every target and the explicit
   incomplete outcome for a cancellation-resistant application callback.
- [ ] Inspect and independently install wheel and sdist on every supported Python.
   Assert namespace, metadata, Python floor, license/notice, upstream base, and
   patch inventory from the artifacts themselves.
- [ ] Generate SHA-256 checksums and a supported provenance/SBOM artifact. Release
   notes include exact upstream base and per-commit patch classification.
- [ ] Configure PyPI Trusted Publishing without storing a long-lived PyPI token.
   Publish only from a reviewed `v1.1.0.post1` tag and approved environment.
- [ ] Add a non-release-blocking scheduled job for latest stable upstream and
    Ergo. It files or reports drift; it does not alter the pinned release gate.
- [ ] Execute fork CI gates through the reviewed workflow, without requiring an absent author's interactive prompt. Agents ask before ad-hoc full local sweeps. Never run the full Chatbook suite for fork-only changes.
- [ ] If external publishing is unavailable, preserve verified local artifacts
    and record the exact blocker. Do not mark publication AC complete or call
    the task Done.

### Required release evidence

```text
Python 3.11–3.14 CI result
deterministic oracle result against built wheel
loopback TLS matrix result
AgentIRC exact version/commit and two-client result
Ergo version, download URL, SHA-256 and two-client result
wheel and sdist filenames plus SHA-256
artifact member/metadata report
owned task/timer shutdown report
privacy-log synthetic-secret report
SBOM/provenance locator
GitHub release and PyPI project/version locators
```

## Cross-task test-support rules

- The scripted server is test-only and implements only the exact frames needed
  by each case. It is not a second IRC server implementation.
- Test events/barriers establish ordering; fixed sleeps are not race evidence.
- Timeouts distinguish operation deadlines from shutdown observation bounds.
- Cancellation tests trigger cancellation after the named lifecycle state is
  observed, not before the response or callback begins.
- Privacy tests inspect every default logger/sink and rendered field, not just
  captured wire bytes.
- Async test helpers store content-only exception type/category rather than a
  traceback retaining live transports.
- Packaging assertions read wheel/sdist members and installed metadata.
- Live target fixtures use synthetic identities/content and bind loopback only.
- Every temporary server/process/path is recorded and cleaned after exact-path
  verification; failed cleanup is reported.

## Documentation and closeout per task

Before a task is marked Done:

- [ ] Check every acceptance criterion only when its evidence exists.
- [ ] Add concise Implementation Notes naming approach, trade-offs, files, tests,
   artifact/commit IDs, and deviations.
- [ ] Record ADR required/path/reason in the task's implementation plan and notes.
- [ ] Run targeted tests and Ruff for touched modules; record exact commands and
   results.
- [ ] Self-review the task diff and compare behavior-sensitive tests against the
   prior task base where useful.
- [ ] Update fork README/governance/changelog when the public contract changes.
- [ ] Add a lessons entry only when an incident reveals reusable evidence not
   already captured by the existing testing/live/backlog lessons.
- [ ] Re-scan task and ADR IDs against current refs/worktrees before merging
   Chatbook governance changes.

## Definition of the first release boundary

The programme is complete only when TASK-21608 publishes externally retrievable artifacts matching the tested hashes. A publication blocker leaves the programme incomplete even when local artifacts pass. Completion does **not** add the dependency to
Chatbook. The next separately designed Chatbook tranche may then file tasks for
typed models/fake adapter, application session manager, private profiles and
credentials, the Network Chat vertical slice, and the real `tldw-pydle`
adapter pinned to the released artifact.
