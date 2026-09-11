# Network Chat IRCv3 and tldw-pydle design

Status: Approved baseline; ADR-149 amendments proposed for review
Reassessed: 2026-09-10
Start here: [PTO handoff](../plans/2026-09-10-network-chat-pto-handoff.md)
Amendment: [ADR-149](../../../backlog/decisions/149-network-chat-handoff-reliability-amendments.md)
Date: 2026-08-23
Decision: [ADR-148](../../../backlog/decisions/148-network-chat-ircv3-and-tldw-pydle-boundary.md)
Implementation: [First-release plan](../plans/2026-08-23-tldw-pydle-first-release.md); TASK-21601 through TASK-21608 are filed and remain To Do

## Summary

Chatbook will gain a lightweight, keyboard-first **Network Chat** screen for
ordinary IRC networks, AgentIRC, and a future IRC service explicitly advertised
by tldw_server. The screen supports connection profiles, channels, direct
messages, channel browsing, favorites, presence, unread and mention state,
common commands, and capability-gated modern IRCv3 behavior.

The IRC client core will be a separately released, tldw-maintained fork of
pydle. The fork remains behind a protocol-neutral Chatbook adapter. It is not
vendored into Chatbook, does not expose mutable pydle state to Textual, and does
not make the Network Chat screen responsible for socket lifetime.

The critical review changed the fork from a bounded collection of feature
patches into a small reliability refactor. Canonical pydle currently logs raw
frames, dispatches inbound messages concurrently, owns reconnect implicitly,
does not fully close its transport, and has incomplete CAP, correlation, SASL,
LIST, and resource-bound semantics. Those defects must be corrected coherently
before the client can be called a stable shared dependency.

## Design outcomes already resolved

- Primary fork repository: `github.com/rmusser01/tldw-pydle`.
- Canonical upstream: `https://codeberg.org/shiz/pydle`.
- Upstream fork point: `develop` commit
  `4efcc3b5096536668dfe772461f19a17f1ddd84e`, verified unchanged on
  2026-08-23.
- Upstream release ancestor: pydle v1.1.0 tag
  `cbc18112e141d357abfb5828262cd98d65628096`.
- Distribution: `tldw-pydle`.
- Import package: `tldw_pydle`.
- Initial version: `1.1.0.post1`.
- Default branch: `develop`.
- Fork Python floor: 3.11; current Chatbook independently requires 3.12+.
- Full upstream Git history and BSD-3-Clause text are preserved.
- Reconnect is owned by Chatbook or another embedding application, not the
  library.
- SASL PLAIN is standard-library code and requires verified TLS; `pure-sasl`
  is not required.
- The first release does not expose SASL EXTERNAL or a certificate credential
  profile.
- Local transcripts remain memory-only by default.

## Goals

- Produce a stable IRCv3 client artifact that can be shared by Chatbook and
  tldw components with client roles.
- Preserve protocol state in server wire order.
- Make every internal task, timer, query, write, and transport close observable
  and bounded.
- Keep authentication material and message content out of ordinary logs.
- Support direct verified TLS, CAP 302, SASL PLAIN, core IRC, message tags,
  server-time, echo-message, BATCH, CHATHISTORY, and channel LIST.
- Offer incremental channel browsing without unbounded accumulation.
- Keep Chatbook's UI, persistence, reconnect, and service-discovery decisions
  outside the fork.
- Make upstream synchronization and downstream patch provenance routine rather
  than archaeological.
- Gate releases on the built artifact, not only an editable checkout.

## Non-goals for the first fork release

- Implementing an IRC server.
- Becoming an IRC bouncer.
- DCC chat or file transfer.
- STARTTLS negotiation or opportunistic plaintext-to-TLS upgrade.
- SASL mechanisms other than PLAIN.
- Client-certificate profile UX.
- Automatic reconnection.
- A scripting or plugin runtime.
- Chatbook profiles, favorites, unread state, transcript persistence, or UI.
- AgentIRC-specific semantics in the generic core.
- Guessing a tldw_server IRC endpoint from an HTTP URL.
- Supporting Python 3.10.

## System ownership

| Concern | Owner | Lifetime | Persistence |
| --- | --- | --- | --- |
| Socket, TLS stream, parser, protocol state, internal tasks and timers | `tldw_pydle` client | One connection | None |
| Ordered protocol event and callback delivery | `tldw_pydle` client | One connection | None |
| Connection generation, reconnect policy and backoff | Chatbook `IrcSessionManager` | Application | None |
| Stable typed events and command results | Chatbook IRC adapter | Application contract | None |
| Bounded recent messages, presence, unread and mentions | Chatbook `IrcBufferStore` | Application session | Memory only |
| Non-secret profiles, favorites and autojoin choices | Chatbook profile repository | Application | Private local store |
| Passwords, tokens, channel keys and certificate material | Chatbook credential provider | Lookup scoped | Environment or credential store |
| Selected network/buffer, rail state and scroll anchor | Network Chat screen | Screen visit | Existing memory-only screen snapshot |
| tldw_server endpoint and authentication metadata | Advertised service descriptor | Active server binding | Existing server authority |

Neither a `tldw_pydle.Client`, its callback object, a mutable users/channels
mapping, nor a raw IRC message crosses the Chatbook adapter boundary.

## Runtime architecture

```text
TCP/TLS reader
    -> bounded byte framing and parsing
    -> ordered protocol reducer
         -> internal state commit
         -> resolve correlated futures
         -> emit immutable internal event
    -> bounded ordered callback/event delivery

application command
    -> validate and encode within byte budget
    -> register correlation before send when required
    -> serialized outbound writer
    -> bounded drain
```

### Why the split is required

Canonical pydle currently creates one task per parsed message. Those tasks can
interleave JOIN, NICK, CAP, WHOIS, BATCH, and history mutations contrary to wire
order. Keeping the task handles would make shutdown more visible but would not
make state deterministic.

Awaiting each existing handler in full is also unsafe. Some handlers issue and
await WHOIS or WHOX while later inbound numerics are responsible for completing
those futures. A globally sequential callback pipeline would wait for a reply
that it has prevented itself from processing.

The reducer therefore performs only bounded protocol work. It may create or
resolve correlation state, update client state, and produce an internal event.
It may not await a future that depends on later wire input or run arbitrary
application code.

## Ordered inbound processing

The connection owns one reader task and one ordered reducer task. An
implementation may combine them when doing so preserves the same ownership and
test seams. Every complete frame receives a monotonic receive sequence.

The reducer applies valid frames in sequence order. Core state changes and
correlated-result updates are complete before an event is offered to callbacks.
Malformed frames produce classified protocol events. A malformed optional
message does not silently corrupt state, and a fatal framing condition closes
the connection with an observable reason.

Feature modules register synchronous capability and reducer hooks. Capability
selection is configured before connection and cannot depend on arbitrary async
application callbacks during CAP negotiation. A reducer hook that needs later
wire input creates correlation state and returns; it never waits inline.

### Callback delivery

Upstream-style async callbacks remain available for compatibility inside the
fork namespace, but they run after state commit through one owned ordered
delivery queue. Chatbook uses the adapter rather than treating those callbacks
as its stable API.

Callback exceptions are captured as content-free diagnostics and exposed
through a defined callback-error event or close result. They do not kill the
reader task, disappear as “Task exception was never retrieved,” or roll back
already committed protocol state.

The queue is bounded. Coalescible presence/state events may replace an older
unconsumed event for the same identity. Message and query-result events are
never silently discarded. If a consumer remains too slow for the bounded
queue, the client closes with a visible slow-consumer outcome without awaiting callback capacity from the reducer. Otherwise a callback awaiting a query can block its own reply. Coalescing removes the old event and appends new state at the latest receive sequence.

## Correlated protocol operations

Every query has an explicit correlation owner and terminal states. Pending
state is published before its outbound command can be observed by the server.
The reducer resolves or fails it on the corresponding terminal numeric,
standard reply, disconnect, timeout, or cancellation.

First-release correlated operations include:

- WHOIS;
- channel LIST;
- capability requests;
- SASL negotiation;
- registration readiness;
- CHATHISTORY BATCH assembly.

Only one operation is allowed when IRC cannot reliably disambiguate concurrent
responses, including LIST and same-target WHOIS where labels are unavailable.
Labeled-response support may permit broader concurrency later, but it is not
assumed.

Timeout/cancellation settles the caller but retains a bounded tombstone for an unlabelled operation until its terminal reply or teardown. Same-key reuse is refused; negotiated IRC CASEMAPPING defines identity. All correlation futures are failed before connection state is discarded.
Cancellation remains distinguishable from server rejection, disconnect,
timeout, truncation, and malformed response.

## Outbound transport

One connection owns one writer and a queue bounded by entries and encoded bytes. Admitted application commands are FIFO; protocol-control traffic has reserved capacity so user pressure cannot starve PONG. Queue admission cannot create unbounded waiters. Registration, CAP, SASL,
queries, PONG, and application messages cannot call `write()`/`drain()`
concurrently.

The writer:

- accepts structured commands rather than raw untrusted line strings at its
  public seam;
- rejects CR, LF, NUL, and invalid target/control input without silent repair;
- budgets encoded bytes after applying tags and encoding;
- splits only operations whose protocol semantics define safe splitting;
- refuses messages that cannot be represented safely;
- uses a configurable bounded `drain()` timeout;
- turns stalled writes into one unexpected-disconnect outcome;
- never logs the encoded frame.

QUIT is best-effort within the close budget. Failure to send QUIT does not
prevent transport teardown.

## Lifecycle contract

The preferred lifecycle is usable directly and as an async context manager:

```text
client = Client(...)
await client.connect(host, port, tls=...)
ready = await client.wait_ready()
...
result = await client.aclose(reason="application shutdown")
```

TASK-21603 must deliver and review the exact typed lifecycle signatures and
result models before dependent work starts. The following behaviors are stable:

- `connect()` establishes one connection generation and returns only after the
  transport is established or fails.
- `wait_ready()` is idempotent within one generation: all awaiters observe the
  same readiness or terminal negotiation result, while the readiness event and
  state transition occur exactly once after numeric `001`.
- `aclose()` is idempotent and safe from connected, connecting, negotiating,
  failed, or already closed states.
- close prevents new commands, settles pending operations, attempts QUIT when
  appropriate, closes the writer, awaits `wait_closed()`, cancels/drains all
  owned tasks and timers, and returns a structured outcome.
- an explicit force-abort path exists for an uncooperative or broken transport
  after the graceful budget expires.
- the library never schedules reconnect.

### Close bounds

Shutdown cannot rely on `asyncio.wait_for(gather(...))` as its only bound,
because a task that swallows cancellation can keep `wait_for` suspended. The
close algorithm first requests cooperative shutdown, observes tasks with
`asyncio.wait()` for bounded intervals, cancels remaining tasks, observes again,
and reports any retained stragglers before aborting the transport. Every done
task has its exception drained.

Python cannot forcibly terminate arbitrary async application code that catches
`CancelledError` and continues forever. A normal or cooperative close therefore
proves zero client-owned tasks. If callback code remains uncooperative after the
close budget while still yielding to the loop, `aclose()` returns a bounded `incomplete` outcome identifying the
retained callback task; all reader, reducer, writer, query and timer work is
settled, the transport is aborted, and the retained callback has no authority
to publish further protocol state. The client retains observability until that
task actually exits and never reports a clean close prematurely.

Callbacks are trusted Python code, not a sandbox. No in-process guarantee can stop code blocking the event loop. Only public command/publication authority is fenced; replacement connections use fresh Client objects.

Release tests include deliberately cancellation-resistant callback work so
both the time bound and the honest incomplete outcome are measured rather than
inferred.

## Connection states

| State | Meaning | Permitted next actions |
| --- | --- | --- |
| `idle` | No generation has started | Connect, close |
| `connecting` | TCP/TLS establishment | Cancel, close |
| `negotiating` | CAP, SASL and registration | Cancel, close |
| `ready` | Numeric `001` received | IRC commands, close |
| `closing` | New work refused; drain underway | Await close |
| `closed` | Transport and owned work settled | Inspect result, create a new generation through the embedding owner |
| `failed` | Generation ended unexpectedly | Inspect failure, close/settle |

Readiness is emitted once per generation. MOTD `376`/`422` may enrich startup
state but does not gate ready, because valid servers such as AgentIRC may not
send either numeric.

## Privacy-safe diagnostics

Raw inbound and outbound frame logging is removed from the default and debug
paths. Log records are created from fixed fields rather than by redacting a
previously formatted frame.

Allowed examples:

- connection ID and generation;
- direction and IRC command name;
- capability name and transition;
- numeric or standard-reply code;
- encoded byte count;
- elapsed duration and timeout category;
- task name and content-free exception type;
- close reason category.

Forbidden examples:

- raw lines or message tags as received;
- hostnames, endpoints, nicknames, accounts, channel names or targets;
- PRIVMSG, NOTICE, topic, real-name, or CTCP bodies;
- PASS and AUTHENTICATE parameters;
- plaintext or base64-encoded synthetic secrets;
- channel keys, OAuth or server tokens;
- certificate private-key bytes or passwords;
- exception messages containing server or transport payload text.

Tests attach a real in-memory handler to every library logger and inspect every
record field, rendered string, exception representation, close result, and
artifact emitted by default test/release paths. Synthetic secrets include
plain and encoded variants. The release also scans built artifacts and example
configuration for insecure or real-looking credentials.

An explicit protocol trace, if added later, requires a separate ephemeral
diagnostic design. It cannot be implemented by turning raw debug logging back
on.

## TLS and authentication

### TLS

First release supports direct TLS and explicit plaintext connections. Chatbook
profiles default to direct TLS and do not expose certificate-verification
disablement as an ordinary recovery action.

The client uses `ssl.PROTOCOL_TLS_CLIENT`, certificate-chain verification,
hostname verification, and server-name indication. It accepts a caller-provided
CA context/reference through a typed boundary. Certificate failure is terminal
for that attempt and never triggers a plaintext retry.

Loopback integration tests use generated test certificates and cover:

- trusted certificate plus matching hostname succeeds;
- trusted certificate plus hostname mismatch fails;
- untrusted certificate fails;
- TLS-required never falls back to plaintext;
- close awaits or aborts the TLS writer within its bound.

### SASL PLAIN

SASL PLAIN is implemented directly:

```text
base64(authzid NUL authcid NUL password)
```

The encoded response follows IRC's 400-byte AUTHENTICATE chunks and sends the
required `+` terminator when the encoded length is an exact multiple of 400.
The implementation accepts bytes at the narrowest possible boundary, avoids
copying secrets into exception or repr output, and releases credential
references after negotiation finishes.

PLAIN is refused unless the current transport is verified TLS. Any future
diagnostic override belongs outside ordinary profile configuration and is not
part of this release.

All SASL timers are owned handles or tasks. The existing upstream continuation
timer bug—passing an already-created coroutine into `call_later()`—is replaced
and regression-tested with a multi-frame challenge and timeout path.

## CAP 302 state machine

The capability model tracks:

- offered capabilities and values;
- application-supported capability policy;
- requested capability names (values belong only to LS/NEW);
- pending request chunks;
- negotiated enabled capabilities;
- semantically supported capabilities exposed by the adapter.

Registration behavior:

1. Send CAP LS 302.
2. Accumulate every continuation row.
3. Compute a deterministic request order from registered feature policy.
4. Split CAP REQ into encoded-line-safe chunks.
5. Process ACK/NAK per request chunk; continuation grammar belongs only to LS/LIST, and values only to LS/NEW.
6. Complete mandatory SASL where configured.
7. Send CAP END exactly once.
8. Emit ready once on `001` after required authentication/capabilities succeed. Allow legacy registration without CAP if none is mandatory. One cancelled waiter cannot cancel the shared result.

After registration, CAP NEW and DEL update offered and enabled state without
sending another registration CAP END. Capability loss cancels or disables only
dependent operations and produces a typed capability-change event.

Capability hooks do not execute arbitrary async application work inside the
ordered reducer. Application policy is registered before connection.

## Message tags, server-time, BATCH and history

Tag parsing follows IRCv3 escaping rules and rejects control or size violations
at the parser boundary. Unknown tags are retained internally within resource
bounds but are not interpreted as trusted application metadata.

`server-time` is parsed into a timezone-aware timestamp. The original raw value
may remain in the bounded internal message model for diagnostics-free protocol
semantics, but ordinary logs never include it with the message body.

BATCH tracks batch ID, type, parameters, parent relation, open/close sequence,
and bounded member identity. Unknown batch types remain representable. Closing
or disconnecting fails incomplete correlated batches honestly.

CHATHISTORY preserves:

- target;
- request anchor and limit;
- BATCH identity;
- server time;
- message ID when present;
- sender, target and message type;
- live-versus-history origin.

History must not mutate live membership/topic state; event-playback support is not implied. Echoes remain authoritative server events, but text/time similarity is not identity and cannot collapse legitimate repeats. A write is not a delivery receipt.

The fork assembles one correlated history result. Chatbook owns display
pagination, buffer retention, cross-page deduplication, and transcript anchors.

## Streaming channel LIST

IRC LIST usually has no true server-side cancellation or universal pagination.
The API therefore models what the client can guarantee rather than pretending
that cancelling a Python future stops the server.

One connection may own one LIST session. The session exposes an asynchronous
stream of:

```text
ListEntry(channel, user_count, topic)
```

and a terminal result containing:

```text
received_count
delivered_count
dropped_count
truncated
cancelled
completion_numeric
error
```

Entries pass through a bounded queue. A caller may set a local delivery limit.
When the consumer cancels or reaches that limit, the client stops delivering
rows and drains/discards remaining LIST replies through numeric `323` while
continuing to process unrelated protocol frames. A drain deadline prevents a
malicious or broken server from holding the caller forever. One bounded LIST tombstone remains until 323 or teardown; another LIST is refused while unrelated chat continues. Terminal counters snapshot caller settlement.

Where ISUPPORT advertises compatible ELIST filters, the adapter may submit a
server-side filter. The API does not claim universal pagination.

LIST tests cover `321`, `322`, `323`, empty results, malformed counts, error
numerics, `TRYAGAIN`, local cancellation, consumer backpressure, disconnect,
and a server that never sends `323`.

## Resource bounds

The fork defines and tests explicit bounds for:

- total inbound framed line bytes;
- IRCv3 tag-section bytes;
- incomplete receive-buffer bytes;
- outbound encoded command bytes;
- SASL challenge/response bytes;
- open BATCH count and members;
- pending correlation count and per-query results;
- LIST queue and total delivered rows;
- callback/event queue;
- close and write-drain duration;
- content-free diagnostic collection.

Protocol limits are counted after encoding, not as Python characters. Defaults
are standards-compatible and have documented absolute safety ceilings. The
implementation plan will pin the exact constants after tests against Ergo and
AgentIRC; the design does not authorize an unbounded “server decides” mode.

An over-limit condition is either a refused outbound operation, a truncated
query with an explicit result, a skipped non-critical message with a visible
gap, or a fatal protocol failure. It is never silent truncation of credentials,
targets, or chat text.

## Fork public compatibility surface

The namespace change makes `tldw-pydle` a distinct distribution, so byte-for-
byte upstream import compatibility is not promised. Within `tldw_pydle`, the
fork retains familiar high-level commands and callback names where they do not
conflict with the reliability architecture.

The stable first-release surface consists of:

- client construction and feature policy;
- explicit lifecycle methods and result types;
- core IRC send/join/part/nick/topic/away commands;
- correlated WHOIS, LIST, and CHATHISTORY operations;
- immutable internal message/event views exposed to adapters;
- capability and connection snapshots;
- typed error, timeout, cancellation, truncation and close outcomes.

Mutable `users`, `channels`, pending dictionaries, raw messages, task sets, and
transport writers are implementation details. Chatbook does not depend on
them even if compatibility access remains available to fork subclasses.

## Namespace migration and upstream synchronization

The fork is created by cloning Codeberg with full history and adding GitHub as
the primary remote while retaining a read-only `upstream` remote. The initial
history is not squashed.

The first downstream commits are deliberately mechanical:

1. Record `UPSTREAM_BASE` and add fork notice/governance files.
2. Rename the package and rewrite imports, entry points, examples, metadata,
   and tests from `pydle` to `tldw_pydle`.
3. Add guards proving the built wheel has no top-level `pydle` package and no
   stale import or script surface.

Behavioral changes begin only after the namespace commit. Generic fixes are
kept separate from tldw lifecycle policy and feature work.

For upstream contribution, a generic fix may be prepared on a clean branch
against the upstream namespace and then represented by a provenance-linked
downstream commit. Downstream release notes map every patch to one of:

- upstream correctness/security;
- adopted open-PR provenance;
- tldw lifecycle policy;
- tldw IRCv3 feature semantics;
- packaging/release infrastructure.

An upstream synchronization records old base, new base, conflicts, retained
patches, dropped patches already absorbed upstream, and the complete gate
result. Opaque squash replacement is prohibited.

## License and notices

The upstream BSD-3-Clause license is retained verbatim, including any unusual
placeholder text in the canonical source. The fork does not silently “repair”
upstream's license wording. A separate NOTICE identifies tldw modifications,
the exact upstream repository and base, and adopted pull-request provenance.

Wheel and sdist tests inspect the actual artifacts for LICENSE, NOTICE,
upstream-base metadata, package namespace, and source URLs.

## CI and release design

### Pull-request CI

- Python 3.11–3.14 on Linux; Python 3.12 transport/TLS/packaging on Windows/macOS.
- Unit and deterministic protocol-oracle tests.
- Real loopback TLS tests.
- Namespace, license, notice, secret-log, resource-bound, and task-leak guards.
- Ruff format and lint checks.
- Wheel and sdist build plus member inspection.
- Clean-environment install and import from each artifact.
- No publishing credentials or privileged workflow trigger.

### Compatibility CI

- AgentIRC pinned to an exact release or commit.
- Ergo pinned to an exact release and checksum.
- Two-client registration, channel chat, direct chat, PING/PONG, tags,
  disconnect and server cleanup.
- CAP/SASL/history coverage where the target supports it.
- The tested artifact is the built wheel, not an editable working tree.

### Scheduled compatibility

- Latest stable Codeberg pydle audit.
- Latest stable Ergo compatibility.
- Dependency and security-advisory review.
- Results create maintenance work but do not rewrite the historical result of
  a pinned release gate.

### Release job

- Build once from a reviewed tag.
- Verify wheel/sdist contents and install both independently.
- Generate SHA-256 checksums and provenance/SBOM data.
- Publish a GitHub release with upstream base and patch inventory.
- Publish to PyPI through Trusted Publishing when configured.
- Use minimal job permissions and actions pinned by full commit SHA.

Chatbook consumes an exact released version in an optional IRC dependency
group only after the artifact passes deterministic, TLS, AgentIRC, and pinned-
Ergo gates.

## Verification matrix

| Area | Required evidence |
| --- | --- |
| Ordering | Scripted server sends state-dependent frames back-to-back; reducer state and emitted events match wire order under delayed/failing callbacks. |
| Correlation | Reply can arrive immediately after command write; future/result existed first and resolves once. Parallel unsupported queries are rejected. |
| Lifecycle | Connect cancel, negotiation cancel, graceful close, transport reset, stalled write and callback failure settle within bounds with zero owned work. A cancellation-resistant callback returns a bounded incomplete outcome naming the retained callback while protocol work reaches zero and late state publication is rejected. |
| Privacy | Plain and encoded SASL secrets, PASS, channel key, chat body and topic are absent from every captured log field, exception representation and default artifact. |
| TLS | Matching trusted cert passes; hostname mismatch and untrusted cert fail; no plaintext fallback. |
| SASL | Success, rejection, timeout, exact-400 chunk terminator, multiple chunks, malformed base64 and disconnect. |
| CAP | Multiline LS, values, deterministic split REQ, ACK/NAK, NEW/DEL, CAP END once, ready on 001 once. |
| LIST | Incremental rows, local limit, cancellation/drain, pressure, no 323, error numerics and disconnect. |
| BATCH/history | Nested/unknown batches, valid history, empty history, incomplete close, time/msgid preservation and bounds. |
| Parser bounds | Oversized line/tag/buffer and malformed escape paths produce classified outcomes without growth or content logs. |
| Packaging | Built wheel/sdist contain only `tldw_pydle`, required license/notice/provenance, and install cleanly on every supported Python. |
| Compatibility | The same built artifact passes deterministic server, AgentIRC and pinned Ergo targets. |

Earlier unpublished spike evidence remains useful as feasibility evidence: the
canonical Codeberg checkout reported 53 upstream tests passing; the isolated
selection suite reported 9 passing with 2 opt-in live skips; and the explicit
Ergo 2.19.0 run reported 2 passing. Those editable-checkout results do not
satisfy the release gate. The release work must reproduce them against the
built artifact.

Per repository policy, implementation will use targeted verification while
work is in progress. A full Chatbook suite is not part of fork work and will be
run only if the user explicitly requests it before a Chatbook integration PR.

## Chatbook adapter boundary

The Chatbook adapter exposes behavior such as:

```text
connect(profile_id)
disconnect(network_id, reason)
join(network_id, channel, key_ref=None)
part(network_id, channel, reason=None)
send_message(network_id, target, text)
send_notice(network_id, target, text)
send_action(network_id, target, text)
set_nick(network_id, nick)
set_away(network_id, message=None)
set_topic(network_id, channel, topic)
request_whois(network_id, nick)
stream_channels(network_id, filter=None, limit=None)
request_history(network_id, target, anchor, limit)
capability_snapshot(network_id)
shutdown()
```

Every operation returns or emits typed Chatbook models. Errors preserve a
stable reason category, user-safe recovery copy, and an optional numeric or
standard-reply code. Raw IRC frames and fork exception messages do not cross
into persistent UI diagnostics.

Connection events carry a monotonic generation. An event from an old
generation cannot mutate a replacement connection after explicit disconnect
or reconnect.

## Chatbook session manager

One application-owned manager is composed by `TldwCli` after its dependencies
are ready and closed during application shutdown. Navigating away from Network Chat detaches or pauses view subscriptions; connections continue. Current routes can be reusable: treat resume/suspend and final unmount separately, resubscribe once with a fresh snapshot, and test both reuse and fresh-screen lifecycles. Initially use the default non-reusable route unless a reuse audit passes.

The manager owns:

- selected profiles and active adapter instances;
- connection generation;
- capped reconnect backoff and jitter;
- explicit-disconnect suppression;
- blocked-versus-retryable failure classification;
- bounded recent buffer and presence state;
- unread, mention and visible history-gap state;
- subscription pressure and screen snapshot delivery;
- complete application shutdown of every adapter.

Authentication rejection, certificate failure, invalid profile and missing
credential enter a blocked state rather than retrying indefinitely. The
Network Chat UI always identifies the problem owner and recovery action.

## Network profiles and credentials

A non-secret profile contains:

- stable ID and display name;
- host, port and direct-TLS requirement;
- verification requirement and optional CA reference;
- nickname, alternate nicknames, username and real name;
- SASL mechanism and credential reference;
- favorite and ordered autojoin channels;
- channel-key credential references;
- reconnect preference;
- UTF-8 encoding policy;
- source authority: manual, server-advertised or built-in example.

Resolved passwords, tokens and keys never enter the profile record, events,
screen snapshots, logs or exception messages. Removing a live profile requires
disconnect confirmation. Parting a channel does not remove its favorite, and
removing a favorite does not part the channel.

## Network Chat screen

Network Chat is a canonical `irc` route associated with the Console
destination. A visible mode strip navigates between Agent Console and Network
Chat; it is real routing, not an in-place widget swap.

The screen follows the Workbench decomposition:

```text
tldw_chatbook/
  IRC/
    models.py
    events.py
    adapter.py
    session_manager.py
    buffer_store.py
    profile_repository.py
    credential_provider.py
    adapters/tldw_pydle_adapter.py
  UI/Screens/irc_screen.py
  UI/IRC_Modules/
    network_rail.py
    transcript_region.py
    member_inspector.py
    channel_browser.py
    composer.py
    controller.py
    wiring.py
  Widgets/IRC/
```

### First-release regions

1. Network and buffer rail with saved networks, joined channels, direct
   messages, favorites, unread, mentions, mute and connection state.
2. Active buffer with identity, topic, modes, bounded transcript, history
   boundary, composer and delivery feedback.
3. Optional member inspector with privileges, account/away state and WHOIS.
4. Channel browser consuming the incremental LIST stream with progress,
   filters, join and favorite actions.

Wide layout shows all three primary columns. Medium layout turns the member
inspector into an explicit overlay/secondary region. Compact layout preserves
the active buffer and provides collapsed rail handles. The inspector disappears
before the network rail.

Screen bindings follow the current keybinding ADR. It does not shadow terminal
convention keys or global `ctrl+p`, `ctrl+q`, `f1`, and `f6`. Footer hints are
generated only for actions that work in the current state and focus context.

No settings are added to deprecated settings surfaces. Global IRC defaults and
profile management eventually live in the canonical F9 Settings screen;
immediate connect, join, favorite and mute actions remain on Network Chat.

## tldw_server discovery

A future active tldw_server capability response advertises an explicit service
descriptor such as:

```json
{
  "service": "irc",
  "protocol": "ircv3",
  "host": "chat.example.internal",
  "port": 6697,
  "tls": "required",
  "auth": {
    "mechanisms": ["PLAIN"],
    "credential_scope": "active-server"
  },
  "capabilities": ["server-time", "batch", "draft/chathistory"]
}
```

The coordinated server contract is a later decision. The descriptor does not
authorize persisting the active server token in an IRC profile. The credential
provider resolves the active credential at connect time. If the descriptor
changes or disappears, the derived profile becomes visibly stale or
unavailable and never falls back to guessed coordinates.

## Implementation workstreams after approval

These are dependency-ordered workstreams, not reserved task IDs. Each becomes
one or more atomic Backlog tasks with its own acceptance criteria and targeted
tests.

1. **Fork provenance and packaging**: create the repository from full Codeberg
   history, pin `UPSTREAM_BASE`, preserve notices, perform the namespace-only
   migration, and establish artifact guards.
2. **Transport privacy and lifecycle**: remove raw logging, serialize writes,
   adopt stalled-write provenance, await transport close, remove implicit
   reconnect, and own tasks/timers.
3. **Ordered reducer and correlation**: separate state commits from callbacks,
   fix send-before-future races, define query outcomes, and prove close under
   callback failure and cancellation resistance.
4. **CAP, TLS and SASL**: implement the CAP 302 state machine, verified TLS
   gates, standard-library PLAIN, owned negotiation timeouts and one-time
   readiness.
5. **Streaming LIST**: structured bounded iterator/session, local cancellation
   and drain behavior, errors and pressure tests.
6. **IRCv3 semantics**: tags, server-time, echo-message, BATCH and CHATHISTORY
   models plus resource bounds.
7. **Compatibility and release**: deterministic oracle, TLS, AgentIRC, pinned
   Ergo, artifact install, supply-chain controls and `1.1.0.post1` publication.
8. **Chatbook model and fake-adapter foundation**: stable typed profiles,
   commands, events and fake adapter with no production fork dependency.
9. **Chatbook session lifecycle**: application composition, generation fencing,
   reconnect, bounded buffers and shutdown.
10. **Private profiles and credentials**: non-secret persistence, favorites,
    autojoin and credential lookup.
11. **Network Chat vertical slice**: route, regions, one network/channel,
    transcript, composer, topic and members through the fake adapter.
12. **Real adapter and browsing**: exact fork artifact, channel/direct chat,
    LIST browser, common commands, unread and favorites.
13. **History and tldw_server integration**: capability-driven history followed
    by a separately approved server descriptor contract.

The fork release is completed before Chatbook adds the real dependency. The
Chatbook fake-adapter and model work may proceed without fork objects, but the
real adapter cannot pin an editable branch or an unpublished artifact.

## First fork release acceptance gates

- Canonical upstream base and complete downstream patch inventory recorded.
- Built wheel and sdist contain `tldw_pydle`, not a top-level `pydle` package.
- LICENSE, NOTICE, source URLs and upstream base are present in both artifacts.
- Python 3.11–3.14 unit and deterministic suites pass.
- Protocol state and callback events preserve scripted wire order.
- Immediate WHOIS replies cannot beat pending-state publication.
- No raw frame, secret, encoded secret or message body appears in default logs
  or failure objects.
- Verified TLS success/failure and no-downgrade tests pass.
- SASL PLAIN success, rejection, chunking, timeout and cleanup tests pass.
- CAP multiline, values, split requests, ACK/NAK, NEW/DEL and one-time END/ready
  tests pass.
- LIST is incremental, bounded and truthful under local cancellation and a
  missing terminal numeric.
- Write stall, read failure, connect cancellation and callback failure finish
  within their declared bounds.
- Normal and cooperative failed close leave zero client-owned tasks and timers.
  A cancellation-resistant callback instead returns a bounded incomplete
  outcome naming the retained callback, with zero protocol tasks/timers and no
  authority for late state publication.
- Tags, server-time, BATCH and CHATHISTORY preserve required identity and obey
  bounds.
- The same built artifact passes two-client AgentIRC and pinned Ergo runs.
- Actions are pinned, permissions are minimal, and release artifacts include
  checksums plus provenance/SBOM data.

## Changes from the earlier draft

The critical review made these deliberate corrections:

- raw frame redaction became content-free structured logging;
- detached handler ownership became ordered state reduction plus post-commit
  callbacks;
- a task registry became complete task/timer/query/transport lifecycle;
- a simple CAP continuation patch became a CAP 302 state machine;
- `pure-sasl` became a dependency-free PLAIN implementation;
- TLS default testing became real certificate and no-downgrade integration
  testing;
- `channel_list() -> list` became a bounded streaming LIST session;
- implicit receive limits became explicit byte/resource contracts;
- the fork floor is 3.11; Chatbook's current floor is independently 3.12;
- “current Ergo” became pinned release CI plus a scheduled latest-stable job;
- namespace isolation gained mechanical-commit and built-artifact guards;
- editable-checkout feasibility evidence no longer counts as release evidence.

The selected library, repository direction, adapter boundary, screen ownership,
memory-only transcript default, and explicit tldw_server discovery seam remain
unchanged.
