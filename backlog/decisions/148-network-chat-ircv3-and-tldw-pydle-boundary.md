# ADR-148: Network Chat IRCv3 and tldw-pydle boundary

Status: Accepted
Date: 2026-08-23
Renumbered: 2026-09-10 from the unlanded ADR-084 draft; that number is now occupied on dev.
Review amendment: [ADR-149](149-network-chat-handoff-reliability-amendments.md) (Proposed; original decision text retained below).
Related Tasks: TASK-21601 through TASK-21608
Supersedes: N/A; consolidates the IRC design and client-selection drafts that
did not land on the current `dev` history

## Decision

Chatbook will add **Network Chat** as a distinct IRCv3 screen under the existing
Console destination. It is a sibling route to the agent Console rather than a
mode inside the agent transcript. Fresh screen instances own only view state.
An application-scoped IRC session manager owns live connections, bounded
in-memory buffers, unread state, reconnect policy, and application shutdown.

Chatbook will use a separately packaged, tldw-maintained fork of pydle behind a
protocol-neutral adapter. The primary repository is intended to be
`rmusser01/tldw-pydle`; the canonical upstream remains
`https://codeberg.org/shiz/pydle`. The fork begins from upstream `develop`
commit `4efcc3b5096536668dfe772461f19a17f1ddd84e`, verified again on
2026-08-23, which descends from the pydle 1.1.0 tag
`cbc18112e141d357abfb5828262cd98d65628096`.

The distribution name is `tldw-pydle`, the import package is `tldw_pydle`, and
the first release line is `1.1.0.postN`. The fork preserves full upstream
history, copyrights, and BSD-3-Clause license text. It adds a notice describing
the fork and exact upstream base; it does not publish or install a top-level
`pydle` package.

The fork requires Python 3.11 or newer. Chatbook already has that floor, and
the reliable lifecycle design benefits from structured concurrency and modern
exception handling. Python 3.10 compatibility is not an initial release goal.

The fork owns transport, parsing, registration, CAP and SASL state machines,
ordered protocol-state application, correlated protocol queries, and every
internal task or timer. Chatbook owns the top-level adapter scope, connection
generation, reconnect decisions, typed application events, profiles,
credential references, message-buffer policy, persistence, and UI.

Incoming frames are parsed and applied to protocol state in wire order. A
state reducer must never await a future whose completion requires later
inbound frames. Correlated operations publish their pending state before their
command is written. User callbacks run only after the corresponding state
commit, through owned bounded delivery machinery; callback failure is
observable but cannot reorder or stall protocol state.

Outbound writes are serialized. Encoded byte limits and a bounded drain
timeout are enforced at the transport boundary. The client has explicit,
idempotent `connect`, `wait_ready`, and asynchronous close behavior. It does
not reconnect implicitly. Close cancels or drains every cooperative owned
reader, dispatcher, callback, timer, and request task, awaits
`writer.wait_closed()` within a bound, and exposes any forced-abort outcome.
Arbitrary callback code that ignores cancellation produces a bounded,
explicitly incomplete close result naming the retained callback; protocol work
still settles and that callback loses authority to publish protocol state.

Persistent and ordinary debug logs are content-free. They may record opaque
network and operation identifiers, command names, capability names, numeric or
standard-reply codes, byte counts, timings, and reason categories. They never
record raw IRC frames, endpoints, nicknames, account or channel names, targets,
message bodies, topics, real names, passwords, channel keys, tokens, SASL
payloads, certificate-key material, or exception messages that may contain
those values. Raw wire capture is not a production logging mode.

Direct TLS is the first-release secure transport. Certificate-chain and
hostname verification are enabled by default, certificate failure never
downgrades to plaintext, and an ordinary Chatbook profile cannot disable
verification. SASL PLAIN is implemented directly using the Python standard
library, including IRC's 400-byte AUTHENTICATE chunking, and is refused unless
verified TLS succeeded. `pure-sasl` is not a required dependency. SASL
EXTERNAL and certificate credential profiles remain unavailable until an
end-to-end implementation and real TLS tests prove them.

CAP 302 uses an explicit registration state machine. Offered, requested,
pending, negotiated, and semantically supported capabilities remain distinct.
The client accumulates continuation rows, handles values plus ACK/NAK/NEW/DEL,
splits deterministic requests within encoded line limits, and sends `CAP END`
exactly once during registration. Registration readiness is emitted exactly
once on numeric `001`; MOTD completion may enrich state but is not the ready
boundary.

The first modern semantic baseline is message tags, server-time,
echo-message, BATCH, and `draft/chathistory`. BATCH membership, timestamps,
message IDs, and history boundaries are preserved internally. Chatbook still
defines stable public event types, history pagination policy, buffer bounds,
and cross-reconnect deduplication.

Channel browsing uses one LIST session per connection and exposes a bounded
asynchronous stream of structured entries rather than accumulating an
unbounded list. Local cancellation stops delivery to the caller and drains or
discards server replies until numeric `323`; it does not claim to cancel the
server's transmission. The terminal result reports completion, truncation,
dropped rows, cancellation, and protocol failure explicitly.

The parser and transport enforce explicit inbound line, tag, receive-buffer,
outbound line, query-result, callback-queue, and shutdown bounds in encoded
bytes. Limits are configurable within a documented safe range. Oversized or
malformed input produces a classified protocol outcome rather than unbounded
allocation or accidental content logging.

The fork has its own CI and release process. Deterministic tests run on every
supported Python version. Release artifacts are built, installed into clean
environments, and inspected to prove the namespace, license/notice files, and
metadata. AgentIRC and a pinned Ergo release are compatibility gates before
Chatbook updates its exact dependency pin. A separate scheduled job may test
the latest stable Ergo, but an unpinned moving target does not define release
reproducibility.

Release automation pins third-party actions by full commit, uses minimal
permissions, does not expose publishing credentials to pull-request jobs, and
publishes checksums plus provenance or an SBOM. PyPI publication uses Trusted
Publishing when configured. The exact upstream base and downstream patch
inventory are present in release metadata and notes.

Generic correctness and security fixes are prepared so they can be offered to
upstream independently. Namespace migration is a separate mechanical commit,
and automated guards detect stale `import pydle`, accidental upstream console
scripts, or a built top-level `pydle` package. Upstream review occurs at least
quarterly, on each upstream release, and on relevant transport, TLS, SASL, or
parser security disclosures.

A future tldw_server integration advertises an explicit IRC service
descriptor containing endpoint, TLS, authentication, and capability metadata.
The client never derives an IRC endpoint by rewriting a REST URL or guessing a
port. Public networks, AgentIRC, and a tldw_server-provided endpoint all use
the same standards-first adapter contract; server-specific extensions remain
optional capability modules.

## Context

Network Chat has a different lifecycle from an agent conversation. It has
long-lived sockets, multiple networks and buffers, presence, capability
negotiation, unread and mention state, channel browsing, reconnection, and
potential server-backed history. Mounting those concerns inside the Console
transcript or letting each screen own a socket would conflict with Chatbook's
fresh-screen navigation and narrow state-owner decisions.

Executable spikes against pydle, AgentIRC, and Ergo showed that canonical
Codeberg pydle is the smallest complete-client base once maintaining a fork is
accepted. The pinned source already contains some fixes absent from its 1.1.0
wheel, but its remaining behavior is not release-ready for Chatbook:

- complete inbound and outbound IRC frames are logged at debug level;
- each inbound message is dispatched through an independent detached task,
  permitting protocol state to change out of wire order;
- implicit reconnect is enabled and transport shutdown does not await the
  writer's close;
- outbound writes are not serialized;
- WHOIS publishes pending correlation state after sending its command;
- CAP state is incomplete for multiline CAP 302 and post-registration NEW/DEL;
- one SASL continuation timeout constructs `call_later` with a coroutine
  object instead of a callback;
- the effective TLS-verification default is unsafe at the client feature seam;
- LIST prior art accumulates shared results and does not model bounded local
  cancellation;
- receive, line, tag, query, callback, and shutdown resource bounds are not a
  coherent public contract.

Keeping only task handles around the existing concurrent handlers would make
shutdown more observable but would not restore deterministic state ordering.
Making every current handler sequential would deadlock handlers that wait for
later WHOIS or WHOX replies. The ordered reducer and post-commit callback
boundary are therefore required architecture rather than optional cleanup.

The earlier SASL prototype used `pure-sasl`. Its latest PyPI release is from
2019. SASL PLAIN's initial response and IRC chunking are small enough to own
and test directly, avoiding a mandatory stale dependency while leaving future
mechanisms to a separate decision.

These decisions cover a long-lived network service, authentication, private
content, dependency ownership, packaging, a cross-module adapter, and durable
screen placement. They require a canonical ADR before implementation tasks
begin.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Use unmodified pydle from PyPI | The wheel predates relevant Codeberg fixes and retains insecure defaults, broken CAP behavior, detached lifecycle, and missing modern semantics. |
| Patch pydle only inside Chatbook | The client is intended for Chatbook and tldw_server consumers. A hidden application patch cannot provide a stable shared artifact, independent CI, or upstream provenance. |
| Keep the upstream `pydle` import namespace | It reduces sync churn but makes co-installation ambiguous and lets the fork silently shadow upstream. Isolation is worth the mechanical migration cost. |
| Use `pure-sasl` for PLAIN | It adds a required dependency whose latest release is from 2019 for a small mechanism the fork can implement and verify directly. |
| Await every existing pydle handler sequentially | Some handlers wait on replies processed by later inbound frames, producing protocol deadlocks. |
| Retain concurrent handlers and add locks | Locks do not define wire-order state commits and can deadlock correlated query paths. |
| Let the library reconnect automatically | Chatbook owns user intent, recovery UI, backoff, credential refresh, and connection generations; hidden reconnect races all of those owners. |
| Return LIST as one Python list | Large networks can exhaust memory and provide no incremental progress or truthful cancellation semantics. |
| Make “latest Ergo” the release gate | A moving dependency makes a previously reproducible release fail without a source change. Pinned release CI plus a scheduled latest-compatibility job separates those concerns. |
| Build a new sans-I/O client around irctokens/ircstates | It provides maximum control but requires building transport, registration, CAP, SASL, correlation, state, feature semantics, and lifecycle. The audited fork remains the smaller complete-client path. |
| Use AgentIRC's client as the universal client | AgentIRC is principally a server and agent-collaboration target. Ordinary IRCv3 interoperability must not depend on its extensions. |
| Persist all received messages by default | That creates a new private transcript store with retention, deletion, search, export, and sync obligations. First release remains memory-only unless server history is available. |

## Consequences

### Benefits

- Chatbook receives a complete, independently testable IRC client artifact
  without exposing fork internals to UI or persistence code.
- Protocol state has deterministic ordering even when application callbacks
  are slow or fail.
- Authentication and chat content stay out of ordinary diagnostics.
- Application-owned reconnect and generation fencing remain truthful.
- Channel browsing, history, and shutdown have explicit resource bounds.
- Public IRC, AgentIRC, and future tldw_server installations share one
  standards-first path.

### Accepted trade-offs

- tldw assumes long-term fork maintenance, release engineering, security
  response, and upstream synchronization.
- The reliable dispatcher is a core refactor rather than a short patch queue.
- Namespace isolation increases mechanical conflict during upstream sync.
- Python 3.10 consumers cannot use the first fork release.
- Memory-only messages are lost at application exit when the server provides
  no history.
- SASL mechanisms other than PLAIN remain unavailable until separately
  implemented and release-gated.
- Local LIST cancellation cannot force an IRC server to stop sending; it can
  only bound local delivery and memory while draining the protocol response.

## Reconsideration triggers

Reconsider the selected core if any of the following remains true after the
first release tranche:

- fork objects or callbacks leak through the Chatbook adapter;
- ordered state application requires invasive rewrites across most feature
  modules rather than the defined dispatch and correlation seams;
- an upstream sync requires repeatedly rebuilding multiple core subsystems;
- a relevant high-severity transport, parser, TLS, or SASL issue cannot be
  fixed or mitigated promptly;
- the artifact cannot pass deterministic, AgentIRC, and pinned-Ergo gates on
  supported Python versions;
- cooperative shutdown cannot prove zero library-internal tasks;
- an incomplete close cannot identify retained uncooperative callback tasks or
  prevent them from publishing late protocol state.

## Links

- [Network Chat and tldw-pydle design](../../Docs/superpowers/specs/2026-08-23-network-chat-ircv3-and-tldw-pydle-design.md)
- [tldw-pydle first-release implementation plan](../../Docs/superpowers/plans/2026-08-23-tldw-pydle-first-release.md)
- [ADR-011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
- [ADR-029: Local Private Data Boundary](029-local-private-data-boundary.md)
- [ADR-031: TUI keybinding and footer-hint conventions](031-tui-keybinding-and-footer-hint-conventions.md)
- [ADR-032: Immutable installed distribution assets](032-immutable-installed-distribution-assets.md)
- [ADR-033: Application session state ownership](033-application-session-state-ownership.md)
- [ADR-036: Application service composition lifecycle](036-application-service-composition-lifecycle.md)
- [Canonical pydle source](https://codeberg.org/shiz/pydle)
- [pydle PR 172: LIST support](https://codeberg.org/shiz/pydle/pulls/172)
- [pydle PR 196: stalled-write timeout](https://codeberg.org/shiz/pydle/pulls/196)
- [IRCv3 specifications](https://ircv3.net/irc/)
- [AgentIRC](https://github.com/agentculture/agentirc)
- [Ergo](https://github.com/ergochat/ergo)
