# ADR-149: Network Chat handoff reliability amendments

Status: Proposed
Date: 2026-09-10
Amends upon acceptance: [ADR-148](148-network-chat-ircv3-and-tldw-pydle-boundary.md)
Related Tasks: TASK-21601 through TASK-21608

## Context

The owner requested a fresh plan assessment and a PR against dev for PTO handoff.
The approved August draft never landed. Its provisional ADR-084 number now
collides, so its decision text is preserved as ADR-148. This separate amendment
records behavioral corrections without silently rewriting an accepted ADR.
Acceptance of this amendment is a reviewer decision; packaging from the original
pinned base does not require waiting for the behavioral decisions below.

Current Chatbook requires Python >=3.12 and its route registry supports retained
screens with resume/suspend. The fork's independently approved >=3.11 floor is
unchanged. Upstream develop advanced eight commits to
`e27b6a2138c81061c2e6a937526fceaedb94d31a`, including the stalled-write fix.
These observations do not establish new fork test results or authorize a base bump.

## Decision proposed

1. Keep `4efcc3b5096536668dfe772461f19a17f1ddd84e` as the audited fork point.
   Audit later upstream changes separately. Adopt the stalled-write change from
   `7c7f5c73bf14b8207a551a6cbf38627984cb7f4e` with provenance, not as an
   unreviewed wholesale base update. Preserve pure-sasl metadata during the
   namespace-only task; remove it with the tested replacement in TASK-21605.
2. A full callback queue must not suspend the protocol reducer. Fail closed with
   a typed slow-consumer result, delivered through independent lifecycle state.
   Otherwise a callback awaiting WHOIS can deadlock while its own reply waits
   behind queue backpressure. Coalesced state is re-enqueued at its new receive
   sequence, preserving ordering relative to intervening message events.
3. Bound outbound queue bytes, entries and admission work, not merely individual
   drain calls. Reserve protocol-control capacity, preserve FIFO application
   admission order, and never await application queue space from the reducer.
   Exhausted control capacity is terminal; no silent PONG loss.
4. Timeout/cancellation settles callers but does not erase wire ambiguity.
   Unlabelled queries retain bounded tombstones until their terminal reply or
   disconnect; reject reuse of the correlation key meanwhile. LIST keeps one
   tombstone after its drain deadline while unrelated chat remains usable.
   Query keys use negotiated IRC CASEMAPPING. Tombstones count against pending
   operation limits; exhaustion rejects new queries rather than growing memory.
5. Bounded incomplete callback shutdown assumes the callback still yields to
   the event loop. Python cannot forcibly stop arbitrary blocking application
   code. Fence public commands/publication and create a fresh client per new
   connection; do not claim Python private attributes sandbox hostile callbacks.
   The close operation owns cleanup independently of any individual awaiter.
6. CAP continuation syntax belongs to LS/LIST, values to LS/NEW. ACK/NAK settles
   each request chunk. Ready requires numeric 001 plus successful mandatory
   authentication; a no-CAP server can register when no capability is mandatory.
   One cancelled readiness waiter cannot cancel everyone else's result.
7. Gate PLAIN on the actual verified SSL transport and approved context, not a
   caller-provided verified flag. Reject credential NULs and invalid PLAIN
   challenges. No HTTP API credential is implicitly an IRC credential.
8. Preserve server echoes without deduplicating by text/time or promising a
   receipt from a socket write. History does not mutate live membership/topic
   state. Do not request event-playback capabilities without explicit support.
9. Navigation does not own connections, whether a screen is destroyed or
   suspended. Later Chatbook work must test both visit lifecycles, detach view
   subscriptions, and reconcile once from a current snapshot on return.
10. Release completion requires published, retrievable artifacts matching tested
    hashes. A publishing blocker leaves the task unfinished. CI is unattended;
    local full-suite consent does not make CI wait for the absent author.
    Add Windows/macOS Python 3.12 transport, TLS and packaging evidence to the
    Linux Python 3.11–3.14 matrix.

## Alternatives and consequences

- Blocking callback delivery preserves all events only until it deadlocks;
  explicit disconnect is preferable to an apparently healthy frozen session.
- Releasing query keys on timeout permits stale responses to corrupt new
  operations. Tombstones trade temporary query availability for correctness;
  reconnect remains an explicit embedding-application action.
- Changing the baseline to current upstream immediately would invalidate the
  prior audit. A reviewed base update remains possible without blocking the
  mechanical fork task.
- Forcible callback termination requires a process boundary and serialization
  contract, disproportionate for trusted embedding callbacks; no such boundary
  is promised by this fork.
- The first fork version remains `1.1.0.post1` as previously approved. That is a
  downstream distribution version, not a claim of upstream API compatibility.
  Review its release naming before publication; a version change must update
  package metadata, task criteria and release workflow together.

No IRC implementation, server endpoint, credentials, publication configuration,
or new persistence schema is introduced by this decision.

## References

- [PTO handoff and validation scenarios](../../Docs/superpowers/plans/2026-09-10-network-chat-pto-handoff.md)
- [Updated execution plan](../../Docs/superpowers/plans/2026-08-23-tldw-pydle-first-release.md)
- [Current route registry](../../tldw_chatbook/UI/Navigation/screen_registry.py)
- [Current Python requirement](../../pyproject.toml)
- [Codeberg change set](https://codeberg.org/shiz/pydle/compare/4efcc3b5096536668dfe772461f19a17f1ddd84e...e27b6a2138c81061c2e6a937526fceaedb94d31a)
- [IRCv3 CAP specification](https://ircv3.net/specs/extensions/capability-negotiation.html)
- [IRCv3 message tags](https://ircv3.net/specs/extensions/message-tags.html)
