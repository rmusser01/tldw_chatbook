# ADR-204: Guardian local self-monitoring and trend awareness

Status: Proposed (2026-09-29) — design spec reviewed against code; implementation pending.
Date: 2026-09-29
Companion spec: [2026-09-29 Guardian × Dreams local design](../../Docs/superpowers/specs/2026-09-29-guardian-x-dreams-local-design.md)
Related: [ADR-029](029-local-private-data-boundary.md) (extended here), [ADR-196](196-dreams-daily-discovery-and-tracking.md) (Dreams' boundary pattern applied), ADR-148 (Console Run Hooks — seam ordering), [TASK-33200](../tasks/task-33200%20-%20Guardian-x-Dreams-bidirectional-awareness-and-trend-analysis-tie-in-tldw_server-chatbook.md)

## Decision

Add a **local Guardian** subsystem (`tldw_chatbook/Guardian/` + `DB/Guardian_DB.py` + a settings section + a daily trend task) porting tldw_server Guardian's *self-monitoring* semantics to chatbook: user-configured pattern rules over the user's own typed Console prompts, humane awareness notices with escalation (notify → redact → block), impulsive-disable cooldowns, ported crisis resources, and a trend analyzer (sustained-topic-share / repetitive-volume) over the recorded hits. A hard opt-in gate lets non-sensitive topic **aggregates** feed Dreams' interest profile — the deferred "conversations-as-signal," delivered as counts, never text. Off by default with zero footprint when disabled.

## Context

Guardian on tldw_server already implements reactive per-message self-monitoring (pattern rules, dedup, escalation, bypass protection, crisis resources) but only for server-connected chats; chatbook's Console — where local users actually type — has no moderation seam at all. Dreams (ADR-196) deliberately deferred conversation-derived interest signal as too privacy-sensitive without a stricter opt-in vehicle. TASK-33200 asks for both halves of the coin: awareness of one's own patterns (Guardian) and discovery of new things (Dreams), interoperating.

## Contracts

1. **Zero footprint when off:** `[guardian] enabled=false` (the default) creates no database file, runs no checks, and adds no measurable send latency; storage is built only by a successful enable through the settings toggle (the Dreams R2 ruling, replicated).
2. **Digest-only storage:** GuardianDB stores message digests, topic labels, and timestamps — never message text and never matched spans. Matched spans appear only transiently in live inline notices. Summaries and trend notices report counts per topic, not excerpts. (The digests add no new exposure class: prompts already persist verbatim in ChaChaNotes `messages`; Guardian records strictly less.)
3. **The discovery gate:** a rule's hits may influence Dreams' outbound discovery **only if** the rule sets `feeds_discovery=1`; rules flagged `is_crisis` can never set it (enforced at the write boundary and test-proven by whole-payload absence). Crisis-adjacent text may never become an outbound search query. This extends ADR-029 and mirrors ADR-196's searchable-goal gate.
4. **Seam ordering vs Run Hooks (ADR-148):** the Guardian checker runs at prompt-queue dispatch *before* user `UserPromptSubmit` hooks; hits count regardless of a later hook block (Guardian measures typed intent); Guardian's own escalated `block` short-circuits first. The hook emission point is untouched.
5. **Fail-open, loud:** checker/store errors never eat a send; they surface one notification per error-signature per visit plus structured logs. Blocking chat on a broken checker is a worse failure than missing a notice.
6. **Cooldown semantics:** an escalated `block` starts that rule's `cooldown_minutes`; while active, both the rule's deactivation and the feature's enable-toggle refuse with a remaining-time notice (config-file escape hatch still works, logged as a bypass).
7. **Not clinical — and crisis rules never silence:** crisis resources (988 / Crisis Text Line / SAMHSA / IASP) port verbatim from tldw_server with its disclaimer, shown wherever a crisis-flagged rule surfaces. **Crisis-flagged rules can never hold or escalate to `redact` or `block`** — redacting distress hides it from the model that should respond to it, and blocking a user mid-crisis from their communication tool is the opposite of the feature's purpose; crisis rules surface `notify` + resources, always (write-boundary enforced, test-pinned). The subsystem performs awareness only. Disabling the feature stops recording and preserves data; only explicit user action clears the store.
8. **Counting vs noticing:** every match records an alert and counts toward escalation; `notification_frequency` dedup gates only notice surfacing — Guardian tracks behavior, not annoyance. Trend-analyzer inputs exclude analyzer-generated rows (no feedback loop).
9. **Retention:** alerts prune at 180 days (configurable); visit summaries persist.

## Consequences

- New per-subsystem store, a new pre-send seam beside the existing refusal gate, a daily scheduler task, and a DreamsDB v3 source bump (`'guardian'`) — the aggregate reader joins the existing getter-injected profile-refresh pipeline with per-source degradation.
- The warm send path gains a compiled-regex match and a cached config gate (budget <5ms); GuardianDB writes are thread-offloaded out of dispatch.
- Crisis content is duplicated across repos and should be kept synchronized (noted here as the maintenance obligation).
- Supervised (guardian→dependent) mode, partner-approval bypass, response-side monitoring, and the server-side Dreams port are explicitly out of scope.

## Alternatives considered

- **Thin client over tldw_server's Guardian** — rejected: makes a local-privacy feature require a server; no offline story; local trend analysis (the point) impossible.
- **Trends only (no rules engine)** — rejected: loses escalation, cooldowns, and crisis resources — the parts that make it a guardian rather than a counter.
- **Post-send analysis only** — rejected: the reminder arrives after the fact, and escalated block (the apex of humane friction) is impossible; pre-send at dispatch honors the course-correct framing.
- **Raw conversation text into Dreams** — rejected at design review: aggregates-only with the feeds_discovery gate, because Dreams topics flow into outbound web-search queries and crisis-adjacent text must never leave the machine that way.

## Current integration ownership amendment (2026-10-12)

Runtime access requires live enablement even when a Guardian store is cached.
Explicit Settings/history inspection has separate authority. Discovery consent is
live and excludes inactive/crisis rules; revocation removes only Guardian-derived
contributions, preserving independently authorized interests. Every delayed check
captures its Console session and visit before work begins; notice/refusal/durable
counter ownership never follows a later active-chat switch. Blank sessions receive
real nonempty ownership before recording. Current draft and acknowledgement fences
remain owned by the prompt queue.

File-backed check/trend work uses finite grouped thread ownership with explicit
drain and retirement of only newly acquired handles. Borrowed/in-memory notification
stores keep their thread affinity. Direct, aggregate, trend and next-visit crisis
notices preserve crisis provenance and the existing resource/disclaimer block.
Guardian remains optional in recovery, uses its selected custom path, and
restored rules/settings confer no automatic recording or execution authority.


### Historical crisis provenance amendment (2026-10-12)

An alert records its crisis classification at observation time; a later rule edit
or deletion cannot reclassify it into discovery material or remove its resource
context. The v1 alert schema has no snapshot or supported payload field, so v2
adds a tri-state crisis_provenance column: 1 crisis, 0 explicitly non-crisis, -1
unproven legacy lineage. Existing v1 rows migrate to -1 without inferring intent
from a mutable current rule. Unproven rows remain local, are excluded from
Dreams discovery input, and any visible aggregate containing them carries the
existing crisis resource/disclaimer block. This conservative compatibility cost
also applies to old summary payloads lacking a classification marker.

Direct checks pass the matched rule's captured classification to the alert write.
Trend notices and visit summaries propagate whether any contributing alert is
crisis or unproven; newly inserted trend rows retain that marker. Discovery needs
both an explicitly non-crisis alert and a currently enabled, non-crisis,
feeds_discovery rule. No new clinical resources or historical text storage is
introduced. Exact populated v1 catalogs remain supported as predecessor inputs;
v1-to-v2 migration is atomic, preserves rows and timestamps, rolls back on failure,
and refuses malformed, future, or hybrid version/schema claims before mutation.
Installed recovery qualifies measured v1 and v2 catalogs at the selected optional
custom path; restore still confers no automatic execution authority.
