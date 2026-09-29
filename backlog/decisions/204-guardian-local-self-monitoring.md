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
2. **Digest-only storage:** GuardianDB stores message digests, topic labels, and timestamps — never message text and never matched spans. Matched spans appear only transiently in live inline notices. Summaries and trend notices report counts per topic, not excerpts.
3. **The discovery gate:** a rule's hits may influence Dreams' outbound discovery **only if** the rule sets `feeds_discovery=1`; rules flagged `is_crisis` can never set it (enforced at the write boundary and test-proven by whole-payload absence). Crisis-adjacent text may never become an outbound search query. This extends ADR-029 and mirrors ADR-196's searchable-goal gate.
4. **Seam ordering vs Run Hooks (ADR-148):** the Guardian checker runs at prompt-queue dispatch *before* user `UserPromptSubmit` hooks; hits count regardless of a later hook block (Guardian measures typed intent); Guardian's own escalated `block` short-circuits first. The hook emission point is untouched.
5. **Fail-open, loud:** checker/store errors never eat a send; they surface one notification per error-signature per visit plus structured logs. Blocking chat on a broken checker is a worse failure than missing a notice.
6. **Cooldown semantics:** an escalated `block` starts that rule's `cooldown_minutes`; while active, both the rule's deactivation and the feature's enable-toggle refuse with a remaining-time notice (config-file escape hatch still works, logged as a bypass).
7. **Not clinical:** crisis resources (988 / Crisis Text Line / SAMHSA / IASP) port verbatim from tldw_server with its disclaimer, shown wherever a crisis-flagged rule surfaces; the subsystem performs awareness only — no diagnosis, intervention, or content blocking beyond the user's own configured rules.
8. **Retention:** alerts prune at 180 days (configurable); visit summaries persist.

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
