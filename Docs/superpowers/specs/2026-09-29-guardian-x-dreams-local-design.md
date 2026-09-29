# Guardian × Dreams — local self-monitoring and trend awareness design

- **Status:** Proposed — awaiting owner review
- **Date:** 2026-09-29
- **Decision:** [ADR-204](../../../backlog/decisions/204-guardian-local-self-monitoring.md)
- **Classification:** Architectural
- **Task:** [TASK-33200](../../../backlog/tasks/task-33200%20-%20Guardian-x-Dreams-bidirectional-awareness-and-trend-analysis-tie-in-tldw_server-chatbook.md) (this spec delivers the local half: chatbook-side Guardian + trend layer + Dreams interop; the server-side Dreams port for connected clients remains a separate future task)
- **Approved product decisions:** local Guardian + trends as ONE slice (one DB, one opt-in, one settings section); pre-send checking at the prompt-queue dispatch seam with escalated `block` able to stop a send; design-review rulings R1–R9 folded (see §Privacy, §Seam ordering, §Notices, §Mitigations).

## Summary

Port tldw_server's Guardian **self-monitoring** (Use Case B) to chatbook natively, and add the trend-analysis layer that makes it "two sides of the same coin" with Dreams: pattern rules over the user's own typed prompts surface humane awareness notices (with escalation, impulsive-disable cooldowns, and crisis resources), a trend analyzer detects doom-loop/fixation patterns over rolling aggregates, and — under a hard opt-in gate — non-sensitive topic aggregates feed Dreams' interest profile as the long-deferred "conversations-as-signal," delivered as counts, never text.

Everything is **off by default**: no storage is created, no checks run, and nothing is recorded unless `[guardian] enabled` is explicitly set in the new Settings section. Framing throughout: supportive awareness tooling, **not a clinical service** — Guardian's existing disclaimer rides every surfaced notice of crisis-flagged rules.

## Background

Verified seams (recon 2026-09-29, on the consolidated Dreams stack):

- **Send chain / insertion point:** the Console send chain runs `ChatScreen.handle_console_send_message` → `_dispatch_console_draft_send` (`UI/Screens/chat_screen.py:19275`) → `ConsolePromptQueueUIController.dispatch` (`UI/Console_Modules/prompt_queue.py:878`) — which already hosts the blocked-reason refusal gate (notify + system-row plumbing injected at `UI/Console_Modules/wiring.py:2213-2218`). There is **no moderation/filter seam** in chatbook today; this design adds one.
- **Console Run Hooks (ADR-148)** already fire user-configured external commands at `UserPromptSubmit` at this same point in the flow — the seam-ordering contract below defines the interplay.
- **Sessions:** Console `session_id` (per chat tab, persisted; `Chat/console_chat_store.py:1633`) and the **visit** (`ChatScreen.on_unmount` → `leave_console_runtime`, `Chat/console_runtime.py:4920`) — visit = the post-visit-summary boundary.
- **Notifications:** `NotificationDispatchService.dispatch(category, ...)` (free-form categories; inbox + `app.notify` toast) and `ChatScreen._append_native_console_system_message` (`chat_screen.py:18890`) — the standard inline transcript row. Post-visit summary rides the "report on next mount" slot precedent (`app.py:3030-3036`).
- **Store template:** `DB/Dreams_DB.py` (per-subsystem `BaseDB`, additive schema-versioned DDL, thread-local connections; app-owned lazy builder `TldwCli.get_dreams_db` at `app.py:6295` with the **build-only-after-enable** ruling; settings toggle as "the one place that may build it", `settings_screen.py:30327-30334`).
- **Dreams consumption contract:** `CycleDeps` getters → `profile_sources.read_*` → `interest_profile.merge_signals` → `_upsert_profile_signals` (protects `user`/`seed` rows). A new reader is the integration point; the profile `source` CHECK (`DB/Dreams_DB.py:84-86`) needs a v3 bump adding `'guardian'`.
- **Crisis resources:** none exist in chatbook — the 988 / Crisis Text Line / SAMHSA / IASP set + disclaimer ports as content from tldw_server's Guardian.
- Server-side Guardian semantics this port mirrors: pattern rules with `except_patterns`, `notification_frequency` dedup (`every_message` / `once_per_conversation` / `once_per_session` / `once_per_day`), display modes (`inline_banner` / `sidebar_note` / `post_session_summary` / `silent_log`), session + rolling-window escalation, `cooldown_minutes` bypass protection, per-run rule compilation cache (30–60s TTL).

## Goals

1. User-configured pattern rules over the user's own typed Console prompts, surfacing humane awareness notices — inline, summarized post-visit, or silently logged.
2. Escalation that can, at its apex, refuse a send — with impulsive-disable cooldowns making deactivation deliberately frictional.
3. Crisis resources with the "not a mental health service" disclaimer wherever a crisis-flagged rule surfaces.
4. Trend detection (doom-loop / fixation / sustained-topic share) over the user's own activity, surfaced through the same notice ladder with its own frequency caps.
5. Dreams interop under a hard gate: opted-in, non-sensitive topic aggregates feed the interest profile — counts, never text.
6. Off by default with zero footprint when disabled: no DB file, no checks, no latency.

## Non-goals

- Supervised guardian→dependent mode (server Guardian's Use Case A) — self-monitoring only.
- `partner_approval` bypass protection — chatbook has no partner concept; v1 is cooldown-only (documented; revisit with a future account/companion concept).
- Server-side Dreams port for connected clients (separate task under TASK-33200's umbrella).
- Clinical framing, diagnosis, or intervention of any kind — awareness notices only.
- Reading anything the user did not type into their own Console (no file scanning, no RAG-content monitoring, no provider-response scanning in v1 — user prompts only; responses deferred).

## Detailed design

### Package layout

```
tldw_chatbook/
  Guardian/
    __init__.py
    settings.py             # [guardian] defaults + accessor (enabled first, False)
    rules_engine.py         # compiled-rule cache, match/except evaluation
    check_pipeline.py       # dispatch-seam hook: match → dedup → escalate → act
    trend_analyzer.py       # rolling aggregates + trend notices
    crisis_resources.py     # ported content + disclaimer
    guardian_view.py        # read-only projections (settings rule table, summaries)
  DB/Guardian_DB.py         # per-subsystem store, DreamsDB template
UI/Screens/settings_screen.py  # Guardian section (mirrors Dreams')
Scheduling/scheduler/handlers/guardian_trend_handler.py  # daily trend task
```

### Data model (GuardianDB v1)

- `guardian_rules(id, name, topic TEXT NOT NULL, pattern, except_patterns TEXT(JSON list), action CHECK('notify','redact','block'), severity CHECK('info','warning','critical'), is_crisis INTEGER DEFAULT 0, notification_frequency CHECK('every_message','once_per_conversation','once_per_session','once_per_day'), display_mode CHECK('inline_banner','post_visit_summary','silent_log') — the server's `sidebar_note` is omitted, chatbook has no sidebar-note surface in v1, escalate_session_threshold INTEGER, escalate_window_threshold INTEGER, escalate_window_days INTEGER, cooldown_minutes INTEGER, feeds_discovery INTEGER DEFAULT 0, enabled INTEGER DEFAULT 1, created_at, updated_at)`. The rule's `topic` is the aggregation key for alerts, trends, and the Dreams reader (v1 convention: one word or short phrase, e.g. "doomscrolling", "fixation: band-name").
- `guardian_alerts(id, rule_id NULL — NULL for trend-analyzer-generated alerts whose `topic` carries the trend kind, session_id, visit_id, topic, message_digest, ts)` — **no message text, no matched span** (the span is shown transiently in the live notice only). `message_digest` = sha256 of the whitespace-stripped prompt (dedup counting only).
- `guardian_escalation_state(rule_id PRIMARY KEY, session_count INTEGER, window_count INTEGER, window_start TEXT, current_action TEXT, cooldown_until TEXT)`.
- `guardian_visit_summaries(visit_id, session_id, created_at, payload TEXT(JSON))` — computed post-visit aggregates (per-topic counts, escalated rules), for the summary surface and trend baselines.
- Indexes (each EXPLAIN-plan-pinned per the repo convention at landing): `alerts(rule_id, ts DESC)`, `alerts(topic, ts DESC)`, `alerts(session_id, ts)`.
- Retention: alerts pruned at `alert_retention_days` (default **180**), configurable; summaries retained indefinitely (small).

### Rules engine + check pipeline

- **Compilation cache:** rules compile (regex + except patterns) once per rules-version with a 60s TTL invalidation on any rule write — the warm send path never recompiles.
- **Pipeline (per send, at dispatch):**
  1. Gate: `[guardian] enabled` false → return immediately (a single config read cached with the compiled rules; effectively zero overhead).
  2. Match: each enabled rule's pattern against the draft, minus `except_patterns` spans. Crisis rules use the ported defaults.
  3. Dedup by `notification_frequency` (session_id for conversation-scoped, visit_id for session-scoped, 24h window for daily).
  4. Escalation: per-rule session counter + rolling-window counter; thresholds replace the base action (`notify → redact → block`) per the server's semantics; `current_action` persisted.
  5. Act: `notify` → inline system row (+ crisis resources when `is_crisis`); `redact` → rewrite the draft in-memory before send (matched span replaced with `[redacted: rule name]`); `block` → refuse through the existing blocked-reason plumbing, notice explains which rule and how to course-correct. `post_visit_summary`/`silent_log` record only.
  6. Record: alert row (digest + topic + ids + ts) in one thread-offloaded hop; escalation counters updated in the same hop.
- **Latency budget:** warm path < 5ms (compiled-regex match + cached config gate); GuardianDB writes off the dispatch thread; a slow/errored store degrades to notice-without-record, never blocks the send.
- **Seam ordering vs ADR-148 hooks (binding contract):** the Guardian checker runs at dispatch **before** user `UserPromptSubmit` hooks fire; the hit is counted **regardless** of whether a hook subsequently blocks the send (Guardian measures typed intent — the draft, not the provider call); Guardian's own `block` short-circuits before hooks run. The hook emission point itself is untouched.
- **Fail-open, loud:** any checker exception → the send proceeds; one `guardian_checker_error` notification per error-signature per visit (then silent), plus loguru with context. A dead checker must be visible, not silent.

### Notices

- **Inline:** system row via the existing transcript channel, `markup=False`, carrying the rule name, the **transient matched span (capped ~80 chars, never persisted)**, and — for `is_crisis` rules — the crisis-resource block + disclaimer verbatim.
- **Post-visit summary:** computed at visit end into `guardian_visit_summaries`, surfaced on next Console mount via the report-slot precedent; per-topic counts and escalated rules only (no excerpts — they were never stored).
- **Trend notices:** rendered like rule notices but produced by the analyzer; **frequency-capped at once per visit per topic** (R5).

### Trend analyzer

- Inputs: `guardian_alerts` only (topic, ts, session_id) + visit summaries. Definitions (v1, deliberately conservative):
  - **Sustained-topic share (fixation proxy):** one topic ≥ `fixation_share_threshold` (default 0.6) of alert hits across `fixation_window_days` (default 7) with ≥ `fixation_min_hits` (default 30) total.
  - **Repetitive-volume (doom-loop proxy):** ≥ `doomloop_hits_per_day` (default 20) hits on the same topic for ≥ 3 consecutive days.
- Each trend rule's notice is itself a `guardian_alerts` row (topic = the trend kind) — so trends feed the same escalation/dedup ladder and are visible in summaries.
- **Cadence (R7):** runs (a) synchronously at visit end (before the summary is written), and (b) as a daily scheduler task (`guardian_trend_handler`, the `dream_track_check` pattern — projection emission, fire-and-forget, budget-free: analysis is local, no LLM, no network).
- False-positive mitigations (R5, named): `except_patterns` on rules (ported defaults include treatment/research/clinical vocabulary for crisis-adjacent rules), `silent_log` default for `info` severity, trend-notice frequency cap, and the disclaimer on every surfaced crisis-adjacent notice. Tuning keys live in `[guardian]` config.

### Dreams interop — the hard gate (R1, binding)

- New `CycleDeps.guardian_db_getter` + `Dreams/profile_sources.read_guardian_topics(guardian_db)`: windowed per-topic hit counts (default 14d), weight per touch, emitted as `{facet: 'topic', text, weight}` rows — **topic labels only, never message content**.
- **`feeds_discovery` gate:** a rule's hits reach `read_guardian_topics` **only if** the rule has `feeds_discovery=1`. The settings surface exposes the toggle per rule with explicit copy; rules with `is_crisis=1` **cannot** set `feeds_discovery` — enforced at the write boundary (`set_rule`/rule-upsert rejects the combination; SQLite CHECKs cannot express cross-column constraints) and pinned by a test proving the rejection plus the whole-payload absence. Crisis-adjacent text may never influence outbound search queries.
- DreamsDB v3 migration: add `'guardian'` to the profile `source` CHECK (additive; v1/v2 files upgrade in place). `_preferred_sources` threads the new origin; `user`/`seed` protection unchanged.
- Each side independently opt-in: `[guardian] enabled` controls recording; per-rule `feeds_discovery` controls discovery influence; `[dreams] enabled` controls consumption (TASK-33200 AC#5).

### Settings section

Mirrors the Dreams section (canonical `settings_screen.py`, ADR-150 classes only): enable toggle (the ONE place storage may be built, with gate copy "Nothing runs, nothing is recorded" when off), the rules table (add/edit/delete via a modal: pattern, excepts, action, severity, frequency, display mode, thresholds, cooldown, crisis flag, feeds_discovery — disabled+explained on crisis rules), cooldown status display (minutes remaining when active), and read-only trend-threshold keys. The **impulsive-disable cooldown (R4)**: while any rule's `cooldown_until` is in the future (set when an escalated `block` fires), the enable toggle and that rule's deactivate action refuse with "available in N min" notices; a config-file disable still works (escape hatch, documented) but is logged as a cooldown bypass.

### Error handling

Checker/store/analyzer failures never eat a send or a visit summary (fail-open + loud, R9); analyzer failures degrade to summary-without-trends; interop failures degrade to Dreams-without-guardian-signal (the existing per-source degrade pattern); retention pruning is best-effort with a logged count.

### Testing

Unit: match/except evaluation; compilation cache invalidation; frequency dedup across the four modes; escalation counter transitions and action replacement; cooldown gating (rule + feature toggle, with the bypass log); redact rewrite; retention prune. Integration: dispatch-seam insertion (fail-open on raising checker; zero-overhead when disabled — a timing test on the warm path); seam ordering (Guardian block short-circuits before hooks; hook-block still counts the hit); visit-end summary + daily handler emission. Interop: `read_guardian_topics` counts; **crisis rule's hits never appear in a synthesized payload** (whole-payload absence assertion, the goals-gate pattern); DreamsDB v3 upgrade. UI: settings section per the Dreams test patterns; notice rendering with `markup=False`.

## Governance

- **ADR:** ADR-204 (privacy boundary extension, new store, hot-path seam, block escalation) — created with this spec.
- Crisis content ports verbatim from tldw_server's Guardian with its disclaimer; changes to that content should stay synchronized across repos (noted in the ADR).
- Backlog tasks created at planning time (three tasks per §Phasing).

## Phasing (three implementation tasks)

1. **GuardianDB + rules engine + dispatch seam** (store, pipeline, inline notices, fail-open, seam ordering, settings enable gate + storage-bootstrap).
2. **Trend analyzer + post-visit summaries + crisis content + daily handler** (aggregates, trend notices, summaries, retention).
3. **Dreams interop + settings rules editor + guide** (guardian_db_getter, reader, v3 bump, feeds_discovery gate, rules UI, user-guide section).

## Follow-ups (out of scope, filed or to file)

- Server-side Dreams port for connected clients (TASK-33200's other half).
- Provider-response monitoring (v1 watches user prompts only).
- `partner_approval` bypass protection (needs a partner concept).
- Trend thresholds auto-tuning from dismissal feedback.
