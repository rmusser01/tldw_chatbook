---
id: TASK-33200
title: >-
  Guardian x Dreams - bidirectional awareness and trend-analysis tie-in
  (tldw_server <-> chatbook)
status: Done
assignee: []
created_date: '2026-09-23 06:07'
labels:
  - guardian
  - dreams
  - wellbeing
  - integration
dependencies: []
priority: low
---

> Renumbering provenance: filed as task-32902 on the pre-divergence docs
> line; dev's older task-32902 (Work-stream the tier-2 P3 polish,
> 2026-09-21) keeps the ID per the 2026-08-21 owner rule (TASK-19601).


## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dreams (chatbook, ADR-196) and Guardian (tldw_server self-monitoring) are two sides of the same coin: Dreams informs the user of new things; Guardian makes the user aware of their own patterns. Tie them together bidirectionally - port Dreams functionality to tldw_server for connected clients and port Guardian self-monitoring to chatbook for local use - and extend Guardian's reactive per-message pattern rules with trend/topic identification over the user's actions, content, and chats. Configured 'topics of consideration' (doomscrolling/doom loops, fixation on a specific thing, self-harm ideation awareness) surface as humane reminders offering an opportunity to course-correct. User-enabled only; must be explicitly configured; disabled by default. Framing: supportive awareness tooling, not a clinical service (match Guardian's existing disclaimer).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Integration design spec covering both directions (Dreams-to-tldw_server port and Guardian-to-chatbook port) exists and is reviewed, with an explicit local-vs-server privacy boundary extending ADR-029
- [x] #2 User-configurable "topics of consideration" analysis works on the chatbook side without requiring a connected tldw_server
- [x] #3 Trend/fixation/doom-loop detection surfaces awareness notices with escalation and anti-impulsive-disable semantics mirroring Guardian's SelfMonitoringService
- [x] #4 Crisis-resource surfacing reuses Guardian's built-in resource set (988 / Crisis Text Line / SAMHSA / IASP) and disclaimer where relevant
- [x] #5 Dreams interest-profile signals and Guardian trend analysis interoperate via a shared topic vocabulary or explicit mapping, with each loop independently opt-in
- [x] #6 All analysis and notice paths are disabled by default and run only when explicitly enabled and configured by the user; nothing leaves the machine without explicit consent
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Grounding: Guardian design doc lives in tldw_server at Docs/Design/Guardian_Self_Monitoring.md (SelfMonitoringService: pattern rules with except_patterns, notification_frequency dedup, display modes incl post_session_summary and silent_log, session + rolling-window escalation, bypass_protection incl partner_approval, crisis resources built in). Guardian today is reactive per-message checking; the trend/analysis layer is new capability on both sides. Dreams side: Docs/superpowers/specs/2026-09-22-dreams-daily-discovery-design.md + backlog/decisions/196-dreams-daily-discovery-and-tracking.md (interest profile + feedback loop are the natural signal seams; conversations-as-signal was deliberately deferred from Dreams v1 and would be revisited here under Guardian's stricter opt-in).

<!-- SECTION:NOTES:IMPLEMENTATION -->
### Local implementation (feat/guardian-local, 2026-09-29) — DONE

Governance: [ADR-204](../decisions/204-guardian-local-self-monitoring.md)
("Guardian local self-monitoring and trend awareness"); design spec
`Docs/superpowers/specs/2026-09-29-guardian-x-dreams-local-design.md`; plan
`.superpowers/sdd/2026-09-29-guardian-local/`. Three commits on
`feat/guardian-local` (b2beeb3724, 0d34d78133, and Task 3).

- **AC1**: ADR-204 + the reviewed spec carry the boundary (digest-only
  storage, feeds_discovery gate, local-vs-server split). The
  Dreams-to-tldw_server port direction is explicitly Non-goal in ADR-204;
  the local design supersedes the bidirectional wording with a local-first
  scope approved at spec review.
- **AC2**: `DB/Guardian_DB.py` (rules/alerts/escalation/summaries, seed
  rules, `GuardianRuleConflict` crisis caps) + `Guardian/rules_engine.py`
  (compiled-regex cache keyed on rules_version) — fully local SQLite, no
  server involvement. `Guardian/check_pipeline.py` runs at Console prompt
  dispatch (pre Run Hooks, fail-open, crisis caps, cooldown arming);
  post-visit summaries in `Guardian/trend_analyzer.py` + the daily task.
- **AC3**: escalation ladder (notify → redact → block) per session/window
  thresholds; `cooldown_active` + anti-impulsive-disable: the Settings
  toggle and rule deactivation refuse while a cooldown binds ("available in
  N min"); config-file escape hatch logged as `guardian_cooldown_bypassed`
  (loguru, once per cooldown event).
- **AC4**: `Guardian/crisis_resources.py` ports 988 / Crisis Text Line /
  SAMHSA / IASP + the disclaimer verbatim; every surfaced crisis notice
  carries the block; crisis rules are capped to `notify` at the write
  boundary and can never feed Dreams.
- **AC5**: `Dreams/profile_sources.read_guardian_topics` counts alerts per
  topic over `feeds_discovery=1` rules only (SQL join at the reader);
  DreamsDB v3 widens the profile `source` CHECK with `'guardian'`
  (faithful-rewind upgrade test); `CycleDeps.guardian_db_getter` wires the
  reader into the profile refresh (origin `'guardian'`, silent degradation
  when None). The whole-payload absence test pins that crisis topic text
  appears NOWHERE in a synthesized outbound payload.
- **AC6**: `[guardian] enabled=false` default: no DB file, no checks, no
  latency; storage is built only by the Settings enable toggle (the ONE
  caller of `TldwCli.get_guardian_db`). Only aggregate topic counts (never
  message text) leave Guardian, and only for explicitly opted-in
  non-crisis rules.

Settings surface: Settings ▸ Domain Defaults ▸ Guardian (`SettingsCategoryId.GUARDIAN`)
— enable toggle (build-on-enable), gate copy, rules table + edit modal
(crisis pinning, cooldown-aware deactivation refusal). User guide:
`Docs/User_Guide/guardian.md`, linked from `dreams.md`.

Tests: `Tests/Guardian/` (rules engine, checker, escalation, summaries,
trends, interop), `Tests/UI/test_settings_guardian.py`,
`Tests/Dreams/test_dreams_db_track.py` (v3).
<!-- SECTION:NOTES:END -->
