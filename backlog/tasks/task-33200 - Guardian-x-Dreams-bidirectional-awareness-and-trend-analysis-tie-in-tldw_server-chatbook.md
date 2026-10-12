---
id: TASK-33200
title: >-
  Guardian x Dreams - bidirectional awareness and trend-analysis tie-in
  (tldw_server <-> chatbook)
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-23 06:07'
updated_date: '2026-10-12 19:39'
labels:
  - guardian
  - dreams
  - wellbeing
  - integration
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dreams (chatbook, ADR-196) and Guardian (tldw_server self-monitoring) are two sides of the same coin: Dreams informs the user of new things; Guardian makes the user aware of their own patterns. Tie them together bidirectionally - port Dreams functionality to tldw_server for connected clients and port Guardian self-monitoring to chatbook for local use - and extend Guardian's reactive per-message pattern rules with trend/topic identification over the user's actions, content, and chats. Configured 'topics of consideration' (doomscrolling/doom loops, fixation on a specific thing, self-harm ideation awareness) surface as humane reminders offering an opportunity to course-correct. User-enabled only; must be explicitly configured; disabled by default. Framing: supportive awareness tooling, not a clinical service (match Guardian's existing disclaimer).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Integration design spec covering both directions (Dreams-to-tldw_server port and Guardian-to-chatbook port) exists and is reviewed, with an explicit local-vs-server privacy boundary extending ADR-029
- [ ] #2 User-configurable "topics of consideration" analysis works on the chatbook side without requiring a connected tldw_server
- [ ] #3 Trend/fixation/doom-loop detection surfaces awareness notices with escalation and anti-impulsive-disable semantics mirroring Guardian's SelfMonitoringService
- [ ] #4 Crisis-resource surfacing reuses Guardian's built-in resource set (988 / Crisis Text Line / SAMHSA / IASP) and disclaimer where relevant
- [ ] #5 Dreams interest-profile signals and Guardian trend analysis interoperate via a shared topic vocabulary or explicit mapping, with each loop independently opt-in
- [ ] #6 All analysis and notice paths are disabled by default and run only when explicitly enabled and configured by the user; nothing leaves the machine without explicit consent
- [ ] #7 Critical introduced Dreams and Guardian privacy, provenance, spend, recovery, navigation, notification and worker-currentness regressions run in the required PR lanes with exact executable membership checks; prior Goal, Personal Context and Buddy targets, order and limits remain intact.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Current-dev integration (2026-10-11): follow Docs/superpowers/plans/2026-10-11-pr2921-current-dev-integration.md and the pinned audit. Reconcile exact feature range with actual sequential dev; fix reviewed consent, URL provenance, captured-chat notice, shared budget, query-goal attribution, disable, native lifetime, recovery and UI contracts in existing owners. Qualify actual integrated source before review/publication. ADR required: existing196/204/207/126 amendments as applicable; no competing new decision.

Updated-source integration: frozen6c074d6eed feature range7b7193d647..head84commits/78paths, Dreamsv6; preserve addressed atomic reservation/generating-goal fixes and fix remaining actual privacy/dispatch/recovery/Guardian/production-Library boundaries plus admitted34801–4. Actual post-PC/Buddy dev and fresh author lease before code/publication; final qualification follows all product changes. Existing ADR196/204/207/126 amendments as needed, no duplicate IDs.
<!-- SECTION:PLAN:END -->

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

The earlier implementation notes below describe frozen source7f18b6ec1063009b916dad51d468629f20687917 and historical source qualification. Current integration is In Progress; reopened acceptance outcomes require the actual fixed current-dev candidate. No integrated test, publication or merged claim is made by this preparation.

Scoped updated-plan preflight found final qualification preceded the admitted follow-up changes and required-CI coverage was absent from task acceptance. Plan order now places all changes before final source/node freeze and qualification; Guardian integration acceptance gains the bounded required-CI outcome. Collections predecessor/rollback/reopen/future-refusal and installed recovery obligations are explicit in extracted Task1. No product or completion claim.

Current source update: author0c941ba2ec984d960ba83376aeace0e830452bbc, 113 nonmerge commits/88 feature paths after actual merged Goal9595facbbb. Independent source review preserves the real shared Toast fix and stronger reconcile/suspension work; remaining integrated proof and actual post-Buddy/latest-dev publication gate are pending. Prior source/plan/qualification claims remain historical. No product change or completion is claimed here.

PR2921 current private integration qualification (2026-10-12):
Implemented local-only Guardian live consent, captured nonempty Console ownership, historical crisis provenance (Guardian v2), finite daily native work/drain and installed optional recovery with inert restored activation. Native ownership proves physical close before worker/process exit and preserves borrowed handles. Existing resource/disclaimer projections cover direct, trend, aggregate and next-visit summary. Required bounded CI groups measured 160 and 80 passing; exact comment-aware membership guards preserve the original 64/23 targets and 187 UI prefix. ADR-204/126 amendments precede changes. Companion-server implementation is outside the approved local scope.
Full source-bound outcomes, original failures, native owners and tradeoffs: .superpowers/sdd/2026-10-11-pr2921-current-dev-integration/task-1-report.md; external pr-review-wave2/pr2921-current-integration. Local implementation is ready for independent review. Actual-base restack, complete independent review and hosted required gates remain pending; preserve In Progress and the existing acceptance checklist until the full DoD is met.
<!-- SECTION:NOTES:END -->

> Renumbering provenance: filed as task-32902 on the pre-divergence docs
> line; dev's older task-32902 (Work-stream the tier-2 P3 polish,
> 2026-09-21) keeps the ID per the 2026-08-21 owner rule (TASK-19601).
