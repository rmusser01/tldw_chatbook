---
id: TASK-32902
title: >-
  Guardian x Dreams - bidirectional awareness and trend-analysis tie-in
  (tldw_server <-> chatbook)
status: To Do
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

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dreams (chatbook, ADR-178) and Guardian (tldw_server self-monitoring) are two sides of the same coin: Dreams informs the user of new things; Guardian makes the user aware of their own patterns. Tie them together bidirectionally - port Dreams functionality to tldw_server for connected clients and port Guardian self-monitoring to chatbook for local use - and extend Guardian's reactive per-message pattern rules with trend/topic identification over the user's actions, content, and chats. Configured 'topics of consideration' (doomscrolling/doom loops, fixation on a specific thing, self-harm ideation awareness) surface as humane reminders offering an opportunity to course-correct. User-enabled only; must be explicitly configured; disabled by default. Framing: supportive awareness tooling, not a clinical service (match Guardian's existing disclaimer).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Integration design spec covering both directions (Dreams-to-tldw_server port and Guardian-to-chatbook port) exists and is reviewed, with an explicit local-vs-server privacy boundary extending ADR-029
- [ ] #2 User-configurable "topics of consideration" analysis works on the chatbook side without requiring a connected tldw_server
- [ ] #3 Trend/fixation/doom-loop detection surfaces awareness notices with escalation and anti-impulsive-disable semantics mirroring Guardian's SelfMonitoringService
- [ ] #4 Crisis-resource surfacing reuses Guardian's built-in resource set (988 / Crisis Text Line / SAMHSA / IASP) and disclaimer where relevant
- [ ] #5 Dreams interest-profile signals and Guardian trend analysis interoperate via a shared topic vocabulary or explicit mapping, with each loop independently opt-in
- [ ] #6 All analysis and notice paths are disabled by default and run only when explicitly enabled and configured by the user; nothing leaves the machine without explicit consent
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Grounding: Guardian design doc lives in tldw_server at Docs/Design/Guardian_Self_Monitoring.md (SelfMonitoringService: pattern rules with except_patterns, notification_frequency dedup, display modes incl post_session_summary and silent_log, session + rolling-window escalation, bypass_protection incl partner_approval, crisis resources built in). Guardian today is reactive per-message checking; the trend/analysis layer is new capability on both sides. Dreams side: Docs/superpowers/specs/2026-09-22-dreams-daily-discovery-design.md + backlog/decisions/178-dreams-daily-discovery-and-tracking.md (interest profile + feedback loop are the natural signal seams; conversations-as-signal was deliberately deferred from Dreams v1 and would be revisited here under Guardian's stricter opt-in).
<!-- SECTION:NOTES:END -->
