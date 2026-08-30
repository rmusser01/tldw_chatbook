# ADR-079: Daily Reports surface and demo seeding

- **Status:** Accepted
- **Date:** 2026-08-29
- **Task:** `TASK-21513` — Daily Reports surface and demo
- **Spec:** [2026-08-29 daily-reports-demo-design](../../Docs/superpowers/specs/2026-08-29-daily-reports-demo-design.md)
- **Amends:** nothing; aligns with [ADR-015](015-shell-destination-ia.md) (Artifacts charter already includes "reports") and [ADR-078](078-research-workspace-authority-and-screen-boundaries.md) (no second universal artifact database).

## Context

The watchlist-briefing pipeline already produces scheduled text briefs with cast
scripts and synthesized audio, but it is invisible as a product: the Artifacts
screen's Reports slot is a hardcoded "none available" placeholder, a new user
must hand-wire watchlist + preset + schedule before anything runs, and scheduled
briefings complete silently (only reminders dispatch notifications).

## Decision

1. **"Daily Report" is a briefing.** The Artifacts screen's Reports slot is fed
   by a read-only view over the existing `briefings`/`briefing_scripts`/
   `briefing_audio` tables across all watchlists. No new artifact store, no new
   tables, no new scheduler task types (ADR-078 direction: presentation adapters
   over canonical owners).
2. **The demo writes real, persistent data by design.** The one-click demo seeds
   a real "Daily Brief" watchlist (three RSS sources incl. Hacker News via
   hnrss.org), a briefing preset, and a 24h cadence, then drives the existing
   run-now seams (claim-path `generate_briefing`, `generate_script`,
   `generate_script_audio`). The seeded setup *is* the user's first daily
   report; it keeps running via the existing `BriefingProjection`/`BriefingJobHandler`.
   Idempotency keys on configured briefing schedules (`list_briefing_schedules`),
   never on names.
3. **Audio is a progressive enhancement.** The cast roster is built from the
   user's existing TTS voice profiles; with zero profiles (or without pydub) the
   demo completes text-only and records an "audio skipped, here's how to enable
   it" hint. `resolve_roster_voices` has no default-voice fallback.
4. **Scheduled briefing completion dispatches one notification** (category
   `"briefing"`, success or attention) through `NotificationDispatchService`,
   policy-gated like `"reminder"`. Stage-by-stage notifications exist only
   during the interactive demo run.
5. **Demo discovery**: Artifacts-screen empty-state CTA and a dismissible
   Watchlists banner (hidden while any briefing schedule exists; dismissal
   persists at `scheduling.daily_report_demo_banner_dismissed`). First-run
   onboarding is follow-up work.
6. **Artifacts v1 surfaces list/play/jump only** (recorded 2026-08-29, after
   the final review flagged the spec §2 action set as an undocumented
   descope): the Reports rows offer labels, per-row Play, and one generic
   Open Watchlists button to the screen root. Canonical read/keep/export
   stay in the Watchlists artifacts pane, one hop away — no second action
   surface to drift out of sync. The spec's kept badge is deferred, not
   dropped: keep state lives in ChaChaNotes (`kept_briefings`), not
   SubscriptionsDB, so the badge needs a cross-DB lookup rather than a join
   extension. Deep-linking rows, in-place preview, keep/export affordances,
   and the kept badge are follow-up work (TASK-21514).

## Consequences

- One artifact authority (SubscriptionsDB briefing tables); the Artifacts
  Reports slot can never disagree with the Watchlists artifacts pane.
- The demo consumes real API quota (LLM + TTS); CTA copy says so.
- Briefing schedules remain rolling cadences (`briefing_cadence_seconds`) — a
  "preferred time of day" remains follow-up work (spec Follow-ups).
