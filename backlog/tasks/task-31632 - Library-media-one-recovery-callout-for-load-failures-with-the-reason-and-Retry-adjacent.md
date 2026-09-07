---
id: TASK-31632
title: >-
  Library media - one recovery callout for load failures with the reason and
  Retry adjacent
status: Done
assignee:
  - '@claude'
created_date: '2026-09-05 06:18'
updated_date: '2026-09-06 05:02'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #5 P1: three load-failure sentences (Couldn't load page 1., Couldn't load media. Check the local Library and retry., Library source services unavailable; retry Library later.) render as bare text with no reason, while the only Retry sits 34 rows below in the pager and the service wall's sole control, Continue, leaves Library for Home. The service wall is a 5-second source-snapshot timeout collapsed into one static string by a bare except, so a transient failure reads as an indefinite outage and never self-heals.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every Media load failure renders in one recovery callout with a tinted state border that names what failed, why, and what to do, with Retry inside the callout
- [x] #2 A snapshot timeout is distinguished from a hard failure and re-tries on return to Library
- [x] #3 Continue either dismisses in place or is renamed to what it does
- [x] #4 Tests cover the three failure paths
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Task 1: load_failure_recovery_state(what, reason, retry_id, stable_selector, kind) beside the policy-denial builder; the browse controller records `failure` (page/facet) with the reason from the shared exception mapping; error_copy unchanged for existing consumers.
2. Task 2: the Media canvas renders `failure` as one ds-recovery-callout (#library-media-load-failure) with Retry inside it where the bare sentence painted; the pager keeps its Retry only for the mutation gate; painted tests at both sizes; live with a scratch profile.
3. Task 3: the source-snapshot asyncio.TimeoutError becomes a warning callout `Library sources did not answer · waited 5 s` with library-source-retry and one automatic retry on return to Library; hard failures keep the service copy plus a privacy-safe reason; the entry canvas renders the wall as a callout with Retry inside and stops offering Continue as the failure's control.
4. SDD per task (review + carried minors), final whole-branch review + fix round, PR G.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Critique #5 P1: three Media load-failure sentences painted bare with no reason and the only Retry 34 rows below; the service wall never self-healed and its Continue ejected to Home. Now every load failure carries a DestinationRecoveryState (severity, retry_id, message `what · reason`) and renders through the product's existing ds-recovery-callout with Retry inside it. Reasons come from PR E's shared mapping (`timed out`, OS/SQLite message, else the exception class — non-OS free text is withheld by the private-media-failure leak pin, so a cold start against a broken DB reads `Couldn't load media · ValueError`; a friendlier map is a rider). The snapshot timeout is told apart from a hard failure (the only place a 5 s figure is true) and retries once on return; the entry canvas no longer ejects via Continue. Re-entry seam: on_screen_resume → _refresh_library_visit_surfaces() already re-ran the snapshot every visit; a severity gate keeps that one automatic retry for a timeout and drops it for a hard failure (a deliberate behaviour change). The timeout branch adds no log line (no-new-logger rule); #library-canvas-error on browse rows still has no Retry (out of this task's scope) — the log-line gap was closed in the final fix round by routing the timeout through the existing warning call with a deadline marker (no new logger call site); #library-canvas-error stays a rider. The final review also found the entry-canvas Retry could go silently inert on a repeated identical failure (dataclass-equality dedup suppressed the repaint) — fixed so a failure snapshot always repaints.
Trade-offs: the callout costs 6 rows at the 36-cell Items floor (shared ds-recovery-callout padding; never clips rows or pager at 100x30); facets have no retry control of their own, so Retry reloads failed facets too; authority_owner is hardcoded "local data source"; two Retry-construction sites stay duplicated (not worth a helper).
Verification: per-file suites vs base in separate processes; whole-file test_library_shell.py diffed against the task-31249 census; live hard-failure callout via a scratch profile whose media_db_path is a directory (Retry inside, no Continue) at 235x52 and 100x30; the timeout callout and the re-entry auto-retry are pinned in app-tests (a live timeout cannot be provoked reliably).
Files: tldw_chatbook/UI/destination_recovery.py (+ Screens re-export), tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py, tldw_chatbook/Widgets/Library/library_media_canvas.py, tldw_chatbook/Widgets/Library/library_entry_canvases.py, tldw_chatbook/UI/Screens/library_screen.py, Docs/User_Guide/library/media-and-conversations.md, tests in Tests/UI/test_library_media_browse_controller.py / test_library_media_render_fixes.py / test_library_shell.py / test_library_entry_compose_once.py.
<!-- SECTION:NOTES:END -->
