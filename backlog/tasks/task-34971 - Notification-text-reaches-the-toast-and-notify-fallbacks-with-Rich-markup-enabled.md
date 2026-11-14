---
id: TASK-34971
title: >-
  Notification text reaches the toast and notify fallbacks with Rich markup
  enabled
status: In Progress
assignee: []
created_date: '2026-10-12 01:46'
updated_date: '2026-10-12 19:39'
labels:
  - notifications
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The shared notification helper passes a notification's title and message, which are often runtime text such as a page or story title, to the toast and to the notify fallback without turning markup off or escaping. The repo's rule for runtime text is to pass markup off for notify and to escape text shown in toasts, because text containing markup characters can raise at render time. The markup census does not see this site because the text arrives in a variable. Dreams tracked-update notices are one more producer of such text; watchlist alerts already were. Found in the review of PR 2921; the helper is shared, so it was left alone there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A notification whose title or message contains square-bracket markup renders literally and never raises, through the toast path and both notify fallbacks
- [ ] #2 Notifications that rely on markup today, if any, are identified and keep their appearance
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Historical root preparation before the second author update:
Read every caller of shared notification helper/dispatcher, audit intentionally formatted producers; make runtime titles/messages literal at actual toast and both notify fallbacks with square-bracket controls. Preserve deliberate producer appearance and existing notification affinity/worker ownership. No ADR required for presentation bug fix.

ADR required: no; existing presentation fix already implemented in the latest author source.
Preserve both markup=False runtime notify sinks, real mounted Textual Toast tests and all author caller-audit notes. Requalify these exact bytes on the integrated branch; do not reimplement the earlier proposed helper change. The unused show_toast branch is historical, not a current product sink.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fixed in the review of PR 2921 (round 4, item C1), in the commit that carries this edit.

Both notify calls now pass markup=False: Utils/NotificationHelper.show_notification and the fallback in NotificationDispatchService._try_transient_delivery. Reproduced first in a Textual run_test with notifications enabled (run_test disables them by default): a title with an unmatched closing tag raised MarkupError and exited the app, a bracketed word was swallowed, and [@click=app.quit] became an action link.

AC1: Tests/Notifications/test_notification_toast_markup.py drives the real dispatch service and the real reminder handler with five such texts and reads the rendered toast; all were red before. The show_toast branch of the helper is unchanged: nothing in the app defines show_toast, so that branch never runs and there is nothing to render; an implementation added later has to take the text literally.

AC2: no caller relies on markup. Searched untruncated: show_notification has one caller (the dispatch service); the dispatch service has 14 calling modules (reminder, automation, briefing x2, guardian trend, watchlists alerts, daily report demo, Dreams tracked update, evaluations, study, quiz, research x2, media ingestion). None passes a literal Rich tag in a title or message and none escapes markup, so nothing changes appearance.

Three test doubles of App.notify with a fixed signature gained the markup parameter (Tests/Scheduling/test_reminder_handler.py, test_automation_end_to_end.py, test_authoring_end_to_end.py).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Historical root integration preparation:
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Source author follow-up admitted into the user-authorized complete PR2921 review/remediation. Historical source description/notes retained. Current integration plan targets frozen updated author6c074d6eed/v6; actual post-Buddy dev and fresh author lease remain required before implementation. No product change or completion claimed.

Author source status was Done with both AC checked on0c941ba2; its complete implementation notes and23 passing independent source positives are retained. Root reopened only the final integrated qualification gate, not the already-fixed presentation logic.

Current source update: author0c941ba2ec984d960ba83376aeace0e830452bbc, 113 nonmerge commits/88 feature paths after actual merged Goal9595facbbb. Independent source review preserves the real shared Toast fix and stronger reconcile/suspension work; remaining integrated proof and actual post-Buddy/latest-dev publication gate are pending. Prior source/plan/qualification claims remain historical. No product change or completion is claimed here.

PR2921 current private integration qualification (2026-10-12):
Preserved the author's actual shared Toast and both notify fixes; no replacement helper was implemented. Existing real mounted Toast tests and producer audit were requalified in required boundary coverage. Author completion/history remains intact; current In Progress concerns combined integration qualification only.
Full source-bound outcomes, original failures, native owners and tradeoffs: .superpowers/sdd/2026-10-11-pr2921-current-dev-integration/task-1-report.md; external pr-review-wave2/pr2921-current-integration. Local implementation is ready for independent review. Actual-base restack, complete independent review and hosted required gates remain pending; preserve In Progress and the existing acceptance checklist until the full DoD is met.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Renumbering provenance

Renumbered from task-34803 to TASK-34971 (2026-10-13): this ID collided
with a concurrently filed task after the branches merged; per the
2026-08-21 owner rule (TASK-19601) the older arrival kept the id.
