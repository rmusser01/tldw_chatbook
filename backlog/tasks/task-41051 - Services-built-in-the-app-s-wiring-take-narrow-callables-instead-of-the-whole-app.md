---
id: TASK-41051
title: "Services built in the app's wiring take narrow callables instead of the whole app"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- architecture
priority: low
references:
- tldw_chatbook/app_service_wiring.py
- tldw_chatbook/app_feature_glue.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ADR-250 rule 5 says consumers get the members they use, not the app. On dev ecad404bd4 (2026-11-15), the app still hands itself to two groups of services:
- **Study, Quiz and Research services**, through `notification_app=self` (`app_service_wiring.py:2409`, `:2418` and `:2468`).
- **Five `lambda: self` getters:**
  - `DailyReportDemoService` (`:2526`);
  - the automation handler (`:2687` and `app_feature_glue.py:101`);
  - `BriefingJobHandler` (`:2729`);
  - `ReminderHandler` (`:2747`).

Found by TASK-36947.4's review. These sites were outside that task's scope; PR #3476 converted only the toast targets that the Console controller reaches. A service that holds the app can reach all of it, which is what the services bundle exists to stop.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed service takes only the members or callables it uses; toasts go through a notify-only target
- [ ] #2 None of them holds or returns the app
- [ ] #3 Their tests pass
<!-- AC:END -->
