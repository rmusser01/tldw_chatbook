---
id: TASK-34817
title: 'Schedules and Watchlists render reminder, rule and source names as markup'
status: In Progress
assignee: []
created_date: '2026-10-12 05:39'
updated_date: '2026-10-12 14:38'
labels:
  - schedules
  - watchlists
  - stability
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Some rows on the Schedules and Watchlists screens put a reminder's title, an alert rule's name or a source's name into text that is parsed as markup. A name containing square brackets can lose part of its text, become a clickable action, or raise an error that ends the app; these sites are already listed as unescaped in the markup census. PR #2921 stopped Dreams from writing brackets into the names it creates (it writes parentheses), but reminders, rules and sources that Dreams created before that change keep their bracketed names, and any name a user types with brackets has the same exposure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A reminder, alert rule or source whose name contains square brackets, including an unmatched closing tag, is shown exactly as written on every Schedules and Watchlists row that shows it
- [ ] #2 No such name can become a clickable action
- [ ] #3 The corresponding rows are removed from the markup census
- [ ] #4 A mounted test renders each of those rows with a bracketed name and fails before the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no; literal presentation bug fix under existing markup/UI contracts.
Audit the actual Schedules/Watchlists reminder/rule/source runtime-name sinks, preserve existing Text and control-character handling, convert only unsafe real sinks to literal text and retire corresponding reviewed census entries. Use mounted red-to-green bracket/action-link controls and ordinary-name positives. Preserve shared Toast fixes; no blanket rewrite, source-name migration or notification-helper reopening.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Current source update: author0c941ba2ec984d960ba83376aeace0e830452bbc, 113 nonmerge commits/88 feature paths after actual merged Goal9595facbbb. Independent source review preserves the real shared Toast fix and stronger reconcile/suspension work; remaining integrated proof and actual post-Buddy/latest-dev publication gate are pending. Prior source/plan/qualification claims remain historical. No product change or completion is claimed here.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->
