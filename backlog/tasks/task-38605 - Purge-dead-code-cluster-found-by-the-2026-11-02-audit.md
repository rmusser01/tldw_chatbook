---
id: TASK-38605
title: Purge dead code cluster found by the 2026-11-02 audit
status: To Do
created_date: 2026-11-02 14:29
labels:
- cleanup
- ui
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cluster of dead surfaces confirmed by repo-wide handler and importer greps: ViewChangeMessage posts, ContentsRebuilt, ChunkingTemplatesChanged, ReminderFormSubmitted, tab_events module, model_download worker branch, Outputs and Sharing panels, Widgets Media, activity_log, dead Utils helpers, indiscriminate encrypt_config, and the mostly test-only state package. See bug-sweep-2026-11-02-dev.md module summaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Dead custom messages are removed or handled
- [ ] #2 tab_events module and model_download branch removed
- [ ] #3 Outputs Sharing panels and Widgets Media either wired in or deleted
- [ ] #4 Dead Utils helpers and encrypt_config removed
- [ ] #5 state package shrunk to live members with the plaintext password field removed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
