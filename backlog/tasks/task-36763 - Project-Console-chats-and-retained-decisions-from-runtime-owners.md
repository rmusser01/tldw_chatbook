---
id: TASK-36763
title: Project Console chats and retained decisions from runtime owners
status: To Do
created_date: 2026-10-25 17:39
dependencies:
- TASK-36760
- TASK-36762
labels:
- console
- web-server
- mobile
- adr-244
priority: high
documentation:
- Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md
- Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md
- backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose current shared Console state and answer supported decisions from the phone even while the relevant chat is in the background or its screen is unmounted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Bounded chat, transcript, run and subagent projections use canonical live and saved identities without duplicate rows, unintended agent starts or unrestricted metadata reads.
- [ ] #2 Current retained decisions are visible and supported answers resolve once through native owners; unsupported kinds remain visibly pending with a terminal path.
- [ ] #3 Answerable-client accounting survives independent client disconnects and preserves native timeout, permission, hook and project-instruction policies.
- [ ] #4 Snapshot/subscription races, overflow resets, stale targets and long inert content preserve currentness and thinking/capture privacy policies without widget scraping.
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
