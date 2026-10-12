---
id: TASK-34917
title: UI Fast Lane shard 3 leaves too little headroom for one hung test to be reported
status: To Do
created_date: 2026-10-12 22:22
labels:
- ci
- tests
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UI Fast Lane shard 3 has a 20-minute cap and ran with a median of 16.4 minutes and a maximum of 18.8 over 35 pull-request runs measured on 2026-10-12. The shard holds a Console test that hangs now and then (test_applying_on_anthropic_submits_none_of_the_hidden_fields). A hung private-profile test costs about 3 minutes 10 seconds before the wrapper reports it when the child stops at its own timeout, and about 3 minutes 18 seconds when it cannot exit. Added to the measured run times, that pushes the shard past its cap in about a fifth of runs (5 of 35 and 7 of 35 respectively, against 4 of 35 before the wrapper change in PR #3091), and the cancelled job then loses exactly the report that says where the test hung. Measured during the review of PR #3091 (TASK-34789).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Shard 3's slowest run over at least 20 pull-request runs leaves room for one hung private-profile test to be reported before the cap
- [ ] #2 The shards stay balanced, with no shard left closer to its cap than shard 3 is today
- [ ] #3 The contract test that pins the shards reflects the new layout
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
