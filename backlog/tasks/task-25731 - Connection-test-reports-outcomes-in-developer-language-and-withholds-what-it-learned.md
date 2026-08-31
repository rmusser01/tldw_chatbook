---
id: TASK-25731
title: >-
  Connection test reports outcomes in developer language and withholds what it
  learned
status: Done
assignee: []
created_date: '2026-08-31 05:10'
updated_date: '2026-08-31 06:45'
labels:
  - console
  - ux-review
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A failed provider test reports that the model listing request had a connection error, naming neither the address tried, the reason, nor a next step. A partial success reports that the listing was reached but the selected model was not confirmed, without naming the models the server just returned. In both cases the product knows more than it says, and the copy is written for the implementer rather than the user.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A failed test names the address tried, the reason, and one concrete next step
- [ ] #2 A successful listing reports the models it discovered
- [ ] #3 Test result copy matches the plain-language standard used elsewhere in setup
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rewrote _FAILURE_DETAILS and the model_unconfirmed verdict in Chat/provider_test_evidence.py from subsystem-framed to user-framed, each naming what happened and the one action that addresses it (e.g. 'The model listing connection was refused.' -> 'Nothing is listening at that address. Start the server, or check the endpoint and port.'). The mixed-signal '✓ Model listing reached; selected model was not confirmed.' now says which half worked and what to do about the other. DEFERRED, needs plumbing not present in this module: naming the address tried and listing the models actually discovered -- ProviderReadinessSnapshot carries neither, only facet enums. 78 tests pass across the consuming suites; no test pinned the old strings.
<!-- SECTION:NOTES:END -->
