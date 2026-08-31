---
id: TASK-25731
title: >-
  Connection test reports outcomes in developer language and withholds what it
  learned
status: Done
assignee: []
created_date: '2026-08-31 05:10'
updated_date: '2026-08-31 06:48'
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
Rewrote _FAILURE_DETAILS and the model_unconfirmed verdict in Chat/provider_test_evidence.py from subsystem-framed to user-framed: each result now names what happened AND the one action that addresses it. The mixed-signal 'Model listing reached; selected model was not confirmed' now says which half worked and what to do about the other.

CORRECTION TO MY FIRST PASS: I claimed no test pinned these strings. That was wrong -- my grep looked for whole sentences, but Tests/Chat/test_provider_readiness.py::test_connection_failed_verdict_names_bounded_probe_category parametrises on FRAGMENTS ('timed out', 'refused', 'unauthorized', 'forbidden', 'http status', 'invalid response', 'connection error'). My first rewrite broke all 7. That test asserts a real property -- the detail must name the bounded probe category so a failure stays diagnosable -- so the right resolution was to satisfy BOTH, not to edit the test. Final copy keeps every category fragment while adding the action, e.g. 'The connection was refused - nothing is listening at that address. Start the server, or check the endpoint and port.' 194 passed.

LESSON: when checking whether copy is pinned, grep for distinctive FRAGMENTS and for the parametrize table, not the full sentence.

DEFERRED, needs plumbing absent from this module: naming the address tried and listing the models actually discovered -- ProviderReadinessSnapshot carries only facet enums, no endpoint or model ids.
<!-- SECTION:NOTES:END -->
