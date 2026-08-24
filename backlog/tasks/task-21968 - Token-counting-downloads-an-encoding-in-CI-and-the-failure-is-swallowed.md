---
id: TASK-21968
title: Token counting downloads an encoding in CI, and the failure is swallowed
status: To Do
assignee: []
labels: [testing, test-integrity, ci]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The single largest source of errors in the core test job, by a wide margin.

Token counting obtains its encoding from a library that fetches a data file on first use and
caches it. The cache lives outside the test sandbox's control and is warm on any machine that
has run the suite before, so developers never see this. In CI it is cold on every run, so
every call attempts a download, the egress guard refuses it, and the caller — which wraps the
lookup in a broad exception handler and returns nothing — hides the refusal. The guard records
the attempt anyway, which is the only reason it is visible at all, and the test fails at
teardown pointing at a network address rather than at token counting.

Measured on one core shard: **1,156 blocked attempts to one address**, with the log naming the
encoding host directly. That address accounted for roughly nine in ten of the blocked attempts
in an earlier baseline as well, so this is long-standing rather than new.

Worth stating because it shaped the investigation: this was initially attributed to a
different library that appears in the same logs, because that one prints retry lines and this
one does not. Forbidding downloads for that library removed its share and left this untouched.
Whatever is done here should be verified by counting attempts to this specific address falling
to zero, not by the absence of an error message.

The fix is not the same shape as forbidding the other library's downloads, because production
genuinely wants an encoding here — token counting is not incidental. The options are to make
the data available to the run rather than fetched, to make the lookup fail immediately instead
of over the network when it cannot be satisfied locally, or to decide the tests that trigger
it should not be reaching real token counting at all. Which is right depends on how many tests
depend on a real count, which should be measured first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No test attempts to download an encoding, verified by attempts to that address falling to zero in a CI shard rather than by the absence of an error
- [ ] #2 Tests that depend on a real token count are identified before choosing an approach, and still get one or are explicitly changed not to need one
- [ ] #3 A failure to obtain an encoding is not silently swallowed into a null result during tests
- [ ] #4 The change is shown not to alter outcomes for the suites that exercise token counting
<!-- AC:END -->

## Notes

Found while confirming TASK-21562 in CI. That change removed the other library's share, and
the shard log then showed this one as essentially the entire remainder.
