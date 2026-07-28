---
id: TASK-1260
title: Path-validation property tests fail under machine load, producing false regression signals
status: In Progress
assignee: []
created_date: '2026-07-28 10:55'
labels:
  - testing
  - flaky
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Utils/test_path_validation_properties.py::TestPathValidationProperties::test_safe_paths_always_validate`
fails intermittently when run alongside other suites, and passes in isolation.

Observed on `origin/dev` @ `048e53cab`:

| Command | Result |
| --- | --- |
| the test alone | passed |
| `Tests/Utils/` alone | 576 passed |
| `Tests/Subscriptions/ Tests/Scheduling/ Tests/Utils/` | **1 failed**, 892 passed |
| the same three, re-run | 893 passed |
| the same three at `e74e37d07` (before the day's changes) | 878 passed |

The tests are Hypothesis `@given` properties with no `settings(...)` override anywhere in the file
and no Hypothesis profile in `Tests/conftest.py`, so they run under the default 200 ms per-example
deadline. `test_safe_paths_always_validate` does real filesystem work per example — a
`TemporaryDirectory` plus up to four `mkdir` calls (the strategy generates 1-5 components and the
loop walks `components[:-1]`, since the last component is the file, not a directory) — which is
exactly the kind of work that crosses 200 ms when the machine is loaded. This repo routinely has
10+ concurrent pytest processes from parallel agents.

**Why this is worth fixing rather than tolerating.** The failure is indistinguishable from a real
regression at the moment it appears. Attributing this one instance cost five separate runs across
two worktrees — running the identical command on a clean baseline, with and without a newly added
file in the same directory, plus a re-run to establish intermittency. Any future change that happens
to be in flight when this fires will pay that cost again.

Other property files in the tree may have the same exposure; the fix should be a Hypothesis profile
rather than a one-line patch to this file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Hypothesis settings profile is registered in Tests/conftest.py and applied to the suite, rather than each property file setting its own deadline
- [x] #2 The profile disables or substantially raises the per-example deadline, so a loaded machine cannot fail a property that holds
- [x] #3 Property tests that do real filesystem or database work are identified, and the profile covers them
- [x] #4 test_safe_paths_always_validate passes when its suite runs concurrently with at least two other large suites under load
- [x] #5 The reason the deadline is relaxed is recorded next to the profile, so it is not "tightened back up" later as an apparent improvement
<!-- AC:END -->

## Implementation Notes

A `tldw` Hypothesis profile is registered and loaded in `Tests/conftest.py` with `deadline=None` and
`HealthCheck.too_slow` suppressed. Registering it once in the root conftest covers every property
file rather than patching the one that happened to fail; the other property suites (DB, path,
validation) carry the same exposure.

**The deadline is disabled rather than raised.** A larger number only moves the threshold a loaded
machine will eventually cross, and the failure it produces is indistinguishable from a real
regression. A deadline that fails a property which *holds* is measuring the machine, not the code.
Timing belongs in benchmarks, not correctness properties. The comment beside the profile says this
explicitly, per AC#5, so it is not "tightened back up" later as an apparent improvement.

**Mutation-checked twice.** A throwaway property sleeping 250ms per example fails with
`DeadlineExceeded` when the profile is not loaded and passes when it is -- confirming the diagnosed
mechanism directly rather than by inference. And setting `deadline=200` in the profile fails
`test_per_example_deadline_is_disabled`, so the permanent guard is not vacuous.

`Tests/test_hypothesis_profile.py` asserts the deadline is disabled, `too_slow` is suppressed, and
the profile is the one this repo registered.

Modified: `Tests/conftest.py`. Added: `Tests/test_hypothesis_profile.py`.
