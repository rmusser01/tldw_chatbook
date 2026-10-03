---
id: TASK-33643
title: Pin Linux os.open ceilings for the Console storage-unit ratchet
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-30 17:54'
labels:
  - perf
  - ci
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #2888 gated the Console storage-unit ratchet in perf-guard.yml. Its os.open ceilings were measured on macOS, so on the Linux runner only the admission and helper-spawn counts gate; the first Linux run (Perf Guard on #2888, 2026-09-30) passed those. Qodo noted the gap: a change that adds directory walks inside existing admissions raises os.open counts without adding admissions, and would pass the Linux gate. A passing run prints no census, so Linux values have to be recorded before they can be pinned.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The storage-unit census reports its measured units in the perf-guard log on every run, not only on failure
- [ ] #2 Linux-specific os.open ceilings, pinned from at least three perf-guard runs with jitter slack, gate on the Linux runner; macOS keeps its own ceilings
- [ ] #3 A deliberate extra directory walk per admission fails the Linux gate (negative control recorded)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. The census appends its measured units to $TLDW_STORAGE_UNIT_CENSUS_LOG on every run; perf-guard prints the file whether the step passes or fails
2. Collect at least three Linux perf-guard runs, pin Linux os.open ceilings with jitter slack, keep the macOS ones
3. Negative control: an extra directory walk per admission fails the Linux gate
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1: the census writes `{case, platform, census}` JSON lines to `$TLDW_STORAGE_UNIT_CENSUS_LOG` (the private-profile child inherits the environment), and perf-guard's storage-unit step sets it to `$RUNNER_TEMP/storage-unit-census.jsonl` and prints it in a log group after pytest, keeping pytest's exit status. Linux pins follow from the runs this records.
<!-- SECTION:NOTES:END -->
