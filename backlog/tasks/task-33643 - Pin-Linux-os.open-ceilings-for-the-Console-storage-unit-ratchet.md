---
id: TASK-33643
title: Pin Linux os.open ceilings for the Console storage-unit ratchet
status: Done
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
- [x] #2 Linux-specific os.open ceilings, pinned from at least three perf-guard runs with jitter slack, gate on the Linux runner; macOS keeps its own ceilings
- [x] #3 A deliberate extra directory walk per admission fails the Linux gate (negative control recorded)
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

AC#2: `LINUX_OS_OPENS_CEILINGS` pins each phase at the highest value seen across three perf-guard runs (six census lines, both evidence variants), with the shared 1.05 jitter slack: typing burst 81 (seen 54-81), typing pause 206 (153-206), credential poll 6.75/tick (3.375-6.75), trace maintenance 23.125/tick (16.5-23.125), trace GC pass 26, visit 1,926 (1,913-1,926). `_ceiling()` gates os.open on the macOS dicts on darwin, on the Linux dict on linux, and not at all elsewhere; admissions and helper spawns gate everywhere. `test_each_platform_gates_os_opens_on_its_own_ceilings` pins the selection.

AC#3: a throwaway draft PR (#2972, closed unmerged) added one extra directory walk per storage admission (an `os.open` per component of the bootstrap root's chain in `_acquire_storage`). Perf-guard run 37097909746 failed the Linux gate in both variants: typing pause 219/246 > 206, trace tick 32.75/41.75 > 23.125, trace GC pass 81 > 26, visit 2,586/2,559 > 1,926; admission counts were unchanged, as expected.
<!-- SECTION:NOTES:END -->
