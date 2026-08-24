---
id: TASK-21280
title: >-
  Library tests see the compact starter rail because the test app factory builds
  a brand-new profile
status: Done
assignee: []
created_date: '2026-08-23 22:05'
updated_date: '2026-08-23 22:57'
labels:
  - testing
  - test-integrity
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The single largest failure cluster in the UI suite. Library tests cannot find the rail
rows they drive, failing with `NoMatches: No nodes match '#library-row-browse-*' on
LibraryScreen()` and equivalents for the rail search input and the Details disclosure.

The rail is behaving correctly. Progressive disclosure composes a compact starter rail —
two rows and an "Explore all tools" button — for a genuinely new profile, and returns
before composing the search input, the Browse and Create sections, and the Details
disclosure. What is wrong is the profile the test harness presents: the shared factory
never writes the persisted lifecycle key, so every app it builds looks brand new, and
the many Library tests written before progressive disclosure existed address rows that
are deliberately not composed for such a profile.

This is the same class of defect the factory already solved once for the first-run setup
wizard, and for the same reason: a test app should present a returning, already-configured
user unless a test asks otherwise.

Note for anyone searching by symptom: TASK-14800 (Done) reports the same exception on the
same selector, but a different and genuine defect — a transient unmount during a rail
recompose, intermittent and load-sensitive. This cluster is deterministic and reproduces
on an idle machine. Same message, different cause.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A factory-built test app presents a Library profile whose rail composes its full Browse and Create sections, its search input and its Details disclosure
- [x] #2 A test that wants a genuinely new profile and the starter rail can still ask for one, and the suites that exercise progressive disclosure are unaffected
- [x] #3 The cause is established by evidence rather than inferred from the message: it is shown to be deterministic rather than a recompose race, and shown to be the disclosure branch rather than a renamed or missing widget
- [x] #4 The recovery is measured on the exact set of nodes that were failing, not asserted from a sample
- [x] #5 The default the fix introduces is itself guarded, including against drift between the value the factory spells and the value the product persists
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce one failing node locally and dump the rail's actual contents, to separate
   "row is late" from "row is never composed".
2. Read the rail composer to find what gates the missing rows.
3. Confirm the gate by flipping it in a probe and re-listing the rail.
4. Fix at the shared factory rather than per test file, mirroring the existing
   `first_run_setup_completed` default.
5. Guard the default, then measure recovery against the harvested failing-node list.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cleared `library_new_profile_admission` in `Tests/UI/app_factory._build_test_app`, and
forwarded the existing `preserve_profile_admission` flag from `test_library_shell.py`'s local
wrapper, which accepted it and then dropped it.

**Cause (AC#3), established rather than inferred.** `app.py:5702` sets
`library_new_profile_admission` from `first_profile_created_this_session()`. The per-test
config sandbox creates a profile for every test, so the flag was True for every app the
shared factory built — each one claiming to be brand new. `Widgets/Library/library_rail.py`
answers that claim correctly: for lifecycle `unknown`/`starter` it composes two rows plus
"Explore all tools" and `return`s, *before* the search input, every Browse/Create section
and the Details disclosure. That early return is why `#library-search-input` and
`#console-rail-section-toggle-library-details` were in the same failure set as the rows.

Two things were ruled out by probe rather than assumed. **Not a recompose race:** with the
shell loaded the rail holds only `library-rail-heading`, `ingest-import-media`,
`create-note`, `rail-explore-all`, and is byte-identical after two further seconds of
pumping. **Not a rename or a missing widget:** clearing the admission flag yields all 15
rows including every selector in the failing set.

**Fix shape.** The first attempt pinned `library.rail_state.lifecycle = "expanded"` in the
config. It worked, but `build_test_app_config` returns `load_settings()`'s cached dict *by
reference*, so it mutated shared settings state, and it pinned a value the screen is meant
to derive. Clearing the admission flag instead lets the screen derive its own: an existing
profile with no persisted lifecycle settles to Expanded, which is the product's own
contract, already pinned by
`test_library_real_existing_config_without_lifecycle_defaults_expanded`. This is also not a
new idea — `test_library_shell.py`, `test_library_file_notes_workspace.py` and
`test_library_prompts_canvas.py` had each hand-rolled the same clearing locally. This
hoists it to the one factory all three go through.

**AC#5 note:** the stated drift risk was between a value the factory spells and the value
the product persists. The final fix spells no lifecycle value at all, so that risk is
dissolved by construction rather than guarded. What is guarded, in
`Tests/UI/test_library_rail_profile_admission.py`, is the default itself, the opt-out, and
— on the DOM rather than on the flag — that the rail composes all seven previously-stranded
selectors.

**Evidence (AC#4), measured on the exact failing set, not a sample.**
The 143 CI nodes carrying the `NoMatches ... on LibraryScreen()` signature were harvested
per-node from all 12 shards of run 32647831275 and committed under
`backlog/docs/test-health-baseline-2026-08-23/ui-cluster-a-nodes.txt`.
- Of the 138 outside `test_library_shell.py`: **130 pass (94.2%)**. The 8 remaining are
  distinct pre-existing causes — 3 are network-egress errors correctly refused by the guard.
- Regression check on `Tests/UI/test_library_shell.py` (the largest shared consumer, 728
  nodes, 23 min): **12 red before, 12 red after, identical sets — zero regressions, zero
  recoveries.** That file was never affected because it already had the local wrapper.
- Every non-passing node in the verification run was cross-checked against the CI red list;
  none is new.

**A caution worth recording.** An earlier diff against the CI list showed "2 regressions".
Re-running one on the pre-change baseline showed it fails there too: it is a pre-existing
local failure that CI's list does not contain, because that run's head is `ac1aa2da5` on a
PR branch and on Linux. Only one of the two was real, and the admission-based fix resolved
it. **A/B against your own before-state; a CI list from another commit is not a baseline.**

Added: `Tests/UI/test_library_rail_profile_admission.py`.
Modified: `Tests/UI/app_factory.py`, `Tests/UI/test_library_shell.py`.
<!-- SECTION:NOTES:END -->
