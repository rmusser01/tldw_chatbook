---
id: TASK-15668
title: 'Repair the two Settings workspaces-category test failures (unowned dev red)'
status: Done
assignee:
  - rmusser01
created_date: '2026-08-11 21:30'
labels:
  - settings
  - tests
  - baseline
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two tests in `Tests/UI/test_settings_workspaces_category.py` fail and were verified pre-existing at commit 2e26bbcad, i.e. not introduced by the supervisor-fleet work. They are unowned. Leaving them red erodes the value of the whole UI suite as a gate, which is the same argument task-3070 was filed on.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both tests pass on dev, or are removed with a recorded reason if the behaviour they assert is gone
- [x] #2 The diagnosis names which commit changed the behaviour, established from history rather than assumed
- [x] #3 Tests/UI/test_settings_workspaces_category.py runs with zero failures
<!-- AC:END -->

## Implementation Plan

1. Verify the premise at current dev base `ecc0a531c8`: run
   `Tests/UI/test_settings_workspaces_category.py` whole; classify any reds
   per the dev-red method (fixed upstream / live defect / known class).
2. Establish from history which commit the two August failures trace to and
   which commit resolved them: commit log of the test file across
   2026-08-11 (filing, verified at `2e26bbcad`) to now; contemporaneous
   records (task-15472 review round 1, the 2026-08-13 sweep inventory, the
   task-15791 closeout ledger, task-18960); confirm whether any production
   commit in the window touched the workspaces-pane surface.
3. A/B the historical test file (2e26bbcad) against current production via a
   file swap (never `git stash`) only if it clarifies the diagnosis.
4. Close with the classification, exact commands and results, and the
   ADR check.

## Implementation Notes

**Classification: fixed upstream / never a durable defect — closed with
evidence, no code change.**

**Verification at current base** (`ecc0a531c8`, this session, under heavy
shared-machine load with other sessions' pytest fleets running):

- `python -m pytest Tests/UI/test_settings_workspaces_category.py -q
  --no-header` → **20 passed, 0 failed** (152.30s). AC#3 holds; the file has
  grown from 13 tests at filing to 20, every one green.

**Diagnosis from history (AC#2).** The premise did not survive: there is no
repair commit to name because the two reds were never a durable code defect.

- The test file was **byte-identical from 2026-08-02 (`be90c95381`) until
  2026-08-18 (`be1a6d45fb`)** — `git log -- Tests/UI/
  test_settings_workspaces_category.py` shows no commit in between. So at
  the 2026-08-11 filing (verified at `2e26bbcad`, 2026-08-11 17:21) and for
  four more days, the file's content was exactly what it had been all
  August.
- It was recorded red twice in that window — the filing (08-11) and the
  2026-08-13 full-suite sweep inventory
  (`Docs/Design/2026-08-13-tests-ui-sweep-inventory.md`: "2
  test_settings_workspaces_category.py", run as 4-way contended chunks) —
  and then verified **green whole with no change** by the task-15791
  closeout (2026-08-14/15): "Personas plus Settings 159 passed", "settings
  modules 174/174", with the settings rows attributed
  resolved/no-action. No production commit in the 08-11..08-15 window
  touched the workspaces-pane surface (the settings commits those days were
  audio-cpp/TTS/provider-lifecycle/keyring/model-catalog-consent).
- Contemporaneous records independently attribute this exact file's
  intermittent reds to load/order flakiness, with the failing tests varying
  run to run:
  - `2fa60adef0` (2026-08-11 18:21 — three hours BEFORE this task was
    filed), task-15472 review round 1: "test_settings_workspaces_category.py
    has PRE-EXISTING, order-independent flakiness of its own — different
    tests in that file fail across repeated standalone runs."
  - task-18960 (committed `997227855b`, 2026-08-20): 3 failures in this
    file under a concurrent CPU-competing sweep, with `event_loop_stall`
    diagnostics up to 4.5s lag; green when isolated and on 3x repeat.
- The flake class was later structurally closed upstream by
  **`149acda36b` (2026-09-18, PR #2707)**, which converted every test in
  the file to `@private_profile_test` (each case re-run in an isolated
  child pytest with a fresh profile) and moved the splash gate to the
  `patch_app_global` seam. An A/B swap of the 08-11 file (via
  `git show 2e26bbcad:...` + copy, restored immediately; no `git stash`)
  against current production confirms the old harness no longer matches
  production's patch surface — the August pair is unreproducible today by
  construction, and its 20 descendants pass.

So for AC#1's first arm: the two tests' current descendants all pass on
dev; nothing needed removal.

ADR required: no
Reason: no code changed; test-health closeout with a history-based
diagnosis only. No storage, boundary, or interface decision involved.

Files changed: this task file only.

**For the class owner:** the supervisor-fleet-era reds in this file were
load flakes; the structural fix (#2707's private-profile isolation) is
already on dev. Nothing further owed here.
