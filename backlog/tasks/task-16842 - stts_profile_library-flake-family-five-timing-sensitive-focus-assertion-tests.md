---
id: TASK-16842
title: 'stts_profile_library flake family: five timing-sensitive focus-assertion tests'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-16'
labels:
  - test-health
  - tts
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_stts_profile_library.py` carries a pre-existing flake family that two
independent reviews hit and characterized (task-15772 review round 2, PR #1691; task-15771
review F4, PR #1699). It is broader than any single pair of tests: across the 15772
reviewer's three full-file runs, **five distinct tests** flaked —

- `test_reference_export_defaults_sanitized_and_bundle_requires_ack`
- `test_windows_clone_export_keeps_sanitized_default_and_disables_bundle`
- `test_delete_shows_advisory_count_but_repository_conflict_is_final`
- `test_unavailable_profile_disables_playground_action_with_clear_recovery`
- `test_import_warns_before_picker_and_stale_successor_requires_reconfirm`

— and the first **reproduced standalone** (single test, own process):
`AssertionError: assert (None is not None)` on `app.focused.id == "bundle-warning-ack"`
after `_wait_until` confirmed the button was *mounted*. So the root cause looks like each
test's own internal focus-settle race (mounted ≠ focused yet), not cross-test pollution.
The 15771 review saw the family degrade under machine load (3 failed normally, 14 failed
in a run at 2x wall-clock). At dev `ee741cf10` the file has had no stabilization commit
since (last touched by 15772's own fix), and one standalone re-run of the first test
passed — consistent with intermittency, not with a fix having landed.

Root-cause the focus-settle pattern (likely one shared helper/idiom around
export/bundle-ack focus) and make the family deterministic — condition-polls on the
actual focus state, not mounted-state proxies or fixed sleeps (the repo's GGUF-settle
lessons apply).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize: run the full file 6x (3 unloaded, 3 under a parallel CPU load) and the
   known standalone reproducer (`test_reference_export_defaults_sanitized_and_bundle_requires_ack`)
   5x in isolation; build the victim table (test, assertion, timing).
2. Root-cause: read the failing assertions' setup (the `_wait_until` mounted-proxy +
   focus-assert idiom, `_select_action_profile`, modal `on_mount` focus scheduling) and
   state the mechanism.
3. Fix the class, not one test: if the race is the shared mounted!=focused idiom, make the
   settle condition poll the *actual* asserted state (focus, availability projection) via
   the shared helper — bounded condition-polls, no fixed sleeps.
4. Re-run the same stress protocol after the fix (6 full-file runs incl. 3 under load +
   standalone x5) and record the before/after table. ruff on touched files.
5. If the race turns out to be a real product focus bug (modal can end up unfocused for a
   user too), stop and escalate instead of loosening the tests.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The mechanism of the focus-settle race is identified and stated (not just retried around)
- [x] #2 All five named tests pass repeatedly under load (e.g. 10 consecutive full-file runs, at least a few under parallel CPU load), with the run evidence recorded
- [x] #3 No fixed-duration sleep is introduced as the fix
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Characterization (before).** File = `Tests/UI/test_stts_profile_library.py`
(163 tests), identical to origin/dev at start. Unloaded: 3/3 full-file runs
green (~62s each). Under 14 parallel CPU burners (14 logical cores): first
loaded run reproduced exactly the two recurring victims —
`test_windows_clone_export_keeps_sanitized_default_and_disables_bundle`
(`assert app.focused is sanitized` → focused was None) and
`test_reference_export_defaults_sanitized_and_bundle_requires_ack`
(`assert app.focused is not None` → None), both immediately after a
`_wait_until` that had confirmed the modal button was *mounted*.

**Mechanism (AC #1) — three stacked causes, none a product bug:**
1. `Widget.focus()` (Textual 8.2.8) defers `screen.set_focus` via
   `app.call_later` → one pump callback AFTER the modal's children become
   queryable. Mounted ≠ focused; `app.focused` is None in that window. Not
   user-visible: the callback message is FIFO-ordered before any input that
   arrives after the modal is on screen — so no product escalation.
2. `pilot.pause(delay)` is a bare `asyncio.sleep`; `pilot.pause()`'s
   `wait_for_idle` is a CPU-idleness heuristic (process-time vs wall-clock).
   An externally loaded machine starves the process, which then reads as
   idle while its queue still holds `RowSelected` / `Callback(set_focus)` /
   `Checkbox.Changed`. Hence the fixed `attempts=100` (~1s nominal) budget in
   `_wait_until` exhausted under load, and every single-pause-then-assert
   idiom sampled pre-settle state — which is why five different tests rotated
   as victims (15771 saw 14 fail at 2x load).
3. `Button.press()` returns early while disabled — a one-shot press issued
   after a heuristic pause (Continue right after toggling the consent
   checkbox; export/action buttons right after row selection) is silently
   swallowed and never retried, surfacing later as an unrelated timeout.
   Availability projection is a fourth settle target: it lands via an async
   observe→apply flow after rows render (preview label "Checking" until
   then), and production correctly re-syncs buttons + detail copy on arrival
   (verified in `_publish_availability`/`_sync_selected_actions`).

**Class fix (test file only, no production change, no fixed sleeps — AC #3):**
- `_wait_until`: attempt-count budget → monotonic wall-clock deadline
  (15s), same 10ms condition-poll; exits on first true predicate.
- New shared settles polling the actual asserted condition:
  `_wait_selected` (selection landed), `_wait_focused` (focus state itself,
  not mounted proxies), `_wait_availability_projected` (label != "Checking"),
  `_acknowledge_bundle_warning` (toggle, settle `not disabled`, then press —
  replaces all six toggle/pause/press sequences).
- Applied at every assert-after-pause site of the class: the five named
  victims plus the same-idiom siblings (repository-page arming, unverified-
  legacy copy, long-identifiers focus pair, stale-rows generation-6
  selection, same-page-refresh selection read, playground text-input focus,
  consent-modal initial focus, review-modal focus walk, windows-import
  arming). `ruff format` also swept a handful of pre-existing unformatted
  constructs in the file (baseline was not format-clean); semantics-neutral.

**Stress evidence (after, AC #2):** 10/10 consecutive full-file runs green —
6 unloaded (~61s) + 4 under the same 14-burner CPU load (79–92s) — plus the
prior standalone reproducer (`test_reference_export_...`) 5/5 in isolation.
Zero failures. ruff check + format clean on the touched file.

**Lesson filed:** `backlog/docs/lessons-testing-evidence.md` — "pilot.pause()
is a CPU-idleness heuristic, not a queue drain".
<!-- SECTION:NOTES:END -->
