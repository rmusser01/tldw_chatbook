# Task 1 Report: Census anti-slack guard (task-27019) + settings-row follow-up filing

## Chosen tolerance + reasoning

`_CENSUS_SLACK_TOLERANCE = 5`, defined and documented in
`Tests/UI/test_library_recompose_ratchet.py` right after
`LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX`.

The size ratchet's own tolerance (200 lines / 10 methods, on a ~44,000-line /
~1,300-method budget) does not transfer by scaling: applied proportionally to
a census of 63 it rounds to <1, which is meaningless for an integer count.
That ratio was never the actual mechanism, either — the size ratchet's 200/10
is sized to absorb *ordinary, unrelated in-file edits* that grow or shrink a
screen's line/method count for reasons that have nothing to do with a
decomposition wave (any edit to a 44k-line file moves the line count a
little). This census has no equivalent noise floor: a
`self.refresh(recompose=True)` / `self.recompose()` statement is never an
incidental byproduct of unrelated work. The count essentially only moves when
someone deliberately adds, removes, or relocates a whole-screen recompose
site — exactly the kind of change (a targeted-seam conversion) this ratchet
exists to police.

So the tolerance is instead sized against this census's own documented drift
history, recorded in the file's existing comments:
- 107 → 80: 23 sites of silent headroom before TASK-22228 caught it.
- 74 → 63: 11 sites of silent headroom before TASK-3's surface-widening audit
  caught it.
- The smallest *single* silent step on record inside that first drift is 6
  sites — "routing the six Reader sub-state presses through
  `_sync_library_media_viewer_or_recompose`" (80 → 74).

Tolerance = 5 sits strictly below that smallest observed step (6), so the
guard would have fired on every drift increment ever recorded against this
pin, including its smallest one, while still forgiving a couple of sites of
legitimate same-PR churn (e.g. sites shuffling across files during the
widened multi-file scan without a net change).

## The four proof outputs

All four runs used `perl -e 'alarm N; exec @ARGV' .venv/bin/python -m pytest
Tests/UI/test_library_recompose_ratchet.py::test_census_pin_is_not_left_slack`
against scratch edits of the `LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX` constant
only, each restored before the next edit. Real measured count throughout:
**63**.

### 1. Fail-when-slack (headroom well beyond tolerance): pin = 73 (slack = 10)

```
FAILED Tests/UI/test_library_recompose_ratchet.py::test_census_pin_is_not_left_slack
E       AssertionError: The Library recompose census pin is 10 sites above the measured count (63 vs pin 73), more than the 5-site tolerance (Tests/UI/test_library_recompose_ratchet.py, _CENSUS_SLACK_TOLERANCE). A wave landed without lowering the ratchet -- set LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX to 63 in the same PR so the gain is locked in, per this file's module docstring and TASK-27019.
E       assert 10 <= 5
1 failed in 25.67s
```

### 2. Boundary fail (one site over tolerance): pin = 69 (slack = 6)

```
FAILED Tests/UI/test_library_recompose_ratchet.py::test_census_pin_is_not_left_slack
E       AssertionError: The Library recompose census pin is 6 sites above the measured count (63 vs pin 69), more than the 5-site tolerance (Tests/UI/test_library_recompose_ratchet.py, _CENSUS_SLACK_TOLERANCE). A wave landed without lowering the ratchet -- set LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX to 63 in the same PR so the gain is locked in, per this file's module docstring and TASK-27019.
E       assert 6 <= 5
1 failed in 25.93s
```

### 3. Boundary pass (exactly at tolerance): pin = 68 (slack = 5)

```
1 passed in 25.69s
```

### 4. Pass-at-pin (real value restored): pin = 63 (slack = 0)

```
6 passed in 50.91s
```
(full-file run: `test_library_screen_whole_screen_recompose_count_is_ratcheted`,
`test_census_pin_is_not_left_slack`, `test_ratchet_counter_measures_its_subject`,
`test_ratchet_counter_blind_spots_are_the_documented_ones`,
`test_census_exemption_matches_only_the_documented_pair`,
`test_ratchet_failure_message_names_the_offending_sites` — all PASSED)

After the boundary runs, the working tree was diffed against `git diff` to
confirm the pin constant landed back at exactly `63` with zero stray
mutation artifacts (`git diff --stat` showed 62 pure insertions, 0
deletions, once the scratch edits were all reverted).

## Mutation testing, both directions (AC #2)

- **Headroom injected → fails**: proofs 1 and 2 above (slack 10 and slack 6,
  the latter being the tightest possible failing case — one site past
  tolerance).
- **Correct/near pin → passes**: proofs 3 and 4 above (slack 5, the tightest
  possible passing case, and slack 0, the real value).

This nails the exact tolerance boundary in both directions, not just two
arbitrary points.

## Backlog CLI outputs

```
$ backlog task edit 27019 -s "In Progress" -a @claude
Updated task TASK-27019

$ backlog task edit 27019 --plan "..."
Updated task TASK-27019

$ backlog task edit 27019 --check-ac 1 --check-ac 2
Updated task TASK-27019

$ backlog task edit 27019 -s Done --notes "..."
Updated task TASK-27019
```

`backlog task 27019 --plain` after closeout shows Status: Done, both ACs
checked (`[x]`), and the full Implementation Notes section present.

```
$ backlog task create "settings_screen.py needs a size-ratchet budget row" \
    -d "Spec 2026-09-01 non-goal follow-up: the ratchet that let library_screen triple also has no settings row; settings_screen.py was 15,922 lines at the 2026-08-02 doctrine baseline." \
    --ac "Budget row added at measured values" \
    --ac "Mutation-checked (dummy method -> fails)"
Created task TASK-27020
File: backlog/tasks/task-27020 - settings_screen.py-needs-a-size-ratchet-budget-row.md
```

`backlog task 27020 --plain` confirms the two `--ac` flags rendered as two
independently-checkable criteria (`#1`, `#2`), not comma-collapsed into one
run-on item — the known CLI trap from `lessons-backlog-hygiene.md`.

## Files changed

- `Tests/UI/test_library_recompose_ratchet.py` — added
  `_CENSUS_SLACK_TOLERANCE = 5` (with its full reasoning docstring) and
  `test_census_pin_is_not_left_slack`. 62 pure insertions, 0 deletions, no
  other lines touched.
- `backlog/tasks/task-27019 - Recompose-census-needs-an-anti-slack-guard-like-the-size-ratchet.md` —
  status → Done, both ACs checked, Implementation Plan and Implementation
  Notes added.
- `backlog/tasks/task-27020 - settings_screen.py-needs-a-size-ratchet-budget-row.md` —
  new file, filed via the CLI per the brief's exact command.

## Verification

- `Tests/UI/test_library_recompose_ratchet.py`: 6 passed (full file, real
  pin=63).
- `./scripts/preflight.sh`: all six checks green (CSS bundle, profile-owned
  path census, production diagnostic inventory, backlog task ids — "No
  duplicate task IDs across 2923 task files" — chachanotes table allowlist,
  index plan pins).
- Commit `477704580` on `refactor/library-decomp-wave2-cold-trio`:
  `test(library): census anti-slack guard (task-27019); file settings ratchet-row task`.

## Self-review

- Diff is additive-only (62 insertions, 0 deletions) to the test file —
  no accidental reformatting or reordering of existing tests.
- The new test follows the file's existing style: same AST-scan helper
  (`_widened_library_surface_sites`), same guidance-in-assertion-message
  pattern as the ceiling test and the size-ratchet model, references
  TASK-27019 and the exact constant name in its failure message so a future
  reader can find the tolerance rationale immediately.
- Tolerance choice is justified from this file's own drift history (already
  documented in the file's comments) rather than an arbitrary round number
  or a naive proportional scaling from the size ratchet — the brief
  explicitly warned the absolute number wouldn't transfer, and the
  docstring states why in both directions (why NOT scaling, and what WAS
  used instead).
- Mutation testing covered the tight boundary (5 vs 6) in addition to an
  obviously-over-tolerance case (10), which is a stronger proof than the
  brief's minimum ask ("headroom injected → fails; exact pin → passes").
- No production code touched — this task is test-file and backlog-file only,
  matching the brief's stated file scope.
- Confirmed via `git diff` after every scratch mutation that the pin
  constant's restored value was byte-identical to its pre-task state (`git
  diff --stat` showed only insertions for the final committed diff).
- One thing worth flagging for whoever picks up Tasks 2-9: this file's
  full-scan tests are slow (~24-25s each, since they AST-parse
  `library_screen.py` at ~44k lines plus every `Library_Modules/*.py` file
  per test, uncached) — pre-existing behavior, not introduced by this
  change, but worth knowing before assuming a hang.

## Concerns

None blocking. The only note is the pre-existing ~25s-per-scan cost on this
test file (unrelated to this change, not a regression).
