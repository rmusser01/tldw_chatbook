# Canvas restored-card readiness diagnostic spike

Date: 2026-09-08. User-approved diagnostic only; no fix or Canvas admission.
Baseline: `8b54cbf884` (product/test checkpoint `55f74aa009`).

## Result

The instrumented run reproduced the missing restored-card acknowledgement and
identified the immediate cause: the test-only F12 adapter tried to find a card
before any Canvas cards were mounted. Successful saved-conversation loading was
not a rendered-card readiness barrier.

Only two test files were temporarily instrumented. No production code, assertion,
action retry, deadline, timeout, dependency or release policy changed. The
diagnostic code has been removed from those files and preserved as an encoded patch.
This is evidence of a failed test run, not passing qualification.

ADR required: no. No architecture/authority contract changed; this diagnoses
TASK-31942 AC1/3 under existing [ADR-125](../../../backlog/decisions/125-lock-safe-private-sqlite-validation.md).

## Observed sequence

Times below are relative to replacement-child F10 load completion. New records
contain only fixed stage labels, monotonic time, capped counts and booleans.

| Event | Relative time | Observed state |
| --- | ---: | --- |
| F10 saved-load completes | 0 ms | Sync in progress and replay requested; 0 cards, matches, mounted matches and active-session matches. |
| Parent sees saved-load receipt | +8.574 ms | Existing `loaded-without-provider` acknowledgement passed. |
| F12 adapter enters | +46.416 ms | Same pending-sync state; still 0 cards/matches. |
| Card button / production open | Not reached | Neither marker appears. |
| Replacement `app.run()` returns | +665.845 ms | Lifecycle subsequently observes process exit 0. |
| Parent cleanup | +15,360.571 ms | Both owned children already have exit code 0; cleanup completes. |

The F12 adapter's existing `next(...)` lookup follows the marker synchronously,
without an intervening await. With no cards, that lookup raises `StopIteration`
before the real button action or production selection handler can execute.
The child returns from `app.run()` and later exits with code 0; this run does not
show a native SQLite signal or SIGBUS. Exit 0 is not a successful workflow.

The initial child is a useful in-run comparison: F12 observes two cards, one
matching mounted/current-session card, and neither sync flag set. It then records
button dispatch, production-open entry/return and the selected/pinned receipt.
The probe did not replace those calls with a mocked successful result.

Relevant unchanged call paths:

- `canvas_live_chatbook_child.py:422–446`: F10 awaits the normal opener and writes
  its logical-load result, without a mounted-card condition.
- `canvas_live_chatbook_child.py:451–488`: F12 immediately looks up the exact card
  and presses its real button.
- `chat_screen.py:17460–17469,17579–17594`: an already-running UI sync coalesces
  the request and returns; replay occurs later.
- `chat_screen.py:20360–20380`: the real card handler checks the active session
  before calling production Canvas selection. The failing adapter never reaches it.

## Command and evidence

From the existing isolated `canvas-v1` worktree, using its normal pytest fixtures
and approved owned browser/loopback execution:

```sh
TLDW_CANVAS_TEST_STAGE_DIAGNOSTICS=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/canvas-v1/output/playwright/mermaid-release/stage-spike.yGyBuz ../../.venv/bin/python -m pytest 'Tests/Canvas/browser/test_canvas_served_flow.py::test_actual_chatbook_console_finalizes_canvas_create_and_update[read-publication-False]' -q -ra --tb=short --show-capture=no
```

Result: **1 failed, 1 inherited Requests warning, 59.36s**, exit 1. Failure at
instrumented test line1039 (original line1029): missing second-F12 receipt.
Startup completed on this run. No source `PYTHONPATH` override was used.

Preserved evidence:

- Exact [throwaway patch](2026-09-08-canvas-card-readiness-spike.patch.json), JSON-encoded
  to preserve whitespace faithfully. `jq -j .patch` emits the original patch.
- Complete [stage records](2026-09-08-canvas-card-readiness-stage-records.json), with
  parent/initial/replacement aliases; original timestamps and fields are unchanged.
- Stage records: `output/playwright/mermaid-release/stage-spike.yGyBuz/`;
  `stage-22365.json` is the parent, `stage-22392.json` the initial child,
  `stage-22818.json` the replacement; `lifecycle.json` preserves the new-named
  source-free lifecycle capture. The prior failure captures were not overwritten.
- Complete output and working report: `canvas-stage-spike-run.log` and
  `canvas-stage-spike-report.md` under
  `.superpowers/sdd/2026-09-07-sqlite-lock-safe-private-validation-implementation/`.

The marker stream is capped at 64 rows per process and card counts at 32; new
rows contain no chat text, generated HTML, SQL, URLs, tokens or raw session IDs.
The existing lifecycle recorder retains its original bounded identifiers.

## Limits and recommendation

This directly establishes premature adapter action in this reproduced run. It
does not prove that a deferred sync would eventually finish without that action,
explain every prior crash, or demonstrate production selection after recovery.
Synchronous marker I/O adds timing overhead; no all-schedules claim is made.

The first-byte body-class observation arrived 4882.879ms after login completion,
close to the existing five-second expectation. That is a measured narrow margin,
not proof of the cause of the separate earlier startup failure. No timeout was
increased or suppressed.

Recommended bounded follow-up: make the synthetic card action wait for and
capture the exact mounted, active-session card within the existing test deadline,
then press it once. If it never becomes available, fail with bounded readiness
state. Preserve real production dispatch, session checks and selection assertions.
Keep the separate startup failure open; do not hide it with retries or a blind
timeout increase. A retained harness correction requires its own approval.

Both live test files were restored with an exact reverse patch after checking
that their diff still matched the saved diagnostic. Their Git blob IDs again
match baseline: child `8269fb633734070b6d370e0c0b44de813d4dc2ee`, served test
`21125a46e2df30fdaa573bc676a57d046aa031f2`. Ruff passes before and after restoration;
whitespace checks pass. No post-restoration rerun without new information.

Independent scoped review accepts the empty-card/F12 conclusion, with no Critical
or Important findings. Its two Minors are recorded: use only observed zero-status
exit language (corrected above), and acknowledge that preserving a new-named
lifecycle snapshot plus cleanup-stage markers goes slightly beyond the literal
startup/card-marker wording. Those additional markers were test-only evidence
plumbing, preserved prior captures and cleanup, and were removed with the spike.
The reviewer independently checked the exact patch and raw records; no tests
were rerun and no broader review was reopened.

TASK-31942 remains In Progress, and Canvas V2 remains disabled with all previously
recorded gates still explicit. The approved diagnostic spike is complete; the
recommended retained harness correction has not been implemented.
