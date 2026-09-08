# Canvas restored-card harness readiness correction

Date: 2026-09-08. Approved bounded follow-up to the
[stage-marker diagnostic](2026-09-08-canvas-card-readiness-spike.md).
Baseline: `a5b3159b89`. Implementation checkpoint: `72af5b63bd`.

## Change

The test-only F12 adapter now waits cooperatively for the exact revision's
mounted card in the current nonempty Console session, with its mounted enabled
exact-revision button. It captures the card/button pair and presses that button
once, without another query or await. Timeout state uses capped counts, booleans
and fixed labels; cancellation propagates.

The existing real card dispatch, production session guard, open-completion
acknowledgement and selected/pinned assertions are retained. The browser test's
first receipt wait (`100 * 0.02s`) and restored receipt wait (`300 * 0.05s`) are
unchanged. No startup expectation or production file changed.

Changed code is limited to:

- `Tests/Canvas/browser/canvas_live_chatbook_child.py`
- `Tests/Canvas/test_live_card_readiness.py`

ADR required: no. This is a test-harness correction under TASK-31942, with
[ADR-125](../../../backlog/decisions/125-lock-safe-private-sqlite-validation.md)
unchanged. There is no new runtime, storage or authority contract.

## Verification

The new helper's initial RED was six failures for its missing API; GREEN was
six passes. Additional boundary coverage and the existing receipt regression
passed together: ten tests. Real Textual cards, buttons and open-request events
exercise delayed mounting, exact/current-session identity, disabled/missing
targets, timeout and cancellation. No provider or browser is needed for these
focused helper tests.

Root reran the committed focused selection:

```sh
../../.venv/bin/python -m pytest Tests/Canvas/test_live_card_readiness.py Tests/Canvas/test_live_receipt.py -q -ra --tb=short --show-capture=no
```

Result: **10 passed, 1 inherited Requests warning, 3.10s**, exit 0.

Root also ran the original failing actual-browser node once, with code frozen:

```sh
../../.venv/bin/python -m pytest 'Tests/Canvas/browser/test_canvas_served_flow.py::test_actual_chatbook_console_finalizes_canvas_create_and_update[read-publication-False]' -q -ra --tb=short --show-capture=no
```

Result: **1 passed, 1 inherited Requests warning, 49.52s**. The unchanged case
checks restored exact/pinned selection, original metadata and persisted rows,
zero restored provider calls, rejection of the old URL, and owned-child cleanup.
The completed pytest output was preserved before an orchestration-wrapper
serialization error on the absent session ID; its separate exit-code field was
not retained. The test was not rerun to replace that reporting artifact.

Changed-file Ruff check and format check pass; both files are formatted, and
whitespace checks pass. No shared dependency was changed to silence the existing
Requests warning. No broad test suite or prior benchmark was rerun.

Full commands/results and review inputs are retained under
`.superpowers/sdd/2026-09-07-sqlite-lock-safe-private-validation-implementation/`:
`task-12-report.md`, `task-12-live-run.log`, `task-12-committed-unit-run.log` and
the scoped review package. The prior lifecycle capture was preserved before the
existing browser harness reused its usual evidence path.

## Review and limits

Independent task-scoped review approved spec compliance and quality with no
Critical or Important findings. Its one Minor is the inherited Requests warning
already disclosed above. The reviewer checked unchanged receipt/selection
assertions; root supplied the completed browser-run evidence that a diff alone
cannot establish. This is not whole-correction or Canvas admission approval.

This run reaches and passes the previously failing restored-card boundary. It
does not explain the separate historical first-byte startup failure or every
native SQLite failure. Existing host semaphore, platform/optional and aggregate
static-check gaps remain explicit. TASK-31942 remains In Progress, and Canvas V2
remains disabled. No PR, push, rebase, merge, host cleanup or dependency action.
