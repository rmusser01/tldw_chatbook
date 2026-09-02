# TASK-21351 Console switcher evidence

Evidence captured on 2026-09-02 from
`codex/task-30001-session-switcher-trust` on macOS 26.5.2 (arm64), Python
3.12.11. These artifacts verify the local Phase 1 switchboard; they make no
claim that the feature is faster than an earlier version.

## Automated behavior

- Focused receipt/marks/FLEET/switcher/outcome suites: passing.
- Production-shaped capture harness: 2 passed.
- Bounded-work benchmark harness: 1 passed.
- Scoped Ruff and `git diff --check`: passing.
- The larger reachable feature rail completed with 766 passed and 36 failed.
  Every reported failure was outside the switcher/receipt test modules, but the
  run is recorded as non-green rather than claimed as baseline-equivalent.
- The full repository suite was not run; repository policy requires a separate
  user opt-in for that sweep.

Reproduction commands:

```bash
../../.venv/bin/pytest \
  Tests/Chat/test_console_activity_receipts.py \
  Tests/Chat/test_conversation_local_marks_service.py \
  Tests/Chat/test_fleet_attention.py \
  Tests/UI/test_console_activity_switcher.py \
  Tests/UI/test_console_activity_outcome_notice.py \
  -q --timeout=60

PYTHONPATH=. ../../.venv/bin/pytest -p Tests.conftest \
  Docs/superpowers/qa/task-21351-console-switcher-activity/capture_evidence.py \
  -q -s --timeout=60

PYTHONPATH=. ../../.venv/bin/pytest -p Tests.conftest \
  Docs/superpowers/qa/task-21351-console-switcher-activity/benchmark_evidence.py \
  -q -s --timeout=60
```

## Compositor captures

- `captures/active-switchboard-120x35.svg`: complete Active grouping and
  operational labels at the maximum approved height.
- `captures/active-switchboard-72x35.svg`: narrow-width fit.
- `captures/history-switchboard-120x35.svg`: bounded History mode.
- `captures/real-ctrl-k-success-selection-160x45.svg`: production Ctrl+K route
  with exact success destination selected.
- `captures/real-success-outcome-notice-160x45.svg`: visible success notice
  before exact acknowledgement.
- `captures/real-failure-mark-seen-160x45.svg`: unsuccessful outcome retaining
  its explicit Mark seen consequence.

The capture assertions additionally enforce a maximum 35-row modal and at most
50 mounted selectable results. The refreshed frames also cover explicit
Active/History scope, consequence-first Enter copy, left-aligned state/title
rows, distilled update counts, theme-semantic accents, and content sizing at
wide and narrow terminal widths.

## Bounded-work measurements

The current machine-local results are in `performance.json`. Key medians:

| Subjects | Modal open | Pure Active filter | Mounted results | Modal rows |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 84.583 ms | 0.006 ms | 5 | 35 |
| 50 | 110.495 ms | 0.055 ms | 50 | 35 |
| 500 | 112.273 ms | 0.544 ms | 50 | 35 |

At 500 subjects, History materialized 50 of 500 reported rows and receipt-cache
refresh materialized the 500 safe receipt records in a median 0.334 ms. Modal
filter timings include the deliberate 200 ms debounce and Textual repaint.

## Native terminal parity status

- iTerm2 is installed and was selected for the macOS equal-cell check, but the
  native accessibility driver could not connect because the host process lacks
  macOS Accessibility/TCC permission. No focus-stealing AppleScript workaround
  was used.
- Windows Terminal parity remains blocked by TASK-20937.6.

Therefore TASK-21351 and TASK-21351.1 remain In Progress. Equal-cell iTerm2 and
Windows Terminal parity is the sole final closeout gate; the automated
production-stylesheet evidence above is available now.
