# Console rail Settings verification — 2026-09-17

TASK-23150 repairs the missed-click tests; TASK-32759 repairs a separate compact
layout defect discovered while checking that diagnosis. Baseline: `0e3b46e021`.
The work remains on draft PR #2704 against dev.

ADR required: no. Existing [ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [ADR-161](../../../../backlog/decisions/161-component-pattern-library.md)
govern this direct composition of existing compact-control and spacing tokens.
No settings ownership or persistence behavior changes.

## Findings and repair

The legacy tests initially failed before reaching their assertions because their
configuration source changed after import. Each of the nine cases now uses the
existing exact-node private-profile helper, preserving its original assertions
and the production recovery guard. With isolation restored, six passed and the
three reported missed-click cases failed (`isolated-baseline.txt`).

Before adding focus or scrolling, an explicit containment assertion measured the
checkbox at `(41,79,36,3)` and the visible pane at `(37,29,111,24)` in all three
cases (`geometry-red.txt`). This confirms the original below-fold diagnosis.
The helper now focuses, waits for scrolling, checks both pane containment and the
actual compositor clip, and presses Space. Original save payload, runtime
activation, failed-save retention, and revert checks remain intact.

Production styling revealed a separate compact defect: at 80×24 the checkbox
extended horizontally beyond the pane (`production-style-red.txt`). Compact
one-row checkbox styling reduced its width to 32 columns, but the inner card clip
was still only 30 (`first-style-attempt.txt`). Removing that inner card's duplicate
horizontal padding at compact widths gives the full label and glyph room. Wide
card padding remains unchanged. Generated Settings CSS was rebuilt from source;
no token values or boot bundle changed.

## Automated evidence

45 distinct targeted cases pass across the recorded runs:

- Nine original rail-label cases in `first-style-attempt.txt`. That run also
  contains the two expected compact failures before the final padding repair.
  These cases use the original destination harness and retain save/failure/revert
  and subsequent Console runtime assertions; adapters are injected in save tests.
- Four production-style cases in `geometry-final.txt`: 190×55 and 80×24, dark and
  light, full viewport/compositor containment, label and glyph paint, keyboard
  staging without premature runtime activation.
- 31 final token, component-pattern, and bundle checks in `governance-final.txt`.
- One boot CSS budget check in `boot-budget.txt`. The unchanged boot bundle is
  620,598 bytes; the ceiling remains 634,050 and the floor 600,000.

Counts do not add overlapping attempts. Ruff check and format pass for the
modified test file and native runner; `git diff --check` and the backlog ID guard
pass. No full repository suite was run.

## Native evidence

`native_check.py` ran the real `TldwCli.run` with LinuxDriver and all rendering
streams attached to an owned tmux terminal. HOME, USERPROFILE, config and data
roots were selected inside a fresh private profile before app imports. The
terminal capability probe ran before app startup.

Final run `/private/tmp/tldw-32759-native-002`, PID 86993, passed four cells:
190×55 and 80×24 in textual-dark and textual-light. Each used Settings search to
focus the rail control, asserted complete clipping containment, toggled with
Space, checked the full label/glyph paint, and toggled back without writing config
or changing the active runtime value. All four SVGs and terminal captures were
inspected. `native-result.json` pins runner and source hashes;
`capture-manifest.json` pins capture bytes.

`lifecycle.json` records app.run returning, exit 0, absent PID, released instance
lock (reacquired non-blockingly), closed owned terminal, normal app-stopping and
Settings-unmount logs, no unhandled/ERROR logs, empty faulthandler, 11 healthy
private databases, zero conversations/messages and unchanged default-profile
fingerprints. Native Save and persistence failure injection are not claimed.

The first native attempt is retained in `native-attempt-001.json`: one wide cell
passed, the compact containment assertion failed, app.run returned and exit 1
was observed. PID 86238 was absent before the final fresh run. It is diagnostic
evidence, not a successful qualification.

Independent read-only review found no actionable code/test findings. Final
lifecycle and test checks were completed after that review.

## Remaining scope

This qualifies the rail-label setting and the original missed-click tests only.
Other Console Behavior controls, instant-apply groups, save/validation journeys,
resize behavior and Storage still need their own review. Initial broader Console
and Storage tests hit the known profile-lifetime fixture failure before UI
behavior; that setup issue is not evidence of a production storage defect.
