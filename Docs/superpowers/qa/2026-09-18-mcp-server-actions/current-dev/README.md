# Current-dev server actions — 2026-09-21

PR2712 keeps every server toolbar press bound to the profile displayed when its
control was created. Retired, hidden, covered or disabled controls cannot act.
Mode departure revokes queued actions synchronously, including a quick return
to Servers. An accepted confirmation retains its original target while another
selection or toolbar refresh proceeds. Compact toolbars use the existing
two-column, token-backed action pattern; wide layout is unchanged.

Base: merged PR2716 dev `9cf5ba67b2396e67c16f3ad7157066b1f7bf740f`.
Saved implementation `0bc91c1ba7` applied without conflicts; the current dev
Workbench lifecycle and cancellation changes were retained. New timing repairs
were made explicitly, rather than selecting one branch's entire files.
ADR required: no new ADR. Existing ADR-150 and
[ADR-161](../../../../../backlog/decisions/161-component-pattern-library.md)
apply; no service, storage, permission or runtime boundary changes.

## Verification

- [230 distinct targeted cases](test-results.json): server ownership/departure,
  Servers canvas, affected Workbench neighbors, native CLI/fixture admission and
  CSS/token/selector guards. All 29 dedicated ownership/departure cases pass.
  The [initial Workbench run](workbench.txt) had one stale test double; the
  [updated deferred-worker contract](worker-contract.txt) passes and verifies
  lazy async callable scheduling with the captured retirement revision.
- [Saved tests on merged dev](baseline.txt): 15 failed, 2 passed; additional
  [mode/disabled/covered regressions](departure-baseline.txt): 9 failed. Applying
  only the saved patch left [7 failures](saved-patch.txt).
- Independent review exposed two further timing gaps. A delayed departure
  refresh discarded a newer Delete press ([red](fresh-consent-red.txt));
  overlapping toolbar pruning could raise `DuplicateIds` before accepted
  deletion dispatched ([red](overlap-red.txt)). Revisions now preserve newer
  controls and suppress superseded publication. All
  [12 departure cases pass](overlap-green.txt).
- [48 CSS/token/selector cases pass](css.txt), including generated bundle
  integrity and the existing selector candidate ceiling. No token values change.
- [All eight derived-artifact guards pass](preflight.txt).
  [Scoped static comparison](static.json) finds no introduced Ruff findings;
  new Python files and modified ranges pass formatting.
- [Independent production and runner review](independent-review.md) found no
  remaining blockers after the timing repairs. No full-suite sweep was run.

## Native evidence

[Current runner](native_check.py), [launch provenance](native-launch.json),
[four real native journeys](native-result.json), and
[clean shutdown](native-lifecycle.json). The real TldwCli uses LinuxDriver and
TTY streams, private configuration/data, real profile validation and real local
profile persistence. Controlled queue and rendering delays expose action races;
service operations and persistence are not replaced. Fixture commands are
`/usr/bin/false`; these journeys intentionally do not connect to external servers.

Each theme/size verifies actual Keep activation and Escape, a queued confirmation
after a mode round trip, a retired Connect after selection changes, and accepted
Alpha deletion while Beta becomes selected. Both fixture IDs are checked for
collisions before creation. Cleanup removes only successfully created IDs and
leaves the original catalog unchanged. Positive compositor dimensions, complete
clipping bounds, painted labels and center hit targets are asserted for activated
controls. Captures wait for notifications to clear.

| Theme / terminal | Safe confirmation | Round trip kept profiles | Retired Connect ignored | Alpha deleted, Beta retained |
| --- | --- | --- | --- | --- |
| Dark / 80×24 | [View](textual-dark-80x24-safe-confirmation.svg) | [View](textual-dark-80x24-round-trip-kept-profiles.svg) | [View](textual-dark-80x24-retired-action-ignored.svg) | [View](textual-dark-80x24-beta-preserved.svg) |
| Light / 80×24 | [View](textual-light-80x24-safe-confirmation.svg) | [View](textual-light-80x24-round-trip-kept-profiles.svg) | [View](textual-light-80x24-retired-action-ignored.svg) | [View](textual-light-80x24-beta-preserved.svg) |
| Dark / 170×48 | [View](textual-dark-170x48-safe-confirmation.svg) | [View](textual-dark-170x48-round-trip-kept-profiles.svg) | [View](textual-dark-170x48-retired-action-ignored.svg) | [View](textual-dark-170x48-beta-preserved.svg) |
| Light / 170×48 | [View](textual-light-170x48-safe-confirmation.svg) | [View](textual-light-170x48-round-trip-kept-profiles.svg) | [View](textual-light-170x48-retired-action-ignored.svg) | [View](textual-light-170x48-beta-preserved.svg) |

All 16 captures were rendered and inspected. Toolbar and confirmation labels fit;
the compact server rail still truncates its longer rows, an existing separate
layout issue. The connected four-button toolbar is qualified by the real-app
headless geometry/paint tests; the native journeys use disconnected profiles.
Compact inspector Test Tool reachability, Audit and other screens remain separate
reviews. This slice does not qualify complete MCP lifecycle or permission flows.

App return and process exit are zero; the app PID is absent, the instance lock
can be reacquired, all ten private databases pass `quick_check`, and no
conversation/message rows remain. Default-file fingerprints and all 14 captured
source hashes match; no network attempts, application errors or faulthandler
output occurred. The owned tmux session was closed. The runner hash matches.
[Export manifest](export-manifest.json) records original and whitespace-normalized
artifact hashes; original captures remain in the owned private profile.

The existing draft PR is updated separately. Current-head CI, accumulated Qodo
review, current-dev/conflict checks and **fresh owner visual approval** still
gate merge. PR2716's approval does not approve this UI slice.
