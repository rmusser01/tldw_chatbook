# Current-dev server actions — 2026-09-21

## PR2769 merged; integration repair complete — 2026-09-21

[PR2769](https://github.com/rmusser01/tldw_chatbook/pull/2769) merged as
`85e7153158` after successful current-head CI (1,152 main passes, 123 admission
passes and one existing expected failure), zero accumulated Qodo findings,
resolved review threads and independent review of the final import repair.
The actual tree exactly matches the locally tested combination with dev
`6622115c27`; no conflicts or UI changes were introduced. TASK-32825 is Done.

The compact Test Tool inspector is the next separate visual review, TASK-32882:
510 targeted cases, four clean native dark/light compact/wide journeys and
16 inspected captures. It remains In Progress pending its own visual approval,
PR CI/review and final current-dev checks. Other MCP and destination reviews
remain open. Earlier checkpoints below are historical.

[Integration merge receipt](integration-merge-closeout.json) and
[combined-state checks](integration-current-dev.txt).

## Merge and integration follow-up — 2026-09-21

[PR2712](https://github.com/rmusser01/tldw_chatbook/pull/2712) merged as
`4b61a5ca8f` after owner visual approval, successful final-head CI
(1,152 main cases; 123 admission passes and one existing expected failure),
resolved Qodo review with zero findings, and conflict-free current-dev review.
All 14 captured native source hashes match the actual merge.
[Merge receipt](merge-closeout.json) separates the CI-tested base from two later
locally tested dev combinations (40 and 38 focused passes).

PR2744 arrived at merge time with four architecture-test/documentation files;
merged runtime matches the last verified combination. Its new size ratchet
exposed Workbench at 6,771 lines against a 6,761-line ceiling:
[66 architecture passes, one failure](post-merge-architecture.txt).
The bounded follow-up moves the existing three-panel clearing sequence into
MCPInspector, retaining Workbench's async worker, exclusive group and await order.
Workbench is now 6,760 lines and the ceiling is tightened to that value.
[106 targeted cases pass](ratchet-followup.txt); independent review found no
blockers, [no new Ruff diagnostics](ratchet-static.json) and
[all changed ranges formatted](ratchet-format.json).
[All nine artifact guards pass](ratchet-preflight.txt).
Approved native visuals remain applicable to this mechanical move; no new native
run is claimed. [PR2769](https://github.com/rmusser01/tldw_chatbook/pull/2769)
contains this repair; final-head CI and merge checks remain pending.

PR2769 CI exposed a separate current-dev integration regression: PR2735's UTC
timestamp helper raised the UI-ready census from 1,026 to 1,027. Following
existing ADR-097, the Advanced execution readiness constant is now imported at
execution time. Permission, admission and audit behavior stay unchanged; no
budget is raised. The isolated import regression failed before the move and
passes afterward. [All 12 boot/import checks pass](boot-import-followup.txt),
measuring 675/686 imported modules and exactly 1,026/1,026 at UI-ready.
[All eight Advanced execution cases pass](advanced-followup.txt), including
Allow, approved Ask, Off and fail-closed gate errors. Three actual-execution
cases needed the existing bootstrap-profile marker: the same three failed with
`raw_source_selection_changed` on unchanged HEAD. Their original assertions and
per-test MCP stores are preserved. An initial combined run was interrupted
when unrelated cases hit that same pre-existing fixture error; it is not claimed
as passing. [No new Ruff diagnostics](boot-import-static.json) and
[five changed ranges formatted](boot-import-format.json). Independent review
found no blockers in either production or the test-profile markers.

Rebased conflict-free onto dev `fce195590b` (PR2738). Its permission-store
duplicate-key rejection is retained. [34 focused integration cases pass](boot-rebase-followup.txt),
plus both new upstream permission-file regressions (2 passes).

Earlier qualification checkpoints below are historical. Compact Test Tool
reachability remains the next separate review.


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

- [232 distinct targeted cases](test-results.json): server ownership/departure,
  Servers canvas, affected Workbench neighbors, native CLI/fixture admission and
  CSS/token/selector guards. All 31 dedicated ownership/departure cases pass.
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
- Qodo's proposed mount-completion race was not reproduced. Installed Textual
  8.2.8 registers children synchronously before returning `AwaitMount`; completing
  that await cannot publish them again. Two additional
  [mount-overlap cases pass](mount-review-green.txt), with independent source/test
  review supporting that conclusion. The [first harness attempt](mount-review.txt)
  waited for an intentionally cancelled exclusive worker; the corrected test
  observes the replacement's actual mount completion. No production change or
  additional lock was needed. [New test static checks pass](mount-static.txt).

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

The existing PR is ready for review. Current-head CI, accumulated Qodo
review, current-dev/conflict checks and **fresh owner visual approval** still
gate merge. PR2716's approval does not approve this UI slice.
