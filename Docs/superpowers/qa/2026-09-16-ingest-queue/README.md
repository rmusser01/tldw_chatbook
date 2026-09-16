# Import queue keyboard continuity — TASK-32667

Baseline: `631cd05954` on `feat/component-pattern-library`.

Queue transitions rebuilt the focused action and sent keyboard input into
Keywords. Focus is now captured at the panel's actual rebuild, restored before
older action callbacks, and left alone when the user has already moved it.
An unavailable replacement action falls back to the source field.

The canvas explicitly reveals focus above its docked import bar and fold hint.
The settled reveal runs after focus and content-size changes, reads current
focus and stops a running animation. Textual's visibility-based reveal can otherwise do nothing while Retry is still
visible, letting the old animation later move it behind the dock. Native failures
painted a blank Retry row at y18; the controlled animation regression reproduces
that exact geometry. A separate trace confirmed that transient preflight layout
clamps scroll as content shrinks; content can then grow without another focus
event. A deferred virtual-size watcher repairs that retained-focus case, following
the existing Library rail pattern. Retry's form-replacement confirmation also
invalidates the button's measured width so the complete label paints.

ADR required: no. This repairs existing interaction contracts under
[ADR-014](../../../../backlog/decisions/014-library-ingest-service-authority-and-recovery.md),
[ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [ADR-161](../../../../backlog/decisions/161-component-pattern-library.md).
Queue ownership, actual import execution and consent semantics are unchanged.
No CSS or token values changed.

## Verification

[Verification](verification.json) records **203 passing targeted checks**,
including 16 new queue journeys. Five inherited warnings remain. The new journeys cover both themes at 170×48 and 80×24, visible
Tab/Shift+Tab focus, the full Retry confirmation, Details disclosure, progress and
state changes, late focus before rebuild, newer focus after an older action,
disabled-action fallback, re-entry, late layout and earlier scroll animations.
Production Library styles, real temporary media SQLite, real local preflight and
an in-memory job registry are used. Jobs are synthetic UI projections; none runs.

The existing select focus test still checks glyph cues, complete value and
unchanged layout dimensions. It compares virtual geometry because revealing the
control above the dock legitimately changes its screen position. The detached
reveal test follows the helper's renamed method and retains its safety assertions.

[Static comparison](static-comparison.json) records zero new Ruff diagnostics;
inherited diagnostics remain. New files and modified ranges are formatted. The two architecture
ceilings already fail at baseline: LibraryScreen is 35,212 lines against 33,204;
the ingest controller is 3,078 against 2,721. This repair adds two lines to each
for label layout invalidation, without adding methods or raising budgets. The
associated slack checks pass. [Sizes](size-comparison.json) records the comparison.
No full repository suite was run.

## Native verification

[native_check.py](native_check.py) runs actual TldwCli/LinuxDriver under exclusive
ownership of a private profile. All configured database paths, user database base
and data directory point into the run directory; keyring is null. The dependency's
real terminal capability probe runs before Textual takes stdin. This qualifies
that probe-primed startup, not ordinary startup timing.

The native journey enters Import, stages a local text fixture, preserves an
unsaved title, Tabs to Retry, and arms—but does not accept—the form-replacement
confirmation. A replacement registry has neither a store nor a runner. Synthetic
failed/active jobs exercise Details, progress, a writing transition, Retry and
Dismiss focus; those recovery actions are not activated. The fixture queue is
cleared directly and the journey repeats after returning to the hub and resizing.
No model, provider, server, installation or import operation is requested.

| State | 170×48 dark | 80×24 light |
|---|---|---|
| Retry focused | [capture](retry-focus-170.svg) | [capture](retry-focus-80.svg) |
| Full replacement confirmation | [capture](retry-consent-170.svg) | [capture](retry-consent-80.svg) |
| Details after queue transition | [capture](queue-details-170.svg) | [capture](queue-details-80.svg) |

[Native result](result.json), [isolation](isolation.json) and
[persistence checks](persistence.json) contain the final receipts. Final run-011 passes both sizes and exits normally with status 0. Its six SVG
captures were rendered and inspected together as the confirmation batch. Ten
private databases pass integrity checks; media/messages/jobs remain empty, the
source bytes and default-profile hashes are unchanged, and the app log has no
ERROR/CRITICAL lines. See [inspection](inspection.json) and [lifecycle](lifecycle.json). Private databases,
full logs and failed-run captures remain in ignored scratch.

## Investigation and limits

Earlier runs are retained as failures, not overwritten: run-001 exposed focus
loss on a queue transition; run-002 let a late optional terminal capability reply
enter the source field; run-003 reused an intentionally armed confirmation;
run-004/006/009/010 reproduced compact obscured focus. Run-005 omitted the required
private data parent and stopped during import. Run-007 tried focusing Start while
preflight temporarily disabled it; the runner now waits for enabled state.
Instrumented run-008 passed, but did not establish a repair: its timing differed
from uninstrumented run-009. Controlled late-layout and animation tests
identified the remaining causes and are retained as regressions. A general deferred-scroll guard did not repair the
layout clamp and was removed.

A trial resize with focus still in Import transferred focus to the Library rail;
this broader shell breakpoint behavior is not qualified here. Re-entry after
resize is covered. A trial that submitted an active synthetic job while batch
Retry was focused correctly hid Retry under the existing availability rule; it
was not retained as an assertion that Retry should remain available during work.

Remaining review includes grouped outcomes, Clear/Recent, actual recovery and
live-resize focus ownership. This evidence does not qualify real ingestion,
remote authorities, provider behavior or restart. No push or dev integration.
