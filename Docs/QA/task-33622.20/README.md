# Character switcher quit-interruption recovery

2026-10-06; branch `codex/task-33622-20-switcher-quit-interruption`,
starting dev `dce291d0fa8c5e27bc999656766d96bd6b00626c`.
Existing ADR-031/120 apply; no new activation outcome or relaxed overlay proof.

## Behavior and boundaries

Only the exact switcher-owned quit question covering an otherwise current
activation explains **Open interrupted by quit confirmation**. The question
remains on top, the canonical opener rolls back its cold runtime and restores
the prior session, and Wait does not automatically replay the open. Query,
selection and explicit recovery survive the incumbent live Active updates;
the retained row's open/current badge still tracks the actual runtime.

Retry, Enter and row click for that retained target use the same fresh immutable
results snapshot. They open only the same typed conversation, with unchanged
authority/revision revalidation and mount, query, mode, epoch, generation,
stack and committed-payload fences. A missing result cannot transfer its action
to a neighboring row.

Conservative `presentation_refused` provenance remains false on real opening,
presentation, rollback and prior-restoration errors. Production best-effort
restoration reports its swallowed repaint/focus errors through its existing
adapter; legacy callers still ignore that result and retain best-effort behavior.
An explicit failed ownership rollback also clears refusal provenance. These
failures retain **Could not open chat**, not the interruption explanation.

## RED / GREEN

- Original copy RED: `/tmp/task33622-20-red-copy-uXGH8B/red.log`,
  two owned-question failures at 120x50/52x20, three negative controls passed.
- Cold Retry RED: `/tmp/task33622-20-retry-red-3uJL4e/retry-red.log`,
  the actual new attempt settled FAILED with its old revision.
- Clean-refusal provenance RED:
  `/tmp/task33622-20-provenance-red-NotfYr/red.log`, one clean-refusal failure,
  four genuine-failure controls passed.
- Enter/click RED: `/tmp/task33622-20-sibling-red-XPXMCY/red.log`,
  both installed inputs settled FAILED. Its first restore tests had an invalid
  adapter assignment, so those failures are not restoration evidence.
- Corrected production restoration RED:
  `/tmp/task33622-20-restore-red-corrected-wPUrkB/red.log`,
  two swallowed-error failures and one clean recovery control passed.
- Final focused GREEN: `/tmp/task33622-20-green-qualified-8LXb03/green.log`,
  **23 passed, 44 deselected in 117.56 s**, no pytest warnings. Real installed
  cold rollback, exact Retry/Enter/click and single-runtime proof; identical-worded
  foreign question, stale request and genuine opening controls; exact/deleted/
  query/mode/overlay/close refresh controls; clean/exception/negative-signal
  recovery and actual production repaint/focus error handling.

The disjoint affected activation/switcher/dialog/quit/governance run completed
with **223 passed, three failed, 23 deselected in 510.97 s**, without warnings
(`affected-first-run.log`). One narrow reconciliation stand-in lacked the newly
read activation fields; its first correction targeted the wrong stand-in.
The actual fixture now declares those two incumbent fields, with no assertion
changes; its complete file passes **12 tests in 1.59 s**
(`green-reconciliation-fixture.log`). Two existing installed controls timed out
waiting three seconds for their held opener to enter; both pass on one separate
confirmation with their **original three-second limits**
(`confirmation-old-limits-and-wrong-fixture.log`, two passes/one fixture failure).
The failed runs remain retained, and the timeouts are not claimed as a proved
production regression or performance qualification. All 249 distinct targeted
nodes have a passing final observation across these runs. No full suite or
warning-free whole-application qualification is claimed.

## Static and visual evidence

All eight changed Python paths format clean; changed tests and the quit helper
are Ruff clean. The production paths retain the same 78 baseline Ruff diagnostics
(69 in workspace, seven in confirmation, one each in coordinator/switcher),
with **zero additions**. Whitespace check passes.

One batched mounted render inspection and one correction confirmation covered
120x50 and 52x20. Full interruption copy, retained selection, Resume badge and
Retry/Cancel fit. Confirmation artifacts are in
`/tmp/task33622-20-green-final-wcg6DZ/quit-{120x50,52x20}.png`, with durable SVGs
`quit-120x50.svg` and `quit-52x20.svg` beside this receipt; their pre-action
view is unchanged by the final Enter/click and restoration signal corrections.
The SVG/Cairo rendering is mounted geometry/copy evidence, not native Terminal
font, viewport, ordinary-quit or Windows qualification.

Independent scoped review found no remaining Critical/Important/Minor issue
after the shared-input and production restoration corrections. All eleven
derived-artifact guards pass (`preflight.log`), including generated CSS,
diagnostic inventory, Backlog integrity, worker contract and UI census.
Portable RED/GREEN/failed/confirmation logs named above are retained alongside
this receipt; original raw logs stay at their temporary execution paths. Only
two assertion-separator trailing-space lines in `red-copy.log` are normalized
for the repository whitespace check; all other copies are byte-identical.
One empty indentation-only line in each portable SVG is likewise normalized;
geometry/text are unchanged and original exports remain at the recorded roots.
Original copy RED SHA256:
`f75d2d9831e0243ec4a7ec24f71f5ff8748b999fb8863ccb6dd9421faec81ba8`.

TASK-31245/31966 native, Windows, participant, full latency and application-owner
qualification stay incomplete. Semantic search, cache/GC policy and the separate
baseline closed-cursor finding are not part of this repair.
