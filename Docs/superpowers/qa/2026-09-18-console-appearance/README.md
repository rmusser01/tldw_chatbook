# Conversation Appearance at compact terminal sizes — TASK-32816

The 80×24 dialog now keeps Apply, Clear and Cancel inside its viewport.
The icon grid can shrink to one complete scrollable row, each icon receives
four cells, the palette reserves a scrollbar row, and the `none` control has
room for its full label. Result, persistence and cancellation handlers are unchanged.
The search field also preserves its cursor when a grid refresh returns focus.

All touched fixed visual values use existing ADR-150/161 tokens. Textual
variables are local to each stylesheet source, so the widget-default builder
now resolves the central token declarations once and copies that map into
each block's existing isolation pass. Theme aliases stay dynamic and local
fallbacks stay local. Generated CSS changes are confined to Appearance; no
stylesheet source, cascade tier or performance ceiling was added.

## Targeted verification

- [Original layout failures](compact-red.txt): eight compact/wide geometry
  cases failed against the original layout.
- [Token-stream failure](token-stream-red.txt): the first tokenized layout
  exposed an unresolved central token in the widget-default stream.
- [Builder regression before the fix](builder-red.txt): the generated output
  retained unresolved central aliases.
- [Mounted matrix](compact-green-builder-assertion-error.txt): all eight layout
  cases passed, including resize retention, keyboard/pointer cancellation,
  first-row icon visibility, palette scrolling, and Apply's exact result.
  A separate new builder assertion incorrectly accessed `Stylesheet.errors`;
  it was corrected to verify the parsed rule count and rerun below.
- [Regression run](regression-green.txt): **61 passed**, including the corrected
  builder test, existing Appearance result/filter/validation cases, token
  governance and generated-bundle synchronization.
- [Parser checks](parser-green.txt): **24 passed**. These pure parser/scoping
  cases use `--noconftest` to avoid unrelated app fixture setup; they do not
  claim mounted or native evidence. Forward references, quoted text, local
  overrides, class-level CSS and both generated streams are covered.
- [Independent review](independent-review.txt): no actionable finding; read-only
  review, with no duplicate test or native run by the reviewer.
- [Lint comparison](lint-delta.json): six existing diagnostics, zero new.
  Both new Python files pass Ruff; [29 changed ranges](format-check.json) pass
  formatting. Existing whole-file lint/format debt is not claimed clean.

[Search-fix verification](filter-green.txt): **5 passed**, covering both slow-typing
regressions and the existing filter, Apply and Escape behavior. Counts across
runs overlap and are not additive. No full test suite was run.

## Native-discovered search failure

[Run 001](native-001/result.json) passed both dark cells, then exposed query
loss in light compact mode: typing `rocket` left `ket`, and Enter selected a
ticket. The [failure capture](native-001/failed-state.svg) preserves that state.
The app shut down normally; the runner returned failure for the assertion.
[Lifecycle checks](lifecycle-001.json) confirm process exit, released lock,
healthy databases, unchanged default configuration and no saved conversations.

The [two-theme regression](filter-red.txt) reproduced the cause deterministically:
a debounced grid refresh focused an icon, then returning to the search Input
selected its existing text. The next key replaced the prefix. Setting
`select_on_focus=False` on this search Input preserves the query and cursor.
This behavior was added to TASK-32816's acceptance criteria before implementation.

## Final native qualification

[All eight captures](GALLERY.md) were rendered and inspected: dark/light at
80×24 and 170×48, initial icons and keyboard-revealed palette. All actions and
initial `none` labels are fully visible. The query remains `rocket`, Enter
selects its matching icon, the last palette swatch is reachable by keyboard,
and pointer selection updates the draft before Cancel discards it.

[Run 002](native/result.json) records all four passing journeys and exact source
hashes. [Lifecycle evidence](lifecycle-002.json) records normal App.run return,
exit 0, independently absent PID, reacquired instance lock, ten healthy SQLite
databases, zero conversation/message rows, unchanged default configuration and
empty error/faulthandler output. The owned terminal was then closed.

Native uses directly constructed dialogs and cancellation-only fixtures with
the real catalog. It does not exercise the normal conversation entry route or
commit an appearance to the database. Prior galleries retain their own source
boundaries; the broader component migration/review remains open.

## Budget checks and teardown disposition

[Budget run](budget-tour-timeout.txt): boot bytes, selector cost and CSS allowlist
passed (three cases). Bytes are **583,433 / 608,090**; selector ceiling **274**
and source thresholds are unchanged. The 15-destination source-count test body
passed, as its [private-child output](tour-timeout-child.txt) records, but the
child did not finish fixture cleanup and the outer test timed out after 180s.
That timeout is retained as a failure; the test body alone is not a green run.
The [diagnostic recheck](tour-diagnostic-timeout.txt) reproduced the timeout.
The child-side [diagnostic plugin](tour_diagnostic_plugin.py.txt) reached the
private process through `PYTEST_PLUGINS`. Its second timed dump identified
`asyncio.Runner.close()` waiting for the default executor while the Meetings
worker compiled the macOS audio helper. The temporary directory disappeared
before archival; the [observed frame excerpt](tour-child-stack-excerpt.txt) is
explicitly a transcription, not a complete original trace.

TASK-32818 isolates only that app instance's system-audio probe in the shared
CSS-tour builder. The real owner, preparation, screens, all fifteen route/body
checks and original limits remain. This does not qualify real system-audio
capture or fix production helper compilation. Independent review found no
bounded issue. The original failures are retained above.

[Final source-budget run](tours-green.txt): **2 passed**, serially, with each
case retaining its original 180-second limit. Both private children exited 0;
their teardown took 1.04s and 1.62s. The [ordinary tour](tour-isolated-child-green.txt)
and [tour plus modal registrations](tour-modal-isolated-child-green.txt) each
visited all fifteen actual destinations. No full suite or audio capability
qualification is claimed. [Source recheck](source-hash-recheck.json) confirms all
nine files bound to the final native receipt still match the inspected captures.
