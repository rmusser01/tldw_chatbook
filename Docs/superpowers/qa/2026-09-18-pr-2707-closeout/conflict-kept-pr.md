## Large SVG failure diagnostics can hide a quick assertion result

**TASK-32821, 2026-09-18.** Both gallery snapshots passed at the baseline.
After an intended button alignment change, the first normal-mode snapshot
comparison consumed a CPU core without a completed report; the owned runner
and child were interrupted and recorded as unqualified. Rerunning the same
assertions with pytest `--assert=plain` promptly reported both expected
mismatches. A semantic SVG diff showed only the action row moved, and the
reviewed updates then passed again in normal mode. When large string diagnostics
stall a failure run, preserve that attempt and use concise assertion reporting
for diagnosis; do not weaken equality, mask geometry or count an interrupted
attempt as a test result.

## CSS consolidation must include standalone production hosts

**TASK-32813, 2026-09-18.** Moving BackupRestoreScreen defaults into the main
app's generated sheets left the separate RecoveryApp unstyled. The real
recovery subprocess caught it; a main-app test alone could not. Register that
screen's existing bundled CSS at its native default scope/tier in the standalone
host. Keep the child test's patch on native `textual.app.App.run` and assert
computed geometry on the actual RecoveryApp. Test hosts also need the split
feature sheets; resolve real screen CSS_PATH sequences before calling them bare.

## A redraw highlight can arrive after the refresh callback that released its guard

**TASK-32796, 2026-09-18.** MCP server navigation cleared tool/finding details,
then the focused table's redraw reopened them. Nine regressions held real
published table messages until after refresh callbacks and reproduced the
selection. Suppress programmatic RowHighlighted/CellHighlighted at publication
with Textual prevent(), across synchronous rebuild/cursor work only. Review
also caught hidden Audit Findings clearing Executions' pending Enter gesture;
include the originating table in dedup identity and reset only that table.
External drill cursor moves need their own boundary after a rebuild ends.

## A pane resize does not report its child's final scrollbar geometry

**TASK-32790, 2026-09-18.** MCP Tools wrapping passed 60-row resize checks,
then independent review clipped one State cell in a five-row Unicode catalog.
The outer scrollbar narrowed the table after the pane's resize callback;
measuring again by hand restored the missing cell. Observe the table's own
Resize and gate rebuilding on changed measured width. The first local message
was also unhandled because Textual converted `MCPToolsTable` to `mcptools_table`;
an explicit message namespace aligned the handler. Exercise short and long
catalogs across scrollbar transitions, not only the terminal dimensions.

**TASK-32792 follow-up.** The sibling Permissions table retained the same gap: a
child-only width42→39 left the canvas geometry unchanged and clipped State.
The native real catalog then exposed a selected row below the viewport after
resize, while an earlier test manually scrolled it back before checking paint.
Observe final child geometry, and reveal the current selected row without test
repair. Height-only20→10 changes need a separate deferred reveal even when
width-gated reflow correctly performs no rebuild. Preserve newer focus and
check the selected cell against both compositor and table viewport.

## A stale review and an admitted write have different lifetimes

**TASK-32780, 2026-09-18.** Preventing delayed Tool Profile reviews after
navigation initially hid errors from already-admitted import/export writes
because their shared exception handlers used the same visit guard. A mounted
publisher held after confirmation then raised `durability_uncertain` after a
category roundtrip; its receipt was empty. Track mutation admission separately:
discard obsolete preparation, but preserve outcomes of writes the user already
confirmed. The same probe then retained its uncertainty receipt.

## A fake payload can preserve the same wrong contract as its consumer

**TASK-32779, 2026-09-18.** Export-review tests invented `payload.rules`,
matching the UI, while the real `ToolProfilePayload` exposes `tools`. A real
service export crashed before filename selection despite passing policy-count
tests. Construct the actual validated payload in presentation fixtures and
retain at least one real capture → review → publication journey. Fixing only
the fixture's field spelling would still leave its contract unvalidated.

## A reused widget ID cannot identify the action that was pressed

**TASK-32778, 2026-09-18.** Holding a real Tool Profiles button event across
row recomposition made Export, Edit, Bind and Remove target the replacement
profile or revision. The handler resolved the old event through a new map
keyed by the same row-index ID. Key captured action context by the originating
control and reject detached origins. Exercise event delivery after refresh;
an immediate click cannot expose this identity substitution.

## App tokens do not cross consolidated widget stylesheet scopes

**TASK-32773, 2026-09-17.** Native light-theme review exposed a black
WorkspaceCreateModal surface. Moving `$ds-*` references into its `BUNDLED_CSS`
then failed native startup: the consolidated widget stylesheet did not share
the app token definitions. Keep the token-backed override in the app dialog
module and a matching Textual theme-variable fallback in the widget. Rebuild
the generated sheets and verify actual startup and computed dark/light colors;
a valid token name in another stylesheet does not prove it resolves here.

## A workspace ID does not identify what a confirmation approved

**TASK-32770, 2026-09-17.** The memory toggle stored only the workspace ID
between its first and second presses. Mounted regressions showed that A→B→A,
category return and modal suspension retained the old acknowledgement; replacing
the saved persona/profile before the second press applied read-write to that new
record. Capture saved and intended values when asking, compare before applying,
and discard the review on navigation. Exercise cancellation as well as acceptance:
the separate imported-profile modal must preserve staging but require a fresh
memory acknowledgement when the user retries after cancelling.

**TASK-32774, 2026-09-18.** First-bind review had the same lifetime gap
across an asynchronous service call: four mounted journeys reproduced an old
dialog after category/workspace A→B→A, an unrelated modal, or a newer persona
draft. Capture the Apply intent before dispatch and invalidate that identity
on navigation or staging changes. Check it again after both review and token
exchange; a delayed token must not commit after the user leaves. Exempt only
the owned review modal from suspension invalidation, while still discarding
its separate memory acknowledgement.


