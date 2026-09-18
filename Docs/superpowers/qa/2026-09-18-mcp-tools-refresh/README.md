# TASK-32794 — MCP Tools refresh continuity

The server selector remains mounted across catalog refresh. Unchanged options keep
an open menu intact; changed labels/order retain the highlighted server identity.
Keyboard and pointer activations commit before Textual can deliver an obsolete
option index. Filters reconcile live values, reject delayed/detached changes, and
let an explicit tool drill supersede older input. A disappearing table or recovery
action moves focus to the retained text filter without moving another control's
focus. The compact recovery button scrolls fully into view when focused.

Existing ADR-150/161 govern this bounded interaction repair. Persistence,
permission authority and tool execution are unchanged.

## Targeted evidence

- [Initial Tools/compact/Workbench run](targeted.txt): 94 passed before the final
  recovery-button scrolling correction, including 30 adjacent Workbench cases.
- [Final Tools and full-app compact replay](final-replay.txt): 66 passed, including
  the two new full-app recovery-action tests. The isolated [recovery replay](empty-final.txt)
  also passed. Overlapping runs are not added to the distinct count.
- [Token/component/bundle governance](governance.txt): 31 passed.
- [Static checks](preflight.txt): touched Python files pass Ruff and formatting.
  No duplicate Backlog task IDs or whitespace errors; [diagnostic inventory](diagnostic-final.txt)
  remains consistent. No production diagnostic calls or CSS tokens changed.

Together these qualify 127 distinct targeted cases. No full repository test suite
was run. [Independent review](independent-review.txt) found no remaining issue in
this scope after the reproduced races and native finding were repaired.

[Initial red](initial-red.txt), [pending-choice red](pending-red.txt),
[menu/drill red](activation-red.txt), [pointer red](pointer-red.txt), and
[full-app empty-state red](empty-red.txt) retain failure evidence. The first pending
probe intercepted the wrong dispatch seam; the corrected run captures the real
posted event. The first empty-state test failed while formatting debug geometry;
[that harness failure](empty-harness-failure.txt) is not counted as a product red.

## Native visual review

Fresh private profiles use the real app, LinuxDriver, TTY output, production
navigation and real catalog reload. Each theme/size verifies an open menu and its
choice, a typed fs_ filter and retained row, and keyboard focus through a controlled
empty/return projection. Only that projection replaces _collect_hub_tools; it does
not qualify a connected external server disconnect. No tools execute, and
permission profiles remain unchanged.

| Theme / size | Open menu after reload | Catalog after reload | Empty focus | Focused recovery action | Restored focus |
| --- | --- | --- | --- | --- | --- |
| Dark 80x24 | [View](textual-dark-80x24-open-refresh.svg) | [View](textual-dark-80x24-catalog-refresh.svg) | [View](textual-dark-80x24-empty-focus.svg) | [View](textual-dark-80x24-recovery-action.svg) | [View](textual-dark-80x24-restored-focus.svg) |
| Dark 170x48 | [View](textual-dark-170x48-open-refresh.svg) | [View](textual-dark-170x48-catalog-refresh.svg) | [View](textual-dark-170x48-empty-focus.svg) | [View](textual-dark-170x48-recovery-action.svg) | [View](textual-dark-170x48-restored-focus.svg) |
| Light 80x24 | [View](textual-light-80x24-open-refresh.svg) | [View](textual-light-80x24-catalog-refresh.svg) | [View](textual-light-80x24-empty-focus.svg) | [View](textual-light-80x24-recovery-action.svg) | [View](textual-light-80x24-restored-focus.svg) |
| Light 170x48 | [View](textual-light-170x48-open-refresh.svg) | [View](textual-light-170x48-catalog-refresh.svg) | [View](textual-light-170x48-empty-focus.svg) | [View](textual-light-170x48-recovery-action.svg) | [View](textual-light-170x48-restored-focus.svg) |

The twenty captures show complete filter focus and the focused recovery action;
compact metadata remains horizontally scrollable. Run001 exposed the invisible
recovery button and is retained as [failed result](failed-run001-native-result.json),
[failed capture](failed-run001-failed-state.svg) and [clean failed-run exit](failed-run001-lifecycle.json).
Run002 qualified the final production source with sixteen captures; its
[result](run002-native-result.json) and [lifecycle](run002-lifecycle.json) are retained.
Run003 adds direct recovery-action captures without further production changes.

The final [native result](native-result.json), [capture manifest](capture-manifest.json)
and [lifecycle](lifecycle.json) pin source/runner hashes, twenty SVGs with terminal
transcripts, app.run return, exit0, process absence before terminal closure,
instance-lock reacquisition, ten healthy private databases, zero conversations
and messages, unchanged default-profile fingerprints and clean shutdown logs.

## Remaining qualification

Inspector definition/draft refresh, schema/raw arguments, execution, diagnostic
action routing, connected-server lifecycles, Servers and Audit remain in the
[MCP review ledger](../../reports/2026-09-18-mcp-review.md).

The previous PR head b46319eb17 failed [PR Fast Lane](https://github.com/rmusser01/tldw_chatbook/actions/runs/35363152779):
test_workbench_mounts_rail_canvas_inspector_and_loads_local_servers saw one rail row
instead of three; teardown then raised NoMatches from the master-save status timer.
The derived-artifact aggregate failed because of that test. This review does not
claim to repair it. All six GGUF checks on that head passed. PR2707 remains draft,
unmerged and subject to separate visual review and merge approval.
