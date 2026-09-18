# TASK-32792 — Permissions table final viewport

The Permissions matrix now observes its own final viewport, so an enclosing
scrollbar cannot leave a stale Tool width that clips State. Width changes retain
the selected row and rebuild only when wrapping changes. Height-only changes
reveal the current selected row without rebuilding. A fresh Enter after a row
rebuild uses the current permission context instead of being swallowed by an
older highlight's activation marker.

## Targeted evidence

| Scope | Evidence |
| --- | --- |
| Final child width/height, production real catalog, repeated dark/light resizing, Unicode/tag/filter identity and fresh-context Enter: 15 passed | [Reflow checks](final-reflow.txt) |
| Existing Permissions mode and handoff: 66 passed before the final height-only reveal callback | [Adjacent checks](adjacent-permissions.txt) |
| Final design token and component governance: 26 passed | [Governance](governance.txt) |

No full repository suite was run. The adjacent run began before the final
height-only callback; the final focused tests and native source receipt qualify
that addition. No service reload or policy mutation is introduced by reflow.

The baseline [child-width probe](child-width-baseline.jsonl) reproduces clipping
with canvas size unchanged: width42→39 leaves Tool width29 when26 is required.
A 60-row matrix also clips after its outer scrollbar appears. The
[final probe](child-width-final.jsonl) fully paints State and converges after one
rebuild. [Child-height red evidence](child-height-red.txt) and the
[final probe](child-height-final.json) cover height20→10, unchanged canvas
geometry and row59 remaining visible without rebuilding or emitting actions.

[Selected-row red evidence](selected-row-red.txt) reproduces the real catalog's
row falling below the viewport. The first native run caught the same failure:
[capture](failed-run001-state.svg), [result](failed-run001-native-result.json),
[clean exit](failed-run001-lifecycle.json). An earlier test had explicitly
scrolled the cursor before checking paint; the new real-catalog regression and
native journey require the application to keep it visible unaided.
[Enter red evidence](enter-red.txt) records the swallowed fresh-context action.

[Independent review](independent-review.txt) found the height-only residual and
confirmed the final correction, latest-focus/hidden-mode guards and action
identity. [Preflight](preflight.txt) records clean new-file lint/format, formatted
changed methods, three unchanged inherited lint findings, no duplicate task IDs,
no diagnostic inventory drift and a clean diff check.

## Native visual review

Final run003 used a fresh private profile, real LinuxDriver/TTY, the production
navigation handler and the real local catalog. Each journey selected the longest
tool name, resized out and back, inspected its exact row/context, filtered it,
cleared the filter and refreshed the matrix. No tool or provider was invoked;
permission profiles stayed unchanged.

| Theme / size | Selected row | Filtered | Refreshed |
| --- | --- | --- | --- |
| Dark 80x24 | [View](textual-dark-80x24-tool-state.svg) | [View](textual-dark-80x24-filtered.svg) | [View](textual-dark-80x24-refreshed.svg) |
| Dark 170x48 | [View](textual-dark-170x48-tool-state.svg) | [View](textual-dark-170x48-filtered.svg) | [View](textual-dark-170x48-refreshed.svg) |
| Light 80x24 | [View](textual-light-80x24-tool-state.svg) | [View](textual-light-80x24-filtered.svg) | [View](textual-light-80x24-refreshed.svg) |
| Light 170x48 | [View](textual-light-170x48-tool-state.svg) | [View](textual-light-170x48-filtered.svg) | [View](textual-light-170x48-refreshed.svg) |

All twelve SVGs were rendered and visually inspected. Compact names wrap beside
the complete State; the selected final row remains visible in the real catalog.
The gallery qualifies this matrix repair, not all surrounding rail/inspector
content, tool execution, or permission-edit workflows.

[Native result](native-result.json) pins the runner and 31 production files;
[capture hashes](capture-manifest.json) pin twelve SVGs and terminal transcripts.
[Lifecycle](lifecycle.json) verifies app.run returned, exit0, process absence
before terminal closure, lock reacquisition, ten healthy databases, zero
conversations/messages, unchanged default-profile fingerprints and no error or
faulthandler output. Run002 succeeded before the final height-only scheduling
correction; its retained pre-final receipts are historical evidence only.

Existing ADR-150/161 apply; this bounded rendering and interaction correction
adds no architectural boundary. The [MCP review ledger](../../reports/2026-09-18-mcp-review.md)
retains master-toggle ordering, refresh/execution, Servers and Audit review.
Draft PR2707 remains subject to separate visual review and merge approval.
