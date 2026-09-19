# TASK-32790 — Readable MCP Tool names and permission states

Long Tool names now wrap within the measured table width while reserving a full
State cell. Server, Tags and Schema retain horizontal keyboard access. Resizing,
filtering and catalog/state refresh restore selection by tool ID; removed or
filtered-out tools fall back safely. Reflow preserves current focus and filters,
observes the table's final viewport, and does not swallow a later Enter.
Existing token styling and service authority are unchanged.

## Targeted evidence

88 distinct targeted cases pass against final production source. No full suite
or provider requests were run.

| Scope | Cases | Evidence |
| --- | --- | --- |
| Full-app dark/light resizing; Unicode/tag/filter/refresh/Enter; 1/5/60-row scrollbar transitions | 5 | [Readability](readability.txt) |
| Existing Tools mode, compact access and real inspector selection/no automatic selection | 44 | [Related Tools checks](related-tools.txt) |
| Workbench selection, catalog drilldown and permission-state propagation | 8 | [Selection checks](selection.txt) |
| Token/component governance and generated CSS reproduction | 31 | [Governance](governance.txt) |

The two existing end-to-end tests now use the repository's private-profile
process boundary; their mounted assertions are unchanged. [Red evidence](red-summary.txt)
records clipped cells, stale Enter suppression, final scrollbar geometry and the
message-namespace correction. [Independent review](independent-review.txt)
replayed catalog sizes 1→5→60→5→1 at widths 34/39/42/90, confirming readable
State, retained identity, convergent reflow and exact Enter behavior with no
spontaneous selections. [Preflight](preflight.txt) records scoped formatting,
Ruff, backlog and diagnostic checks.

## Native visual review

The real app ran with LinuxDriver, TTY rendering, an instance lock and a fresh
private profile. Production navigation opened MCP and keyboard controls selected
the real catalog's longest tool, retained it through compact/wide resizing,
opened its exact inspector, scrolled to metadata, and applied text/server filters.
Permission profiles remained unchanged. No tool was executed.

| Theme / size | Tool and State | Horizontal metadata | Filtered selection |
| --- | --- | --- | --- |
| Dark 80x24 | [View](textual-dark-80x24-tool-state.svg) | [View](textual-dark-80x24-metadata.svg) | [View](textual-dark-80x24-filtered.svg) |
| Dark 170x48 | [View](textual-dark-170x48-tool-state.svg) | [View](textual-dark-170x48-metadata.svg) | [View](textual-dark-170x48-filtered.svg) |
| Light 80x24 | [View](textual-light-80x24-tool-state.svg) | [View](textual-light-80x24-metadata.svg) | [View](textual-light-80x24-filtered.svg) |
| Light 170x48 | [View](textual-light-170x48-tool-state.svg) | [View](textual-light-170x48-metadata.svg) | [View](textual-light-170x48-filtered.svg) |

All twelve final captures were rendered and inspected. [Native result](native-result.json)
pins the runner and 31 production sources; [capture hashes](capture-manifest.json)
track original and normalized SVG/terminal captures. [Lifecycle](lifecycle.json)
confirms normal exit0, app.run return, PID absence before terminal closure,
instance-lock reacquisition, ten healthy private databases, no conversations or
messages, unchanged default-profile fingerprints and no error/faulthandler output.

Run001 failed because its runner assumed the final server option owned the chosen
tool; [failed receipt](failed-run001-native-result.json) and [clean exit](failed-run001-lifecycle.json)
are retained. The corrected runner chooses the exact server. Run002 passed the
real-catalog journey before the final child-resize namespace correction and is
[historical only](pre-final-run002-native-result.json), with its own [exit receipt](pre-final-run002-lifecycle.json).
Run003 alone qualifies final production source.

## Remaining scope

The unchanged workspace-root copy visible in these captures is incorrect for
Console; independent review also reproduced lost root drafts and out-of-order
root saves. The [MCP review ledger](../../reports/2026-09-18-mcp-review.md)
records these next repairs, further refresh/focus behavior, execution, connected
servers and other modes. This is not whole-destination completion.

Existing [ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [ADR-161](../../../../backlog/decisions/161-component-pattern-library.md) govern
this presentation repair. No new ADR is required. Draft PR2707 remains subject
to separate visual review and merge approval.
