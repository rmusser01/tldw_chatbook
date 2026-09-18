# TASK-32789 — Reachable MCP Tools controls and rows

Tools now owns a scrollable canvas and reveals its current focused control after
layout. For a focused table, it also reveals the selected row after resize.
The complete local-tools toggle wraps within the pane; compact filters stack
using the workbench's existing compact class. All visual values use existing
tokens. Service authority and persistence behavior are unchanged.

## Targeted evidence

88 distinct targeted cases pass. No full suite or provider requests were run.

| Scope | Cases | Evidence |
| --- | --- | --- |
| Complete on/off labels, usable filters and first/last rows, read-only resize/filtering, newer focus/mode ownership, retained selected row with 60 tools | 6 | [Final access checks](access.txt) |
| Existing Tools mode plus permission readability and Settings handoff | 46 | [MCP regressions](mcp-regressions.txt) |
| Token/component governance, generated CSS sync and structural CSS performance | 36 | [Governance](governance.txt) |

The broader 46-case run preceded the final selected-row correction. Its 36 Tools
mode cases were repeated against final source: [36 passed](final-tools-mode.txt).
Those repeated cases are not additional distinct tests. The final six access
checks and native journeys qualify the correction directly.

[Red evidence](red-summary.txt) records four initial geometry failures, the
first toggle-width correction and two independently discovered cursor failures.
The [independent review](independent-review.txt) caught tests navigating after
resize and thereby masking the long-catalog cursor failure. Its final replay
kept row 59 visible through repeated compact/wide resizing without additional
navigation. [Preflight](preflight.txt) records clean Ruff, formatted new files and
changed methods, two unchanged pre-existing module formatting differences, and
passing backlog/diagnostic inventory checks.

## Native visual review

The real app used LinuxDriver, TTY streams, a held instance lock and a fresh
private profile. The production navigation handler opened MCP; focus and keyboard
then operated its real controls and services. Each cell persisted the local-tools
switch off and back on, read its full label, reached first/last rows, retained the
last row through compact/wide resizing, inspected the exact selected tool, and
used text and server filters. Permission profiles stayed unchanged. No tool was
executed. This qualifies configuration persistence, not a subsequent Console run.

| Theme / size | Switch off | Last row after resize | Text filter | Server filter |
| --- | --- | --- | --- | --- |
| Dark 80x24 | [View](textual-dark-80x24-switch-off.svg) | [View](textual-dark-80x24-last-row.svg) | [View](textual-dark-80x24-filtered.svg) | [View](textual-dark-80x24-server-filter.svg) |
| Dark 170x48 | [View](textual-dark-170x48-switch-off.svg) | [View](textual-dark-170x48-last-row.svg) | [View](textual-dark-170x48-filtered.svg) | [View](textual-dark-170x48-server-filter.svg) |
| Light 80x24 | [View](textual-light-80x24-switch-off.svg) | [View](textual-light-80x24-last-row.svg) | [View](textual-light-80x24-filtered.svg) | [View](textual-light-80x24-server-filter.svg) |
| Light 170x48 | [View](textual-light-170x48-switch-off.svg) | [View](textual-light-170x48-last-row.svg) | [View](textual-light-170x48-filtered.svg) | [View](textual-light-170x48-server-filter.svg) |

All sixteen captures were rendered and inspected. Mounted baseline captures at
[80x24](before-80x24.svg) and [120x40](before-120x40.svg) show the original defects;
their controlled service fixture differs from native inventory. [Baseline geometry](baseline-geometry.json)
and [baseline capture hashes](baseline-capture-manifest.json) preserve that distinction.

[Native result](native-result.json) pins the runner and 31 production sources.
[Capture hashes](capture-manifest.json) preserve original and normalized stored
hashes. The [lifecycle receipt](lifecycle.json) records normal exit 0, app.run
returning, PID absence before terminal closure, instance-lock reacquisition,
ten healthy private databases, zero conversations/messages, unchanged default
profile fingerprints and no error/faulthandler output.

## Scope still open

The compact Tools table still uses horizontal scrolling for columns beyond its
available width. Long tool names can push State out of the leftmost viewport;
simultaneous Tool/State readability remains a separate Tools review item.
Workspace-root guidance also still incorrectly describes the configured root as
the next Console run's authority; ADR-102 assigns that compatibility root to the
standalone local MCP server. Root-save feedback, draft/reload behavior, connected
server and execution workflows remain open in the [MCP review ledger](../../reports/2026-09-18-mcp-review.md).

Existing [ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [ADR-161](../../../../backlog/decisions/161-component-pattern-library.md) govern
this presentation repair. Draft PR2707 requires separate visual review and merge
approval.
