# TASK-32785 — Visible MCP permission Edit handoff

Settings Edit now reveals the requested profile's permission table after its
content finishes laying out. The Permissions canvas scrolls within the existing
workbench; focusing the selector or filter reveals that control. Home/End/arrow
reading on the canvas cannot change an offscreen permission row: Space cycles
state only while the table owns focus. Delayed reveals follow current focus and
respect newer modes and dialogs. Profile/revision admission is unchanged.

## Targeted evidence

129 distinct targeted cases pass across the runs below. No full suite or
provider requests were run.

| Scope | Cases | Evidence |
| --- | --- | --- |
| Real Settings route, cold/restored state, current authority, first/last/filtered rows, controls and read-only canvas scrolling; dark/light × compact/wide | 4 | [Routed](routed.txt) |
| Existing deep links, stale authority, named-profile edits and row state cycling | 31 | [Workbench](workbench.txt) |
| Existing permission mode projection/filter/keyboard/CSS contracts | 62 | [61 passed plus old token assertion](mode-first.txt), [corrected final case](mode-final.txt) |
| Token/component governance, generated CSS sync and structural CSS ratchet | 32 | [Governance](governance.txt) |

[Red evidence](red-summary.txt) records clipping before repair and why overflow
alone did not fix focus arriving before layout. The older mode tests needed the
existing private-profile process wrapper; the same setup failure reproduced on
HEAD. One old assertion also expected a numeric CSS literal that had already
become a token. It now requires that token and its unchanged 70% value; all other
original assertions remain unchanged. Dependency-version warnings from requests
are environmental and remain visible in the logs.

[Independent review](independent-review.txt) found no introduced blocker and
verified newer dialog/mode ownership. Scoped Ruff introduces no diagnostics;
new files and changed production methods pass formatting. Backlog and diagnostic
inventory checks pass. The only final production change after the routed and
workbench runs was formatting the existing reveal guard; final native hashes
below pin that exact source.

## Native visual review

The real app ran with LinuxDriver, TTY streams, a held instance lock and real
private Tool Profile services. Each cell imported a real service-exported pack,
opened Edit, navigated to the last row, scrolled horizontally to State, saved a
global permission change, reached its feedback, and returned through Settings
Edit at the fresh revision. The default policy and export fixture stayed
unchanged. Focusing the canvas and pressing End/Space made no policy write.

| Theme / size | Matrix | Last row | State column | Saved feedback | Return visit |
| --- | --- | --- | --- | --- | --- |
| Dark 80x24 | [View](textual-dark-80x24-matrix.svg) | [View](textual-dark-80x24-last-row.svg) | [View](textual-dark-80x24-state-column.svg) | [View](textual-dark-80x24-feedback.svg) | [View](textual-dark-80x24-restored.svg) |
| Dark 170x48 | [View](textual-dark-170x48-matrix.svg) | [View](textual-dark-170x48-last-row.svg) | [View](textual-dark-170x48-state-column.svg) | [View](textual-dark-170x48-feedback.svg) | [View](textual-dark-170x48-restored.svg) |
| Light 80x24 | [View](textual-light-80x24-matrix.svg) | [View](textual-light-80x24-last-row.svg) | [View](textual-light-80x24-state-column.svg) | [View](textual-light-80x24-feedback.svg) | [View](textual-light-80x24-restored.svg) |
| Light 170x48 | [View](textual-light-170x48-matrix.svg) | [View](textual-light-170x48-last-row.svg) | [View](textual-light-170x48-state-column.svg) | [View](textual-light-170x48-feedback.svg) | [View](textual-light-170x48-restored.svg) |

All 20 final captures were rendered and inspected. At 80 columns, existing table
columns require horizontal scrolling (End/Home or Left/Right); the State view
above verifies access. This repair qualifies vertical table/control visibility,
not a redesign of the whole MCP destination. Compact rail/header wrapping and
simultaneous tool/state readability remain part of its broader layout review.

[Native result](native-result.json), [capture hashes](capture-manifest.json) and
[lifecycle receipt](lifecycle.json) pin the final runner/source. Ctrl+Q returned 0;
the app PID was absent before terminal closure, the instance lock was reacquired,
all 11 private databases passed integrity checks, conversations/messages stayed
empty, default-profile fingerprints were unchanged, and no error/faulthandler
logs were emitted. Run001 also completed four cells, but preceded guard formatting
and the added horizontal-column assertion: [historical result](pre-final-run001-native-result.json),
[clean lifecycle](pre-final-run001-lifecycle.json). Run002 is final evidence.

ADR-107 and ADR-150 govern this bounded repair. The [Tool Profiles ledger](../../reports/2026-09-18-tool-profiles-review.md)
retains concurrent-workflow and broader destination review. Draft PR 2707 remains
open and needs its own visual review and merge approval.
