# Roleplay draft recovery — TASK-32820

The partial-save recovery dialog used an unstyled full-screen Container. This
repair composes the shared dialog title and action row, with a centered,
token-backed frame and complete failed-domain text. Retry, Stay and Escape keep
their original results and handlers. Source styles live in the dialog component
module; the app bundle is rebuilt, never hand-edited.

## Baseline and targeted verification

The [original native compact capture](../2026-09-18-css-consolidation/native/textual-dark-80x24-roleplay-recovery.svg)
shows the unframed layout. [Six red layout cases](layout-red-results.json)
reproduce its full-screen frame at 52×20, 80×24 and 170×48 in both themes.
The [initial layout run](layout-initial-green-results.json) then passes all six
with full compositor text, containment, focus and action-result assertions.
[The original consumer run](consumers-initial-green-results.json) passes 19 cases,
including real mounted partial-save/Stay draft retention, aggregate discard/Stay,
and four recovery dialog types in dark/light compact/wide views.

Independent review identified a shared CSS cascade mismatch: `.dialog-buttons`
loads after `.button-group-right` and centers the action row. The
[alignment red case](alignment-red-001.txt) reproduces this before a scoped
right-alignment rule. Other dialogs' alignment remains outside this repair.
The [final selected cases](final-selected-cases.txt) include the stronger six-cell
layout matrix, incumbent Roleplay consumers, design-token/bundle checks and
unchanged CSS byte/selector/source ceilings. The first final run passed 30/31:
its [selector-budget failure](selector-budget-red.txt) counted 275 against 274.
The broad scoped Static rule now targets only the title class and failed-domain
ID, preserving its color values. The [final rekey run](final-rekey-results.json) passes all 21 affected cases,
including the six painted layout cases. The [combined exact-case ledger](qualified-cases.json)
qualifies **31 distinct targeted cases**, retaining each final log and earlier
failures. The ten unrepeated cases cover unchanged navigation/action behavior,
hover/disabled/Cancel states, allowlist and source-count journeys; the rekey only
changes how the same two text colors are indexed. The two source-count tours
passed through teardown. No source, selector or byte limit was raised.

Final boot CSS is **584,038 / 608,090 bytes**; broad selector candidates are
**274 / 274**. Token and generated-bundle guards pass. Production Python, the new
mounted test and native runner pass Ruff; changed-function/new-file formatting
and authored whitespace checks pass. The layout detector reports no findings;
that mechanical scan is supplementary to actual rendering.
[Preflight](preflight-initial.txt) passed its six local guards; its public pinned
Mermaid input was unavailable under restricted networking. The unchanged
[Mermaid check rerun with network access](preflight-mermaid-recheck.txt) reproduces
all six outputs. Together these qualify all seven preflight checks; the first
nonzero invocation is retained. The fetched-ref/worktree allocation recheck
confirms TASK-32820 has no other owner.

## Native visual and lifecycle qualification

[All six final captures](GALLERY.md) were rendered and inspected. Each shows
complete text, a centered frame, right-aligned readable controls and visible
Stay focus. The [native receipt](native/result.json) records the initial Retry
focus, Tab/Enter Stay, pointer Retry and Escape results in every cell, using
LinuxDriver with both output streams and the rendering console attached to TTYs.

[Independent lifecycle checks](lifecycle-001.json) confirm normal App.run return,
exit 0, absent PID, reacquired profile lock, ten healthy SQLite databases,
zero conversation/message rows, unchanged default settings and no error or
faulthandler output. Eight source hashes match the inspected capture source.
The owned terminal was closed after shutdown. [Independent review](independent-review.txt)
found no outstanding issue after the local alignment and receipt corrections.

## Scope

Mounted checks use the current app styles and isolated profiles. The native
runner opens the real modal directly in TldwCli with the four possible failed
domains and observes its Retry/Stay/Escape callback results. That is native modal
layout and interaction evidence; the normal partial-save entry path and actual
save-worker behavior are covered only by their stated mounted fixtures. No full
suite is claimed. TASK-31243 and broader component review remain open.

ADR required: no. This implements existing navigation recovery (ADR-120), design
language (ADR-150) and modal component patterns (ADR-161). PR2707 stays draft and
unmerged, subject to its own visual review and merge approval.
