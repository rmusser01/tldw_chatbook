# TASK-32781 — Tool Profile action focus

Unchanged listings retain their controls. Changed listings serialize replacement
and restore focus by profile/action, with a same-profile or Import fallback when
an action disappears. Newer navigation and dialog focus take precedence. Render
workers no longer cancel an in-progress teardown. Queued controls retain their
original authority and cannot acquire a replacement profile's revision.

Full-paint assertions exposed a second defect: at 80×24 the Remove button was
ten columns wide but only eight were visible. Compact action rows now use a
two-column grid with existing design tokens. Wide rows retain their layout.

## Targeted evidence

112 final-source cases pass; no full suite or provider requests were run.

| Scope | Cases | Evidence |
| --- | --- | --- |
| Unchanged/reordered listings, disappearance/disable, newer focus/dialog, overlapping renders, cancellation | 19 | [Focus and governance](focus-governance.txt) |
| Token/component governance | 26 | [Focus and governance](focus-governance.txt) |
| Review lifetime | 23 | [Workflows](workflows.txt) |
| Existing Settings Tool Profiles workflows | 30 | [Workflows](workflows.txt) |
| Export recovery and publication outcomes | 9 | [Workflows](workflows.txt) |
| Generated CSS synchronization | 5 | [Workflows](workflows.txt) |

The pre-fix focus run failed 16 of 19 cases. After the focus repair, the three
remaining failures all exposed compact Remove clipping. All 12 existing
queued-action/loading lifecycle cases also passed at that intermediate source,
before the compact-only stylesheet adjustment; they are not included in the
112 final-source count. [Failure provenance](red-summary.txt).

## Native visual qualification

The real app used LinuxDriver with both TTY streams, a private profile and an
owned tmux session. Each of four cells used a real service-exported fixture to
inspect, revise, cancel, import unbound, cancel removal, reorder the listing,
and remove the imported profile through the actual UI/service. Every cell
verified unchanged policy bytes on cancellation, the unbound import, retained
Remove focus after reorder, a permanent tombstone after removal, and a visible
Import continuation. The source archive remained unchanged.

| Theme / viewport | Options | Import review | Imported | Removal review | Reordered focus | Removed |
| --- | --- | --- | --- | --- | --- | --- |
| Dark 80×24 | [View](textual-dark-80x24-options.svg) | [View](textual-dark-80x24-review.svg) | [View](textual-dark-80x24-imported.svg) | [View](textual-dark-80x24-remove-review.svg) | [View](textual-dark-80x24-refreshed.svg) | [View](textual-dark-80x24-removed.svg) |
| Dark 170×48 | [View](textual-dark-170x48-options.svg) | [View](textual-dark-170x48-review.svg) | [View](textual-dark-170x48-imported.svg) | [View](textual-dark-170x48-remove-review.svg) | [View](textual-dark-170x48-refreshed.svg) | [View](textual-dark-170x48-removed.svg) |
| Light 80×24 | [View](textual-light-80x24-options.svg) | [View](textual-light-80x24-review.svg) | [View](textual-light-80x24-imported.svg) | [View](textual-light-80x24-remove-review.svg) | [View](textual-light-80x24-refreshed.svg) | [View](textual-light-80x24-removed.svg) |
| Light 170×48 | [View](textual-light-170x48-options.svg) | [View](textual-light-170x48-review.svg) | [View](textual-light-170x48-imported.svg) | [View](textual-light-170x48-remove-review.svg) | [View](textual-light-170x48-refreshed.svg) | [View](textual-light-170x48-removed.svg) |

All 24 captures were rendered and visually inspected. Compact import review
scrolls its body while actions remain visible. The compact profile actions show
complete labels in two rows; wide actions remain on one row. Focused Remove and
its Import fallback are fully painted in all four cells. Captures show the
current viewport; data assertions qualify rows or receipts outside that view.

[Native result](native-result.json), [capture hashes](capture-manifest.json),
and [lifecycle receipt](lifecycle.json) record the exact runner and source.
Normal Ctrl+Q shutdown returned exit 0; the process was absent before terminal
closure, the instance lock was reacquired, all 11 private databases passed
integrity checks, conversation/message counts remained zero and default-profile
fingerprints were unchanged. No error or faulthandler output was recorded.

Ruff adds no diagnostics to existing Settings/panel baselines (114/1). New tests,
native runner, panel and the changed Settings method pass scoped formatting.
Backlog and diff guards pass. Independent focus and compact-layout reviews
found no introduced blocker. [Verification](verification.json).

Existing ADR-107 and ADR-150 apply; no storage, service or permission policy
changed. Remaining removal boundaries and MCP Edit/Bind handoffs are tracked
in the [review ledger](../../reports/2026-09-18-tool-profiles-review.md).
