# TASK-32788 — Compact MCP readability

The MCP introduction now grows to show its complete text. Compact Source controls
retain enough label space for both Local and Server. Permission rows wrap long
Tool labels while reserving the complete State column at the left scroll position.
Tags retain horizontal keyboard access. Row keys, filtering, exact profile action
authority, and newer focus ownership remain unchanged.

## Targeted evidence

134 distinct targeted cases pass across the following runs. No full suite or
external provider requests were run.

| Scope | Cases | Evidence |
| --- | --- | --- |
| Production-styled introduction and Local/Server keyboard selection; full Tool/State paint at 80, 100, 120 and 170 columns; Unicode, resize, filtering, tags, row identity, focus and exact action context | 6 | [Final readability run](readability.txt) |
| Existing permission mode, Settings handoff and source rail behavior | 92 | [MCP regressions](mcp-targeted.txt) |
| Token/component governance, generated CSS sync and structural CSS performance | 36 | [Governance](governance.txt) |

The 92-case run started before the final scrollbar reservation correction. The
six final cases and native source hashes qualify that correction. Counts are
distinct within this task, but overlap earlier tasks and must not be added to
their totals.

[Baseline geometry](baseline-geometry.json) and [red evidence](red-summary.txt)
separate reproduced defects from fixture mistakes. Initially, Local wrapped into
two rows, the introduction clipped to one row, and the Tool column pushed State
offscreen. A Unicode fixture then exposed a first-fix gap: the dynamic scrollbar
size was zero before wrapping made the scrollbar appear. Reserving its configured
width fixes the two-cell clipping. Tests check actual compositor content and cell
geometry, not only underlying row text.

[Independent review](independent-review.txt) found no introduced blocker and
independently checked that selected rows remain visible through resize without
test-driven scrolling. [Preflight](preflight.txt) records unchanged existing Ruff
diagnostics, clean new-file lint/format, backlog IDs and diagnostic inventory.
No token values or service contracts changed; ADR-150 and ADR-161 apply.

## Native visual review

The real app used LinuxDriver, TTY streams, a held instance lock and real private
Tool Profile services. Each dark/light × 80x24/170x48 cell imported a real
service-exported pack, opened its exact profile through Settings Edit, read the
complete introduction and Local source, viewed the long server label beside its
State, reached the last row, saved one permission change and returned at the fresh
revision. Canvas Space made no write; the default profile remained unchanged.

Local and Server keyboard selection are both qualified by the mounted production
tests. The native journey uses Local; it does not qualify a connected external
server or all MCP server/tool/audit workflows.

| Theme / size | Introduction and Source | Complete Tool and State | Saved feedback | Return visit |
| --- | --- | --- | --- | --- |
| Dark 80x24 | [View](textual-dark-80x24-source.svg) | [View](textual-dark-80x24-wrapped-state.svg) | [View](textual-dark-80x24-feedback.svg) | [View](textual-dark-80x24-restored.svg) |
| Dark 170x48 | [View](textual-dark-170x48-source.svg) | [View](textual-dark-170x48-wrapped-state.svg) | [View](textual-dark-170x48-feedback.svg) | [View](textual-dark-170x48-restored.svg) |
| Light 80x24 | [View](textual-light-80x24-source.svg) | [View](textual-light-80x24-wrapped-state.svg) | [View](textual-light-80x24-feedback.svg) | [View](textual-light-80x24-restored.svg) |
| Light 170x48 | [View](textual-light-170x48-source.svg) | [View](textual-light-170x48-wrapped-state.svg) | [View](textual-light-170x48-feedback.svg) | [View](textual-light-170x48-restored.svg) |

All sixteen final captures were rendered and inspected. Compare the mounted
[80-column baseline](before-80x24.svg) and [120-column baseline](before-120x40.svg)
with the native final captures above; their fixtures differ, so these are not
pixel-equivalence comparisons.

[Native result](native-result.json) pins the runner and 29 production sources.
[Capture hashes](capture-manifest.json) record original and whitespace-normalized
stored files. The [lifecycle receipt](lifecycle.json) records exit 0, app.run
returning, PID absence before terminal closure, successful instance-lock
reacquisition, eleven healthy private databases, zero conversations/messages,
unchanged default-profile fingerprints and no error/faulthandler output.

The [Tool Profiles ledger](../../reports/2026-09-18-tool-profiles-review.md) and
[component ledger](../../reports/2026-09-17-design-system-completion-audit.md)
retain broader destination review. Draft PR 2707 needs separate visual review
and merge approval.
