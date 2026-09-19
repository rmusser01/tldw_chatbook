# PR2724 current-dev conflict and visual review

PR2724 was rebased from `b0eca0deb0d3f2e091d086e6667f43dda625d6da`
onto merged dev `29b0a31df4701160a3c805e1bf490c76b9353964` (PR2740).
The rebased original commit is `c635002f9a`; the follow-up hardens unavailable
Audit controls and refreshes verification on that integrated source.

## Conflict choices

| File | What was retained |
| --- | --- |
| [Completion audit](../../reports/2026-09-17-design-system-completion-audit.md) | Kept dev's session-grant revocation, rule-action and permission-navigation sections, then appended the Audit navigation section from PR2724. |
| [MCP ledger](../../reports/2026-09-18-mcp-review.md) | Kept dev's closeout and session-revocation history plus PR2724's post-merge and Audit navigation notes. Added a current checkpoint to distinguish merged PR2731/2734/2740 from historical pending statements. |

No application source conflict required choosing a side. The inspector and
workbench merged automatically. The merged session-revocation, rule-action and
permission-navigation controls remain intact. The only subsequent product
change rejects queued Audit presses when their button or owning view is
hidden, invisible, disabled or covered by another screen.

## Verification

[145 distinct targeted cases](current-dev/qualified-cases.json), all seven
preflight guards and independent review pass. Three existing inspector setup
errors were repaired with the documented private-profile test wrapper; original
assertions are unchanged. See the [full evidence](README.md).

## Visual review

The [24 fresh native captures](GALLERY.md) show the integrated source in dark/light
at 120×40 and 170×48. Both focused actions are fully painted, visible and owned
by the expected control; the correct destination is selected after its filter
is cleared. Each unavailable-tool action stays in Audit and shows the warning.
All captures were rendered and inspected.
This review covers both Audit actions, destinations after a nonmatching filter,
and warnings for unavailable targets. No styles or design-token values changed.
PR2726 retains same-ID catalog freshness; the wider component review stays open.

PR2724 remains a draft. Approval of PR2740 does not approve this PR's visuals or merge.
