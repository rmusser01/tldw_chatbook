---
id: TASK-32834
title: Keep MCP Audit selection tied to its visible execution
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 03:45'
updated_date: '2026-09-21 15:21'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the execution shown in MCP Audit consistent with the selected visible row while filters and newest-first log refreshes change the table.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Filtering or refreshing away a selected execution clears its detail and drill actions without selecting a replacement.
- [x] #2 Newer log entries preserve an unambiguous selected execution and exact tool drilldown; queued stale row activations cannot resolve to a different execution.
- [x] #3 Targeted regressions and private native dark/light checks verify filtering, refresh, and exact drilldown.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/170-table-repopulation-selection-boundary.md; existing ADR-150/161 also apply. Reason: integrate the saved execution-selection repair within current table/inspector boundaries, without log schema or authority changes.
1. Resume PR2720 on an isolated branch and preserve both histories in the two documentation conflicts; no product conflicts occurred.
2. Run saved selection regressions and relevant Audit/navigation/architecture checks against the current runtime. Repair concrete integration failures while retaining exact displayed-event identity and absent/ambiguous selection clearing.
3. Modernize the saved native entry point to the existing shared CLI/private-profile/provenance/lifecycle conventions before launch; exercise affected admission checks.
4. Rebase onto actual merged PR2770 dev, qualify dark/light compact/wide native selection/filter/refresh/drill behavior, and retain exact source and cleanup receipts.
5. Update existing PR2720 with current tests, conflicts and native evidence; obtain its own visual approval before the authorized CI/Qodo/dev-check merge. Filter layout and inspector guidance remain separate saved PRs.
6. Independent review reproduced duplicate contraction and retired gestures across mode/subview round trips. Require uniqueness in both snapshots when restoring by metadata, and renew execution row keys while retiring selection on departure; preserve current live activation and same-snapshot duplicate rows.
7. Address Qodo by documenting nullable message arguments, extracting the duplicated uniqueness check for isolated unit coverage, retaining mounted integration coverage, and explaining the required deferred-import admission boundary. Recheck targeted behavior and native visuals; no layout or authority change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserve execution selection by snapshot row key during filtering and a unique match in both snapshots during refresh. Absent/ambiguous entries clear detail/actions. Departing Audit or Executions retires keys and selection; delayed gestures cannot target replacement rows. Independent review reproduced and verified duplicate-contraction and round-trip gesture fixes.
Current integration: 305 targeted cases and nine artifact guards pass. Four unchanged CSS literal assertions remain documented baseline debt. No new Ruff findings; changed code formatting checked. Twenty inspected private native captures cover dark/light compact/wide selection, refresh, filtering and exact same-name tool navigation. Exit, fixture cleanup, lock release, ten DBs, unchanged defaults and source hashes verified; synthetic metadata, two real local catalogs, no tool execution or external network.
PR2770 merged as ba5aa6e9ec60bb93aec4adc6aae219deb9266b78 with a tree identical to the qualified parent. Saved PR2720 documentation conflicts retain both histories; no product conflicts. Existing shared native admission and stdio fixture replace the outdated runner boundary and duplicate fixture.
Evidence: Docs/superpowers/qa/2026-09-18-mcp-audit-selection/current-dev/README.md. Fresh owner visual approval, current-head CI/review and current-dev review remain PR2720 merge gates. Compact filters and guidance remain saved PR2721/2722. Task remains In Progress until closeout.
ADR required: no; existing backlog/decisions/170-table-repopulation-selection-boundary.md and ADR150/161 apply. No schema, runtime authority or token change.

Owner approved PR2720 at 6180cf54c9. Rebased without conflicts onto f1a027c12b (PR2766 Evals); all fourteen captured sources and the runner are unchanged. Fresh replay passes 27 selection/startup checks, nine artifact guards and all four private native cells with normal cleanup. Twenty captures match approved content apart from timestamps and caret blinking. Evidence: Docs/superpowers/qa/2026-09-18-mcp-audit-selection/approved-rebase/README.md. Current-head CI/Qodo and final dev review remain the authorized merge gates.

Qodo review at 919ae96fb5 reports zero bugs and three rule findings. Follow-up plan: document nullable selection arguments, extract the duplicated old/new uniqueness check into one pure helper and add isolated matching/key-retirement coverage while preserving mounted regressions, and retain deferred runtime imports with a technical explanation because private admission and environment selection must precede app imports. Correct the adjacent Findings docstring that still describes the retired index symmetry. No layout, authority or schema change; verify focused behavior and a fresh native replay before push.

Qodo follow-up verified: 37 focused tests (8 isolated, 15 mounted, 14 architecture), all nine artifact guards, no added Ruff findings and changed-range formatting pass. Independent review clear. Native run 004 passes all four theme/size cells; twenty captures preserve approved content with timestamp/caret-only differences. Lifecycle, two fixture exits, ten DBs, defaults/sentinels and source hashes verified. Evidence: Docs/superpowers/qa/2026-09-18-mcp-audit-selection/qodo-followup/README.md. Current-head CI/review and latest-dev gate remain.

Merged PR2720 as 7a758b8196bd3083ae9f878551d01dde7f4ec1ad after owner approval, addressed Qodo findings and successful required current-head CI (1152 main +123 admission, one expected failure). PR2739 landed eight seconds earlier, so the actual merge differs from the CI candidate and was separately verified: 382 targeted passes, five proven upstream size-ratchet failures, all nine artifact guards, independent integration review clear and twenty approved-equivalent native captures with clean lifecycle. Audit reduces Workbench by seven lines; existing TASK-32809 owns the incoming size debt. Evidence: Docs/superpowers/qa/2026-09-18-mcp-audit-selection/merge-closeout/README.md. No new ADR required; ADR170/150/161 remain applicable.
<!-- SECTION:NOTES:END -->
