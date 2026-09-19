---
id: TASK-32835
title: Keep MCP Audit filters readable and reachable in compact terminals
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-19 04:02'
updated_date: '2026-09-21 15:22'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users read and operate all Audit filters and reach the filtered execution table in a compact terminal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Text filter values and full decision and initiator labels paint within their controls in dark and light compact layouts.
- [x] #2 Keyboard focus can reach each filter and the execution table, including after resize, without losing filter values.
- [x] #3 Wide layouts remain usable; targeted tests, design governance and private native evidence verify the supported sizes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
# Saved PR2721 integration preparation

Scope: TASK-32835 readable/reachable MCP Audit filters. Integrate only after PR2720 merge confirmation, on a fresh branch from merged dev; the next visual approval applies only to this slice.

ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: existing responsive Audit controls and focus contract, no new state/storage/authority boundary.

1. Preserve PR2720 closeout and mark TASK-32834 Done only after required current-head CI, resolved review, unchanged final dev and confirmed identical merge tree.
2. Start a fresh integration branch from actual merged dev and apply saved commit 8ba5f72f0b6c045756e6b1761e83bbf819410744. The read-only three-way preview shows two documentation conflicts in the MCP review ledger; retain both histories and label saved 2026-09-18 qualification historical. Product and source CSS hunks combine without conflicts in the preview. Rebuild generated CSS from sources.
3. Reopen TASK-32835 and add the current integration plan before making follow-up changes. Run ten saved real-app layout regressions, all legacy Audit behavior cases, the final selection/unit cases and relevant governance/architecture checks. Retain any failed first attempt as unqualified evidence; change only confirmed integration defects.
4. Modernize the saved native runner with the existing shared CLI/private-profile parser, checkout pinning before any project import, app warm_up_image_protocol, network guard, source/module provenance, private logging and normal shutdown receipts. No connected fixture server is needed: this layout journey uses real private JSONL metadata with no execution. Add it to shared CLI admission coverage.
5. Run fresh dark/light 80x24 and 170x48 terminal journeys with real keyboard typing, decision/initiator menus, filtered last-row access and retained values. Verify painted labels, selected-row visibility, real TTY streams, app exit/lock release, healthy private databases, unchanged defaults/sentinels and no network attempts. Inspect captures.
6. Obtain independent diff review, targeted formatting/lint and all artifact guards. Update existing PR2721 against dev; show compact/wide visual evidence and explicit documentation-conflict choices. Wait for this slice's fresh visual approval before its own CI/Qodo/current-dev merge.

Saved runner issues already confirmed by source inspection: direct private dependency probe_terminal import, old profile validation before checkout pinning, raw positional indexing/no shared CLI admission, no network guard, limited provenance. Existing current runners supply the bounded established replacements.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stacked Audit filters at full width across supported pane sizes and enabled execution-pane scrolling. Deferred focus/resize reveal keeps the current control and table cursor reachable without changing filter values. Existing ADR-150/161 apply; no new ADR, tokens, policy or runtime boundary. Plan refinement: the same column is used at wide sizes because intermediate triad widths also crowd the controls.

107 distinct targeted cases pass, including all 69 legacy Audit cases and ten new real-app dark/light cases at 80x24, 100x30, 120x40, 170x48. Seven preflight guards pass; no introduced Ruff diagnostics; new/changed ranges formatted. All 18 private native captures were inspected, with clean exit/lock release, ten healthy databases and unchanged defaults. Independent review found no actionable issue. Initial and intermediate failures, including the standalone legacy fixture import-order limitation, remain in Docs/superpowers/qa/2026-09-18-mcp-audit-filters/README.md.

Modified Audit UI/CSS, generated sheets, rendered layout tests and review ledgers. Existing CSS literal pins now test computed geometry. Based on fresh dev; PR2720 owns the separate selection repair. Next review is inspector guidance ownership. Follow-up PR requires its own current-head CI and final visual approval; wider component review stays open.
<!-- SECTION:NOTES:END -->
