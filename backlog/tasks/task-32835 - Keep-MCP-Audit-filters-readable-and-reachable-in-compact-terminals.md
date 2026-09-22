---
id: TASK-32835
title: Keep MCP Audit filters readable and reachable in compact terminals
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-19 04:02'
updated_date: '2026-09-22 03:28'
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
- [x] #4 Published QA receipts normalize machine-specific paths while preserving verifiable source and export hashes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
# Saved PR2721 integration preparation

Scope: TASK-32835 readable/reachable MCP Audit filters. Integrate only after PR2720 merge confirmation, on a fresh branch from merged dev; the next visual approval applies only to this slice.

ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: existing responsive Audit controls and focus contract, no new state/storage/authority boundary.

1. Preserve PR2720 closeout and mark TASK-32834 Done only after required current-head CI, resolved review and actual merge confirmation. Compare actual tree with the qualified CI candidate; if dev advances at merge, separately qualify the actual merge and document the deviation.
2. Start a fresh integration branch from actual merged dev and apply saved commit 8ba5f72f0b6c045756e6b1761e83bbf819410744. The read-only three-way preview shows two documentation conflicts in the MCP review ledger; retain both histories and label saved 2026-09-18 qualification historical. Product and source CSS hunks combine without conflicts in the preview. Rebuild generated CSS from sources.
3. Reopen TASK-32835 and add the current integration plan before making follow-up changes. Run ten saved real-app layout regressions, all legacy Audit behavior cases, the final selection/unit cases and relevant governance/architecture checks. Retain any failed first attempt as unqualified evidence; change only confirmed integration defects.
4. Modernize the saved native runner with the existing shared CLI/private-profile parser, checkout pinning before any project import, app warm_up_image_protocol, network guard, source/module provenance, private logging and normal shutdown receipts. No connected fixture server is needed: this layout journey uses real private JSONL metadata with no execution. Add it to shared CLI admission coverage.
5. Run fresh dark/light 80x24 and 170x48 terminal journeys with real keyboard typing, decision/initiator menus, filtered last-row access and retained values. Verify painted labels, selected-row visibility, real TTY streams, app exit/lock release, healthy private databases, unchanged defaults/sentinels and no network attempts. Inspect captures.
6. Obtain independent diff review, targeted formatting/lint and all artifact guards. Update existing PR2721 against dev; show compact/wide visual evidence and explicit documentation-conflict choices. Wait for this slice's fresh visual approval before its own CI/Qodo/current-dev merge.

Saved runner issues already confirmed by source inspection: direct private dependency probe_terminal import, old profile validation before checkout pinning, raw positional indexing/no shared CLI admission, no network guard, limited provenance. Existing current runners supply the bounded established replacements.

7. Post-approval Qodo closeout: document callback event Args, share the synthetic fixture count and derive filtered expectations; export QA receipts through a small stdlib path-normalizing copier while retaining original/copy hashes. Normalize all QA additions in this PR, including preceding selection closeout; retain raw receipts only in the private local profile. Add focused exporter tests. Rebased without conflicts onto dev ea2d7b22a8; rerun targeted Audit/native qualification because incoming app startup and MCP support code changed. UI approval remains valid only if capture comparison confirms unchanged appearance. No new ADR: QA publication metadata repair and existing callback documentation, no product authority/storage contract change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Current integration on merged PR2720 dev 7a758b8196: stacked full-width Audit filters and execution-pane scrolling preserve readable values, keyboard access and the current table cursor across resize. Existing ADR-150/161 apply; no new token values, selection contract, policy or runtime boundary. Product and source CSS applied cleanly. Both conflicting review-document histories were retained and older filter evidence labeled historical.

294 distinct targeted cases pass (110 Audit/layout/selection/token, 133 runner admission, 51 governance/build). One dimension-governance case fails on twelve declarations proven identical in merged dev; no allowances were weakened. Nine artifact preflight guards pass. No introduced Ruff diagnostics; 14 new-file/changed-range formatting checks pass. Independent review has no actionable findings. Current qualification and all retained failures: Docs/superpowers/qa/2026-09-18-mcp-audit-filters/current-dev/README.md. Earlier 107-case/seven-guard evidence is historical.

Modernized native runner admission and checkout pinning, replaced obsolete image-protocol probe with app helper, and added network blocking, private logging and source/module provenance. All 18 real native dark/light compact/wide captures were inspected. Typing, keyboard menus, retained values and last-row access pass. Normal shutdown, released lock, ten healthy private databases, zero chats/messages, unchanged defaults/sentinels, matching 15 source/runner hashes and zero network attempts pass. Synthetic Audit metadata only; no tools execute or servers connect.

Plan deviation for preceding PR2720 closeout: performance PR2739 landed eight seconds before merge, so actual merge tree differed from CI candidate. The actual merged tree was separately tested and visually qualified, with the deviation retained in its merge-closeout receipt; no identical-tree claim remains.

PR2721 requires its own fresh visual approval, current-head CI, accumulated Qodo review and final dev/conflict review before merging. TASK-32835 stays In Progress. PR2722 inspector guidance and wider component review remain separate. No full suite was run.

Post-approval closeout: rebased conflict-free onto dev ea2d7b22a8; added callback Args docs, shared fixture count and a tested stdlib receipt exporter. All 25 affected added QA receipts now use stable host-path placeholders with source/export hashes preserved. Independent review caught and verified the fix for output aliases: new destinations only, all inputs validated before writes. 381 distinct current-source checks pass; six incoming MCP failures reproduce on untouched dev. Nine artifact guards pass. All 18 native Audit bodies match the approved layout after timestamp/style-ID normalization; nine whole frames differ only in top navigation scroll. Fresh private lifecycle/source checks pass. Evidence: Docs/superpowers/qa/2026-09-18-mcp-audit-filters/current-dev/final-review/README.md. Current-head CI/Qodo and final dev/tree verification still gate merge; task remains In Progress.
<!-- SECTION:NOTES:END -->
