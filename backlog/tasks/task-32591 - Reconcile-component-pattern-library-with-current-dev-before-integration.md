---
id: TASK-32591
title: Reconcile component-pattern library with current dev before integration
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 00:18'
updated_date: '2026-09-18 02:14'
labels:
  - design-system
  - ui
  - audit
dependencies: []
references:
  - Docs/superpowers/reports/2026-09-14-component-first-ui-audit.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The completed design-system branch cannot merge cleanly into current dev. The audit found five conflicts, including the source stylesheet and the obsolete generated Console sheet. Reconcile the histories while preserving the design-system contracts and intervening feature behavior before broader UI changes depend on this branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The merge candidate includes current dev and has no unresolved conflicts; the integration base and preserved feature changes are recorded.
- [x] #2 Canonical ownership, source literal floors and generated bundle/split reproducibility pass on the merge candidate; current upstream Console rules remain represented in their owning sources.
- [x] #3 Targeted Console and Library shell/file-notes tests covering the merged changes pass, with any demonstrated pre-existing failures identified separately.
- [x] #4 The boot-CSS budget is met without raising its ratchet, and the proposed merge has unique Backlog task IDs.
- [x] #5 Python files changed by the integration pass fatal syntax and undefined-name checks.
- [x] #6 Source-tree CSS freshness checks support the integrated multi-module split registry without exceptions, require missing generated sheets for complete splits, and retain partial-tree behavior.
- [x] #7 The approved PR candidate passes the unchanged startup module and CSS-selector ratchets, preserves the Models source-mode keyboard journey at 80x24, and keeps the Pattern Gallery command usable on demand.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/161-component-pattern-library.md (existing); backlog/decisions/150-design-token-system-and-design-language.md
Reason: reconcile existing implementations without changing their architecture or design-language contracts.

1. Refresh origin/dev and merge its exact commit into the existing component-pattern worktree with no automatic merge commit.
2. Preserve upstream Library test coverage and behavior; transplant upstream Console styling into the decomposed owning sources and tokenize any newly introduced fixed values.
3. Rebuild generated bundle/split styles; run ownership, literal-floor, reproducibility and boot-budget checks plus the affected Console and Library tests.
4. Review the integration diff against both parents, check Backlog ID uniqueness, record exact evidence and commit the verified merge candidate.
5. Final native exit-log review exposed a stale split.module access in startup freshness detection. Add a regression for multi-source split outputs and partial trees, migrate the caller to split.modules, run the complete CSS staleness test module, and re-boot the private profile with a clean startup log.
6. Post-approval CI repair (2026-09-17): compare the failing Models journey with dev, preserve its one-row section-title spacing in the Models owner, load the gallery only when requested, and narrow four action/input selectors without changing computed styling. Rebuild generated CSS; run the existing failing tests, then affected mounted journeys, gallery command/snapshots, performance and governance checks. Record CI and visual evidence before merging. ADR required: no; existing ADR-150/161 govern local styling and ADR-097 governs unchanged startup ratchets.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reconciled origin/dev fd30614dcdc1e6cbd39b1532769d3e10be9b12b6 with design head 705a39bdb8, preserving all 89 added/changed upstream style declarations in their decomposed owners and rebuilding generated artifacts. Corrected two incoming style-floor violations, the retired Console split reference, multiline-comment selector parsing, thirteen Library harness stylesheet pins, and one type-checking import. Refreshed the diagnostic manifest for five previously deleted widgets (ten removed calls). Older dev TASK-32532 retained its ID; the younger component-pattern family and twelve subtasks are now TASK-32596, with references updated.

Evidence: 68 successful governance/build checks followed by the repaired guard in a 30-pass targeted harness run; 36 targeted layout checks; fatal Ruff across 277 changed Python files. Boot CSS is 612,733/634,050 bytes without a raised budget. All derived-artifact checks pass (Mermaid checked with network access to pinned inputs). Native scratch-profile startup rendered at 120x40 and 80x24; no provider request sent. Existing incoming documentation whitespace is preserved, and historical design-branch whitespace remains tracked by the audit. No full test suite run.

ADR check: existing ADR-161 and ADR-150 apply; no new architecture decision. Full details, limitations and selected evidence: Docs/superpowers/reports/2026-09-14-component-integration.md.

Final native exit-log review exposed a caught startup AttributeError after the merge: _generated_css_is_stale still used split.module while the integrated registry now exposes split.modules. The app rendered committed styles, so initial render captures did not prove freshness detection worked. Reopened this task and added AC #6 before repair. The caller now requires all source files before requiring a split output, preserving partial-tree behavior. Three new complete/partial split regressions failed before the fix; the complete CSS freshness module now passes all 18 tests. Fatal Ruff and test formatting pass. A new scratch-profile native run rendered Console, logged no CSS error, and exited with code 0; its command, diagnostics, exit code and tested-source hashes are retained under Docs/superpowers/qa/2026-09-14-component-fixes/startup-fixed-*.

The subsequent component repairs also close the original More/Settings/gallery/documentation follow-ups. Current boot CSS is 613053/634050 bytes and the full design-branch whitespace comparison is clean. Final evidence and remaining review boundaries: Docs/superpowers/reports/2026-09-14-component-audit-fixes.md. The caught-startup-error incident is recorded in backlog/docs/lessons-testing-evidence.md. Existing ADR-150/161 still apply; no new ADR is required.

Post-approval CI repair (2026-09-17): moved the Pattern Gallery provider into app.py and deferred its screen import until command execution; re-keyed four CSS action/input rules to existing unique IDs with unchanged specificity/declarations; restored one-row spacing only on direct llama.cpp/llamafile headings. Startup census is back within 1022 and broad-selector count within 274, with no raised ratchets. Final four regression targets pass; performance modules 9 pass, governance/build 32 pass, gallery discovery/search 2 pass, trust dialogs 12 pass. Original gallery snapshots and model-catalog assertions pass in isolated profiles; older shared-profile setup and bare-screen RAG test failures are documented separately. Four native dark/light Models captures at 80x24 were inspected; normal exit0, ten healthy private databases, released lock, unchanged default fingerprints. Independent review found no remaining issues after narrowing the Models override to exclude vLLM. Existing ADR-150/161 and ADR-097 apply. Exact evidence: Docs/superpowers/qa/2026-09-17-pr-2704-ci-repair/README.md. No full suite run.
<!-- SECTION:NOTES:END -->
