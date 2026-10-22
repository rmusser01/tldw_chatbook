---
id: TASK-35648
title: The Library panels sheet is over the 2,000 active-line CSS ceiling on dev
status: Done
assignee:
- '@claude'
created_date: 2026-10-20 03:07
updated_date: 2026-10-22 22:59
dependencies: []
labels:
- follow-up
- css
- tests
priority: medium
references:
- tldw_chatbook/css/features/_library_panels.tcss
- Tests/UI/test_component_pattern_governance.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of Phase 9's rebase onto dev 88fe3108e5 (2026-10-20). `Tests/UI/test_component_pattern_governance.py::test_sheet_size_ceiling` fails on plain dev: `css/features/_library_panels.tcss` has 2,004 comment-stripped active lines against the 2,000 ceiling (`SHEET_ACTIVE_LINE_CEILING`, ADR-161 spec 3.5, TASK-24451). The file has 2,982 lines in all. Its recent growth is TASK-34000.25's Library Reader work (47f5bbea32, 8b9041834f, bbc7c2c6d2). The test runs in no PR lane (not in scripts/ui_pr_gate_census.txt or any workflow), so dev's CI never reported it. The ceiling exists to stop the next 12,000-line grab-bag sheet; the test says a sheet over it splits by vocabulary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 test_sheet_size_ceiling passes on dev with the ceiling unchanged
- [x] #2 The Library panels rules are split by vocabulary into sheets that each stay under the ceiling, and the CSS bundle still reproduces (preflight)
- [x] #3 The Library screens render the same at 211x44 and 235x52 before and after the split
- [x] #4 The ceiling test runs in a PR lane, so the next sheet that crosses it fails a PR
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the sheet at base dev 28beeb9ed5 before splitting anything: b8cc1403fd (TASK-34000.24, 2026-10-21) already moved the Notes workbench vocabulary to features/_library_notes_panels.tcss, so check whether test_sheet_size_ceiling is still red.
2. If it is green, do not split further (no criterion asks for headroom). Verify the existing split instead: rebuild with the Notes rules put back at the tail of _library_panels.tcss, then compare the generated sheets rule for rule, and the rendered Library surfaces at 211x44 and 235x52, against the tree as it is.
3. Gate the ceiling test in the UI Fast Lane: append its node id at the census tail (no existing entry changes shard), raise MINIMUM_FILES, and record the reason.
4. Run Tests/CI, the governance and CSS build tests, and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**The split had already landed.** b8cc1403fd (TASK-34000.24, 2026-10-21) moved the sheet's last 100 lines (81 active) verbatim to `features/_library_notes_panels.tcss`, listed directly after `_library_panels.tcss` both in `CSS_MODULES` and in the library `ScreenOwnedSplit`. Those lines are three groups: the lasting-sync vocabulary (sync canvas, authority and status lines, root, review and history rows, attention actions), the compact row wrappers, and the New note heading and viewport. The rest of the Notes vocabulary, 52 selector lines, stays in `_library_panels.tcss`: `.library-notes-row`, `#library-notes-list`, the tree pager, folder rows, the create row and header, and the editor fields, among others. That commit was filed after this task and does not cite it. At base dev 28beeb9ed5, `_library_panels.tcss` has 1,946 active lines and `test_sheet_size_ceiling` passes. No further split was made: no criterion asks for headroom, and a Reader carve would be speculative. This task therefore verifies that split and gates the test.

- **AC#1.** The test passes at base and at head with `SHEET_ACTIVE_LINE_CEILING` unchanged (green under `-n 4`, 3.8 s call). Red control: a temporary 60-line append took the sheet to 2,006 and the test failed with "features/_library_panels.tcss: 2006 active lines > 2000". The edit was reverted.
- **AC#2.** The lasting-sync vocabulary, the compact row wrappers and the New note heading and viewport are their own sheet (81 active lines), `_library_panels` has 1,946, and every other `CSS_MODULES` sheet passes the same test. The temporary "unsplit" tree appended the moved lines from b8cc1403fd^ back onto `_library_panels.tcss`, deleted the Notes sheet and dropped its two manifest entries. After a rebuild, its `tldw_cli_modular.tcss` and `screen_agentic_library.tcss` match the split tree's once comments and whitespace are stripped (376,898 and 82,529 chars). Only banners and header comments differ, so cascade order is unchanged. Preflight passes, including generated stylesheets.
- **AC#3.** A temporary scratchpad harness, never committed, mounted `LibraryProductionCSSHarness` (`TldwCli.CSS_PATH`, two seeded notes) through `run.sh` on a private profile. It covered five states (home, Notes browse, note editor, New note, Manage sync folders with one root) at 211x44 and 235x52, and exported `export_screenshot()` SVGs. The unsplit tree against the split tree (base 28beeb9ed5): 10 of 10 SVGs byte-identical. Noise control: two base runs, 22 of 22 identical. Sensitivity control: with the moved rules removed, both sync-roots SVGs changed. The other four states did not change, because at these widths the wrapper rules change nothing visible (they act at compact sizes). For them, the evidence is the rule-for-rule sheet identity above.
- **AC#4.** `Tests/UI/test_component_pattern_governance.py::test_sheet_size_ceiling` is appended at the tail of `scripts/ui_pr_gate_census.txt`, so no existing entry moves. `MINIMUM_FILES` rose by 1 (588 to 589 at dev 28beeb9ed5; 603 to 604 after the rebase onto dev's TASK-36130..36132 nodes). Its shard depends on how many entries precede it, so the notes do not pin one. The node is static: it reads `css/` and imports only the stdlib-only `build_css`, it costs about 4-6 s with its private-profile child, and sibling nodes of the same file already run in that lane's install set. The census checker passes, and `Tests/CI` (`-n 4`) reports 1025 passed, 1 skipped.

Files: `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
<!-- SECTION:NOTES:END -->
