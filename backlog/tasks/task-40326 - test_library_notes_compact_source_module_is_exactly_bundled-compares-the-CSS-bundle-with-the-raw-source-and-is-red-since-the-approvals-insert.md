---
id: TASK-40326
title: "test_library_notes_compact_source_module_is_exactly_bundled compares the CSS bundle with the raw source and is red since the approvals insert"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-13 02:15'
created_date: 2026-11-12 19:36
labels:
- follow-up
- tests
- css
- dev-red
priority: low
references:
- Tests/UI/test_css_build_integrity.py
- tldw_chatbook/css/build_css.py
- tldw_chatbook/css/components/_agentic_terminal.tcss
dependencies:
- TASK-39171
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (load stream, TASK-39171 recorded it red on the clean base). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** A bundle-integrity guard that is permanently red stops guarding anything, and real drift in the agentic CSS module would hide behind it. The file is on no PR lane; TASK-33910.21 (To Do) would lane it but does not name this defect.

**Evidence.** `Tests/UI/test_css_build_integrity.py::test_library_notes_compact_source_module_is_exactly_bundled` (`:436`) compares the bundle's `components/_agentic_terminal.tcss` section tokens with the raw source's. Since a2fcbe61bc (2026-10-15), `build_css.py:1094-1100` replaces `/* @console-approvals */` (`_agentic_terminal.tcss:1509`) with the 218-line `features/_console_approvals.tcss`, so the bundle's section holds `#approval-batch-body` twice and the source once. Not run here; reasoned from the source.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test compares the bundle with the source as built (inserts applied), passes at dev, and fails when a hand edit is planted in the bundle
- [x] #2 The file runs on a PR lane, or the notes say which open task lanes it
<!-- AC:END -->

## Implementation Plan

1. Show the node red on base 26b2e677dd (run.sh, the file at `-n 4`).
2. Move the build's `/* @console-approvals */` insert into one `build_css` helper that the bundle loop and the test both call, so the test compares the bundle with the source as built.
3. Show it green, check the rebuilt bundle is byte-identical, and plant a hand edit in the bundle's agentic section (temporary) to show it red.
4. Lane: the census has no re-deal margin this round, so record the open task that lanes the file.

## Implementation Notes

**Fixed: the node compares the bundle with the source as built.** `build_css.module_text_as_built(css_dir, module)` now holds the `/* @console-approvals */` insert (a2fcbe61bc); the bundle loop and the test both call it, so the two cannot drift apart again. The rebuilt bundle is byte-identical to the committed one (checked with a fresh `build_css` into a temp file).

**Red, then green** (run.sh, base 26b2e677dd). Red: the file at `-n 4`, 1 failed / 30 passed (load 44), `At index 2691 diff: ('selector_start_id', '#approval-batch-body') != ('selector_start', 'ConsoleScopePickerModal')`, as the description reasoned. Green: the node alone, 1 passed. Negative control: a `margin: 7;` hand-planted into the bundle's agentic section (temporary, restored) turns it red (`('declaration_name', 'margin:') != ('declaration_name', 'height:')`).

**Covering run.** `test_css_build_integrity.py`, `test_css_bundle_sync_guard.py`, `test_css_generated_comments.py`, `test_console_mcp_approval.py` (pins the insert slot): 395 passed at `-n 4` (load 61-65).

**Lane (AC#2).** Not laned here: the file's 31 nodes are private-profile children (11-15 s each at `-n 4`) and the UI lane has no re-deal margin this round. TASK-33910.21 (To Do) AC#2 puts `Tests/UI/test_css_build_integrity.py` in a required PR CI job.

**Files.** `tldw_chatbook/css/build_css.py`, `Tests/UI/test_css_build_integrity.py`.
