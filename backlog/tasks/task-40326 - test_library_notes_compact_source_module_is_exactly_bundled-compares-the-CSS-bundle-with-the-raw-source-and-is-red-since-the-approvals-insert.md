---
id: TASK-40326
title: "test_library_notes_compact_source_module_is_exactly_bundled compares the CSS bundle with the raw source and is red since the approvals insert"
status: To Do
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
- [ ] #1 The test compares the bundle with the source as built (inserts applied), passes at dev, and fails when a hand edit is planted in the bundle
- [ ] #2 The file runs on a PR lane, or the notes say which open task lanes it
<!-- AC:END -->
