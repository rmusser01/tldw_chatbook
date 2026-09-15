---
id: TASK-32657
title: Review Skills Files and supporting-file journeys
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 20:54'
updated_date: '2026-09-15 21:08'
labels:
  - library
  - skills
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the component review through the read-only Skills Files inventory and navigation, preserving file contents and editor drafts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Empty and populated Files states show truthful paths, byte sizes and text/binary labels from real local bundles.
- [x] #2 Long names and many-file inventories remain readable and keyboard-scrollable at 170x48 and 80x24 in both themes.
- [x] #3 Switching between Files and other Skill modes preserves unsaved edits and does not alter supporting files or trust authority.
- [x] #4 Targeted checks and a private native run qualify the reviewed states, with documented findings and evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/009-local-skill-trust-boundary.md; backlog/decisions/076-library-lifecycle-progressive-disclosure.md; backlog/decisions/086-library-adaptive-reader-shell.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md (existing)
Reason: Review and repair the existing read-only inventory and mode navigation without changing storage, file authority or application structure.

1. Exercise empty, nested text/binary, long-name and many-file bundles through production CSS; check keyboard scrolling, mode focus and unsaved draft preservation.
2. Fix reproduced presentation or interaction defects with existing tokens and components; keep the inventory read-only and add targeted regression coverage.
3. Run affected UI/state/service/governance checks and private native size/theme verification; confirm exact file hashes and normal exit.
4. Record QA, update guide/audit/task notes, obtain bounded review and commit locally. No full suite, provider execution or dev integration.

Allocation: fresh origin fetch; all-history and live-worktree sweep, plus all-ref source/doc content scan. Evidence: .superpowers/sdd/2026-09-15-skills-files/task-id-sweep.json.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reviewed the existing read-only Files inventory; no production defect or code repair was needed. Added four production-CSS journeys for empty, binary, UTF-8, zero-byte, nested/long-path and 65-supporting-file states at both sizes and themes. Real Tab/Shift+Tab mode changes preserve drafts; exact hashes of all bundle and real private trust files, plus trusted status, remain unchanged.

64 distinct targeted checks pass (45 reader/state, 4 journeys, 2 selected service, 13 token/bundle governance). Both new Python files pass Ruff and formatting; diff whitespace and QA links were checked. Initial wide-path assertion failures came from adjoining pane text in the test crop and were corrected without a production edit. Independent follow-up review found no remaining findings.

Private native run-001 passes at 170x48 dark and 80x24 light; six rendered captures inspected. Normal Ctrl+Q returns exit 0, owned shell closed, all 66 file hashes unchanged, native trust stays uninitialized with no manifest, ten SQLite integrity checks pass, and message count is zero. No full suite, provider/script execution, app restart or dev integration.

Updated Docs/User_Guide/library/skills.md, the Library workflow audit and Docs/superpowers/qa/2026-09-15-skills-files/README.md with evidence and limits. Added Tests/UI/test_library_skill_files_journeys.py and the native evidence runner. Plan deviation: no UI repair was warranted; coverage qualifies existing behavior.

ADR required: no. Existing backlog/decisions/009-local-skill-trust-boundary.md, 076-library-lifecycle-progressive-disclosure.md, 086-library-adaptive-reader-shell.md, 150-design-token-system-and-design-language.md and 161-component-pattern-library.md apply. Storage, authority, tokens and production runtime are unchanged. Next review: Library Collections.
<!-- SECTION:NOTES:END -->
