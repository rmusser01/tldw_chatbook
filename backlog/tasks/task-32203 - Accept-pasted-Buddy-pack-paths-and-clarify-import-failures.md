---
id: TASK-32203
title: Accept pasted Buddy pack paths and clarify import failures
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 17:02'
updated_date: '2026-09-09 19:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A user importing Trenchcoat on dev receives a generic failure. The published archive works with an absolute path, but common home-relative and quoted path inputs fail identically. Make the import input usable and provide actionable failure messages without changing archive validation or existing selections.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Buddy management imports valid native packs entered as absolute paths, home-relative paths, or quoted local paths.
- [x] #2 Missing files, invalid archives, unsupported packs, stale sources and storage publication failures give actionable path-free messages while preserving prior Buddy settings.
- [x] #3 Existing no-follow source checks and rejection of links, unsafe input and malformed pack content remain enforced; focused regressions and the actual Trenchcoat archive are verified.
- [x] #4 Equivalent quoted, home-relative and absolute retry paths reuse the installed Buddy after a settings-write failure, using shared input validation and without resolving links.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md (existing); backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md (existing)
Reason: Routine import-input and error-copy correction; immutable review, no-follow checks, publication and ownership contracts stay unchanged.
1. Reproduce the published Trenchcoat archive and common path inputs on latest dev.
2. Add focused failing regressions through Buddy management using real archive validation and SQLite publication, plus failure/selection preservation cases.
3. Normalize home-relative and quoted local paths in the UI worker; retain no-follow validation and provide safe actionable errors for input, review and publication.
4. Run targeted library, coordinator and modal checks; repeat the actual pack import and review the diff.
5. PR #2551 Qodo: normalize through a shared strict Pydantic input boundary, use the lexical normalized path consistently for retry cache identity, cover failed-save retries and update public Args/Returns/Raises documentation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented pasted-path handling in Buddy Management: current-home shorthand and matching outer quotes are normalized without resolving links or executing shell text. Missing/non-regular files, invalid/unsupported archives, stale sources, read permissions, storage installation and concurrent settings changes have distinct path-free recovery messages. Source-read OSError uses the existing failed category at the snapshot pinning boundary; decoder/content failures remain invalid. Previous selections and import retry identities remain intact. Updated the import placeholder and Buddy guide.

Validation on dev base e574c81d22: 98 targeted tests passed across new management-import regressions, coordinator, library, modal and native importer. Initial regressions failed on the old path/error behavior; additional source-mutation, permission and double-slash-home regressions also failed before their fixes. Ruff lint/format, compile and git diff --check passed. Independent read-only review found the permission-message issue and verified it and home-prefix handling fixed.

The published Trenchcoat archive is 217491 bytes, SHA-256 620f06958112d9be1dce0d6592842f47ada9c8c4d9f359c0ed92d878af4ab80f (Git blob 30207a60400e92f079f5a3bb76adfdb38dd2e75c confirmed with GitHub API). Actual bytes passed the mounted headless Textual dialog with absolute, home-relative, single-quoted and double-quoted-home inputs, independent publication and native preview in disposable profiles. This is not native-terminal or Windows qualification. The original reporter OS, download method and exact input remain unconfirmed, so the fix addresses a reproduced path/error problem without asserting their precise root cause.

ADR required: no new ADR; existing ADR-139 and ADR-074 ownership, review and validation boundaries are preserved.

PR #2551 Qodo follow-up: rebased on dev 04f6ae4eca. Reproduced duplicate installation after a failed settings write with three equivalent path spellings (3 failing real-SQLite regressions). Shared strict BuddyImportPathInput plus validate_buddy_import_path now provide one normalized lexical value for archive review and retry cache lookup/insertion, retaining links as distinct paths. Added argument/return/error contracts for apply_choice and read_buddy_archive. Invalid boundary values remain path-free.

Verification after review fixes: 108 targeted tests passed; mounted Trenchcoat import/preview passed again for all four path forms. Ruff format, compilation and diff checks passed. The five Buddy-specific Python files have no Ruff findings; input_validation.py retains the same 8 unrelated pre-existing findings as the rebased base, with zero new findings. Independent read-only review found no remaining issues.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

This task previously held id TASK-32172, colliding with the older
"Library-Notes-Database-Notes-offers-no-date-ordering-after-Sort-left-the-folder-tree"
task (created 2026-09-09 09:10, on `origin/fix/library-notes-docs` and the
other Library ▸ Notes critique-9 branches, PR #2558), which arrived first.
This task was created 2026-09-09 17:02 (add commit 401e344f30 on
`codex/buddy-import-trenchcoat`, merged to dev as PR #2551). Per the owner
rule decided 2026-08-21 in TASK-19601 (**the older arrival by
`created_date` keeps the id regardless of status; the younger task
renumbers with a provenance note**), it renumbered to TASK-32203. The other
TASK-32172 holder is the older arrival and keeps the id.

The renumber landed on 32203 rather than the next free id above the swept
remote maximum (32198): a concurrent session already held an uncommitted
`task-32199` in the `test-health` worktree, and `task-32201`/`task-32202`
were held in another, so 32199-32202 would have traded one collision for
another. No inbound reference to TASK-32172 existed anywhere in the tree
outside this file's own front matter, so nothing else moved.
