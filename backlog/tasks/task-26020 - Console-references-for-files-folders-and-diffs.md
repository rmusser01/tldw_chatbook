---
id: TASK-26020
title: 'Console: @-references for files, folders and diffs'
status: In Progress
assignee: []
created_date: '2026-08-31 15:45'
updated_date: '2026-09-02 00:39'
labels:
  - console
  - context
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
There is no way to put a file into the prompt by name. Verified on origin/dev: a named grep for expand reference, @folder, @diff, @staged and preprocess reference across Chat/ and Widgets/Console/ returns zero; the $-sigil mention path exists but resolves skills only (Chat/console_command_suggestions.py:163, Chat/console_skill_resolver.py:36,156). Users must attach or paste. Hermes expands @file with line ranges, @folder, @diff, @staged, @git and @url inline before send, with binary and size guards. Chatbook already has the composer suggestion surface to hang completion off and an attachment reader for the file access.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A reference to a workspace file in the composer expands to that file's content before send, with an optional line range
- [ ] #2 References to a folder listing and to the working-tree diff are supported
- [x] #3 Expansion respects the existing allowed file roots and sensitive-path denials - a reference cannot read what the tools cannot read
- [x] #4 Binary files and oversized files are refused with a clear message rather than injected
- [ ] #5 The composer offers completion for reference targets, reusing the existing suggestion surface
- [ ] #6 The transcript shows what was expanded so the user can see what was actually sent
- [x] #7 Text containing an at-sign that is not a reference (an email address, a decorator) is left untouched - asserted by tests
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pure parser find_reference_candidates (emails/decorators excluded)\n2. Pure expand_references with injected resolver + git_runner\n3. Impure resolver reusing is_within + allowed_file_roots + sensitive-paths + binary/size guards\n4. run_git_reference for @diff/@staged\n5. Send-path application + transcript records + composer completion = TASK-26044 (app-context)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped the tested @-reference engine; send-path application + completion split to TASK-26044 (app-context, needs live verification).

Engine (Chat/console_references.py, new):
- find_reference_candidates: an @ preceded by a word char is an email/handle, excluded (AC#7). Trailing punctuation trimmed.
- expand_references(text, resolve, git_runner): rebuilds the text, inlining a real allowed file (optional #L range) / folder listing / git diff|staged; a candidate resolving to nothing (decorator/typo) is left LITERAL (AC#7); a refusal is recorded and its content NEVER injected. Returns the rewritten text + a ReferenceRecord per expansion (the AC#6 data).
- resolve_reference / build_console_reference_resolver: reuses file_operation_tools.is_within + workspace_file_roots.allowed_file_roots(write=False) + Utils.sensitive_paths -> a reference can never read what the file tools cannot (AC#3, tested against a real tmp root incl. outside-roots refusal). Binary (NUL/non-text ratio) and >256KB files are refused (AC#4). Nonexistent -> literal.
- run_git_reference: git diff / git diff --staged in the launch cwd, output-bounded (AC#2).

Fully met at the engine level: AC#3 (roots/sensitive), AC#4 (binary/size), AC#7 (non-reference @ untouched). AC#1/#2/#6 are engine-complete; applying the expansion to the draft before send + rendering the records in the transcript is TASK-26044. AC#5 (composer completion) is TASK-26044.

Tests: Tests/Chat/test_console_references.py (19: parser email/decorator/line-range, expander file/folder/diff/staged/refused, resolver against a real tmp workspace incl. outside-roots/binary/oversized refusal).

Files: tldw_chatbook/Chat/console_references.py (new), Tests/Chat/test_console_references.py.
<!-- SECTION:NOTES:END -->
