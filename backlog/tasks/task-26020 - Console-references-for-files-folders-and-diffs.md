---
id: TASK-26020
title: 'Console: @-references for files, folders and diffs'
status: To Do
assignee: []
created_date: '2026-08-31 15:45'
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
- [ ] #3 Expansion respects the existing allowed file roots and sensitive-path denials - a reference cannot read what the tools cannot read
- [ ] #4 Binary files and oversized files are refused with a clear message rather than injected
- [ ] #5 The composer offers completion for reference targets, reusing the existing suggestion surface
- [ ] #6 The transcript shows what was expanded so the user can see what was actually sent
- [ ] #7 Text containing an at-sign that is not a reference (an email address, a decorator) is left untouched - asserted by tests
<!-- AC:END -->
