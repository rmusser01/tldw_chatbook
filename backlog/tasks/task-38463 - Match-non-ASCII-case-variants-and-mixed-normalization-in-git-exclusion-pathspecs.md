---
id: TASK-38463
title: Match non-ASCII case variants and mixed normalization in git exclusion pathspecs
status: To Do
created_date: 2026-10-31 12:10
labels:
- security
- tools
priority: medium
references:
- backlog/tasks/task-37655 - Unicode-normalize-path-exclusion-and-sensitive-path-matching.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-37655 made exclusion and sensitive-path matching insensitive to NFC/NFD spelling, and git exclusion pathspecs are now emitted in both spellings. Two gaps remain in the git pathspecs, which decide what git_diff and git_status output can include. First, git's `icase` magic folds ASCII only, so an exclusion written as `NAÏVE.TXT` does not exclude a tracked `naïve.txt`, while the Python matcher treats them as the same file. Second, a name stored in a mixed normalization (neither pure NFC nor pure NFD) matches neither emitted spelling. Both predate TASK-37655. Either can let an excluded file's content reach a model through git output.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 git_diff and git_status exclude a file whose exclusion rule differs from its stored name only by non-ASCII case, in both directions.
- [ ] #2 A file whose stored name is in mixed normalization is excluded by a rule written in NFC or NFD.
- [ ] #3 Regression tests use a real temp git repo with core.precomposeunicode both on and off, and ASCII pathspec output stays byte-identical.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
