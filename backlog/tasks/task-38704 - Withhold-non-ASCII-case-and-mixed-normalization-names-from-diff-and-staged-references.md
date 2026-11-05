---
id: TASK-38704
title: Withhold non-ASCII case and mixed-normalization names from @diff and @staged references
status: To Do
created_date: 2026-11-03 22:40
labels:
- security
- console
priority: medium
references:
- backlog/tasks/task-38463 - Match-non-ASCII-case-variants-and-mixed-normalization-in-git-exclusion-pathspecs.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-38463 closed a leak in the git_diff and git_status tools. Exclusion pathspecs missed files whose stored names differ from the rule only by non-ASCII case (git's `icase` folds ASCII only) or by normalization: with `core.precomposeunicode=true`, the macOS default, git rewrites argv pathspecs to NFC, so NFD-stored and mixed-normalization names match no spelling. The tools now scan the changed names, check each with the folded matcher, and withhold denied ones byte-exactly, failing closed.

The Console's `@diff` and `@staged` references (`Chat/console_references.run_git_reference`) build their own `git diff` from `_denylist_pathspecs` and do not use that scan. An excluded file named in a non-ASCII case variant or a different normalization can still put its content into the model's context through those references. Open PR #3361 (TASK-38580..38613) rewrites `run_git_reference` and computes the exclusions from the repository top. That fixes a separate gap, where a denylisted file above the launch directory was not excluded. Implement this task on top of that version, after #3361 merges, to avoid conflicting with it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `@diff` and `@staged` exclude a file whose exclusion rule differs from its stored name only by non-ASCII case, in both directions, or by normalization (NFC, NFD or mixed), with core.precomposeunicode on and off.
- [ ] #2 When the excluded names cannot be determined (scan failure or limit exceeded), the reference refuses without naming or including excluded content.
- [ ] #3 ASCII-only repositories produce byte-identical `@diff` and `@staged` output, and regression tests use a real temp git repo.
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
