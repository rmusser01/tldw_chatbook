---
id: TASK-38463
title: Match non-ASCII case variants and mixed normalization in git exclusion pathspecs
status: In Progress
assignee:
  - '@claude'
created_date: 2026-10-31 12:10
updated_date: '2026-11-03'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Probe git 2.39.5 (Apple) with exact index bytes: confirm the gaps with precompose on and off (finding: with precompose ON git precomposes argv, so an NFD or mixed stored name leaks under BOTH rule spellings, and no literal pathspec can name it).
2. Write regression tests first (real temp repos, names written byte-exact via `hash-object` + `update-index --index-info`), precompose on and off: non-ASCII case both directions, mixed normalization under NFC and NFD rules, a directory rule, git_status, and an ASCII argv byte-identical pin. Show them red.
3. git_status: filter every parsed porcelain-v2 `-z` entry through the folded matcher before counting (the entries are already parsed to render them), behind the unchanged pathspecs.
4. git_diff: the folded matcher decides. Enumerate the stored non-ASCII names the current pathspecs let through (`diff --name-only --no-renames -z` with the same mode and pathspecs plus explicit non-ASCII glob positives), match each with the folded matcher, and exclude matches with byte-exact `:(exclude,glob)` pathspecs (each non-ASCII byte as a one-byte bracket, which also leaves the element invalid UTF-8 so argv precomposition cannot rewrite it; literal on Windows). Collapse a subtree match to its stored directory prefix; cap total bytes and enumeration output, fail closed past either. After the real diff, re-run the same query; any matched name left means a race or an encoding git did not honour, so refuse.
5. `_denylist_pathspecs`: add a rule's folded spelling when the rule is non-ASCII but folds to ASCII (ASCII stored names are never enumerated). ASCII rules unchanged.
6. Regenerate the remote worker bundle, adjust its ratchet row, run the affected suites, preflight, ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
