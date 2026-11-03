---
id: TASK-38463
title: Match non-ASCII case variants and mixed normalization in git exclusion pathspecs
status: Done
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
- [x] #1 git_diff and git_status exclude a file whose exclusion rule differs from its stored name only by non-ASCII case, in both directions.
- [x] #2 A file whose stored name is in mixed normalization is excluded by a rule written in NFC or NFD.
- [x] #3 Regression tests use a real temp git repo with core.precomposeunicode both on and off, and ASCII pathspec output stays byte-identical.
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
The folded matcher (`fold_path_part` via `_is_relative_sensitive_path`) now decides what git_diff and git_status may name. The existing pathspecs stay ahead of it unchanged, so ASCII argv and output are byte-identical.

- Root cause, measured on git 2.39.5 (Apple) with names written byte-exact through `hash-object` + `update-index --index-info`: `icase` folds ASCII only. With `core.precomposeunicode=true` (`git init`'s macOS default), git precomposes argv. An NFD-stored name therefore leaked under BOTH the NFC and the NFD rule, and a mixed name leaked even under its own exact bytes. No literal pathspec can name either. TASK-37655's test only covered precompose off.
- `git_status`: `_parse_status_porcelain_v2` drops every `-z` entry the matcher denies, before counting, so the truncation note cannot reveal a withheld entry either. Every rendered path is checked, with no extra git call and no race.
- `git_diff`: `_withheld_names` runs `diff --name-only --no-renames -z` with the same mode and pathspecs, plus positive `:(glob)` pathspecs that select names with a non-ASCII byte in any component. It returns the names the matcher denies. `_withheld_name_pathspecs` excludes them byte-exactly with `_exact_exclude_pathspec`: a `:(exclude,glob)` that writes each non-ASCII byte as a one-byte bracket. Wildmatch compares bytes, and the element is not valid UTF-8, so precomposition leaves it alone. Windows git does not precompose, so it gets `:(exclude,literal)`. A subtree match collapses to its stored directory prefix (`/**`).
- Fail closed: after the real diff the same scan runs again. Any denied name left means the file changed mid-call or git ignored the exclusion, so the output is dropped and the call refuses with a message that names nothing. A truncated scan (`_WITHHELD_SCAN_MAX_BYTES`, 8 MB), a failed scan, or added pathspecs over `_WITHHELD_PATHSPEC_BUDGET` (16,000 chars, inside Windows' 32K command line) also refuse.
- TOCTOU: the post-run scan catches a denied name that is still changed when it runs. The residual is a file that changes and reverts entirely inside one call, or a ref that moves between the diff and the re-scan.
- `_denylist_pathspecs`: a non-ASCII rule that folds to ASCII (`Straße` -> `strasse`) also gets that spelling, because the scan never lists ASCII names. ASCII rules are unchanged. Its rule lookup moved into `_repo_exclusions`, so git_diff/git_status hand the same rules to the matcher; that helper is a named non-entry exemption in the choke-point tripwire.
- `_run_git_bytes`' timeout error now shows the argv only up to `--`. The pathspecs after it name excluded paths, now including the stored spellings this change adds.
- Other entry points: git_log is metadata only (unfiltered by design), git_blame checks its single path through the matcher, and git_branches names no paths. code review's `ReviewGitReader.changed` already re-checks each name with `safe_path`. The remote and pinned worker runs these same functions through the regenerated bundle.
- Tests (`Tests/Tools/test_git_tool_sensitive_paths.py`) use real temp repos with byte-exact names. Each case runs with precompose on and off and checks every diff mode (worktree, `--cached`, range, stat) plus git_status for both content and name. The 8 cases are non-ASCII case in both directions, case plus normalization, mixed name vs NFC and NFD rules, a directory rule, a name rule, and a rule that folds to ASCII. Also covered: the live-denylist path, the post-run refusal, both bounds, the timeout echo, and an ASCII argv byte-identical pin with an unmatched non-ASCII file present. 18 of the 23 new tests were red before the fix (17 at commit ca64a4c629, and the timeout-echo test when it was added). The 4 precompose-off pure-NFD cases already passed there, because the NFD rule puts case on ASCII base letters; their precompose-on twins were red. The ASCII pin is a regression guard. All green after.
- Files: `Tools/git_tool_impls.py`, `Tools/remote_worker_bundle.py` (regenerated), `Tests/Tools/test_git_tool_sensitive_paths.py` (new tests; the explicit-magic AST pin also accepts `_withheld_name_pathspecs`), `Tests/Tools/test_local_tool_sensitive_paths.py` (exemption), `Tests/Architecture/test_module_size_ratchet.py` (bundle row 5,722 -> 5,795, dated; dev's committed bundle was fresh), `backlog/docs/lessons-testing-evidence.md`.
- Verified 2026-11-03 on claude/bd-38463 (base origin/dev 335f984bd2) through the scratch wrapper, against a detached dev worktree. The branch and dev failed the same 22 ids with the same exceptions: 20 `RecoveryRequired: raw_source_selection_changed`, and 2 non-repo tests whose basetemp sits inside the main checkout's repo. Under a throwaway `bootstrap_profile` probe, the 20 ran 70/70 green on both trees. `Tests/Architecture` (serial) failed the same 43 ids on both trees (4044 passed, 5 skipped each); the size ratchet is green. Preflight passed; it has no remote-bundle step, and `Tests/Tools/test_remote_worker_bundle.py` covers drift. Ruff check on the touched source shows only three findings that dev already has (RUF100, BLE001, SIM102).

**Review round (PR review, 2026-11-03).** Four findings, one commit each. Every commit regenerates the bundle and sets its size-ratchet row.
- **Stored-byte brackets (dba8491c05).** `_exact_exclude_pathspec` encoded names with `os.fsencode`, but `_withheld_names` decodes them as UTF-8 with surrogateescape. Under a non-UTF-8 filesystem encoding, `é` became `[\xe9]`, so the exclusion missed, the post-run scan found the name, and every git_diff was refused. The brackets now come from `char.encode("utf-8", "surrogateescape")`, the inverse of the decode. No other site encodes names for a pathspec: `_denylist_pathspecs` passes rule strings through argv, which the filesystem encoding round-trips. RED: `test_a_byte_exact_exclusion_re_encodes_names_as_they_were_decoded` swaps the module's `os` for a Latin-1 `fsencode` and got `[\udce9]`.
- **Neutral refusal (f96a69a090).** Three refusals now raise one sentence, `git diff output withheld: could not verify path exclusions`: a truncated scan, the pathspec budget, and the post-run re-scan. The old wording announced that excluded paths existed, and its "retry" invited a loop on a refusal that can repeat every time. The bound and post-run tests pin the exact text; 3 were RED before.
- **Scope-limited scan (3013846380).** git ORs positive pathspecs, so a path-scoped scan also listed every non-ASCII name in the repository. Withheld names outside the scope then spent the 8 MB and 16 KB bounds. A scoped scan now passes only the diff's own pathspecs. That is every name under the scope, so the scan stays fail-closed there. RED: `test_a_scoped_git_diff_ignores_withheld_names_outside_its_scope` was refused under both bounds. `test_a_scoped_git_diff_still_withholds_inside_its_scope` guards the fail-closed side and was green before and after.
- **One rule snapshot (b031b25b17).** git_status and git_diff called `_repo_exclusions` twice per call. They now call it once and pass the result to both `_denylist_pathspecs(exclusions=...)` and the folded matcher. RED: `test_git_status_and_git_diff_read_the_exclusion_rules_once` counted 2 lookups.
- **Bundle and ratchet.** The bundle grew from 5,795 to 5,810 lines, an exact fresh build. Its ratchet row is set to that, with a dated comment.
- **Evidence.**
  - Passed: `Tests/Tools/test_git_tool_sensitive_paths.py` 60; `Tests/Tools/test_remote_worker_bundle.py` 16; the size ratchet 201 (serial); preflight.
  - Failed, environmental: `Tests/Tools/test_git_tool_impls.py` still fails the same 2 non-repo tests, because the scratch basetemp sits inside the main checkout's repository.
  - ruff: no new findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
git_diff and git_status now withhold a stored name whenever the folded matcher denies it: non-ASCII case either way, NFD or mixed normalization, with core.precomposeunicode on or off. git_status filters its parsed entries. git_diff excludes scanned names with byte-exact pathspecs and refuses if a re-scan after the run still finds one. ASCII argv is unchanged.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
