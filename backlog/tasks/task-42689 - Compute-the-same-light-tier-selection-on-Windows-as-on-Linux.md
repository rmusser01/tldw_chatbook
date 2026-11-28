---
id: TASK-42689
title: Compute the same light-tier selection on Windows as on Linux
status: Done
assignee:
  - '@claude'
created_date: '2026-11-27 18:58'
updated_date: '2026-11-27 19:45'
labels:
  - ci
dependencies:
  - TASK-42521
references:
  - scripts/ci/tier_policy.py
  - Tests/CI/test_tiered_ci.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`scripts/ci/tier_policy.py` chooses what the required light job runs for a dev PR. `backlog/docs/lessons-testing-evidence.md` (TASK-33621.197) tells authors to run it locally before arming a PR, and the owner and several parallel sessions work on Windows. There the same call returns, without any error, a plan with no importers and no readers, so an author runs too few files and learns the rest from CI. On dev c0a999302f (2026-11-27, Windows 10, Python 3.12.10) `Tests/CI/test_tiered_ci.py` fails 53 of its 157 nodes on Windows and none on Linux CI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 On a Windows checkout, `select_changes` returns the plan CI's Linux job printed for the same tree and change list, shown on real PR runs
- [x] #2 `Tests/CI/test_tiered_ci.py` passes on Windows with no platform skip
- [x] #3 Tests that supply Windows' path separator and default text encoding themselves, on any host, go red when either reaches the policy's paths or text again
- [x] #4 For real change lists the plan is unchanged where paths are POSIX and reads are UTF-8, as on CI's Linux
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the 53 failures on dev. Change the two `str(f.relative_to(root))` index keys to `as_posix()` and count again.
2. Diagnose the nodes still red. Fix a separator cause in the policy or the test; skip a node only if it needs something Windows does not have.
3. Check the other platform default the policy takes, the text encoding, against the real tree.
4. Pin the fixes with a Windows stand-in that runs on Linux, since no CI lane runs `Tests/CI` on Windows.
5. Compare Windows plans with the plans CI printed for real PR runs.
6. Run preflight, open a draft PR, and read Linux from the PR's own full run (`scripts/ci/` is a full-run prefix).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
On Windows the selector now returns the plan CI's Linux job prints. Two platform defaults had reached it.

**Path separators.** `select_changes` keyed its index of `Tests/` by `str(f.relative_to(root))`, which has backslashes on Windows, while every path it is compared with comes from git. It now uses `.as_posix()`. `_importers` built a package prefix from an f-string of `Path(current).parent` (`Tests\Pkg/` on Windows), so a package `__init__.py` lost its own files as dependents; that node's cause was not known when this task was filed. The untracked-fallback listing had the same `str()`. Its consumers read only file and folder names, so reverting it turns nothing red; it is changed so every path string in the policy has one form.

**Text encoding.** The four reads (workflow, UI census, `Tests/` corpus, changed application sources) used the locale's encoding, cp1252 on Windows. No result differs on today's tree: for all 6,438 tracked `Tests/` paths the importers, readers, precise readers and folder readers are the same under both decodings, and so are the lane tables and the schema verdicts of 3,044 modules. But 1,834 of the 1,849 test files with non-ASCII bytes tokenise differently under cp1252, so a name written next to a smart quote would have diverged with no error. All four reads go through `_read` (UTF-8). This was not part of the original request, so it has its own commit.

**Tests.** `test_diff_preserves_deletions_and_odd_filenames` needed no platform skip. Windows cannot create a file name with a newline, but the diff under test reads commits, so the test now commits its rename as git objects (`git mktree -z`, `git commit-tree`) with its assertion unchanged. `_WindowsCheckout` is a root whose `relative_to` returns a `PureWindowsPath` and whose `read_text` defaults to cp1252. Two tests use it, because no CI lane runs `Tests/CI` on Windows and Linux cannot show either default.

**Evidence** (Windows 10, Python 3.12.10; branch on dev 585d8ab46a, measured 2026-11-27).

- AC 2: `Tests/CI/test_tiered_ci.py` was 53 failed, 104 passed on dev c0a999302f. It is now 159 passed, none skipped (two new nodes). The whole `Tests/CI` folder is 2,067 passed, 16 skipped and 1 failed, the symlink node below.
- AC 1: for PR runs 46048726684, 46042456647, 46067447397, 46079802490 and 46074517470 the merge tree rebuilt with `git merge-tree --write-tree` has the tree id of CI's merge commit, and Windows returns the plan the "CI preflight" job printed (same `full`, `targets` and `reason`). Before the fix the first two got 24 of CI's 51 targets and 9 of its 22.
- AC 3: reverting the corpus keys turns 53 nodes red, the package prefix 2, and any one read `test_a_windows_checkout_reads_the_tree_as_utf8`. The encoding runs used Python's UTF-8 mode, where the default read is UTF-8 as on Linux, so that red comes from the stand-in. The path runs were on a Windows host only; there the stand-in and the host agree, so they do not show the stand-in working alone.
- AC 4: a Linux stand-in on Windows (POSIX path flavour, UTF-8 reads) returns CI's plan for the same five runs with the old policy and with the new one. The real Linux check is the PR's own full run. Its result is recorded in the PR, since recording it here would cost another push and full run.

Not addressed: `Tests/CI/test_canvas_mermaid_asset_checker.py::test_a_symlinked_copied_output_fails_the_pin` still needs the Windows symlink privilege (`WinError 1314`).

Files: `scripts/ci/tier_policy.py`, `Tests/CI/test_tiered_ci.py`, `scripts/ci/README.md`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
