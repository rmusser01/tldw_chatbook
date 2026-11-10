---
id: TASK-39126
title: "The Architecture guards job's runtime-policy owner node parses all 9,001 repository Python files in one node, and its CI call takes up to 113 s against a 180 s per-node timeout"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-07 02:03
updated_date: 2026-11-10 19:20
labels:
- follow-up
- ci
- tests
- performance
priority: low
references:
- Tests/Architecture/test_profile_owned_path_inventory.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-39054
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and in the notes of TASK-39054 and TASK-39060 (round-11 branch fu/r11-arch). Line numbers are dev's.

**The node.**
- The Architecture guards job runs `Tests/Architecture/test_profile_owned_path_inventory.py::test_compatibility_and_runtime_policy_constants_have_no_new_runtime_owners` with `--timeout=180` (`derived-artifacts.yml:1240`, `:1247`).
- The node (`test_profile_owned_path_inventory.py:312`) walks `REPO_ROOT.rglob("*.py")` and parses each file: 9,001 files, 214.5 MB of source.

**Measured times.** CI reports per-node times, because `pyproject.toml:662` sets `--durations=25 --durations-min=1.0`. The node's own call time:

| Run (2026-11-06) | The node's call | Job total |
|---|---|---|
| 42310788537 | 113.30 s | 404.3 s |
| 42309351181 | 112.74 s | 400.2 s |
| 42306963606 | 61.90 s | 243.1 s |

- The slowest is 113 s against a 180 s timeout, only 1.6x headroom.

- Locally, per TASK-39054's and TASK-39060's notes:
  - the node's call took 177.4 s at load about 79 on base, and 332.9 s alone at load 65-106 on the branch;
  - it timed out in a review's job run (load 84-91) and passed at load 24-57.
- A slower CI runner would fail the Architecture job on this node, not on a real regression.

**Not part of this.** The `test_persistent_diagnostic_inventory` task_15103 ledger nodes, also named in this follow-up, run on no PR lane. They read frozen revisions a shallow checkout cannot serve, and timed out only in local runs, so they do not threaten the job.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The node's own call time is measured on CI and locally, with load recorded
- [ ] #2 The node keeps at least 2x headroom under its per-node timeout across three CI job runs, recorded
- [x] #3 Negative controls: a new `DEFAULT_RUNTIME_POLICY_PATH` assignment and a new base-data-dir CLI consumer each still turn it red
<!-- AC:END -->

## Implementation Plan

1. Measure the node's call time locally before the change, with load.
2. Parse only files whose text can name a guarded constant: read every file outside `Docs/` as before, and parse it only if its text (NFKC-normalized when not ASCII, because the parser folds identifiers that way) contains `BASE_DATA_DIR_CLI` or `DEFAULT_RUNTIME_POLICY_PATH`. Neither finding can come from a file that does not spell one of them, so the findings are unchanged. Pin the prefilter in the already-laned `test_base_data_dir_census_detects_qualified_and_aliased_consumers` (a full-width spelling must still be admitted), so no lane or pin list changes.
3. Negative controls: a temporary new `DEFAULT_RUNTIME_POLICY_PATH` assignment and a temporary new `BASE_DATA_DIR_CLI` consumer each turn the node red; reverted.
4. Measure the node's call time locally after the change, with load; CI times are collected by the coordinator after merge.

## Implementation Notes

**Approach.** The node still reads every `*.py` file outside `Docs/`, but parses only a file whose text names `BASE_DATA_DIR_CLI` or `DEFAULT_RUNTIME_POLICY_PATH` (`_may_name_runtime_owner`). Both findings are AST nodes whose identifier is one of those two names (a load of the constant, a qualified `.BASE_DATA_DIR_CLI`, an `import ... BASE_DATA_DIR_CLI as x` that makes `x` count, or an assignment target), so a file that never spells either name cannot hold a finding. The parser NFKC-normalizes identifiers, so non-ASCII text is NFKC-normalized before the check (1.45 s for the 2,941 non-ASCII files). On the current tree 7 of 9,053 files are parsed.

**The parse side effect, and how it is covered now.** Parsing every file also failed this node on any unparseable `.py` outside `Docs/`. With the prefilter, a file that names neither constant is no longer parsed (it still cannot hide a finding: a file naming either constant is still parsed and still raises). Review r13a-rev1 found that no other required PR-time check parses the non-package roots: preflight's profile census covers only `tldw_chatbook/`, pytest collection only the `Tests/` files a lane runs, and `lint.yml` is not required. A new node in the same file and job, `test_python_outside_the_package_and_tests_parses`, parses every git-tracked `.py` (`git ls-files -z -- '*.py'`, which reads the index and works in CI's shallow checkout) outside `tldw_chatbook/`, `Tests/` and `Docs/`: 134 files (scripts/ 37, Helper_Scripts/ 33, Packaging/ 27, packages/ 24, .github/ 6, native/ 4, qa/ 1, two top-level files). Untracked local folders (build/, venv/, node_modules/) are not repository source and are not walked (review r13a-rev2: a filesystem walk took ~45 s over 9,876 files in the main checkout). It parses bytes, so a UTF-8 BOM or a PEP 263 cookie parses as Python reads it (a BOM file and a `latin-1` cookie file both fail when read as UTF-8 text). The failure message names every unparseable file, one per line. Its call took 0.60-0.92 s at load 41-48. Plants in this worktree, then removed and unstaged: two broken files left untracked -> passed (not repository source); the same two `git add -N`'d -> red, naming both (`Helper_Scripts/_r13a_broken_b.py:1: '(' was never closed` and `scripts/_r13a_broken_a.py:1: invalid syntax`); a git-added BOM file and `latin-1` cookie file -> passed. These ran in this worktree's index, not a scratch copy, because `run.sh` accepts only `fu-*` worktrees. Still not parse-checked by any PR-time check: `Tests/` files that no lane runs (the old node parsed them as a side effect).

**Pin.** The already-laned `test_base_data_dir_census_detects_qualified_and_aliased_consumers` now also asserts that the prefilter admits its source and a full-width spelling (`\uff22ASE_DATA_DIR_CLI`, which the parser reads as `BASE_DATA_DIR_CLI`), and skips a file naming neither. Red before the NFKC line (temporary removal): `assert _may_name_runtime_owner(full_width)` -> `assert False`; green with it. The new syntax node is laned by node id in `derived-artifacts.yml`'s architecture-guards step and in `ARCHITECTURE_GUARD_TARGETS` (`Tests/CI/test_ci_queue_pressure_contract.py`, whose completeness pin requires every test in this file to be listed).

**Local measurements** (the node inside the job's exact command, serial):

| Run | Code | Load (start -> end) | Node's call |
|---|---|---|---|
| base1 | dev ae9f9e8d21 | 72 -> 57 | 180.20 s, `Failed: Timeout (>180.0s)` |
| base2 | dev ae9f9e8d21 | 25 -> 40 | 172.71 s (passed) |
| final1 / final2 / final3 | this branch | 65 -> 37 / 37 -> 31 / 31 -> 28 | 8.58 s / 4.22 s / 5.48 s |

Alone at load about 40-45: 10.6-12.8 s per run (the controls below).

**AC #1 and #2 are left open:** CI times are collected by the coordinator after merge. Expect the node well under 90 s; locally it went from 172.7-180 s to 4.2-8.6 s at higher load than base2's.

**Negative controls** (each a temporary `scripts/_r13a_planted.py`, removed): no plant -> 1 passed; `DEFAULT_RUNTIME_POLICY_PATH = Path("x")` -> red (`['scripts/_r1...bootstrap.py'] == ['tldw_chatbo...bootstrap.py']`); `from tldw_chatbook.config import BASE_DATA_DIR_CLI as data_dir` + a use -> red (`[('scripts/_r...anted.py', 2)] == []`); `import tldw_chatbook.config as cfg` + `cfg.BASE_DATA_DIR_CLI` -> red; a full-width `\uff24EFAULT_RUNTIME_POLICY_PATH = 1` -> red.

**Files.** `Tests/Architecture/test_profile_owned_path_inventory.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
